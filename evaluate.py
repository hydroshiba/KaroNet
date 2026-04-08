import torch
import copy
import os
import csv
import datetime
import random
import numpy as np

# Force PyTorch to let weaker GPUs compile (Bypass <68 SMs hardcoded limit)
import torch._inductor.utils
torch._inductor.utils.is_big_gpu = lambda *args: True

import torch._inductor.config
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

from tqdm import tqdm
from torch import nn, optim
from component import board, architecture, agent, trainer, simulator, loss
from concurrent.futures import ProcessPoolExecutor, as_completed

state_dicts_global = []
_worker_cache = {}

def eval_pair(args):
	# Crucial for PyTorch multiprocessing performance to avoid thread contention deadlocks
	torch.set_num_threads(1)
	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	
	from torch.nn.utils import fuse_conv_bn_eval
	
	i, j, name_i, name_j, rounds_count, model_path, arch_name = args
	device = torch.device('cuda')
	ArchClass = getattr(architecture, arch_name)

	if i not in _worker_cache:
		net = ArchClass().to(device).half().eval()
		if name_i != "untrained":
			# Load onto CPU first perfectly prevents PCI-E SSD transfer locks stalling other workers' GPU loops!
			cpu_state = torch.load(os.path.join(model_path, f'{name_i}.pth'), map_location='cpu')
			net.load_state_dict(cpu_state)
		# Fuse BatchNorm for ~11% speedup if architecture has it
		if hasattr(net, 'backbone') and isinstance(net.backbone, nn.Sequential):
			for k in range(len(net.backbone) - 1):
				if isinstance(net.backbone[k], nn.Conv2d) and isinstance(net.backbone[k+1], nn.BatchNorm2d):
					net.backbone[k] = fuse_conv_bn_eval(net.backbone[k], net.backbone[k+1])
					net.backbone[k+1] = nn.Identity()
		_worker_cache[i] = torch.compile(net, mode='max-autotune', dynamic=True)

	if j not in _worker_cache:
		net = ArchClass().to(device).half().eval()
		if name_j != "untrained":
			cpu_state = torch.load(os.path.join(model_path, f'{name_j}.pth'), map_location='cpu')
			net.load_state_dict(cpu_state)
		# Fuse BatchNorm for ~11% speedup if architecture has it
		if hasattr(net, 'backbone') and isinstance(net.backbone, nn.Sequential):
			for k in range(len(net.backbone) - 1):
				if isinstance(net.backbone[k], nn.Conv2d) and isinstance(net.backbone[k+1], nn.BatchNorm2d):
					net.backbone[k] = fuse_conv_bn_eval(net.backbone[k], net.backbone[k+1])
					net.backbone[k+1] = nn.Identity()
		_worker_cache[j] = torch.compile(net, mode='max-autotune', dynamic=True)

	net_1 = _worker_cache[i]
	net_2 = _worker_cache[j]
	wins, draws, losses = evaluate(board.Board(), net_1, net_2, rounds=rounds_count)
	return i, j, (wins, draws, losses)

def _get_moves(agent, boards, active, player):
	"""Get moves for all active boards. Batches inference for Neural agents."""
	if hasattr(agent, 'model') and hasattr(agent, 'device'):
		states_np = np.array([boards[i]._board_1d for i in active])
		states = torch.from_numpy(states_np).to(agent.device, non_blocking=True).mul_(player)
		with torch.no_grad():
			pols, _ = agent.model(states)
		pols[states != 0] = float('-inf')
		if agent.temperature == 0.0:
			moves = pols.argmax(dim=1).cpu().tolist()
		else:
			probs = torch.softmax(pols / agent.temperature, dim=1)
			moves = torch.multinomial(probs, 1).squeeze(1).cpu().tolist()
		return {idx: moves[j] for j, idx in enumerate(active)}
	return {idx: agent.play(boards[idx], player) for idx in active}

def evaluate(gameboard, net1, net2, rounds=1000, k=4):
	device = torch.device('cuda')
	
	half = rounds // 2
	total = half * 2
	
	win_filters = torch.zeros(4, 1, 5, 5, device=device, dtype=torch.float16)
	win_filters[0, 0, 2, :] = 1
	win_filters[1, 0, :, 2] = 1
	for i in range(5): win_filters[2, 0, i, i] = 1
	for i in range(5): win_filters[3, 0, i, 4-i] = 1
	
	states = torch.zeros(total, 225, dtype=torch.float16, device=device)
	active = torch.ones(total, dtype=torch.bool, device=device)
	winners = torch.zeros(total, dtype=torch.long, device=device)
	
	for i in range(k):
		player = 1 if i % 2 == 0 else -1
		empty = (states == 0)
		rand_weights = torch.rand(total, 225, device=device)
		rand_weights[~empty] = -1
		moves = rand_weights.argmax(dim=1)
		states[torch.arange(total), moves] = player

	step = k

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.allow_tf32 = True
	torch.backends.cuda.matmul.allow_tf32 = True
	
	net1_mask_p1 = torch.cat([torch.ones(half, dtype=torch.bool, device=device), torch.zeros(half, dtype=torch.bool, device=device)])
	net1_mask_pmin1 = ~net1_mask_p1
	pols = torch.empty(total, 225, device=device, dtype=torch.float16)

	while active.any():
		player = 1 if step % 2 == 0 else -1
		net1_mask = net1_mask_p1 if player == 1 else net1_mask_pmin1
		net1_active = active & net1_mask
		net2_active = active & (~net1_mask)

		pols.fill_(float('-inf'))
		
		# Dynamic slicing perfectly shrinks the batch mathematically, avoiding unnecessary TFLOPS
		if net1_active.any():
			with torch.no_grad():
				out_1, _ = net1((states[net1_active] * player).view(-1, 1, 15, 15))
				pols[net1_active] = out_1
		if net2_active.any():
			with torch.no_grad():
				out_2, _ = net2((states[net2_active] * player).view(-1, 1, 15, 15))
				pols[net2_active] = out_2
				
		active_idx = active.nonzero(as_tuple=True)[0]
		
		states_valid = states[active_idx] == 0
		pols_active = pols[active_idx]
		pols_active[~states_valid] = float('-inf')
		
		moves = pols_active.argmax(dim=1)
		states[active_idx, moves] = player
		
		with torch.no_grad():
			player_masks = (states[active_idx].view(-1, 1, 15, 15) == player).to(torch.float16)
			conv_out = torch.nn.functional.conv2d(player_masks, win_filters, padding=2)
			wins = (conv_out == 5).reshape(len(active_idx), -1).any(dim=1)

		new_won_idx = active_idx[wins]
		winners[new_won_idx] = player
		active[new_won_idx] = False

		draws = (states[active_idx] == 0).sum(dim=1) == 0
		new_draw_idx = active_idx[draws]
		active[new_draw_idx] = False

		step += 1

	winners_c0 = winners[:half]
	winners_c1 = winners[half:]

	wins_x_all = (winners_c0 == 1).sum().item()
	wins_o_all = (winners_c1 == -1).sum().item()

	draws_all = (winners_c0 == 0).sum().item()
	draws_o_all = (winners_c1 == 0).sum().item()
	
	losses_x_all = (winners_c0 == -1).sum().item()
	losses_o_all = (winners_c1 == 1).sum().item()

	return ([wins_x_all, wins_o_all], [draws_all, draws_o_all], [losses_x_all, losses_o_all])

def compute_elos(snapshot_perf, num_stages):
	# Minorize-Maximization for Bradley-Terry MLE
	wins = np.zeros((num_stages, num_stages))
	games = np.zeros((num_stages, num_stages))

	for idx, perf in enumerate(snapshot_perf):
		stage1 = idx
		for stage2, (w, d, l) in perf.items():
			wt = w[0] + w[1]
			dt = d[0] + d[1]
			lt = l[0] + l[1]
			wins[stage1, stage2] += wt + (dt / 2.0)
			wins[stage2, stage1] += lt + (dt / 2.0)
			games[stage1, stage2] += wt + dt + lt
			games[stage2, stage1] += wt + dt + lt

	W_total = np.maximum(np.sum(wins, axis=1), 1e-8)
	gamma = np.ones(num_stages)

	for _ in range(1000):
		gamma_next = np.copy(gamma)
		for i in range(num_stages):
			denom = 0.0
			for j in range(num_stages):
				if i != j and games[i, j] > 0:
					denom += games[i, j] / (gamma[i] + gamma[j])
			if denom > 0:
				gamma_next[i] = W_total[i] / denom
		
		# Normalize against baseline (stage 0)
		gamma_next /= gamma_next[0]
		
		if np.max(np.abs(gamma_next - gamma)) < 1e-5:
			break
		gamma = gamma_next

	elos = 1000.0 + 400.0 * np.log10(gamma)
	return elos.tolist()

if __name__ == '__main__':
	import multiprocessing as mp
	import sys
	import os
	import re
	import argparse
	
	mp.set_start_method('spawn', force=True)
	
	parser = argparse.ArgumentParser(description="Evaluate GomokuNet models")
	parser.add_argument('--path', type=str, required=True, help="Path to the model folder containing .pth files")
	parser.add_argument('--rounds', type=int, default=50, help="Amount of rounds per pair")
	args = parser.parse_args()
	
	path = args.path
	rounds = args.rounds
	
	def natural_keys(text):
		return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

	pth_files = [f for f in os.listdir(path) if f.endswith('.pth')]
	pth_files.sort(key=natural_keys)
	
	if not pth_files:
		print("No .pth files found in the specified path.")
		sys.exit(1)
		
	# Detect architecture from the first snapshot
	sample_state = torch.load(os.path.join(path, pth_files[0]), map_location='cpu')
	detected_arch_name = None
	for name in dir(architecture):
		obj = getattr(architecture, name)
		if isinstance(obj, type) and issubclass(obj, nn.Module) and name != 'Module':
			try:
				net = obj()
				net.load_state_dict(sample_state, strict=True)
				detected_arch_name = name
				break
			except Exception:
				pass
				
	if not detected_arch_name:
		print("Could not detect architecture from the saved state dict!")
		sys.exit(1)
		
	print(f"Detected architecture: {detected_arch_name}")
	
	model_names = ["untrained"] + [f[:-4] for f in pth_files]
	num_snapshots = len(model_names)
	 
	snapshot_perf = [{} for _ in range(num_snapshots)]
	
	tasks = []
	for i in range(num_snapshots):
		for j in range(i):
			tasks.append((i, j, model_names[i], model_names[j], rounds, path, detected_arch_name))
			
	pbar = tqdm(total=len(tasks), desc="Evaluating snapshots")
	with mp.Pool(8) as pool:
		for res in pool.imap_unordered(eval_pair, tasks):
			i, j, res_tuple = res
			snapshot_perf[i][j] = res_tuple
			pbar.update(1)
	pbar.close()
	
	elos = compute_elos(snapshot_perf, num_snapshots)

	with open(os.path.join(path, "round_robin.csv"), "w", newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["A", "B", "A_Wins_as_X", "A_Draws_as_X", "A_Losses_as_X", "A_Wins_as_O", "A_Draws_as_O", "A_Losses_as_O"])
		for idx, perf in enumerate(snapshot_perf):
			for j in range(idx):
				wins, draws, losses = perf[j]
				writer.writerow([model_names[idx], model_names[j], wins[0], draws[0], losses[0], wins[1], draws[1], losses[1]])
	
	with open(os.path.join(path, "elo.csv"), "w", newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Snapshot", "Elo"])
		for idx, elo in enumerate(elos):
			writer.writerow([model_names[idx], elo])
