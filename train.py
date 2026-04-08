import torch
import copy
import os
import csv
import datetime
import random
import numpy as np

from tqdm import tqdm
from torch import nn, optim
from component import board, architecture, agent, trainer, simulator, loss

class RandomAgent:
	def play(self, board, player):
		legal = board.legal_moves()
		return legal[torch.randint(len(legal), (1,)).item()]

def _get_moves(agent, boards, active, player):
	"""Get moves for all active boards. Batches inference for Neural agents."""
	if hasattr(agent, 'model') and hasattr(agent, 'device'):
		states = torch.stack([boards[i].board * player for i in active]).to(agent.device)
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

def evaluate(gameboard, agent1, agent2, rounds=1000, k=3):
	half = rounds // 2
	wins = [0, 0]
	draws = [0, 0]

	pbar = tqdm(total=rounds)
	for config in range(2):
		agent_x = agent1 if config == 0 else agent2
		agent_o = agent2 if config == 0 else agent1

		boards = [type(gameboard)() for _ in range(half)]

		# Play k random moves on each board
		for board_ in boards:
			for i in range(k):
				if board_.terminal(): break
				player = 1 if i % 2 == 0 else -1
				legal = board_.legal_moves()
				board_.make_move(legal[torch.randint(len(legal), (1,)).item()], player)

		active = [idx for idx in range(half) if not boards[idx].terminal()]
		step = k

		while active:
			player = 1 if step % 2 == 0 else -1
			current_agent = agent_x if player == 1 else agent_o
			moves = _get_moves(current_agent, boards, active, player)

			still_active = []
			for idx in active:
				boards[idx].make_move(moves[idx], player)
				if boards[idx].terminal():
					result = boards[idx].evaluate()
					if config == 0:
						if result == 1: wins[0] += 1
						elif result == 0: draws[0] += 1
					else:
						if result == -1: wins[1] += 1
						elif result == 0: draws[1] += 1
				else:
					still_active.append(idx)

			pbar.update(len(active) - len(still_active))
			active = still_active
			step += 1

	pbar.close()
	losses = [half - wins[0] - draws[0], half - wins[1] - draws[1]]
	print(f"{'':10} {'as X':>6} {'as O':>6} {'Percentage':>10}")
	print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6}" + f" {(wins[0] + wins[1]) / rounds * 100:>9.2f}%")
	print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}" + f" {(draws[0] + draws[1]) / rounds * 100:>9.2f}%")
	print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}" + f" {(losses[0] + losses[1]) / rounds * 100:>9.2f}%")

	return (wins, draws, losses)

# Lol
def ordinal(n):
	return f"{n}{'th' if 11 <= n % 100 <= 13 else ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"

def compute_elos(snapshot_perf, num_stages):
	# Minorize-Maximization for Bradley-Terry MLE
	wins = np.ones((num_stages, num_stages)) * 0.5  # Laplace smoothing prior
	games = np.ones((num_stages, num_stages)) * 1.0
	np.fill_diagonal(wins, 0)
	np.fill_diagonal(games, 0)

	for idx, perf in enumerate(snapshot_perf):
		stage1 = idx + 1
		for stage2, (w, d, l) in perf.items():
			wins[stage1, stage2] += w + (d / 2.0)
			wins[stage2, stage1] += l + (d / 2.0)
			games[stage1, stage2] += w + d + l
			games[stage2, stage1] += w + d + l

	W_total = np.sum(wins, axis=1)
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

if __name__ == "__main__":
	model = architecture.GomokuNet()
	eval_rounds = 1000

	# Play games and train	
	snapshots = []
	snapshot_perf = []
	snapshot_elos = []
	games = []
	round_schedule = [256 for i in range(256)]

	epochs = 10
	buffer_size = 131072 # Virtually unlimited
	train_size = 2048
	optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer, T_0=32, T_mult=1, eta_min=1e-5
	)
 
	mcts_sims = 100.0
	mcts_sims_gamma = 1.01089
	mcts_sims_max = 1600.0
 
	sim = simulator.MCTS(temperature=1.0, parallel=256, dirichlet_alpha=0.03, dirichlet_frac=0.25, temp_decay=0.95, temp_min=0.05)
	trn = trainer.MonteCarlo(optimizer, loss.AlphaZero(), loss_freq=1)
	
	latest_elos = [1000.0]
	
	# Save snapshots along the way
	path = f"model/{model.__class__.__name__}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/"
	if not os.path.exists(path):
		os.makedirs(path)

	print("Training the model with self-play")
	print(f"Model architecture: {model.__class__.__name__}")
	print(f"Self-play method: {sim.__class__.__name__}")
	print(f"Training method: {trn.__class__.__name__}")
	print(f"Loss function: {trn.loss_fn.__class__.__name__}")
	print("=" * 97)

	for i, rounds in enumerate(round_schedule):
		# Save snapshot before training to evaluate against it later and track progress
		snapshots.append(copy.deepcopy(model))
		
		# Define top 20 Elo queue
		queue_indices = sorted(range(len(snapshots)), key=lambda j: latest_elos[j], reverse=True)[:20]
		
		# Self-play and training
		print(f"Stage {i + 1}: Simulating {rounds} games...")
		if i == 0:
			opponent_idx = 0
			print("Opponent: Itself (first iteration)")
		else:
			valid_indices = [j for j in queue_indices if j in snapshot_perf[-1]]
			if valid_indices:
				opponent_idx = max(valid_indices, key=lambda j: (snapshot_perf[-1][j][2], j))
				losses = snapshot_perf[-1][opponent_idx][2]
			else:
				opponent_idx = random.choice(queue_indices)
				losses = "?"
			elo = latest_elos[opponent_idx]
			print(f"Opponent: {ordinal(opponent_idx)} snapshot with Elo {elo:.2f} (Losses: {losses})")

		current = agent.Neural(model)
		opponent = agent.Neural(snapshots[opponent_idx])

		current_games = sim.simulate(board.Board(), agent1 = current, agent2 = opponent, rounds=rounds, simulations=int(mcts_sims))
		games.extend(current_games)
		games = games[-buffer_size:]

		new_games = current_games
		old_games = games[:-len(new_games)]
		old_samples = min(len(old_games), train_size - len(new_games))
		if old_samples > 0: old_games = random.sample(old_games, old_samples)
		else: old_games = []

		train_games = new_games + old_games
		print(f"Training on {len(new_games)} newly generated games and {len(old_games)} old games sampled from buffer...")
		trn.train(model, train_games, epochs=epochs, batch_size=128)
		
		# Evaluate against snapshots
		print("Evaluating against previous snapshots opening with first 4 random moves:")
		performance = {}

		eval_queue = sorted(list(set(queue_indices + [len(snapshots)])))
		
		for stage in eval_queue:
			print(f"{ordinal(i + 1)} stage vs {ordinal(stage)} snapshot:" if stage < len(snapshots) else f"{ordinal(i + 1)} stage vs Itself:")
			opponent = snapshots[stage] if stage < len(snapshots) else model
			wins, draws, losses = evaluate(board.Board(), agent.Neural(model), agent.Neural(opponent), rounds=eval_rounds, k=4)
			w, d, l = sum(wins), sum(draws), sum(losses)
			performance[stage] = (w, d, l)

		snapshot_perf.append(performance)
		
		# Compute Elo values after evaluations
		latest_elos = compute_elos(snapshot_perf, len(snapshots) + 1)
		snapshot_elos.append(latest_elos)
		print(f"{ordinal(i + 1)} stage Elo rating: {latest_elos[-1]:.2f}")
		
		scheduler.step()
		print("=" * 97)
  
		# Save and update stuff
		mcts_sims = min(mcts_sims * mcts_sims_gamma, mcts_sims_max)
		torch.save(model.state_dict(), path + f"snapshot_{i + 1}.pth")
	print()
 
	# Save game W/D/L data for analysis
	with open(path + "log.csv", "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerow(["stage", "opponent_stage", "wins", "draws", "losses", "elo_stage", "elo_opponent_stage"])
		for idx, (perf, elos) in enumerate(zip(snapshot_perf, snapshot_elos)):
			for op_stage, (wins, draws, losses) in perf.items():
				writer.writerow([idx + 1, op_stage + 1, wins, draws, losses, elos[idx + 1], elos[op_stage]])
    
	print(f"Model and snapshots saved to {path}")