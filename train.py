import torch
import copy
import os
import gc
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

			active = still_active
			step += 1

	losses = [half - wins[0] - draws[0], half - wins[1] - draws[1]]
	# print(f"{'':10} {'as X':>6} {'as O':>6} {'Percentage':>10}")
	# print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6}" + f" {(wins[0] + wins[1]) / rounds * 100:>9.2f}%")
	# print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}" + f" {(draws[0] + draws[1]) / rounds * 100:>9.2f}%")
	# print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}" + f" {(losses[0] + losses[1]) / rounds * 100:>9.2f}%")

	return (wins, draws, losses)

# Lol
def ordinal(n):
	return f"{n}{'th' if 11 <= n % 100 <= 13 else ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"

def compute_elos(snapshot_perf, num_stages):
	# Minorize-Maximization for Bradley-Terry MLE
	wins = np.ones((num_stages, num_stages)) * 0.005  # Laplace smoothing prior
	games = np.ones((num_stages, num_stages)) * 0.01
	np.fill_diagonal(wins, 0)
	np.fill_diagonal(games, 0)

	for idx, perf in enumerate(snapshot_perf):
		stage1 = idx + 1
		for stage2, stats in perf.items():
			w, d, l = stats[0], stats[1], stats[2]
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

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
 
def augment_games(games):
	if not games:
		return []

	size = int(games[0][0][0].shape[0] ** 0.5)

	states, players, moves, policies, results = [], [], [], [], []
	game_boundaries = [0]
	
	for game in games:
		for state, player, move, policy, result in game:
			states.append(state)
			players.append(player)
			moves.append(move)
			policies.append(policy)
			results.append(result)
		game_boundaries.append(len(states))
		
	num_steps = len(states)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	states_t = torch.stack(states).to(device).view(num_steps, size, size)
	policies_t = torch.stack(policies).to(device).view(num_steps, size, size)
	moves_t = torch.tensor(moves, device=device)
	
	moves_y = moves_t // size
	moves_x = moves_t % size
	
	aug_states, aug_policies, aug_moves = [], [], []
	
	for rot in range(4):
		for flip in [False, True]:
			s_2d = torch.rot90(states_t, k=rot, dims=[1, 2])
			p_2d = torch.rot90(policies_t, k=rot, dims=[1, 2])
			
			m_y, m_x = moves_y.clone(), moves_x.clone()
			for _ in range(rot):
				m_y, m_x = size - 1 - m_x, m_y.clone()
				
			if flip:
				s_2d = torch.flip(s_2d, dims=[2])
				p_2d = torch.flip(p_2d, dims=[2])
				m_x = size - 1 - m_x
				
			aug_states.append(s_2d.reshape(num_steps, -1).cpu())
			aug_policies.append(p_2d.reshape(num_steps, -1).cpu())
			aug_moves.extend((m_y * size + m_x).cpu().tolist())
			
	cat_states = torch.cat(aug_states, dim=0)
	cat_policies = torch.cat(aug_policies, dim=0)
	
	# Since players and results are invariant, extend them 8x
	all_players = players * 8
	all_results = results * 8
	
	new_games = []
	for i in range(8):
		offset = i * num_steps
		for j in range(len(games)):
			start = offset + game_boundaries[j]
			end = offset + game_boundaries[j+1]
			
			game_list = [
				(
					cat_states[k],
					all_players[k],
					aug_moves[k],
					cat_policies[k],
					all_results[k]
				)
				for k in range(start, end)
			]
			new_games.append(game_list)
			
	return new_games

def increase_O(games, percentage = 0.25): # Should NEVER be used with DeepQ
	for game in games:
		# Filter positions from games where O won
		# An O win is identifiable by a negative evaluation value and the last move being O's
		if game[-1][1] == -1 and game[-1][-1] < 0:
			o_positions = [step for step in game if step[1] == -1]
			target_amount = int(len(game) * percentage)
			if target_amount > 0 and o_positions:
				if len(o_positions) >= target_amount:
					game.extend(random.sample(o_positions, target_amount))
				else:
					game.extend(random.choices(o_positions, k=target_amount))
	
	return games

if __name__ == "__main__":
    # Set fixed seed
	seed = 97
	seed_everything(seed)
	
	model = architecture.GomokuNet()
	eval_rounds = 1000

	# Play games and train	
	snapshots = []
	snapshot_perf = []
	snapshot_elos = []
	games = []
	round_schedule = [128 for i in range(1024)]

	epochs = 5
	buffer_size = 1048576
	train_size = 8192
	optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer, T_0=32, T_mult=1, eta_min=1e-5
	)

	mcts_sims = 200.0
	mcts_O_sims_ratio = 1.6
	mcts_sims_gamma = 1.021897
	mcts_sims_max = 1200.0
 
	trn = trainer.MonteCarlo(optimizer, loss.AlphaZero(policy_weight=0.5, value_weight=0.5), loss_freq=1)
	sim = simulator.MCTS(
    	temperature=1.0,
		temp_decay=0.98,
		temp_min=0.05,
    	parallel=128, wave_size=8,
    	dirichlet_alpha=0.3,
    	dirichlet_frac=0.25,
		value_blend_lambda=1.0,
		value_blend_decay=0.998876,
		value_blend_min=0.8
    )
	
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
		# Clear GPU cache to prevent OOM issues during self-play and training
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
    
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
			if random.randint(0, 19) == 0:
				# Occasionally evaluate against a random snapshot from past anchors for more diverse feedback
				indices = list(range(0, len(snapshots), 32))
				opponent_idx = random.choice(indices) if indices else 0
			else:
				valid_indices = [j for j in queue_indices if j in snapshot_perf[-1] and j != 0]
				if valid_indices:
					top_opponents = sorted(valid_indices, key=lambda j: snapshot_perf[-1][j][2], reverse=True)[:3]
					opponent_idx = random.choice(top_opponents)
					losses = snapshot_perf[-1][opponent_idx][2]
				else:
					valid_queue = [j for j in queue_indices if j != 0]
					opponent_idx = random.choice(valid_queue) if valid_queue else 0
					losses = "?"
			
			elo = latest_elos[opponent_idx]
			print(f"Opponent: {ordinal(opponent_idx)} snapshot with Elo {elo:.2f}")

		current = agent.Neural(model)
		opponent = agent.Neural(snapshots[opponent_idx])

		simulations = int(mcts_sims)
		current_games = sim.simulate(
    		board.Board(),
    		agent1 = current,
    		agent2 = opponent,
      		rounds = rounds,
    		X_sims = simulations,
			O_sims = int(simulations * mcts_O_sims_ratio)
		)

		print(f"{len(current_games)} self-played games statistics:")
		print(f"Longest game: {max(len(g) for g in current_games)} plies")
		print(f"Shortest game: {min(len(g) for g in current_games)} plies")
		print(f"Average game length: {sum(len(g) for g in current_games) / len(current_games):.2f} plies")
		print(f"X wins: {sum(1 for g in current_games if g[-1][-1] > 0)}")
		print(f"O wins: {sum(1 for g in current_games if g[-1][-1] < 0)}")
		print(f"Draws: {sum(1 for g in current_games if g[-1][-1] == 0)}")
		print('-' * 50)
		
		current_games = increase_O(current_games, 1.25) # More O player data to balance it out
		current_games = augment_games(current_games) # 8x augmentation
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
		performance = {}
		eval_queue = queue_indices + [len(snapshots)] + [i for i in range(0, len(snapshots), 32)]
		eval_queue = sorted(list(set(eval_queue)))  # Remove duplicates
		
		for stage in tqdm(eval_queue, desc="Evaluating against previous snapshots opening with first 4 random moves"):
			# print(f"{ordinal(i + 1)} stage vs {ordinal(stage)} snapshot:" if stage < len(snapshots) else f"{ordinal(i + 1)} stage vs Itself:")
			opponent = snapshots[stage] if stage < len(snapshots) else model
			wins, draws, losses = evaluate(board.Board(), agent.Neural(model), agent.Neural(opponent), rounds=eval_rounds, k=4)
			w, d, l = sum(wins), sum(draws), sum(losses)
			performance[stage] = (w, d, l, wins, draws, losses)

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

		# Save game W/D/L data for analysis
		with open(path + "log.csv", "w", newline='') as f:
			writer = csv.writer(f)
			writer.writerow(["stage", "opponent_stage", "wins_X", "wins_O", "draws_X", "draws_O", "losses_X", "losses_O", "elo_stage", "elo_opponent_stage"])
			for idx, (perf, elos) in enumerate(zip(snapshot_perf, snapshot_elos)):
				for op_stage, stats in perf.items():
					_, _, _, wins, draws, losses = stats
					writer.writerow([idx + 1, op_stage, wins[0], wins[1], draws[0], draws[1], losses[0], losses[1], elos[idx + 1], elos[op_stage]])

	print()
	print(f"Model and snapshots saved to {path}")

	# Call evaluate.py on the newly trained model folder
	import subprocess
	import sys
	print(f"Running evaluation on {path}...")
	subprocess.run([sys.executable, "evaluate.py", "--path", path, "--rounds", "100"])