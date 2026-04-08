import torch
import torch.nn.functional as F
from tqdm import tqdm

class EpsilonGreedy:
	def __init__(self, max_epsilon=0.1, min_epsilon=0.1, decay=1.0):
		self.epsilon = max_epsilon
		self.min_epsilon = min_epsilon
		self.decay = decay

	def simulate(self, board, agent, rounds=1000):
		num_cells = len(board.board)
		device = agent.device
		model = agent.model

		boards = [type(board)() for _ in range(rounds)]
		players = [1] * rounds
		logs = [[] for _ in range(rounds)]
		active = list(range(rounds))
		games = []

		pbar = tqdm(total=rounds)
		while active:
			rand_vals = torch.rand(len(active))
			moves = {}
			neural_indices = []

			# Random moves
			for j, idx in enumerate(active):
				if rand_vals[j] < self.epsilon:
					legal = boards[idx].legal_moves()
					moves[idx] = legal[torch.randint(len(legal), (1,)).item()]
				else:
					neural_indices.append(idx)

			# Batched neural inference
			if neural_indices:
				states = torch.stack([boards[idx].board * players[idx] for idx in neural_indices]).to(device)
				with torch.no_grad():
					pols, _ = model(states)
				pols[states != 0] = float('-inf')
				best_moves = pols.argmax(dim=1).cpu().tolist()
				for j, idx in enumerate(neural_indices):
					moves[idx] = best_moves[j]

			# Apply moves, log, check terminal
			still_active = []
			completed = 0
			for idx in active:
				move = moves[idx]
				policy = F.one_hot(torch.tensor(move), num_classes=num_cells).float()
				logs[idx].append((boards[idx].board.clone(), players[idx], move, policy))
				boards[idx].make_move(move, players[idx])
				players[idx] *= -1

				if boards[idx].terminal():
					result = boards[idx].evaluate()
					games.append([(s, p, m, pol, result) for s, p, m, pol in logs[idx]])
					completed += 1
				else:
					still_active.append(idx)

			pbar.update(completed)
			active = still_active

		pbar.close()
		self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
		return games