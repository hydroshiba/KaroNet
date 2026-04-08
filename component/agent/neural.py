import torch

class Neural:
	def __init__(self, model, device=None, temperature=0.0):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.device = device
		self.temperature = temperature
		self.model = model.to(self.device)
		self.model.eval()

	def play(self, board, player):
		move_values = self.model.policy((board.board * player).to(self.device)).cpu()
		legal_moves = board.legal_moves()
		if self.temperature == 0.0:
			best_move = [move for move in legal_moves if move_values[move] == move_values[legal_moves].max()][0]
			return best_move
		logits = move_values[legal_moves]
		probs = torch.softmax(logits / self.temperature, dim=0)
		return legal_moves[torch.multinomial(probs, 1).item()]