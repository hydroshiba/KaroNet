class Minimax:
	def __init__(self, depth, evaluation=None, pruning=False):
		self.evaluation = evaluation
		self.pruning = pruning
		self.depth = depth
	
	def minimax(self, board, depth, player, alpha=float('-inf'), beta=float('inf')):
		if board.terminal():
			return board.evaluate() * player

		if depth <= 0:
			if self.evaluation is not None: return self.evaluation.evaluate(board.board * player)
			else: return 0  # Neutral evaluation at leaf nodes
		
		eval = float('-inf')
		
		for move in board.legal_moves():
			board.make_move(move, player)
			eval = max(eval, -self.minimax(board, depth - 1, -player, -beta, -alpha))
			if self.pruning:
				alpha = max(alpha, eval)
				if alpha >= beta:
					board.undo_move(move)
					break
			board.undo_move(move)

		return eval

	def play(self, board, player):
		if self.depth <= 0:
			if self.evaluation is not None:
				moves = board.legal_moves()
				return moves[self.evaluation.policy(board.board * player)[moves].argmax()]
			else: return board.legal_moves()[0]  # Arbitrary legal move
    
		eval = float('-inf')
		best_move = None
		
		alpha = float('-inf')
		beta = float('inf')
		
		for move in board.legal_moves():
			board.make_move(move, player)
			if self.pruning: move_eval = -self.minimax(board, self.depth - 1, -player, -beta, -alpha)
			else: move_eval = -self.minimax(board, self.depth - 1, -player)
			board.undo_move(move)
   
			if move_eval > eval:
				eval = move_eval
				best_move = move

			if self.pruning:
				alpha = max(alpha, eval)
		
		return best_move