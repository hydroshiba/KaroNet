import torch
import numpy as np

class Board:
	"""
	Board storing data natively as numpy so clone() is a pure numpy copy.

	self.board  = torch.from_numpy(self._board_1d)  — torch tensor is a view.
	self._board_np  — 2-D float32 numpy array (size×size), shared with board.
	self._board_1d  — flat view of _board_np, used for direct indexed writes.

	Empty-cell tracking with swap-and-pop (O(1) add/remove):
	  _empty_arr[0 .. _n_empty-1]  — currently empty cells
	  _pos_of[cell]                — index of cell in _empty_arr
	"""

	def __init__(self, size=15, win_length=5):
		self.size       = size
		self.win_length = win_length
		n               = size * size

		self._board_np  = np.zeros((size, size), dtype=np.float32)
		self._board_1d  = self._board_np.ravel()          # flat, shared memory
		self.board      = torch.from_numpy(self._board_1d)  # torch view (no copy)

		self._move_stack = []
		self._last_move  = -1   # mirrors _move_stack[-1]; avoids list.copy() in clone()
		self._n_empty    = n
		self._empty_arr  = np.arange(n, dtype=np.int32)
		self._pos_of     = np.arange(n, dtype=np.int32)

	def make_move(self, move, player):
		if self._board_1d[move] == 0:
			self._board_1d[move] = player
			pos                      = self._pos_of[move]
			n                        = self._n_empty - 1
			last                     = self._empty_arr[n]
			self._empty_arr[pos]     = last
			self._pos_of[last]       = pos
			self._n_empty            = n
			self._move_stack.append(move)
			self._last_move          = move
			return True
		return False

	def undo_move(self, move):
		self._board_1d[move]         = 0
		pos                          = self._n_empty
		self._empty_arr[pos]         = move
		self._pos_of[move]           = pos
		self._n_empty               += 1
		if self._move_stack and self._move_stack[-1] == move:
			self._move_stack.pop()
		self._last_move = self._move_stack[-1] if self._move_stack else -1

	def legal_moves(self):
		return self._empty_arr[:self._n_empty].tolist()

	def terminal(self):
		return self.evaluate() != 0 or self._n_empty == 0

	def evaluate(self):
		if self._last_move == -1:
			return 0
		move   = self._last_move
		r, c   = divmod(move, self.size)
		brd    = self._board_np
		player = brd[r, c]
		if player == 0:
			return 0
		size       = self.size
		win_length = self.win_length
		for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
			count = 1
			for sign in (1, -1):
				for k in range(1, win_length):
					nr = r + sign * dr * k
					nc = c + sign * dc * k
					if 0 <= nr < size and 0 <= nc < size and brd[nr, nc] == player:
						count += 1
					else:
						break
			if count >= win_length:
				return int(player)
		return 0

	def view(self):
		return self.board.view(self.size, self.size)

	def clone(self):
		new             = Board.__new__(Board)
		new.size        = self.size
		new.win_length  = self.win_length
		new._board_np   = self._board_np.copy()       # pure numpy copy, no .numpy() overhead
		new._board_1d   = new._board_np.ravel()
		new.board       = torch.from_numpy(new._board_1d)   # zero-copy torch view
		new._move_stack = []           # MCTS never calls undo_move on clones; skip list copy
		new._last_move  = self._last_move   # just an int — O(1)
		new._n_empty    = self._n_empty
		new._empty_arr  = self._empty_arr.copy()
		new._pos_of     = self._pos_of.copy()
		return new
