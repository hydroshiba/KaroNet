import torch
import math
import numpy as np
from tqdm import tqdm


class MCTSNode:
	"""
	board is lazily materialised — None until first selected.
	virtual_loss tracks in-flight simulations for wave batching.

	Lazy expansion: after neural net evaluation, priors are stored in two
	parallel arrays (_unexp_moves: list[int], _unexp_priors: np.ndarray)
	rather than creating child nodes. np.argmax on the priors array is a
	C-level operation, replacing the O(n) Python max() dict scan.
	Child MCTSNode objects are created on demand during _select.
	"""
	__slots__ = ('parent', 'move', 'player', 'children',
	             'W', 'N', 'prior', 'board', 'virtual_loss',
	             '_unexp_moves', '_unexp_priors', '_child_player')

	def __init__(self, parent, move, player, prior, board=None):
		self.parent         = parent
		self.move           = move
		self.player         = player
		self.children       = {}
		self.W              = 0.0
		self.N              = 0
		self.prior          = prior
		self.board          = board
		self.virtual_loss   = 0
		self._unexp_moves   = None   # np.ndarray int32 (slice view, swap-and-shrink)
		self._unexp_priors  = None   # np.ndarray float32
		self._child_player  = 0


class MCTS:
	"""
	MCTS simulator with wave-batched GPU inference and virtual loss.

	Instead of one GPU call per simulation, we collect wave_size leaves
	per game before any GPU call, producing batches of
	parallel × wave_size states per forward pass.

	GPU calls per move:  ceil(simulations / wave_size)   (was: simulations)
	Batch size per call: parallel × wave_size            (was: parallel)

	Virtual loss pushes concurrent selections within the same wave to
	explore different branches. It is applied on selection and reversed
	before real backprop.

	Parameters
	──────────────────────────────────────────────────────────
	simulations : int   Rollouts per move.
	parallel    : int   Concurrent games (GPU batch = parallel × wave_size).
	wave_size   : int   Leaves collected before each GPU call. 4-16 is good.
	c_puct      : float  Exploration constant.
	dirichlet_alpha : float  Root noise concentration (~0.3 for 15×15).
	dirichlet_frac  : float  Root noise weight (AlphaZero uses 0.25).
	temperature : float  Move sampling temperature.
	virtual_loss_weight : float  Penalty per in-flight simulation.
	"""

	def __init__(self,
	             parallel=16,
	             wave_size=8,
	             c_puct=1.5,
	             dirichlet_alpha=0.3,
	             dirichlet_frac=0.25,
	             temperature=1.0,
				 temp_decay=0.95,
				 temp_min=0.05,
	             virtual_loss_weight=1.0):
		self.parallel            = parallel
		self.wave_size           = wave_size
		self.c_puct              = c_puct
		self.dirichlet_alpha     = dirichlet_alpha
		self.dirichlet_frac      = dirichlet_frac
		self.temperature         = temperature
		self.virtual_loss_weight = virtual_loss_weight
		self.temp_decay          = temp_decay
		self.temp_min            = temp_min

	# ── Clone validation ──────────────────────────────────────────────────────
	def _validate_clone(self, board):
		cloned = board.clone()
		if board.board.data_ptr() == cloned.board.data_ptr():
			raise RuntimeError(
				"board.clone() is shallow — board tensor shares storage.\n"
				"Implement clone() as:\n"
				"    def clone(self):\n"
				"        c = YourBoard.__new__(YourBoard)\n"
				"        c.board = self.board.clone()\n"
				"        return c"
			)
		legal = cloned.legal_moves()
		if legal:
			snapshot = board.board.clone()
			cloned.make_move(legal[0], 1)
			if not torch.equal(board.board, snapshot):
				raise RuntimeError(
					"board.clone() is not independent: a move on the clone "
					"modified the original board."
				)

	# ── Lazy materialisation ──────────────────────────────────────────────────
	def _materialise(self, node):
		if node.board is None:
			node.board = node.parent.board.clone()
			node.board.make_move(node.move, node.parent.player)

	# ── PUCT selection ────────────────────────────────────────────────────────
	def _select(self, root):
		node = root
		while node.children or node._unexp_moves is not None:
			sqrt_N     = math.sqrt(node.N + 1)
			best_score = -float('inf')
			best_child = None
			c_puct     = self.c_puct
			vl_weight  = self.virtual_loss_weight

			# Score expanded children
			for child in node.children.values():
				effective_n = child.N + child.virtual_loss
				effective_w = child.W + child.virtual_loss * vl_weight
				q     = -(effective_w / effective_n) if effective_n > 0 else 0.0
				u     = c_puct * child.prior * sqrt_N / (1 + effective_n)
				score = q + u
				if score > best_score:
					best_score = score
					best_child = child

			# Check best unexpanded move (N=0 → score = c_puct * prior * sqrt_N)
			# .argmax() method avoids numpy wrapper overhead vs np.argmax()
			unexp_priors = node._unexp_priors
			if unexp_priors is not None:
				idx        = int(unexp_priors.argmax())
				top_prior  = float(unexp_priors[idx])
				u_unexp    = c_puct * top_prior * sqrt_N
				if u_unexp > best_score:
					best_unexp_mv = node._unexp_moves[idx]
					# Swap-and-shrink: O(1), numpy slice view (no realloc)
					last = len(node._unexp_moves) - 1
					if idx != last:
						node._unexp_moves[idx]   = node._unexp_moves[last]
						unexp_priors[idx]        = unexp_priors[last]
					node._unexp_moves  = node._unexp_moves[:last] if last > 0 else None
					node._unexp_priors = unexp_priors[:last]      if last > 0 else None
					child = MCTSNode(
						parent=node, move=best_unexp_mv,
						player=node._child_player,
						prior=top_prior, board=None,
					)
					node.children[best_unexp_mv] = child
					self._materialise(child)
					return child  # New leaf — needs evaluation

			if best_child is None:
				break

			self._materialise(best_child)
			node = best_child
		return node

	# ── Virtual loss ──────────────────────────────────────────────────────────
	def _apply_virtual_loss(self, node):
		cur = node
		while cur is not None:
			cur.virtual_loss += 1
			cur = cur.parent

	def _remove_virtual_loss(self, node):
		cur = node
		while cur is not None:
			cur.virtual_loss -= 1
			cur = cur.parent

	# ── Expansion ─────────────────────────────────────────────────────────────
	def _expand(self, node, pol_np, value):
		# pol_np is a 1-D numpy float32 array (already on CPU)
		brd    = node.board
		n      = brd._n_empty
		legal  = brd._empty_arr[:n]        # numpy slice — no tolist() needed
		logits = pol_np[legal].copy()      # copy so we can modify in-place
		logits -= logits.max()
		exp    = np.exp(logits)
		s      = exp.sum()
		if s > 0:
			priors = exp / s
		else:
			priors = np.ones(n, dtype=np.float32) / n
		node._unexp_moves  = legal.tolist()   # Python list for move keys
		node._unexp_priors = priors.astype(np.float32, copy=False)
		node._child_player = -node.player
		return value if math.isfinite(value) else 0.0

	# ── Dirichlet noise ───────────────────────────────────────────────────────
	def _add_dirichlet(self, root):
		exp_moves  = list(root.children.keys())
		exp_priors = [root.children[mv].prior for mv in exp_moves]

		unexp_moves  = root._unexp_moves  or []
		unexp_priors = root._unexp_priors  # numpy array or None

		n = len(exp_moves) + len(unexp_moves)
		if n == 0:
			return
		noise = torch.distributions.Dirichlet(
			torch.full((n,), self.dirichlet_alpha)
		).sample().tolist()
		frac           = self.dirichlet_frac
		one_minus_frac = 1 - frac

		for i, mv in enumerate(exp_moves):
			root.children[mv].prior = one_minus_frac * exp_priors[i] + frac * noise[i]

		if unexp_priors is not None:
			n_exp = len(exp_moves)
			for j in range(len(unexp_moves)):
				unexp_priors[j] = one_minus_frac * float(unexp_priors[j]) + frac * noise[n_exp + j]
			# priors updated in-place; no need to replace _unexp_priors

	# ── Backpropagation ───────────────────────────────────────────────────────
	def _backprop(self, node, value):
		sign = 1
		while node is not None:
			node.N += 1
			node.W += value * sign
			sign   *= -1
			node    = node.parent

	# ── Move selection ────────────────────────────────────────────────────────
	def _pick_move(self, root, num_cells, move_number):
		if not root.children:
			legal      = root.board.legal_moves()
			move       = legal[torch.randint(len(legal), (1,)).item()]
			visit_dist = torch.zeros(num_cells)
			visit_dist[legal] = 1.0 / len(legal)
			return move, visit_dist

		legal_moves = list(root.children.keys())
		visits      = torch.tensor(
			[root.children[mv].N for mv in legal_moves], dtype=torch.float
		)
		visit_dist = torch.zeros(num_cells)
  
		temp = self.temperature * (self.temp_decay ** move_number)
		temp = max(temp, self.temp_min)

		if temp == 0.0 or visits.sum() == 0:
			if visits.sum() == 0:
				priors = torch.tensor([root.children[mv].prior for mv in legal_moves])
				if torch.isnan(priors).any() or priors.sum() == 0:
					priors = torch.ones(len(legal_moves))
				best = legal_moves[priors.argmax().item()]
			else:
				best = legal_moves[visits.argmax().item()]
			visit_dist[best] = 1.0
			return best, visit_dist

		logits                  = torch.log(visits + 1e-9) / temp
		probs                   = torch.softmax(logits, dim=0)
		visit_dist[legal_moves] = probs
		move                    = legal_moves[torch.multinomial(probs, 1).item()]
		return move, visit_dist

	# ── Core: parallel games with wave-batched inference ─────────────────────
	def _run_parallel(self, board_class, agent1, agent2, n, pbar, simulations):
		boards    = [board_class() for _ in range(n)]
		players   = [1]     * n
		logs      = [[]     for _ in range(n)]
		done      = [False] * n
		results   = [None]  * n
		num_cells = len(boards[0].board)
		# Pre-allocated state buffer: avoids per-wave torch.stack + tensor allocs
		max_batch  = n * self.wave_size
		_state_buf = np.empty((max_batch, num_cells), dtype=np.float32)

		agents = {1: agent1, -1: agent2}

		while not all(done):
			active = [i for i in range(n) if not done[i]]

			roots = {
				i: MCTSNode(
					parent=None, move=None, player=players[i],
					prior=1.0,   board=boards[i].clone()
				)
				for i in active
			}
			noise_applied = {i: False for i in active}
			sims_done     = {i: 0 for i in active}

			while any(sims_done[i] < simulations for i in active):

				# ── Collect one wave of leaves ────────────────────────────────
				terminal_nodes     = []   # (game_idx, node, value)
				non_terminal_nodes = []   # (game_idx, node)

				for i in active:
					remaining = simulations - sims_done[i]
					if remaining <= 0:
						continue
					n_wave = min(self.wave_size, remaining)

					for _ in range(n_wave):
						node = self._select(roots[i])
						brd  = node.board
						# Inline terminal: evaluate() once, avoid double call via terminal()
						res  = brd.evaluate()
						if res != 0 or brd._n_empty == 0:
							terminal_nodes.append((i, node, float(res * node.player)))
						else:
							self._apply_virtual_loss(node)
							non_terminal_nodes.append((i, node))

					# Dirichlet noise after root's first expansion
					if not noise_applied[i] and (roots[i].children or roots[i]._unexp_moves):
						self._add_dirichlet(roots[i])
						noise_applied[i] = True

				# ── Single GPU call for all non-terminal leaves ───────────────
				if non_terminal_nodes:
					for p_turn in (1, -1):
						p_nodes = [(i, node) for (i, node) in non_terminal_nodes if node.player == p_turn]
						if not p_nodes:
							continue
						
						agent  = agents[p_turn]
						model  = agent.model
						device = agent.device
						nb     = len(p_nodes)

						for j, (_, node) in enumerate(p_nodes):
							np.multiply(node.board._board_1d, node.player, out=_state_buf[j])
						states = torch.from_numpy(_state_buf[:nb]).to(device, non_blocking=True)

						with torch.no_grad():
							pol_batch, val_batch = model(states)

						pol_np   = pol_batch.cpu().numpy()   # one bulk PCIe transfer
						val_list = val_batch.squeeze(-1).tolist()

						for j, (i, node) in enumerate(p_nodes):
							value = self._expand(
								node,
								pol_np[j],
								val_list[j],
							)
							self._remove_virtual_loss(node)
							self._backprop(node, value)
							sims_done[i] += 1

				# ── Backprop terminals ────────────────────────────────────────
				for i, node, value in terminal_nodes:
					self._backprop(node, value)
					sims_done[i] += 1

			# ── Advance each active game by one move ──────────────────────────
			for i in active:
				move, visit_dist = self._pick_move(roots[i], num_cells, len(logs[i]))
				logs[i].append(
					(boards[i].board.clone(), players[i], move, visit_dist)
				)
				boards[i].make_move(move, players[i])
				players[i] *= -1

				if boards[i].terminal():
					results[i] = boards[i].evaluate()
					done[i]    = True
					pbar.update(1)

		games = []
		for i in range(n):
			games.append([
				(state, pl, mv, policy, results[i])
				for state, pl, mv, policy in logs[i]
			])
		return games

	# ── Public interface ──────────────────────────────────────────────────────
	def simulate(self, board, agent1, agent2, rounds=256, simulations=100):
		agent1.model.eval()
		agent2.model.eval()
		board_class = type(board)

		self._validate_clone(board)

		games = []
		pbar  = tqdm(total=rounds, desc="MCTS 2-agent play")

		i = 0
		while i < rounds:
			batch_n = min(self.parallel, rounds - i)
			batch   = self._run_parallel(board_class, agent1, agent2, batch_n, pbar, simulations)
			games.extend(batch)
			i += batch_n

		pbar.close()
		return games
