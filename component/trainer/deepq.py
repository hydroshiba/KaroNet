import torch
import copy
from tqdm import tqdm

class DeepQ:
	def __init__(self, optimizer, loss_fn, gamma=0.95, target_update_freq=10, loss_freq=1):
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.gamma = gamma
		self.target_update_freq = target_update_freq
		self.loss_freq = loss_freq

	def train(self, model, games, epochs=10, batch_size=256):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device).train()
		if device.type == 'cuda':
			torch.backends.cudnn.benchmark = True

		target_model = copy.deepcopy(model)
		target_model.eval()

		states, players, moves, next_states, dones, results = [], [], [], [], [], []
		for game in games:
			for i, (state, player, move, policy, result) in enumerate(game):
				terminal = (i == len(game) - 1)
				next_state = torch.zeros_like(state) if terminal else game[i + 1][0]
				states.append(state)
				players.append(float(player))
				moves.append(int(move))
				next_states.append(next_state)
				dones.append(1.0 if terminal else 0.0)
				results.append(float(result * player))

		# Move everything to GPU memory once since Gomoku board is small (15x15)
		states_tsr = torch.stack(states).to(device, non_blocking=True)
		players_tsr = torch.tensor(players, device=device).unsqueeze(1)
		moves_tsr = torch.tensor(moves, device=device)
		next_states_tsr = torch.stack(next_states).to(device, non_blocking=True)
		dones_tsr = torch.tensor(dones, device=device)
		results_tsr = torch.tensor(results, device=device)

		# Perspective flips computed globally once to save time inside the batch loop
		b_states_global = states_tsr * players_tsr
		b_next_states_global = next_states_tsr * (-players_tsr)

		dataset_len = len(states_tsr)
		
		# Define mixed precision scaler
		use_amp = device.type == 'cuda'
		scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

		for epoch in tqdm(range(epochs)):
			if epoch % self.target_update_freq == 0:
				target_model.load_state_dict(model.state_dict())

			total_loss = 0.0
			batches = 0
			
			# Shuffle everything globally once per epoch to avoid inner-loop dynamic indexing overhead
			perm = torch.randperm(dataset_len, device=device)
			
			shuffled_states = b_states_global[perm]
			shuffled_moves = moves_tsr[perm]
			shuffled_next_states = b_next_states_global[perm]
			shuffled_dones = dones_tsr[perm]
			shuffled_results = results_tsr[perm]

			for i in range(0, dataset_len, batch_size):
				end_idx = min(i + batch_size, dataset_len)
				actual_batch_size = end_idx - i
				
				# Slice views natively without memory allocation
				b_states_view = shuffled_states[i:end_idx]
				b_moves_view = shuffled_moves[i:end_idx]
				b_next_states_view = shuffled_next_states[i:end_idx]
				b_dones_view = shuffled_dones[i:end_idx]
				b_results_view = shuffled_results[i:end_idx]

				# Bellman target & FP16 Forward prop
				with torch.autocast(device_type=device.type, enabled=use_amp):
					with torch.no_grad():
						next_preds_curr, _ = model(b_next_states_view)
						next_preds_curr.masked_fill_(b_next_states_view != 0, float('-inf'))
						best_next_moves = next_preds_curr.argmax(dim=1)

						next_preds_target, _ = target_model(b_next_states_view)
						max_next_q = next_preds_target[torch.arange(actual_batch_size, device=device), best_next_moves]
						q_values = b_results_view * b_dones_view - self.gamma * max_next_q * (1 - b_dones_view)

					predictions, values = model(b_states_view)
					loss = self.loss_fn(predictions, b_moves_view, q_values, values, b_results_view)

				# Backward prop with scaled gradients
				self.optimizer.zero_grad(set_to_none=True)
				scaler.scale(loss).backward()
				
				# Unscale prior to clipping to maintain bounds
				scaler.unscale_(self.optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				
				scaler.step(self.optimizer)
				scaler.update()

				total_loss += loss.item()
				batches += 1

			if batches > 0 and epoch % self.loss_freq == 0:
				tqdm.write(f"Epoch {epoch + 1}/{epochs} — loss: {total_loss / batches:.4f}")

		final_loss = total_loss / batches if batches > 0 else 0.0
		print(f"Final training loss: {final_loss:.4f}")
