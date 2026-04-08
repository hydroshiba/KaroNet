import torch
from tqdm import tqdm

class MonteCarlo:
	def __init__(self, optimizer, loss_fn, loss_freq=10):
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.loss_freq = loss_freq

	def train(self, model, games, epochs=1000, batch_size=256):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device).train()
		if device.type == 'cuda':
			torch.backends.cudnn.benchmark = True

		states, players, moves, policies, results = [], [], [], [], []
		for game in games:
			for state, player, move, policy, result in game:
				states.append(state)
				players.append(player)
				moves.append(move)
				policies.append(policy)
				results.append(result * player)

		states = torch.stack(states).to(device)
		players = torch.tensor(players).to(device).float()
		moves = torch.tensor(moves).to(device).long()
		policies = torch.stack(policies).to(device)
		results = torch.tensor(results).to(device).float()

		use_amp = device.type == 'cuda'
		scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

		for epoch in tqdm(range(epochs)):
			perm = torch.randperm(len(states))
			total_loss = 0.0
			batches = 0

			for i in range(0, len(states), batch_size):
				idx = perm[i:i + batch_size]
				batch_states = states[idx] * players[idx].unsqueeze(1)   # [B, len]
				batch_moves   = moves[idx]      # [B]
				batch_results = results[idx]    # [B]
				batch_policies = policies[idx]  # [B, actions]

				with torch.autocast(device_type=device.type, enabled=use_amp):
					predictions, values = model(batch_states)
					loss = self.loss_fn(predictions, batch_moves, batch_policies, values, batch_results)

				self.optimizer.zero_grad(set_to_none=True)
				scaler.scale(loss).backward()

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
