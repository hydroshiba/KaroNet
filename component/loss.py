import torch
import torch.nn.functional as F
from torch import nn

class MSEDual:
	def __init__(self):
		pass

	def __call__(self, predictions, moves, targets, values, results):
		if targets.dim() == 1 or targets.shape == moves.shape: policies = predictions[range(len(moves)), moves]
		else: policies = predictions
		policy_loss = F.mse_loss(policies, targets)
		value_loss = F.mse_loss(values.squeeze(-1), results)
		q_reg_loss = predictions.pow(2).mean() * 0.01
		return policy_loss * 0.5 + value_loss * 0.5 + q_reg_loss

# Do not use, currently broken
class PolicyGradient:
	def __init__(self):
		self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

	def __call__(self, predictions, moves, targets, values, results):
		policy_loss = (self.cross_entropy(predictions, moves) * results).mean()
		value_loss = F.mse_loss(values.squeeze(-1), results)
		return policy_loss * 0.5 + value_loss * 0.5

class AlphaZero:
	def __init__(self, policy_weight=0.5, value_weight=0.5):
		self.policy_weight = policy_weight
		self.value_weight = value_weight

	def __call__(self, predictions, moves, targets, values, results):
		# Cast to float32: in fp16 (AMP), large logits can overflow to +inf.
		# Multiple +inf values make log_softmax produce NaN (inf - inf).
		# Clamp to a safe range before log_softmax to prevent this.
		log_probs = F.log_softmax(predictions.float().clamp(-1e4, 1e4), dim=-1)
		policy_loss = -(targets.float() * log_probs).sum(dim=-1).mean()
		value_loss = F.mse_loss(values.squeeze(-1).float(), results.float())
		return self.policy_weight * policy_loss + self.value_weight * value_loss