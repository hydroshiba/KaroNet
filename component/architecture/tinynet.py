import torch
from torch import nn
import torch.nn.functional as F

class TinyNet(nn.Module):
	def __init__(self):
		super().__init__()
  
		self.backbone = nn.Sequential(
			nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(),
			nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(),
			nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten()
		)

		self.policy_head = nn.Linear(32, 225)
		self.value_head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())
		self.apply(self.weight_init)

	def weight_init(self, module):
		if isinstance(module, (nn.Conv2d, nn.Linear)):
			nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
			if module.bias is not None:
				nn.init.zeros_(module.bias)

	def forward(self, x):
		x = x.view(-1, 1, 15, 15)
		x = self.backbone(x)
		return self.policy_head(x), self.value_head(x)
	
	def evaluate(self, x):
		with torch.no_grad():
			_, val = self.forward(x)
		return val.item()

	def policy(self, x):
		with torch.no_grad():
			pol, _ = self.forward(x)
		return pol.squeeze(0)