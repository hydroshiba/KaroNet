import torch
from torch import nn, optim

class GomokuNet(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.backbone = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=5, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=5, padding=2),
			nn.BatchNorm2d(128),
			nn.ReLU(),
		)

		self.policy_head = nn.Sequential(
			nn.Conv2d(128, 1, kernel_size=1),
			nn.Flatten(),
		)
		
		self.value_head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(128, 1),
			nn.Tanh()
		)

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