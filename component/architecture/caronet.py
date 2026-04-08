import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

	def forward(self, x):
		identity = self.shortcut(x)
		out = F.relu(self.conv1(x))
		out = self.conv2(out)
		return F.relu(out + identity)

class CaroNet(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.backbone = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=5, padding=2, bias=False),
			nn.ReLU(),
			ResidualBlock(64, 64),
			ResidualBlock(64, 64)
		)

		self.policy_head = nn.Sequential(
			nn.Conv2d(64, 2, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(2 * 15 * 15, 225)
		)
		
		self.value_head = nn.Sequential(
			nn.Conv2d(64, 2, kernel_size=1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(2 * 15 * 15, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Tanh()
		)

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