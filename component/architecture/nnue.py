import torch
from torch import nn
import torch.nn.functional as F

NNUE_CARO_SQUARES = 15 * 15
NNUE_FEATURES = 2 * NNUE_CARO_SQUARES * NNUE_CARO_SQUARES

NNUE_L1 = 128
NNUE_L2 = 32
NNUE_L3 = 32

# Currently in work

class NNUE(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.features = nn.EmbeddingBag(
			NNUE_FEATURES,
			NNUE_L1,
			mode='sum',
		)
  
		self.policy_head = nn.Linear(NNUE_L1, NNUE_CARO_SQUARES)
		self.value_head = nn.Sequential(
			nn.Linear(NNUE_L1, NNUE_L2), nn.Hardtanh(0, 1),
			nn.Linear(NNUE_L2, NNUE_L3), nn.Hardtanh(0, 1),
			nn.Linear(NNUE_L3, 1), nn.Tanh()
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