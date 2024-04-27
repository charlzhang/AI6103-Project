import torch
from torch import nn


class NormLoss(nn.Module):
	def __init__(self):
		super(NormLoss, self).__init__()

	def forward(self, latent_source, latent_transferred):
		difference = latent_source - latent_transferred
		return torch.sum(difference.norm(2, dim=(1, 2))) / difference.shape[0]