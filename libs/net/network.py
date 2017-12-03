import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Conv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=True, bn=True):
		super(Conv2d, self).__init__()
		padding = int((kernel_size - 1) / 2) if same_padding else 0
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
		self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.003, affine=True) if bn else None
		self.relu = nn.ReLU(inplace=True) if relu else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x