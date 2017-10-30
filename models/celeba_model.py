# -*- coding: utf-8 -*-
from base_model import *


class G_celeba(nn.Module):
	'''
	Generator model for Celeba-HQ in paper.
	'''
	def __init__(self):
		super(G_celeba, self).__init__()
		self.block = []
		self.block += [SameResolution3LayerBlock(512, 512, 3, True, 4, 3)]
		self.block += [SameResolution3LayerBlock(512, 512, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(512, 512, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(512, 512, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(512, 256, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(256, 128, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(128, 64, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(64, 32, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(32, 16, 3, False, 3, 3)]
		self.n_level = len(self.block)  # 9
		
	def forward(self, x, level=None):
		if level is None:
			level = self.n_level
		for i in range(level):
			x = self.block[i](x)
		return x


class G_celeba_256(nn.Module):
	def __init__(self):
		super(G_celeba_256, self).__init__()
		self.block = []
		self.block += [SameResolution3LayerBlock(512, 512, 3, True, 4, 3)]
		self.block += [SameResolution3LayerBlock(512, 512, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(512, 256, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(256, 128, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(128, 64, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(64, 32, 3, False, 3, 3)]
		self.block += [SameResolution3LayerBlock(32, 16, 3, False, 3, 3)]
		self.n_level = len(self.block)  # 7
		
	def forward(self, x, level=None):
		if level is None:
			level = self.n_level
		for i in range(level):
			x = self.block[i](x)
		return x


class D_celeba_1024(nn.Module):
	pass


class D_celeba_256(nn.Module):
	pass
