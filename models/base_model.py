# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn 
from torch.nn.init import kaiming_normal
import math


class PixelNormLayer(nn.Module):
	def __init__(self, eps=1e-8):
		super(PixelNormLayer, self).__init__()
		self.eps = 1e-8
		self.norm = lambda x: x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5
	
	def forward(self, x):
		return self.norm(x)


class WScaleLayer(nn.Module):
	def __init__(self, layer):
		super(WScaleLayer, self).__init__()
		weight = layer.weight
		scale = None
		if weight is not None:
			scale = Variable(torch.sqrt(torch.mean(weight.data ** 2)), requires_grad=False)
		b = layer.bias
		layer.bias = None
		self.scale = scale
		self.b = b

	def forward(self, x):
		size = x.size()
		if self.scale:
			x *= self.scale
		if self.b is not None:
			x += b.view(1, -1, 1, 1)
		return x


class MinibatchStatConcatLayer(nn.Module):
	'''Minibatch stat concatenation layer. 
	-func is the function to use for the activations across minibatch
	-averaging tells how much averaging to use ('all', 'spatial', 'none')
	Currently only support averaging='all'
	'''
	def __init__(self, avergaing='all'):
		super(MinibatchStatConcatLayer, self).__init__()
		self.avergaing = avergaing.lower()
		if self.avergaing != 'all':
			raise NotImplementedError("Currently only support averaging='all'.")

	def forward(self, x):
		std = torch.std(x, 0, keepdim=True, unbiased=False)
		res = torch.mean(std, keepdim=True).expand(x.size[0], 1, x.size(1), x.size(2))
		return torch.cat([x, res], 1)


class ConvBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, padding, to_rgb=False):
		super(ConvBlock, self).__init__()
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.padding = padding
		self.to_rgb = to_rgb

		self.conv = nn.Conv2d(self.in_channel, self.out_channel, self.kernel_size, 1, self.padding, bias=False)
		if self.to_rgb:
			self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
			kaiming_normal(self.conv.weight, a=0.2)
			self.norm = PixelNormLayer()
		else:
			kaiming_normal(self.conv.weight, a=1.0)
		self.ws = WScaleLayer(self.conv)
		
		if self.to_rgb:
			self.feature = nn.Sequential([self.conv, self.ws])
		else:
			self.feature = nn.Sequential([self.conv, self.ws, self.lrelu, self.norm])

	def forward(self, x):
		return self.feature(x)


class SameResolution3LayerBlock(nn.Module):
	def __init__(self, in_channel, out_channel, channel=3, first_block=False, nks1=3, nks2=3):
		super(SameResolution3LayerBlock, self).__init__()
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.channel = channel
		self.first_block = first_block
		self.nks1 = nks1
		self.nks2 = nks2

		self.block = []
		if self.first_block:
			b = [ConvBlock(self.in_channel, self.out_channel, self.nks1, self.nks1-1)]
		else:
			b = [nn.Upsample(scale_factor=2, mode='nearest')]
			b += [ConvBlock(self.in_channel, self.out_channel, self.nks1, 1)]
		self.block += b
		self.block += [ConvBlock(self.out_channel, self.out_channel, self.nks2, 1)]
		self.feature = nn.Sequential(self.block)

		self.to_rgb = ConvBlock(self.out_channel, self.channel, 1, 1, True)
	
	def forward(self, x, to_rgb=False):
		x = self.feature(x)
		if self.to_rgb:
			return self.to_rgb(x)
		else:
			return x






