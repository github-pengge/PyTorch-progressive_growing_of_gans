# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.init import kaiming_normal, calculate_gain
import numpy as np
import sys
if sys.version_info.major == 3:
	from functools import reduce


class PixelNormLayer(nn.Module):
	def __init__(self, eps=1e-8):
		super(PixelNormLayer, self).__init__()
		self.eps = eps
	
	def forward(self, x):
		return x / (torch.mean(x**2, dim=1, keepdim=True) + 1e-8) ** 0.5

	def __repr__(self):
		return self.__class__.__name__ + '(eps = %s)' % (self.eps)


class WScaleLayer(nn.Module):
	def __init__(self, incoming):
		super(WScaleLayer, self).__init__()
		self.incoming = incoming
		self.scale = (torch.mean(self.incoming.weight.data ** 2)) ** 0.5
		self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)
		self.bias = None
		if self.incoming.bias is not None:
			self.bias = self.incoming.bias
			self.incoming.bias = None

	def forward(self, x):
		x = self.scale * x
		if self.bias is not None:
			x += self.bias.view(1, self.bias.size()[0], 1, 1)
		return x

	def __repr__(self):
		param_str = '(incoming = %s)' % (self.incoming.__class__.__name__)
		return self.__class__.__name__ + param_str


def mean(tensor, axis, **kwargs):
	if isinstance(axis, int):
		axis = [axis]
	for ax in axis:
		tensor = torch.mean(tensor, axis=ax, **kwargs)
	return tensor


class MinibatchStatConcatLayer(nn.Module):
	def __init__(self, averaging='all'):
		super(MinibatchStatConcatLayer, self).__init__()
		self.averaging = averaging.lower()
		if 'group' in self.averaging:
			self.n = int(self.averaging[5:])
		else:
			assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
		self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

	def forward(self, x):
		shape = list(x.size())
		target_shape = shape.copy()
		vals = self.adjusted_std(x, dim=0, keepdim=True)
		if self.averaging == 'all':
			target_shape[1] = 1
			vals = torch.mean(vals, keepdim=True)
		elif self.averaging == 'spatial':
			if len(shape) == 4:
				vals = mean(vals, axis=[2,3], keepdim=True)  # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
		elif self.averaging == 'none':
			target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
		elif self.averaging == 'gpool':
			if len(shape) == 4:
				vals = mean(x, [0,2,3], keepdim=True)  # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
		elif self.averaging == 'flat':
			target_shape[1] = 1
			vals = torch.FloatTensor([self.adjusted_std(x)])
		else:  # self.averaging == 'group'
			target_shape[1] = self.n
			vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
			vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
		vals = vals.expand(*target_shape)
		return torch.cat([x, vals], 1)

	def __repr__(self):
		return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)


class MinibatchDiscriminationLayer(nn.Module):
	def __init__(self, num_kernels):
		super(MinibatchDiscriminationLayer, self).__init__()
		self.num_kernels = num_kernels

	def forward(self, x):
		pass



class GDropLayer(nn.Module):
	def __init__(self, mode='mul', strength=0.4, axes=(0,1), normalize=False):
		super(GDropLayer, self).__init__()
		self.mode = mode.lower()
		assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode'%mode
		self.strength = strength
		self.axes = [axes] if isinstance(axes, int) else list(axes)
		self.normalize = normalize
		self.gain = None

	def forward(self, x, deterministic=False):
		if deterministic or not self.strength:
			return x

		rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
		if self.mode == 'drop':
			p = 1 - self.strength
			rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
		elif self.mode == 'mul':
			rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
		else:
			coef = self.strength * x.size(1) ** 0.5
			rnd = np.random.normal(size=rnd_shape) * coef + 1

		if self.normalize:
			rnd = rnd / np.linalg.norm(rnd, keepdims=True)
		rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
		if x.is_cuda:
			rnd = rnd.cuda()
		return x * rnd

	def __repr__(self):
		param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (self.mode, self.strength, self.axes, self.normalize)
		return self.__class__.__name__ + param_str


class LayerNormLayer(nn.Module):
	def __init__(self, incoming, eps=1e-4):
		super(LayerNormLayer, self).__init__()
		self.incoming = incoming
		self.eps = eps
		self.gain = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
		self.bias = None

		if self.incoming.bias is not None:
			self.bias = self.incoming.bias
			self.incoming.bias = None

	def forward(self, x):
		x = x - mean(x, axis=range(1, len(x.size())))
		x = x * torch.inverse(torch.sqrt(mean(x**2, axis=range(1, len(x.size())), keepdim=True) + self.eps))
		x = x * self.gain
		if self.bias is not None:
			x += self.bias
		return x

	def __repr__(self):
		param_str = '(incoming = %s, eps = %s)' % (self.incoming.__class__.__name__, self.eps)
		return self.__class__.__name__ + param_str


def resize_activations(v, so):
	si = list(v.size())
	so = list(so)
	assert len(si) == len(so) and si[0] == so[0]

	# Decrease feature maps.
	if si[1] > so[1]:
		v = v[:, :so[1]]

	# Shrink spatial axes.
	if len(si) == 4 and (si[2] > so[2] or si[3] > so[3]):
		assert si[2] % so[2] == 0 and si[3] % so[3] == 0
		ks = (si[2] / so[2], si[3] / so[3])
		v = F.avg_pool2d(v, kernel_size=ks, stride=ks, ceil_mode=False, padding=0, count_include_pad=False)

	# Extend spatial axes.
	shape = [1, 1]
	for i in range(2, len(si)):
		if si[i] < so[i]:
			assert so[i] % si[i] == 0
			shape += [so[i] // si[i]]
		else:
			shape += [1]
	v = v.repeat(*shape)

	# Increase feature maps.
	if si[1] < so[1]:
		z = torch.zeros((v.shape[0], so[1] - si[1]) + so[2:])
		v = torch.cat([v, z], 1)
	return v


# class LODSelectLayer(nn.Module):
# 	def __init__(self, pre, lods, nins, first_incoming_lod=0):
# 		super(LODSelectLayer, self).__init__()
# 		self.pre = pre
# 		self.lods = lods
# 		self.nins = nins
# 		self.first_incoming_lod = first_incoming_lod

# 	def forward(self, x, y=None, cur_lod=0, ref_idx=0, min_lod=None, max_lod=None):
# 		# v = [resize_activations(input, x[self.ref_idx].size()) for input in x]
# 		# lo = np.clip(int(np.floor(min_lod - self.first_incoming_lod)), 0, len(v)-1) if min_lod is not None else 0
# 		# hi = np.clip(int(np.ceil(max_lod - self.first_incoming_lod)), lo, len(v)-1) if max_lod is not None else len(v)-1
# 		# t = cur_lod - self.first_incoming_lod
# 		# r = v[hi]
# 		# for i in range(hi-1, lo-1, -1): # i = hi-1, hi-2, ..., lo
# 		# 	if t < i+1:
# 		# 		r = v[i] * ((i+1)-t) + v[i+1] * (t-i)
# 		# if lo < hi:
# 		# 	if t <= lo:
# 		# 		r = v[lo]
# 		# return r
# 		v = []
# 		if self.pre is not None:
# 			x = self.pre(x)
# 		for i in range(ref_idx):  # ref_idx: physical index
# 			if i == 0 and y is not None:
# 				x = self.lods[i](x, y)
# 			else:
# 				x = self.lods[i](x)
# 			out = self.nins[i](x)
# 			v += [out]
# 		target_shape = v[-1].size()
# 		t = cur_lod - self.first_incoming_lod  # cur_lod is float!
# 		lo = np.clip(int(np.floor(min_lod - self.first_incoming_lod)), 0, len(v)-1) if min_lod is not None else 0
# 		hi = np.clip(int(np.ceil(max_lod - self.first_incoming_lod)), lo, len(v)-1) if max_lod is not None else len(v)-1
# 		r = v[hi]
# 		for i in range(hi-1, lo-1, -1):
# 			if t < i+1:
# 				r = resize_activations(v[ref_idx-i+1], target_shape) * (i+1-t) + resize_activations(v[ref_idx-i+2], target_shape) * (t-i)
# 		if lo < hi:
# 			if t <= lo:
# 				r = v[lo]
# 		return r


class GSelectLayer(nn.Module):
	def __init__(self, pre, chain, post):
		super(GSelectLayer, self).__init__()
		assert len(chain) == len(post)
		self.pre = pre
		self.chain = chain
		self.post = post
		self.N = len(self.chain)

	def forward(self, x, y=None, cur_level=None, insert_y_at=None):
		if cur_level is None:
			cur_level = self.N  # cur_level: physical index
		if y is not None:
			assert insert_y_at is not None

		min_level, max_level = int(np.floor(cur_level-1)), int(np.ceil(cur_level-1))
		min_level_weight, max_level_weight = max_level-cur_level, min_level+1-cur_level

		_from, _to, _step = 0, max_level+1, 1

		if self.pre is not None:
			x = self.pre(x)

		out = {}
		print('G: level=%s, size=%s' % ('in', x.size()))
		for level in range(_from, _to, _step):
			if level == insert_y_at:
				x = self.chain[level](x, y)
			else:
				x = self.chain[level](x)

			print('G: level=%d, size=%s' % (level, x.size()))

			if level == min_level:
				out['min_level'] = self.post[level](x)
			if level == max_level:
				out['max_level'] = self.post[level](x)
				x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
						out['max_level'] * max_level_weight
		return x


class DSelectLayer(nn.Module):
	def __init__(self, pre, chain, inputs):
		super(DSelectLayer, self).__init__()
		assert len(chain) == len(inputs)
		self.pre = pre
		self.chain = chain
		self.inputs = inputs
		self.N = len(self.chain)

	def forward(self, x, y=None, cur_level=None, insert_y_at=None):
		if cur_level is None:
			cur_level = self.N  # cur_level: physical index
		if y is not None:
			assert insert_y_at is not None

		max_level, min_level = int(np.floor(self.N-cur_level)), int(np.ceil(self.N-cur_level))
		min_level_weight, max_level_weight = int(cur_level+1)-cur_level, cur_level-int(cur_level)

		_from, _to, _step = max_level, self.N, 1

		if self.pre is not None:
			x = self.pre(x)

		out = {}
		print('D: level=%s, size=%s' % ('in', x.size()))
		for level in range(_from, _to, _step):
			if level == max_level:
				tmp = self.inputs[level](x)
				if level == insert_y_at:
					tmp = self.chain[level](tmp, y)
				else:
					tmp = self.chain[level](tmp)
				out['max_level'] = tmp
				continue
			if level == min_level:
				out['min_level'] = self.inputs[level](x)
				x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
						out['max_level'] * max_level_weight
			if level == insert_y_at:
				x = self.chain[level](x, y)
			else:
				x = self.chain[level](x)
			print('D: level=%d, size=%s' % (level, x.size()))

		return x


class ConcatLayer(nn.Module):
	def __init__(self):
		super(ConcatLayer, self).__init__()

	def forward(self, x, y):
		return torch.cat([x, y], 1)


class ReshapeLayer(nn.Module):
	def __init__(self, new_shape):
		super(ReshapeLayer, self).__init__()
		self.new_shape = new_shape  # not include minibatch dimension

	def forward(self, x):
		assert reduce(lambda u,v: u*v, self.new_shape) == reduce(lambda u,v: u*v, x.size()[1:])
		return x.view(-1, *self.new_shape)


def he_init(layer, nonlinearity='conv2d', param=None):
	nonlinearity = nonlinearity.lower()
	if nonlinearity not in ['linear', 'conv1d', 'conv2d', 'conv3d', 'relu', 'leaky_relu', 'sigmoid', 'tanh']:
		if not hasattr(layer, 'gain') or layer.gain is None:
			gain = 0  # default
		else:
			gain = layer.gain
	elif nonlinearity == 'leaky_relu':
		assert param is not None, 'Negative_slope(param) should be given.'
		gain = calculate_gain(nonlinearity, param)
	else:
		gain = calculate_gain(nonlinearity)
	kaiming_normal(layer.weight, a=gain)

