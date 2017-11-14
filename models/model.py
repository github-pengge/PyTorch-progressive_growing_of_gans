# -*- coding: utf-8 -*-
from base_model import *


def G_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, 
		to_sequential=True, use_wscale=True, use_batchnorm=False, use_pixelnorm=True):
	layers = incoming
	layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
	he_init(layers[-1], init, param)  # init layers
	if use_wscale:
		layers += [WScaleLayer(layers[-1])]
	layers += [nonlinearity]
	if use_batchnorm:
		layers += [nn.BatchNorm2d(out_channels)]
	if use_pixelnorm:
		layers += [PixelNormLayer()]
	if to_sequential:
		return nn.Sequential(*layers)
	else:
		return layers


def NINLayer(incoming, in_channels, out_channels, nonlinearity, init, param=None, 
			to_sequential=True, use_wscale=True):
	layers = incoming
	layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]  # NINLayer in lasagne
	he_init(layers[-1], init, param)  # init layers
	if use_wscale:
		layers += [WScaleLayer(layers[-1])]
	if not (nonlinearity == 'linear'):
		layers += [nonlinearity]
	if to_sequential:
		return nn.Sequential(*layers)
	else:
		return layers


class Generator(nn.Module):
	def __init__(self, 
				num_channels        = 1,        # Overridden based on dataset.
				resolution          = 32,       # Overridden based on dataset.
				label_size          = 0,        # Overridden based on dataset.
				fmap_base           = 4096,
				fmap_decay          = 1.0,
				fmap_max            = 256,
				latent_size         = None,
				normalize_latents   = True,
				use_wscale          = True,
				use_pixelnorm       = True,
				use_leakyrelu       = True,
				use_batchnorm       = False,
				tanh_at_end         = None):
		super(Generator, self).__init__()
		self.num_channels = num_channels
		self.resolution = resolution
		self.label_size = label_size
		self.fmap_base = fmap_base
		self.fmap_decay = fmap_decay
		self.fmap_max = fmap_max
		self.latent_size = latent_size
		self.normalize_latents = normalize_latents
		self.use_wscale = use_wscale
		self.use_pixelnorm = use_pixelnorm
		self.use_leakyrelu = use_leakyrelu
		self.use_batchnorm = use_batchnorm
		self.tanh_at_end = tanh_at_end

		R = int(np.log2(resolution))
		assert resolution == 2**R and resolution >= 4
		if latent_size is None: 
			latent_size = self.get_nf(0)

		negative_slope = 0.2
		act = nn.LeakyReLU(negative_slope=negative_slope) if self.use_leakyrelu else nn.ReLU()
		iact = 'leaky_relu' if self.use_leakyrelu else 'relu'
		output_act = nn.Tanh() if self.tanh_at_end else 'linear'
		output_iact = 'tanh' if self.tanh_at_end else 'linear'

		pre = None
		lods = nn.ModuleList()
		nins = nn.ModuleList()
		layers = []

		if self.normalize_latents:
			pre = PixelNormLayer()

		if self.label_size:
			layers += [ConcatLayer()]

		layers += [ReshapeLayer([latent_size, 1, 1])]
		layers = G_conv(layers, latent_size, self.get_nf(1), 4, 3, act, iact, negative_slope, 
					False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm) 
		net = G_conv(layers, latent_size, self.get_nf(1), 3, 1, act, iact, negative_slope, 
					True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)  # first block
		
		lods.append(net)
		nins.append(NINLayer([], self.get_nf(1), self.num_channels, output_act, output_iact, None, True, self.use_wscale))  # to_rgb layer

		for I in range(2, R):  # following blocks
			ic, oc = self.get_nf(I-1), self.get_nf(I)
			layers = [nn.Upsample(scale_factor=2, mode='nearest')]  # upsample
			layers = G_conv(layers, ic, oc, 3, 1, act, iact, negative_slope, False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
			net = G_conv(layers, oc, oc, 3, 1, act, iact, negative_slope, True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
			lods.append(net)
			nins.append(NINLayer([], oc, self.num_channels, output_act, output_iact, None, True, self.use_wscale))  # to_rgb layer

		self.output_layer = GSelectLayer(pre, lods, nins)

	def get_nf(self, stage):
		return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

	def forward(self, x, y=None, cur_level=None, insert_y_at=None):
		return self.output_layer(x, y, cur_level, insert_y_at)


def D_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, 
		to_sequential=True, use_wscale=True, use_gdrop=True, use_layernorm=False, gdrop_param=dict()):
	layers = incoming
	if use_gdrop:
		layers += [GDropLayer(**gdrop_param)]
	layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
	he_init(layers[-1], init, param)  # init layers
	if use_wscale:
		layers += [WScaleLayer(layers[-1])]
	layers += [nonlinearity]
	if use_layernorm:
		layers += [LayerNormLayer()]
	if to_sequential:
		return nn.Sequential(*layers)
	else:
		return layers


class Discriminator(nn.Module):
	def __init__(self, 
				num_channels    = 1,        # Overridden based on dataset.
				resolution      = 32,       # Overridden based on dataset.
				label_size      = 0,        # Overridden based on dataset.
				fmap_base       = 4096,
				fmap_decay      = 1.0,
				fmap_max        = 256,
				mbstat_avg      = 'all',
				mbdisc_kernels  = None,
				use_wscale      = True,
				use_gdrop       = True,
				use_layernorm   = False):
		super(Discriminator, self).__init__()
		self.num_channels = num_channels
		self.resolution = resolution
		self.label_size = label_size
		self.fmap_base = fmap_base
		self.fmap_decay = fmap_decay
		self.fmap_max = fmap_max
		self.mbstat_avg = mbstat_avg
		self.mbdisc_kernels = mbdisc_kernels
		self.use_wscale = use_wscale
		self.use_gdrop = use_gdrop
		self.use_layernorm = use_layernorm

		R = int(np.log2(resolution))
		assert resolution == 2**R and resolution >= 4
		gdrop_strength = 0.0

		negative_slope = 0.2
		act = nn.LeakyReLU(negative_slope=negative_slope)
		iact = 'leaky_relu'
		gdrop_param = {'mode': 'prop', 'strength': gdrop_strength}

		nins = nn.ModuleList()
		lods = nn.ModuleList()
		pre = None

		nins.append(NINLayer([], self.num_channels, self.get_nf(R-1), act, iact, negative_slope, True, self.use_wscale))

		for I in range(R-1, 1, -1):
			ic, oc = self.get_nf(I), self.get_nf(I-1)
			net = D_conv([], ic, ic, 3, 1, act, iact, negative_slope, False, 
						self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
			net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False, 
						self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
			net += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
			lods.append(nn.Sequential(*net))
			# nin = [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
			nin = []
			nin = NINLayer(nin, self.num_channels, oc, act, iact, negative_slope, True, self.use_wscale)
			nins.append(nin)

		net = []
		ic = oc = self.get_nf(1)
		if self.mbstat_avg is not None:
			net += [MinibatchStatConcatLayer(averaging=self.mbstat_avg)]
			ic += 1
		net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False, 
					self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
		net = D_conv(net, oc, self.get_nf(0), 4, 0, act, iact, negative_slope, False,
					self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)

		if self.mbdisc_kernels:
			net += [MinibatchDiscriminationLayer(num_kernels=self.mbdisc_kernels)]

		oc = 1 + self.label_size
		lods.append(NINLayer(net, self.get_nf(0), oc, 'linear', 'linear', None, True, self.use_wscale))

		self.output_layer = DSelectLayer(pre, lods, nins)

	def get_nf(self, stage):
		return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

	def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
		self.__dict__['strength'] = gdrop_strength  # update gdrop_strength
		return self.output_layer(x, y, cur_level, insert_y_at)
