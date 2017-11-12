# -*- coding: utf-8 -*-
from base_model import *


class G_celeba(nn.Module):
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
		super(G_celeba, self).__init__()
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
		cur_lod = 0.0
		if latent_size is None: 
			latent_size = self.get_nf(0)

		negative_slope = 0.2
		act = nn.LeakyReLU(negative_slope=negative_slope) if self.use_leakyrelu else nn.ReLU()
		iact = 'leaky_relu' if self.use_leakyrelu else 'relu'

		layers = []
		pre = None
		lods = nn.ModuleList()
		nins = nn.ModuleList()

		if self.normalize_latents:
			pre = PixelNormLayer()

		if self.label_size:
			layers += [ConcatLayer()]

		layers += [ReshapeLayer([latent_size, 1, 1])]
		net = self.conv(layers, latent_size, self.get_nf(1), 4, 3, act, iact, negative_slope, True)  # first block
		
		lods.append(net)
		nins.append(self.NINLayer([], self.get_nf(1), self.num_channels, 'linear', 'linear', None, True))  # to_rgb layer

		for I in range(2, R):  # following blocks
			ic, oc = self.get_nf(I-1), self.get_nf(I)
			layers = [nn.Upsample(scale_factor=2, mode='nearest')]  # upsample
			layers = self.conv(layers, ic, oc, 3, 1, act, iact, negative_slope, False)
			net = self.conv(layers, oc, oc, 3, 1, act, iact, negative_slope, True)
			lods.append(net)
			append(self.NINLayer([], oc, self.num_channels, 'linear', 'linear', None, True))  # to_rgb layer

		self.output_layer = LODSelectLayer(pre, lods, nins)


	def conv(self, incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None, to_sequential=True):
		layers = incoming
		layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
		he_init(layers[-1], init, param)  # init layers
		if self.use_wscale:
			layers += [WScaleLayer(layers[-1])]
		layers += [nonlinearity]
		if self.use_batchnorm:
			layers += [nn.BatchNorm2d(out_channels)]
		if self.use_pixelnorm:
			layers += [PixelNormLayer()]
		if to_sequential:
			return nn.Sequential(*layers)
		else:
			return layers

	def NINLayer(self, incoming, in_channels, out_channels, nonlinearity, init, param, to_sequential=True):
		layers = incoming
		layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]  # NINLayer in lasagne
		he_init(layers[-1], init, param)  # init layers
		if self.use_wscale:
			layers += [WScaleLayer(layers[-1])]
		if not (nonlinearity == 'linear'):
			layers += [nonlinearity]
		if to_sequential:
			return nn.Sequential(*layers)
		else:
			return layers

	def get_nf(self, stage):
		return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

	def forward(self, x, y=None, cur_lod=0, ref_idx=0, min_lod=None, max_lod=None):
		return self.output_layer(x, y, cur_lod, ref_idx, min_lod, max_lod)


class D_celeba():
	def __init__(self, ):
		pass
