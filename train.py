# -*- coding: utf-8 -*-
import os, sys, time
sys.path.append('utils')
sys.path.append('models')
from celeba_model import *
from data import *
import torch
from torch.autograd import Variable, grad
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
import argparse, math


parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', default='', type=str, help='gpu(s) to use.')
# parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch size.')
parser.add_argument('-bunch', '--bunch', default=600, type=int, 
	help='after #k number of images, fade in the next block or stabilize.')
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate.')
parser.add_argument('-l', '--lam', default=10.0, type=float, help='weight of gradient penalty term.')
parser.add_argument('-lc', '--lipschitz', default=1.0, type=float, help='lipschitz constant.')
parser.add_argument('-e', '--epsilon', default=1e-3, type=float, help='weight of drifting loss term.')
parser.add_argument('-sample_freq', '--sample_freq', default=500, type=int, help='sample frequence.')
parser.add_argument('-save_freq', '--save_freq', default=500, type=int, help='save model frequence.')
parser.add_argument('-sample_dir', '--sample_dir', default='exp/<time>/samples', type=str, help='Samples folder.')
parser.add_argument('-ckpt_dir', '--ckpt_dir', default='exp/<time>/ckpts', type=str, help='Checkpoints folder.')
# parser.add_argument('-gan', '--gan', default='wgan_gp', type=str, help='GAN type: wgan_gp/lsgan.')

args = parser.parse_args()
gpu = args.gpu
cuda = len(gpu) > 0
bunch = args.bunch * 1000
lr = args.learning_rate
lam = args.lam
lc = args.lipschitz
epsilon = args.epsilon
sample_freq = args.sample_freq
save_freq = args.save_freq
sample_dir = args.sample_dir
ckpt_dir = args.ckpt_dir
# gan = args.gan
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

current_time = time.strftime('%Y-%m-%d %H%M%S')
if '<time>' in sample_dir:
	sample_dir = sample_dir.replace('<time>', current_time)
if '<time>' in ckpt_dir:
	ckpt_dir = ckpt_dir.replace('<time>', current_time)

if not os.path.exists(sample_dir):
	os.makedirs(sample_dir)
if not os.path.exists(ckpt_dir):
	os.makedirs(ckpt_dir)

with open(os.path.join(ckpt_dir, 'opt.txt'), 'w') as f:
	kwargs = args._get_kwargs()
	for k, v in kwargs:
		f.writelines('%s: %s' % (k, v))

resol2bs = {4:32, 8:32, 16:32, 32:32, 64:32, 128:32, 256:16}
celeba = CelebA(True)
noise = RandomNoiseGenerator(512, 'gaussian')
level = int(math.log2(256)) - 1

# networks
G = G_celeba_256()
D = D_celeba_256()

# registe on gpu
if cuda:
	G.cuda()
	D.cuda()

optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0, 0.99))
optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0, 0.99))

for i in range(1, level):
	resolution = 2 ** (i+1)
	batch_size = resol2bs[resolution]
	num_epochs = bunch // batch_size
	for phase in ['fade_in', 'stabilize']:
		# TODO: dealing with phase='fade_in'
		if phase == 'fade_in':
			pass
		for epoch in range(num_epochs):
			start_time = time.time()

			# clear grad
			D.zero_grad()

			# real images
			x_real = Variable(torch.from_numpy(celeba(batch_size, size=resolution)))
			# latent noise
			z = Variable(torch.from_numpy(noise(batch_size)))
			if cuda:
				x_real = x_real.cuda()
				z = z.cuda()

			# fake images
			x_fake = G(z, level=i)

			# discriminator output of real images
			d_real = D(x_real, level=i)
			# discriminator output of fake images
			d_fake = D(x_fake.detach(), level=i)
			# discriminator output of mixing images
			mixing_factors = Variable(torch.rand(x_real.size(0), 1, 1, 1))
			if cuda:
				mixing_factors = mixing_factors.cuda()
			x_mix = x_real * (1 - mixing_factors) + x_fake.detach() * mixing_factors
			d_mix = D(x_mix, level=i)

			# compute loss of D
			gradients = grad(outputs=d_mix, inputs=x_mix, grad_outputs=torch.ones(d_mix.size()).cuda() \
				if cuda else torch.ones(d_mix.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
			grad_penalty = ((gradients.norm(2, dim=1) - lc) ** 2).mean() * lam / (lc**2)
			drifting_loss = torch.mean(d_real ** 2) * epsilon
			w_dist = torch.mean(d_real) - torch.mean(d_fake)
			d_loss = -w_dist + grad_penalty + drifting_loss
			
			# update D
			d_loss.backward()
			optim_D.step()
			
			# update G
			G.zero_grad()
			d_fake = D(x_fake, level=i)
			g_loss = -torch.mean(d_fake)
			g_loss.backward()
			optim_G.step()
			
			# sample
			if epoch % sample_freq == 0:
				celeba.save_imgs(x_fake.cpu().data.numpy(), 
					file_name=os.path.join(sample_dir, '%dx%d-%s-epoch_%d-generated' % (resolution, resolution, phase, epoch)))
				celeba.save_imgs(x_real.cpu().data.numpy(), 
					file_name=os.path.join(sample_dir, '%dx%d-%s-epoch_%d-real' % (resolution, resolution, phase, epoch)))

			# save model
			if epoch % save_freq == 0:
				torch.save(G.state_dict(), os.path.join(ckpt_dir, '%dx%d-%s-epoch_%d-G.pth' % (resolution, resolution, phase, epoch)))
				torch.save(D.state_dict(), os.path.join(ckpt_dir, '%dx%d-%s-epoch_%d-D.pth' % (resolution, resolution, phase, epoch)))

