import torch
from torch.autograd import Variable
import sys
sys.path.append('./models')
sys.path.append('./utils')
from models.model import *
from utils.data import CelebA


G = Generator(num_channels=3, resolution=1024, fmap_max=512, fmap_base=8192, latent_size=512)
D = Discriminator(num_channels=3, resolution=1024, fmap_max=512, fmap_base=8192)

param_G = G.named_parameters()
print('G:')
for name, p in param_G:
	print(name, p.size())

print('\n')

param_D = D.named_parameters()
print('D:')
for name, p in param_D:
	print(name, p.size())

print(G)
print(D)

G.cuda(1)
D.cuda(1)
data = CelebA()
z = Variable((torch.rand(3, 512)-0.5)*2).cuda(1)
x = G(z, cur_level=1.2)
# x = Variable(torch.from_numpy(data(3, size=8))).cuda(1)
print('x:', x.size())
d = D(x, cur_level=1.2, gdrop_strength=0.2)
d = torch.mean(d)
print(d)
d.backward()

print('G:')
for name, p in G.named_parameters():
	if p.grad is not None:
		print(name, p.size(), p.grad.mean().data[0])

print('D:')
for name, p in D.named_parameters():
	if p.grad is not None:
		print(name, p.size(), p.grad.mean().data[0])
