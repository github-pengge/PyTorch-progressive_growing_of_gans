import torch
from torch.autograd import Variable
import sys
sys.path.append('./models')
from model import *


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
z = Variable(torch.randn(3, 512)).cuda(1)
x = G(z, cur_level=8.2)
print('x:', x.size())
d = D(x, cur_level=8.2, gdrop_strength=0.2)
d = torch.mean(d)
d.backward()
