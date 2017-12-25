import sys
sys.path.append('models')
from models import AutoencodingDiscriminator
import torch
from torch.autograd import Variable
D = AutoencodingDiscriminator(3, 32)
print(D)
x = Variable(torch.randn(1,3,4,4))
xx = D(x, cur_level=1)  # , cur_level=4
print(xx.size())

for name, p in D.named_parameters():
	print(name, p.size())