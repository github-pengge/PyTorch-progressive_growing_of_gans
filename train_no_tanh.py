# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.autograd import Variable
import sys, os, time
sys.path.append('utils')
sys.path.append('models')
from data import CelebA, RandomNoiseGenerator
from model import Generator, Discriminator
import argparse
import numpy as np
from scipy.misc import imsave


class PGGAN():
    def __init__(self, G, D, data, noise, opts):
        self.G = G
        self.D = D
        self.data = data
        self.noise = noise
        self.opts = opts

        gpu = self.opts['gpu']
        self.use_cuda = len(gpu) > 0
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        current_time = time.strftime('%Y-%m-%d %H%M%S')
        self.opts['sample_dir'] = os.path.join(os.path.join(self.opts['exp_dir'], current_time), 'samples')
        self.opts['ckpt_dir'] = os.path.join(os.path.join(self.opts['exp_dir'], current_time), 'ckpts')
        os.makedirs(self.opts['sample_dir'])
        os.makedirs(self.opts['ckpt_dir'])

        self.bs_map = {2**R: self.get_bs(2**R) for R in range(2, 11)}
        self.rows_map = {32: 8, 16: 4, 8: 4, 4: 2, 2: 2}

        # save opts
        with open(os.path.join(os.path.join(self.opts['exp_dir'], current_time), 'options.txt'), 'w') as f:
            for k, v in self.opts.items():
                print('%s: %s' % (k, v), file=f)
            print('batch_size_map: %s' % self.bs_map, file=f)

    def get_bs(self, resolution):
        R = int(np.log2(resolution))
        if R < 7:
            bs = 32 / 2**(max(0, R-4))
        else:
            bs = 8 / 2**(min(2, R-7))
        return int(bs)

    def registe_on_gpu(self):
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def create_optimizer(self):
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.opts['lr'], betas=(self.opts['beta1'], self.opts['beta2']))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.opts['lr'], betas=(self.opts['beta1'], self.opts['beta2']))

    def create_criterion(self):
        # w is for gan
        if self.opts['gan'] == 'lsgan':
            self.adv_criterion = lambda p,t,w: torch.mean((p-t)**2)  # sigmoid is applied here
        elif self.opts['gan'] == 'wgan_gp':
            self.adv_criterion = lambda p,t,w: (-2*t+1) * torch.mean(p)
        elif self.opts['gan'] == 'gan':
            lambda p,t,w: -w*(torch.mean(t*torch.log(p+1e-8)) + torch.mean((1-t)*torch.log(1-p+1e-8)))
        else:
            raise ValueError('Invalid/Unsupported GAN: %s.' % self.opts['gan'])

    def compute_adv_loss(self, prediction, target, w):
        return self.adv_criterion(prediction, target, w)

    def compute_additional_g_loss(self):
        return 0.0

    def compute_additional_d_loss(self):  # drifting loss and gradient penalty, weighting inside this function
        return 0.0

    def _get_data(self, d):
        return d.data[0] if isinstance(d, Variable) else d

    def compute_G_loss(self):
        g_adv_loss = self.compute_adv_loss(self.d_fake, True, 1)
        g_add_loss = self.compute_additional_g_loss()
        self.g_adv_loss = self._get_data(g_adv_loss)
        self.g_add_loss = self._get_data(g_add_loss)
        return g_adv_loss + g_add_loss

    def compute_D_loss(self):
        d_adv_loss = self.compute_adv_loss(self.d_real, True, 0.5) + self.compute_adv_loss(self.d_fake, False, 0.5)
        d_add_loss = self.compute_additional_d_loss()
        self.d_adv_loss = self._get_data(d_adv_loss)
        self.d_add_loss = self._get_data(d_add_loss)
        return d_adv_loss + d_add_loss

    def postprocess(self):
        # TODO: weight cliping or others
        pass

    def _numpy2var(self, x):
        var =  Variable(torch.from_numpy(x))
        if self.use_cuda:
            var = var.cuda()
        return var

    def add_noise(self, x):
        # TODO: support more method of adding noise.
        if self.opts.get('no_noise', False):
            return x

        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + torch.mean(self.d_real).data[0] * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        noise = self._numpy2var(np.random.randn(*x.size()).astype(np.float32) * strength)
        return x + noise

    def preprocess(self, z, real):
        self.z = self._numpy2var(z)
        self.real = self._numpy2var(real)

    def forward_G(self, cur_level):
        self.d_fake = self.D(self.fake, cur_level=cur_level)
    
    def forward_D(self, cur_level, detach=True):
        self.fake = self.G(self.z, cur_level=cur_level)
        self.d_real = self.D(self.add_noise(self.real), cur_level=cur_level)
        self.d_fake = self.D(self.fake.detach() if detach else self.fake, cur_level=cur_level)
        # print('d_real', self.d_real.view(-1))
        # print('d_fake', self.d_fake.view(-1))
        # print(self.fake[0].view(-1))

    def backward_G(self):
        g_loss = self.compute_G_loss()
        g_loss.backward()
        self.optim_G.step()
        self.g_loss = self._get_data(g_loss)

    def backward_D(self, retain_graph=False):
        d_loss = self.compute_D_loss()
        d_loss.backward(retain_graph=retain_graph)
        self.optim_D.step()
        self.d_loss = self._get_data(d_loss)

    def report(self, it, num_it, phase, resol):
        formation = 'Iter[%d|%d], %s, %s, G: %.3f, D: %.3f, G_adv: %.3f, G_add: %.3f, D_adv: %.3f, D_add: %.3f'
        values = (it, num_it, phase, resol, self.g_loss, self.d_loss, self.g_adv_loss, self.g_add_loss, self.d_adv_loss, self.d_add_loss)
        print(formation % values)

    def train(self):
        # prepare
        self.create_optimizer()
        self.create_criterion()
        self.registe_on_gpu()

        to_level = int(np.log2(self.opts['target_resol']))
        from_level = int(np.log2(self.opts['first_resol']))
        assert 2**to_level == self.opts['target_resol'] and 2**from_level == self.opts['first_resol'] and to_level >= from_level >= 2
        cur_level = from_level

        for R in range(from_level-1, to_level-1):
            batch_size = self.bs_map[2 ** (R+1)]
            train_kimg = int(self.opts['train_kimg'] * 1000)
            transition_kimg = int(self.opts['transition_kimg'] * 1000)
            if R == to_level-1:
                transition_kimg = 0
            cur_nimg = 0
            _len = len(str(train_kimg + transition_kimg))
            _num_it = (train_kimg + transition_kimg) // batch_size
            for it in range(_num_it):
                # determined current level: int for stabilizing and float for fading in
                cur_level = R + float(max(cur_nimg-train_kimg, 0)) / transition_kimg 
                cur_resol = 2 ** int(np.ceil(cur_level+1))
                phase = 'stabilize' if int(cur_level) == cur_level else 'fade_in'

                # get a batch noise and real images
                z = self.noise(batch_size)
                x = self.data(batch_size, cur_resol)

                # preprocess
                self.preprocess(z, x)

                # update D
                self.optim_D.zero_grad()
                self.forward_D(cur_level, detach=True)  # TODO: feed gdrop_strength
                self.backward_D()

                # update G
                self.optim_G.zero_grad()
                self.forward_G(cur_level)
                self.backward_G()

                # report 
                self.report(it, _num_it, phase, cur_resol)
                
                cur_nimg += batch_size

                # sampling
                if (it % self.opts['sample_freq'] == 0) or it == _num_it-1:
                    self.sample(os.path.join(self.opts['sample_dir'], '%dx%d-%s-%s.png' % (cur_resol, cur_resol, phase, str(it).zfill(6))))

                # save model
                if (it % self.opts['save_freq'] == 0 and it > 0) or it == _num_it-1:
                    self.save(os.path.join(self.opts['ckpt_dir'], '%dx%d-%s-%s' % (cur_resol, cur_resol, phase, str(it).zfill(6))))

    def sample(self, file_name):
        batch_size = self.z.size(0)
        n_row = self.rows_map[batch_size]
        n_col = int(np.ceil(batch_size / float(n_row)))
        white_space = np.ones((self.real.size(1), self.real.size(2), 3))
        samples = []
        i = j = 0
        for row in range(n_row):
            one_row = []
            # fake
            for col in range(n_col):
                one_row.append(self.fake[i].cpu().data.numpy())
                one_row.append(white_space)
                i += 1
            one_row.append(white_space)
            # real
            for col in range(n_col):
                one_row.append(self.real[j].cpu().data.numpy())
                if col < n_col-1:
                    one_row.append(white_space)
                j += 1
            samples += [np.concatenate(one_row, axis=2)]
        samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])
        imsave(file_name, samples)

    def save(self, file_name):
        g_file = file_name + '-G.pth'
        d_file = file_name + '-D.pth'
        torch.save(self.G.state_dict(), g_file)
        torch.save(self.D.state_dict(), d_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='', type=str, help='gpu(s) to use.')
    parser.add_argument('--train_kimg', default=600, type=float, help='# * 1000 real samples for each stabilizing training phase.')
    parser.add_argument('--transition_kimg', default=600, type=float, help='# * 1000 real samples for each fading in phase.')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2 for adam')
    parser.add_argument('--gan', default='lsgan', type=str, help='model: lsgan/wgan_gp/gan, currently only support lsgan or gan with no_noise option.')
    parser.add_argument('--first_resol', default=4, type=int, help='first resolution')
    parser.add_argument('--target_resol', default=256, type=int, help='target resolution')
    parser.add_argument('--drift', default=1e-3, type=float, help='drift, only available for wgan_gp.')
    parser.add_argument('--sample_freq', default=500, type=int, help='sampling frequency.')
    parser.add_argument('--save_freq', default=5000, type=int, help='save model frequency.')
    parser.add_argument('--exp_dir', default='./exp', type=str, help='experiment dir.')
    parser.add_argument('--no_noise', action='store_true', help='do not add noise to real data.')

    # TODO: support conditional inputs

    args = parser.parse_args()
    opts = {k:v for k,v in args._get_kwargs()}

    latent_size = 512
    sigmoid_at_end = args.gan in ['lsgan', 'gan']

    G = Generator(num_channels=3, latent_size=latent_size, resolution=args.target_resol, fmap_max=latent_size, fmap_base=8192, tanh_at_end=False)
    D = Discriminator(num_channels=3, resolution=args.target_resol, fmap_max=latent_size, fmap_base=8192, sigmoid_at_end=sigmoid_at_end)
    print(G)
    print(D)
    data = CelebA()
    noise = RandomNoiseGenerator(latent_size, 'gaussian')
    pggan = PGGAN(G, D, data, noise, opts)
    pggan.train()
