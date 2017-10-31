# -*- coding: utf-8 -*-
from base_model import *


class G_celeba_1024(nn.Module):
    '''
    Generator model for Celeba-HQ in paper.
    '''
    def __init__(self):
        super(G_celeba_1024, self).__init__()
        self.n_level = 9
        self.blk0 = GSameResolution3LayerBlock(512, 512, 3, True, 4, 3)
        self.rgb0 = GConvBlock(512, 3, 1, 0, True)
        self.blk1 = GSameResolution3LayerBlock(512, 512, 3, False, 3, 3)
        self.rgb1 = GConvBlock(512, 3, 1, 0, True)
        self.blk2 = GSameResolution3LayerBlock(512, 512, 3, False, 3, 3)
        self.rgb2 = GConvBlock(512, 3, 1, 0, True)
        self.blk3 = GSameResolution3LayerBlock(512, 512, 3, False, 3, 3)
        self.rgb3 = GConvBlock(512, 3, 1, 0, True)
        self.blk4 = GSameResolution3LayerBlock(512, 256, 3, False, 3, 3)
        self.rgb4 = GConvBlock(256, 3, 1, 0, True)
        self.blk5 = GSameResolution3LayerBlock(256, 128, 3, False, 3, 3)
        self.rgb5 = GConvBlock(128, 3, 1, 0, True)
        self.blk6 = GSameResolution3LayerBlock(128, 64, 3, False, 3, 3)
        self.rgb6 = GConvBlock(64, 3, 1, 0, True)
        self.blk7 = GSameResolution3LayerBlock(64, 32, 3, False, 3, 3)
        self.rgb7 = GConvBlock(32, 3, 1, 0, True)
        self.blk8 = GSameResolution3LayerBlock(32, 16, 3, False, 3, 3)
        self.rgb8 = GConvBlock(16, 3, 1, 0, True)
        self.block = [self.blk0, self.blk1, self.blk2, self.blk3, self.blk4, self.blk5, self.blk6, self.blk7, self.blk8]
        self.to_rgb = [self.rgb0, self.rgb1, self.rgb2, self.rgb3, self.rgb4, self.rgb5, self.rgb6, self.rgb7, self.rgb8]
        
    def forward(self, x, level=None):
        if level is None:
            level = self.n_level
        assert 1 <= level <= self.n_level
        x = x.view(-1, 512, 1, 1)
        for i in range(level):
            x = self.block[i](x)
        x = self.to_rgb[level-1](x)
        return x


class G_celeba_256(nn.Module):
    def __init__(self):
        super(G_celeba_256, self).__init__()
        self.n_level = 7
        self.blk0 = GSameResolution3LayerBlock(512, 512, 3, True, 4, 3)
        self.rgb0 = GConvBlock(512, 3, 1, 0, True)
        self.blk1 = GSameResolution3LayerBlock(512, 512, 3, False, 3, 3)
        self.rgb1 = GConvBlock(512, 3, 1, 0, True)
        self.blk2 = GSameResolution3LayerBlock(512, 256, 3, False, 3, 3)
        self.rgb2 = GConvBlock(256, 3, 1, 0, True)
        self.blk3 = GSameResolution3LayerBlock(256, 128, 3, False, 3, 3)
        self.rgb3 = GConvBlock(128, 3, 1, 0, True)
        self.blk4 = GSameResolution3LayerBlock(128, 64, 3, False, 3, 3)
        self.rgb4 = GConvBlock(64, 3, 1, 0, True)
        self.blk5 = GSameResolution3LayerBlock(64, 32, 3, False, 3, 3)
        self.rgb5 = GConvBlock(32, 3, 1, 0, True)
        self.blk6 = GSameResolution3LayerBlock(32, 16, 3, False, 3, 3)
        self.rgb6 = GConvBlock(16, 3, 1, 0, True)
        self.block = [self.blk0, self.blk1, self.blk2, self.blk3, self.blk4, self.blk5, self.blk6]
        self.to_rgb = [self.rgb0, self.rgb1, self.rgb2, self.rgb3, self.rgb4, self.rgb5, self.rgb6]
        
    def forward(self, x, level=None):
        if level is None:
            level = self.n_level
        assert 1 <= level <= self.n_level
        x = x.view(-1, 512, 1, 1)
        for i in range(level):
            x = self.block[i](x)
        x = self.to_rgb[level-1](x)
        return x


class D_celeba_1024(nn.Module):
    def __init__(self):
        super(D_celeba_1024, self).__init__()
        self.n_level = 9
        self.blk0 = DDownsmple3LayerBlock(16, 32, 3, False, 3, 3)
        self.rgb0 = DConvblock(3, 16, 1, 0)
        self.blk1 = DDownsmple3LayerBlock(32, 64, 3, False, 3, 3)
        self.rgb1 = DConvblock(3, 32, 1, 0)
        self.blk2 = DDownsmple3LayerBlock(64, 128, 3, False, 3, 3)
        self.rgb2 = DConvblock(3, 64, 1, 0)
        self.blk3 = DDownsmple3LayerBlock(128, 256, 3, False, 3, 3)
        self.rgb3 = DConvblock(3, 256, 1, 0)
        self.blk4 = DDownsmple3LayerBlock(256, 512, 3, False, 3, 3)
        self.rgb4 = DConvblock(3, 512, 1, 0)
        self.blk5 = DDownsmple3LayerBlock(512, 512, 3, False, 3, 3)
        self.rgb5 = DConvblock(3, 512, 1, 0)
        self.blk6 = DDownsmple3LayerBlock(512, 512, 3, False, 3, 3)
        self.rgb6 = DConvblock(3, 512, 1, 0)
        self.blk7 = DDownsmple3LayerBlock(512, 512, 3, False, 3, 3)
        self.rgb7 = DConvblock(3, 512, 1, 0)
        self.blk8 = DDownsmple3LayerBlock(512, 512, 3, True, 3, 4)
        self.rgb8 = DConvblock(3, 512, 1, 0)
        self.block = [self.blk0, self.blk1, self.blk2, self.blk3, self.blk4, self.blk5, self.blk6, self.blk7, self.blk8]
        self.from_rgb = [self.rgb0, self.rgb1, self.rgb2, self.rgb3, self.rgb4, self.rgb5, self.rgb6, self.rgb7, self.rgb8]

    def forward(self, x, level=None):
        if level is None:
            level = self.n_level
        assert 1 <= level <= self.n_level
        x = self.from_rgb[self.n_level-level](x)
        for i in range(level):
            x = self.block[self.n_level-level+i](x)
        return x


class D_celeba_256(nn.Module):
    def __init__(self):
        super(D_celeba_256, self).__init__()
        self.n_level = 7
        self.blk0 = DDownsmple3LayerBlock(16, 32, 3, False, 3, 3)
        self.rgb0 = DConvblock(3, 16, 1, 0)
        self.blk1 = DDownsmple3LayerBlock(32, 64, 3, False, 3, 3)
        self.rgb1 = DConvblock(3, 32, 1, 0)
        self.blk2 = DDownsmple3LayerBlock(64, 128, 3, False, 3, 3)
        self.rgb2 = DConvblock(3, 64, 1, 0)
        self.blk3 = DDownsmple3LayerBlock(128, 256, 3, False, 3, 3)
        self.rgb3 = DConvblock(3, 256, 1, 0)
        self.blk4 = DDownsmple3LayerBlock(256, 512, 3, False, 3, 3)
        self.rgb4 = DConvblock(3, 512, 1, 0)
        self.blk5 = DDownsmple3LayerBlock(512, 512, 3, False, 3, 3)
        self.rgb5 = DConvblock(3, 512, 1, 0)
        self.blk6 = DDownsmple3LayerBlock(512, 512, 3, True, 3, 4)
        self.rgb6 = DConvblock(3, 512, 1, 0)
        self.block = [self.blk0, self.blk1, self.blk2, self.blk3, self.blk4, self.blk5, self.blk6]
        self.from_rgb = [self.rgb0, self.rgb1, self.rgb2, self.rgb3, self.rgb4, self.rgb5, self.rgb6]

    def forward(self, x, level=None):
        if level is None:
            level = self.n_level
        assert 1 <= level <= self.n_level
        x = self.from_rgb[self.n_level-level](x)
        for i in range(level):
            x = self.block[self.n_level-level+i](x)
        return x
