#-*- coding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

import warnings
warnings.filterwarnings('ignore')

import numpy as np


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(
            2).unsqueeze(3).expand_as(x) * x
        return out


class DSOD(nn.Module):
    """docstring for DSOD"""

    def __init__(self, phase, size=300, cfg=None):
        super(DSOD, self).__init__()
        self.phase = phase
        self.size = size
        self.cfg = cfg

        self.Stem = nn.Sequential(
            conv_bn_relu(3, cfg.init_features,
                         kernel_size=3, stride=2, padding=1),
            conv_bn_relu(cfg.init_features, cfg.init_features,
                         kernel_size=3, stride=1, padding=1),
            conv_bn_relu(cfg.init_features, 2 * cfg.init_features,
                         kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        num_features = 2 * cfg.init_features
        channels = [num_features + cfg.growth_rate *
                    _ for _ in np.cumsum(cfg.block_config)]

        self.Block12 = nn.Sequential(
            _DenseBlock(cfg.block_config[0], num_features,
                        cfg.bottleneck_1x1_num, cfg.growth_rate),
            _Transition(channels[0], channels[0], pool=True, ceil_mode=True),
            _DenseBlock(cfg.block_config[1], channels[
                        0], cfg.bottleneck_1x1_num, cfg.growth_rate),
            _Transition(channels[1], channels[1]),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = bn_relu_conv(channels[1], 256, 1)

        self.Block34 = nn.Sequential(
            _DenseBlock(cfg.block_config[2], channels[
                        1], cfg.bottleneck_1x1_num, cfg.growth_rate),
            _Transition(channels[2], channels[2]),
            _DenseBlock(cfg.block_config[3], channels[
                        2], cfg.bottleneck_1x1_num, cfg.growth_rate),
            _Transition(channels[3], channels[3])
        )
        self.conv3 = bn_relu_conv(channels[3], 256, 1)

        self.Extra = nn.ModuleList([
            LHRH(512, 512, ceil_mode=True),
            LHRH(512, 256, ceil_mode=True),
            LHRH(256, 256, ceil_mode=True),
            LHRH(256, 256)])

        n_channels = [channels[1], 512, 512, 256, 256, 256]
        self.L2Norm = nn.ModuleList()
        self.loc = nn.ModuleList()
        self.conf = nn.ModuleList()

        for i, x in enumerate(n_channels):
            n = cfg.anchor_config.anchor_nums[i]
            self.L2Norm.append(L2Norm(x, 20))
            self.loc.append(
                nn.Conv2d(x, n * 4, kernel_size=3, stride=1, padding=1))
            self.conf.append(
                nn.Conv2d(x, n * self.cfg.num_classes, kernel_size=3, stride=1, padding=1))

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        x = self.Stem(x)
        x = self.Block12(x)
        sources += [x]

        x = self.pool2(x)
        x2 = self.conv2(x)

        x = self.Block34(x)
        x = self.conv3(x)
        x = torch.cat((x2, x), dim=1)
        sources += [x]

        for m in self.Extra:
            x = m(x)
            sources += [x]

        for i, x in enumerate(sources):
            x = self.L2Norm[i](x)
            _loc = self.loc[i](x)
            _loc = _loc.permute(0, 2, 3, 1).contiguous().view(
                _loc.size(0), -1, 4)
            loc += [_loc]

            _conf = self.conf[i](x)
            _conf = _conf.permute(0, 2, 3, 1).contiguous().view(
                _conf.size(0), -1, self.cfg.num_classes)
            conf += [_conf]

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.cfg.num_classes))  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.cfg.num_classes)
            )
        return output

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if 'bias' in m.state_dict().keys():
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def bn_relu_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(nn.BatchNorm2d(in_channels),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


class _DenseLayer(nn.Module):
    """docstring for _DenseLayer"""

    def __init__(self, in_channels, growth_rate, bottleneck_1x1_num):
        super(_DenseLayer, self).__init__()
        self.conv1 = bn_relu_conv(
            in_channels, bottleneck_1x1_num, kernel_size=1, stride=1)
        self.conv2 = bn_relu_conv(
            bottleneck_1x1_num, growth_rate, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        new_features = self.conv2(self.conv1(x))
        return torch.cat([x, new_features], dim=1)


class _DenseBlock(nn.Module):
    """docstring for _DenseBlock"""

    def __init__(self, num_layers, in_channels, bottleneck_1x1_num, growth_rate):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in xrange(num_layers):
            layer = _DenseLayer(in_channels + i *
                                growth_rate, growth_rate, bottleneck_1x1_num)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _Transition(nn.Module):
    """docstring for _Transition"""

    def __init__(self, in_channels, out_channels, pool=False, ceil_mode=False):
        super(_Transition, self).__init__()
        self.conv1 = bn_relu_conv(
            in_channels, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2, ceil_mode=ceil_mode) if pool else nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        return out


# Learning Half and Reusing Half
class LHRH(nn.Module):

    def __init__(self, in_channels, out_channels, ceil_mode=False):
        super(LHRH, self).__init__()

        self.conv1_1 = bn_relu_conv(in_channels, int(out_channels / 2), 1)
        self.conv1_2 = bn_relu_conv(int(out_channels / 2), out_channels // 2, 3,
                                    padding=1 * ceil_mode, stride=2)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=ceil_mode)
        self.conv2 = bn_relu_conv(in_channels, out_channels // 2, 1)

    def forward(self, x):
        out1 = self.conv1_2(self.conv1_1(x))
        out2 = self.conv2(self.pool2(x))
        return torch.cat([out1, out2], 1)


def build_net(phase='train', size=300, config=None):
    if not phase in ['test', 'train']:
        raise ValueError("Error: Phase not recognized")

    if size != 300:
        raise NotImplementedError(
            "Error: Sorry only DSOD300 are supported!")

    return DSOD(phase, size, config)

if __name__ == '__main__':
    inputs = torch.randn(2, 3, 300, 300)
    net = DSOD('train')
    out = net(inputs)
