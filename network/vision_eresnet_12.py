# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch, time
import numpy as np
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from e2cnn import gspaces   
from e2cnn import nn as enn
import warnings
from colorama import init, Fore
init()  # Init Colorama
warnings.filterwarnings("ignore")

class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()
            return block_mask * x * (countM / count_ones)
        else: return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]
        offsets = torch.stack([  
               torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
               torch.arange(self.block_size).repeat(self.block_size), ]).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()
            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
        block_mask = 1 - padded_mask
        return block_mask

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=5, stride=stride, padding=2, bias=False)

class IRL_Block(nn.Module): # Inner Residual Layers
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=False):
        super(IRL_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.maxpool = nn.MaxPool2d(stride, ceil_mode=True)
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.pool = pool
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        
        if self.pool: out = self.maxpool(out)

        out = self.dropout(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out

class OSL_Block1(nn.Module): # Outer Synthetic Layer
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=True):
        super(OSL_Block1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride, ceil_mode=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.pool = pool
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.pool: out = self.maxpool(out)
        
        out = self.dropout(out)  

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out

class OSL_Block2(nn.Module): # Outer Synthetic Layer
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=True):
        super(OSL_Block2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride, ceil_mode=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.pool = pool
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.pool: out = self.maxpool(out)
        
        out = self.dropout(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out

# VisionEResnet12
class LE_Vision_EResnet(nn.Module):
    def __init__(self, InnerBlock, OuterBlock1, OuterBlock2, mode=True, method='fft', truncate_ratio=0.85, net_channel='narrow', 
                 drop_block=False, irl_dropout_rate=0.0, osl_dropout_rate=0.0, drop_rate=0.0, rot_elems=8, dropblock_size=5):
        self.mode, self.method, self.truncate_ratio = mode, method, truncate_ratio
        self.net_channel, self.rot_elems = net_channel, rot_elems
        super(LE_Vision_EResnet, self).__init__()
        
        # Equivariant Geometric Constraint Module(EGCM):
        self.r2_act = gspaces.Rot2dOnR2(N=self.rot_elems)
        if self.mode and (self.method == 'fft'):
            self.EGCM = enn.SequentialModule(
                enn.R2Conv( enn.FieldType(self.r2_act, 6*[self.r2_act.trivial_repr]), 
                            enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr]), kernel_size=5), 
                enn.ReLU(enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])), 
                enn.R2Conv( enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr]), 
                            enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr]), kernel_size=3),
                enn.ReLU(enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])),
                )
            # self.EGCM = self.EGCM.cuda()
            
            self.inplanes = 6 * self.rot_elems
        elif self.mode: self.inplanes = 6
        else: self.inplanes = 3

        if self.net_channel == 'middle':
            self.feat_dim = [512, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 64, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 128, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 256, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 512, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'narrow': 
            self.feat_dim = [480, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 48, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 96, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 224, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 480, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'normal': 
            self.feat_dim = [640, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 80, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 160, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 320, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 640, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        else: raise ValueError('Wrong Network Channel Type!')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def fft_input(self, X, truncate_ratio):
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device)
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft

    def vision_layer(self, X):
        if self.method =='fft':   
            x_rec = self.fft_input(X, self.truncate_ratio) 
            x_cat = torch.cat([X, x_rec], dim=1)
        else: raise ValueError('Wrong vision method!')
        return x_cat

    def forward(self, x):
        if self.mode: x0 = self.vision_layer(x)
        else: x0 =x
        if self.mode and (self.method == 'fft'): 
            x1 = enn.GeometricTensor(x0, enn.FieldType(self.r2_act, 6*[self.r2_act.trivial_repr]))
            x1 = self.EGCM(x1).tensor         
        else: x1 = x0
        x2 = self.IRL_layer1(x1)
        x3 = self.IRL_layer2(x2)
        x4 = self.OSL_layer1(x3)
        le_out = self.OSL_layer2(x4)
        return le_out

class RE_Vision_EResnet(nn.Module):
    def __init__(self, InnerBlock, OuterBlock1, OuterBlock2, mode=True, method='fft', truncate_ratio=0.85, net_channel='narrow', 
                 drop_block=False, irl_dropout_rate=0.0, osl_dropout_rate=0.0, drop_rate=0.0, rot_elems=8, dropblock_size=5):
        self.mode, self.method, self.truncate_ratio = mode, method, truncate_ratio
        self.net_channel, self.rot_elems = net_channel, rot_elems
        super(RE_Vision_EResnet, self).__init__()
        
        # Equivariant Geometric Constraint Module(EGCM):
        self.r2_act = gspaces.Rot2dOnR2(N=self.rot_elems)
        if self.mode and (self.method == 'fft'):
            self.EGCM = enn.SequentialModule(
                enn.R2Conv( enn.FieldType(self.r2_act, 6*[self.r2_act.trivial_repr]), 
                            enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr]), kernel_size=5), 
                enn.ReLU(enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])), 
                enn.R2Conv( enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr]), 
                            enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr]), kernel_size=3),
                enn.ReLU(enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])),
                )
            self.inplanes = 6 * self.rot_elems
            # self.EGCM = self.EGCM.cuda()
        elif self.mode: self.inplanes = 6
        else: self.inplanes = 3

        if self.net_channel == 'middle':
            self.feat_dim = [512, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 64, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 128, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 256, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 512, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'narrow': 
            self.feat_dim = [480, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 48, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 96, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 224, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 480, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'normal': 
            self.feat_dim = [640, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 80, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 160, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 320, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 640, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        else: raise ValueError('Wrong Network Channel Type!')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def fft_input(self, X, truncate_ratio):
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device)
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft

    def vision_layer(self, X):
        if self.method =='fft':   
            x_rec = self.fft_input(X, self.truncate_ratio) 
            x_cat = torch.cat([x_rec, X], dim=1)
        else: raise ValueError('Wrong vision method!')
        return x_cat

    def forward(self, x):
        if self.mode: x0 = self.vision_layer(x) 
        else: x0 =x
        if self.mode and (self.method == 'fft'): 
            x1 = enn.GeometricTensor(x0, enn.FieldType(self.r2_act, 6*[self.r2_act.trivial_repr]))
            x1 = self.EGCM(x1).tensor  
        else: x1 = x0
        x2 = self.IRL_layer1(x1)
        x3 = self.IRL_layer2(x2)
        x4 = self.OSL_layer1(x3)
        re_out = self.OSL_layer2(x4)
        return re_out

class LE_Vision_EResnet_split_vision(nn.Module):
    def __init__(self, InnerBlock, OuterBlock1, OuterBlock2, mode=True, method='fft', truncate_ratio=0.85, net_channel='narrow', 
                 drop_block=False, irl_dropout_rate=0.0, osl_dropout_rate=0.0, drop_rate=0.0, rot_elems=8, dropblock_size=5):
        self.mode, self.method, self.truncate_ratio = mode, method, truncate_ratio
        self.net_channel, self.rot_elems = net_channel, rot_elems
        super(LE_Vision_EResnet_split_vision, self).__init__()
        
        # Equivariant Geometric Constraint Module(EGCM):
        self.r2_act = gspaces.Rot2dOnR2(N=self.rot_elems)
        self.EGCM = enn.SequentialModule(
                enn.R2Conv( enn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr]), 
                            enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr]), kernel_size=5), 
                enn.ReLU(enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr])), 
                enn.R2Conv( enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr]), 
                            enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr]), kernel_size=3),
                enn.ReLU(enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr])),
                )
        self.inplanes = 3 * self.rot_elems
        # self.EGCM = self.EGCM.cuda()

        if self.net_channel == 'middle':
            self.feat_dim = [512, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 64, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 128, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 256, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 512, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'narrow': 
            self.feat_dim = [480, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 48, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 96, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 224, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 480, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'normal': 
            self.feat_dim = [640, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 80, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 160, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 320, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 640, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        else: raise ValueError('Wrong Network Channel Type!')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def fft_input(self, X, truncate_ratio):
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device) 
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft

    def vision_layer(self, X):
        if self.method =='fft':   
            x_rec = self.fft_input(X, self.truncate_ratio) 
            x_cat = torch.cat([X, x_rec], dim=1)
        else: raise ValueError('Wrong vision method!')
        return x_cat

    def forward(self, x):
        x0 =x
        x1 = enn.GeometricTensor(x0, enn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr]))
        x1 = self.EGCM(x1).tensor
        x2 = self.IRL_layer1(x1)
        x3 = self.IRL_layer2(x2)
        x4 = self.OSL_layer1(x3)
        le_out = self.OSL_layer2(x4)
        return le_out

class RE_Vision_EResnet_split_vision(nn.Module):
    def __init__(self, InnerBlock, OuterBlock1, OuterBlock2, mode=True, method='fft', truncate_ratio=0.85, net_channel='narrow', 
                 drop_block=False, irl_dropout_rate=0.0, osl_dropout_rate=0.0, drop_rate=0.0, rot_elems=8, dropblock_size=5):
        self.mode, self.method, self.truncate_ratio = mode, method, truncate_ratio
        self.net_channel, self.rot_elems = net_channel, rot_elems
        super(RE_Vision_EResnet_split_vision, self).__init__()
        
        # Equivariant Geometric Constraint Module(EGCM):
        self.r2_act = gspaces.Rot2dOnR2(N=self.rot_elems)
        self.EGCM = enn.SequentialModule(
                enn.R2Conv( enn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr]), 
                            enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr]), kernel_size=5), 
                enn.ReLU(enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr])), 
                enn.R2Conv( enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr]), 
                            enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr]), kernel_size=3),
                enn.ReLU(enn.FieldType(self.r2_act, 3*[self.r2_act.regular_repr])),
                )
        self.inplanes = 3 * self.rot_elems
        # self.EGCM = self.EGCM.cuda()
        
        if self.net_channel == 'middle':
            self.feat_dim = [512, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 64, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 128, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 256, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 512, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'narrow': 
            self.feat_dim = [480, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 48, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 96, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 224, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 480, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'normal': 
            self.feat_dim = [640, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 80, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 160, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 320, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 640, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        else: raise ValueError('Wrong Network Channel Type!')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def fft_input(self, X, truncate_ratio): 
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device)
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft

    def vision_layer(self, X):
        if self.method =='fft':   
            x_rec = self.fft_input(X, self.truncate_ratio) 
            x_cat = torch.cat([x_rec, X], dim=1)
        else: raise ValueError('Wrong vision method!')
        return x_cat

    def forward(self, x):
        x0 =x
        x1 = enn.GeometricTensor(x0, enn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr]))
        x1 = self.EGCM(x1).tensor 
        x2 = self.IRL_layer1(x1)
        x3 = self.IRL_layer2(x2)
        x4 = self.OSL_layer1(x3)
        re_out = self.OSL_layer2(x4)
        return re_out

class LE_Vision_EResnet_split_egcm(nn.Module):
    def __init__(self, InnerBlock, OuterBlock1, OuterBlock2, mode=True, method='no', truncate_ratio=0.85, net_channel='narrow', 
                 drop_block=False, irl_dropout_rate=0.0, osl_dropout_rate=0.0, drop_rate=0.0, rot_elems=8, dropblock_size=5):
        self.mode, self.method, self.truncate_ratio = mode, method, truncate_ratio
        self.net_channel, self.rot_elems = net_channel, rot_elems
        super(LE_Vision_EResnet_split_egcm, self).__init__()
        
        self.replace = nn.Sequential(
            nn.Conv2d(6,  48, kernel_size=5, stride=1, padding=2, bias=False),  # 5x5 convolutional layer
            nn.ReLU(),                                                          # Activation function
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),  # 3x3 convolutional layer
            nn.ReLU()                                                           # Activation function 
            )
        self.inplanes = 6 * self.rot_elems

        if self.net_channel == 'middle':
            self.feat_dim = [512, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 64, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 128, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 256, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 512, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'narrow': 
            self.feat_dim = [480, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 48, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 96, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 224, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 480, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'normal': 
            self.feat_dim = [640, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 80, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 160, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 320, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 640, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        else: raise ValueError('Wrong Network Channel Type!')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def fft_input(self, X, truncate_ratio): 
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device)
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft

    def vision_layer(self, X):
        x_rec = self.fft_input(X, self.truncate_ratio) 
        x_cat = torch.cat([X, x_rec], dim=1)
        return x_cat

    def forward(self, x):
        if self.mode: x0 = self.vision_layer(x)
        else: x0 =x
        x1=self.replace(x0)
        x2 = self.IRL_layer1(x1)
        x3 = self.IRL_layer2(x2)
        x4 = self.OSL_layer1(x3)
        le_out = self.OSL_layer2(x4)
        return le_out
    
class RE_Vision_EResnet_split_egcm(nn.Module):
    def __init__(self, InnerBlock, OuterBlock1, OuterBlock2, mode=True, method='no', truncate_ratio=0.85, net_channel='narrow', 
                 drop_block=False, irl_dropout_rate=0.0, osl_dropout_rate=0.0, drop_rate=0.0, rot_elems=8, dropblock_size=5):
        self.mode, self.method, self.truncate_ratio = mode, method, truncate_ratio
        self.net_channel, self.rot_elems = net_channel, rot_elems
        super(RE_Vision_EResnet_split_egcm, self).__init__()
        
        self.replace = nn.Sequential(
            nn.Conv2d(6,  48, kernel_size=5, stride=1, padding=2, bias=False),  # 5x5 convolutional layer
            nn.ReLU(),                                                          # Activation function
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),  # 3x3 convolutional layer
            nn.ReLU()                                                           # Activation function 
            )
        self.inplanes = 6 * self.rot_elems
        
        if self.net_channel == 'middle':
            self.feat_dim = [512, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 64, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 128, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 256, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 512, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'narrow': 
            self.feat_dim = [480, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 48, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 96, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 224, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 480, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'normal': 
            self.feat_dim = [640, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 80, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 160, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 320, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 640, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        else: raise ValueError('Wrong Network Channel Type!')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def fft_input(self, X, truncate_ratio):
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device)
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft

    def vision_layer(self, X):
        x_rec = self.fft_input(X, self.truncate_ratio) 
        x_cat = torch.cat([x_rec, X], dim=1)
        return x_cat

    def forward(self, x):
        if self.mode: x0 = self.vision_layer(x)
        else: x0 =x
        x1=self.replace(x0)
        x2 = self.IRL_layer1(x1)
        x3 = self.IRL_layer2(x2)
        x4 = self.OSL_layer1(x3)
        re_out = self.OSL_layer2(x4)
        return re_out

class LE_Vision_EResnet_split_vision_egcm(nn.Module):
    def __init__(self, InnerBlock, OuterBlock1, OuterBlock2, mode=True, method='no', truncate_ratio=0.85, net_channel='narrow', 
                 drop_block=False, irl_dropout_rate=0.0, osl_dropout_rate=0.0, drop_rate=0.0, rot_elems=8, dropblock_size=5):
        self.mode, self.method, self.truncate_ratio = mode, method, truncate_ratio
        self.net_channel, self.rot_elems = net_channel, rot_elems
        super(LE_Vision_EResnet_split_vision_egcm, self).__init__()
        
        self.replace = nn.Sequential(
            nn.Conv2d(6,  48, kernel_size=5, stride=1, padding=2, bias=False),  # 5x5 convolutional layer
            nn.ReLU(),                                                          # Activation function
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),  # 3x3 convolutional layer
            nn.ReLU()                                                           # Activation function 
            )
        self.inplanes = 6 * self.rot_elems

        if self.net_channel == 'middle':
            self.feat_dim = [512, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 64, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 128, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 256, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 512, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'narrow': 
            self.feat_dim = [480, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 48, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 96, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 224, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 480, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'normal': 
            self.feat_dim = [640, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 80, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 160, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 320, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 640, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        else: raise ValueError('Wrong Network Channel Type!')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def fft_input(self, X, truncate_ratio):
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device)
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft

    def vision_layer(self, X):
        x_rec = self.fft_input(X, self.truncate_ratio) 
        x_cat = torch.cat([X, x_rec], dim=1)
        return x_cat

    def forward(self, x):
        x1 = torch.cat([x, x], dim=1)
        x1 = self.replace(x1)
        x2 = self.IRL_layer1(x1)
        x3 = self.IRL_layer2(x2)
        x4 = self.OSL_layer1(x3)
        le_out = self.OSL_layer2(x4)
        return le_out

class RE_Vision_EResnet_split_vision_egcm(nn.Module):
    def __init__(self, InnerBlock, OuterBlock1, OuterBlock2, mode=True, method='no', truncate_ratio=0.85, net_channel='narrow', 
                 drop_block=False, irl_dropout_rate=0.0, osl_dropout_rate=0.0, drop_rate=0.0, rot_elems=8, dropblock_size=5):
        self.mode, self.method, self.truncate_ratio = mode, method, truncate_ratio
        self.net_channel, self.rot_elems = net_channel, rot_elems
        super(RE_Vision_EResnet_split_vision_egcm, self).__init__()
        
        self.replace = nn.Sequential(
            nn.Conv2d(6,  48, kernel_size=5, stride=1, padding=2, bias=False),  # 5x5 convolutional layer
            nn.ReLU(),                                                          # Activation function
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),  # 3x3 convolutional layer
            nn.ReLU()                                                           # Activation function 
            )
        self.inplanes = 6 * self.rot_elems
        
        if self.net_channel == 'middle':
            self.feat_dim = [512, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 64, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 128, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 256, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 512, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'narrow': 
            self.feat_dim = [480, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 48, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 96, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 224, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 480, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        elif self.net_channel == 'normal': 
            self.feat_dim = [640, 10, 10]
            self.IRL_layer1 = self._make_layer(InnerBlock, 80, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.IRL_layer2 = self._make_layer(InnerBlock, 160, stride=2, dropout_rate=irl_dropout_rate, drop_rate=drop_rate, pool=True)
            self.OSL_layer1 = self._make_layer(OuterBlock1, 320, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
            self.OSL_layer2 = self._make_layer(OuterBlock2, 640, stride=2, dropout_rate=osl_dropout_rate, drop_rate=drop_rate, drop_block=drop_block, block_size=dropblock_size, pool=True)
        else: raise ValueError('Wrong Network Channel Type!')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dropout_rate=0.0, drop_rate=0.0, drop_block=False, block_size=1, pool=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def fft_input(self, X, truncate_ratio):
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device)
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft

    def vision_layer(self, X):
        x_rec = self.fft_input(X, self.truncate_ratio) 
        x_cat = torch.cat([x_rec, X], dim=1)
        return x_cat

    def forward(self, x):
        x1 = torch.cat([x, x], dim=1)
        x1 = self.replace(x1)
        x2 = self.IRL_layer1(x1)
        x3 = self.IRL_layer2(x2)
        x4 = self.OSL_layer1(x3)
        re_out = self.OSL_layer2(x4)
        return re_out

#########################
# VisionEResnet12-normal
#########################
def Le_Vision_EResnet_12(mode_=True, method_='fft', tru_ratio_=0.85, net_channel_='normal', drop_block=True, **kwargs):
    """Constructs a LE_Vision_EResnet model."""
    model = LE_Vision_EResnet(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'LE_Vision_EResnet_12参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model
def Re_Vision_EResnet_12(mode_=True, method_='fft', tru_ratio_=0.85, net_channel_='normal', drop_block=True, **kwargs):
    """Constructs a RE_Vision_EResnet model."""
    model = RE_Vision_EResnet(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'RE_Vision_EResnet_12参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model

#########################
# VisionEResnet12-narrow
#########################
def Le_Vision_EResnet_12_narrow(mode_=True, method_='fft', tru_ratio_=0.85, net_channel_='narrow', drop_block=True, **kwargs):
    """Constructs a LE_Vision_EResnet model."""
    model = LE_Vision_EResnet(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'LE_Vision_EResnet_12参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model

def Re_Vision_EResnet_12_narrow(mode_=True, method_='fft', tru_ratio_=0.85, net_channel_='narrow', drop_block=True, **kwargs):
    """Constructs a RE_Vision_EResnet model."""
    model = RE_Vision_EResnet(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'RE_Vision_EResnet_12参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model

#########################
# VisionEResnet12-middle
#########################
def Le_Vision_EResnet_12_middle(mode_=True, method_='fft', tru_ratio_=0.85, net_channel_='middle', drop_block=True, **kwargs):
    """Constructs a LE_Vision_EResnet model."""
    model = LE_Vision_EResnet(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'LE_Vision_EResnet_12参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model

def Re_Vision_EResnet_12_middle(mode_=True, method_='fft', tru_ratio_=0.85, net_channel_='middle', drop_block=True, **kwargs):
    """Constructs a RE_Vision_EResnet model."""
    model = RE_Vision_EResnet(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'RE_Vision_EResnet_12参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model


######################################
# VisionEResnet12-normal_split_vision
######################################
def Le_Vision_EResnet_12_split_vision(mode_=True, method_='fft', tru_ratio_=0.85, net_channel_='normal', drop_block=True, **kwargs):
    """Constructs a LE_Vision_EResnet model."""
    model = LE_Vision_EResnet_split_vision(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'LE_Vision_EResnet_12_split_vision参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model

def Re_Vision_EResnet_12_split_vision(mode_=True, method_='fft', tru_ratio_=0.85, net_channel_='normal', drop_block=True, **kwargs):
    """Constructs a RE_Vision_EResnet model."""
    model = RE_Vision_EResnet_split_vision(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'RE_Vision_EResnet_12_split_vision参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model

######################################
# VisionEResnet12-normal_split_egcm
######################################
def Le_Vision_EResnet_12_split_egcm(mode_=True, method_='no', tru_ratio_=0.85, net_channel_='normal', drop_block=True, **kwargs):
    """Constructs a LE_Vision_EResnet model."""
    model = LE_Vision_EResnet_split_egcm(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'LE_Vision_EResnet_12_split_egcm参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model

def Re_Vision_EResnet_12_split_egcm(mode_=True, method_='no', tru_ratio_=0.85, net_channel_='normal', drop_block=True, **kwargs):
    """Constructs a RE_Vision_EResnet model."""
    model = RE_Vision_EResnet_split_egcm(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'RE_Vision_EResnet_12_split_egcm参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model

###########################################
# VisionEResnet12-normal_split_vision_egcm
###########################################
def Le_Vision_EResnet_12_split_vision_egcm(mode_=False, method_='no', tru_ratio_=0.85, net_channel_='normal', drop_block=True, **kwargs):
    """Constructs a LE_Vision_EResnet model."""
    model = LE_Vision_EResnet_split_vision_egcm(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'LE_Vision_EResnet_12_split_vision_egcm参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model

def Re_Vision_EResnet_12_split_vision_egcm(mode_=False, method_='no', tru_ratio_=0.85, net_channel_='normal', drop_block=True, **kwargs):
    """Constructs a RE_Vision_EResnet model."""
    model = RE_Vision_EResnet_split_vision_egcm(IRL_Block, OSL_Block1, OSL_Block2, mode=mode_, method=method_, truncate_ratio=tru_ratio_, net_channel=net_channel_, 
                              drop_block=drop_block, irl_dropout_rate=0.2, osl_dropout_rate=0.2, drop_rate=0.0, **kwargs)
    print(Fore.RED+'*********'* 10)
    print(Fore.BLUE+'RE_Vision_EResnet_12_split_vision_egcm参数:')
    print(Fore.BLUE+'Vision_method:{0},  EGCM_mode:{1},  FFT_truncate_ratio:{2}'.format(mode_, method_, tru_ratio_))
    print(Fore.RED+'*********'* 10)
    return model


if __name__ == '__main__':
    
    #########################################################################
    # 实例化模型：(normal, middle, narrow, no_vision, no_egcm, no_vision_egcm)
    #########################################################################
    model_depth='no_egcm'
    if model_depth=='normal':
        model_le, model_re = Le_Vision_EResnet_12().cuda(), Re_Vision_EResnet_12().cuda()               # 模型实例化(normal)
    elif model_depth=='narrow':
        model_le, model_re = Le_Vision_EResnet_12_narrow().cuda(), Re_Vision_EResnet_12_narrow().cuda() # 模型实例化(narrow)
    elif model_depth=='middle':
        model_le, model_re = Le_Vision_EResnet_12_middle().cuda(), Re_Vision_EResnet_12_middle().cuda() # 模型实例化(middle)
    elif model_depth=='no_vision':
        model_le, model_re = Le_Vision_EResnet_12_split_vision().cuda(), Re_Vision_EResnet_12_split_vision().cuda() # 模型实例化(middle)
    elif model_depth=='no_egcm':
        model_le, model_re = Le_Vision_EResnet_12_split_egcm().cuda(), Re_Vision_EResnet_12_split_egcm().cuda() # 模型实例化(middle)
    elif model_depth=='no_vision_egcm':
        model_le, model_re = Le_Vision_EResnet_12_split_vision_egcm().cuda(), Re_Vision_EResnet_12_split_vision_egcm().cuda() # 模型实例化(middle)
    print(model_le)
    print(model_re)

    # 输入+ 推理 +输出
    Input_tensor = torch.rand(1, 3, 84, 84)
    print('Input tensor:',Input_tensor.shape)

    s_time1 = time.time()
    for i in range(10): result_le = model_le(Input_tensor.cuda())
    f_time1 = time.time()
    period_1 = "{:.4f}".format(f_time1 - s_time1)
    print('le_time:', period_1) 

    s_time2 = time.time()
    for i in range(10): result_re = model_re(Input_tensor.cuda())
    f_time2 = time.time()
    period_2 = "{:.4f}".format(f_time2 - s_time2)
    print('re_time:', period_2) 
    
    print('Le_Vision_EResnet_12 Result:', result_le.shape)
    print('Re_Vision_EResnet_12 Result:', result_re.shape)