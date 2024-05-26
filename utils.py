import numpy as np
import os
import glob
import argparse
import torch
import random

import network.vision_econv4 as vision_econv4
import network.vision_eresnet_12 as vision_eresnet12

model_dict = dict(
    # VisionECov4 & VisionEResnet12
    Le_Vision_EConv4=vision_econv4.Le_Vision_EConv4,
    Re_Vision_EConv4=vision_econv4.Re_Vision_EConv4,
    Le_Vision_EResnet12=vision_eresnet12.Le_Vision_EResnet_12,
    Re_Vision_EResnet12=vision_eresnet12.Re_Vision_EResnet_12,
    # VisionECov4变体：
    Le_Vision_EConv4_narrow=vision_econv4.Le_Vision_EConv4_narrow,
    Re_Vision_EConv4_narrow=vision_econv4.Re_Vision_EConv4_narrow,
    Le_Vision_EConv4_middle=vision_econv4.Le_Vision_EConv4_middle,
    Re_Vision_EConv4_middle=vision_econv4.Re_Vision_EConv4_middle,
    # VisionEResnet12变体：
    Le_Vision_EResnet12_narrow=vision_eresnet12.Le_Vision_EResnet_12_narrow,
    Re_Vision_EResnet12_narrow=vision_eresnet12.Re_Vision_EResnet_12_narrow,
    Le_Vision_EResnet12_middle=vision_eresnet12.Le_Vision_EResnet_12_middle,
    Re_Vision_EResnet12_middle=vision_eresnet12.Re_Vision_EResnet_12_middle,
    # VisionEResnet构件消融实验：
    Le_Vision_EResnet12_split_vision=vision_eresnet12.Le_Vision_EResnet_12_split_vision,
    Re_Vision_EResnet12_split_vision=vision_eresnet12.Re_Vision_EResnet_12_split_vision,
    Le_Vision_EResnet12_split_egcm=vision_eresnet12.Le_Vision_EResnet_12_split_egcm,
    Re_Vision_EResnet12_split_egcm=vision_eresnet12.Re_Vision_EResnet_12_split_egcm,
    Le_Vision_EResnet12_split_vision_egcm=vision_eresnet12.Le_Vision_EResnet_12_split_vision_egcm,
    Re_Vision_EResnet12_split_vision_egcm=vision_eresnet12.Re_Vision_EResnet_12_split_vision_egcm,
    )


# 加载断点
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # 其他需要加载的内容
    return model, optimizer, epoch

def display_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        current_lr = param_group["lr"]
        # print(f'current lr: {param_group["lr"]}')
    return current_lr

def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    print(best_file)
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def load_model(model, dir):
    model_dict = model.state_dict()
    file_dict = torch.load(dir)['state']
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False