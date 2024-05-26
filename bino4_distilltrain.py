import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
import gc
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm, gc
from colorama import init, Fore
init()  # Init Colorama
color_code = Fore.BLUE  
try: from apex import amp 
except Exception:
    amp = None
    print('WARNING: could not import pygcransac')
    pass
from data.datamgr import SimpleDataManager, SetDataManager
from utils import *
from methods.bino1_template import Bino_BaselineTrain

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def train_distill(params, base_loader, val_loader, model, model_t, stop_epoch):
    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=params.distill_lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    if params.amp_opt_level != "O0" and amp is not None:  model, optimizer = amp.initialize(model, optimizer, opt_level=params.amp_opt_level)

    loss_fn = nn.CrossEntropyLoss()
    loss_div_fn = DistillKL(4)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=params.milestones, gamma=params.gamma)
    
    if not os.path.isdir(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)

    path_file = os.path.join(params.checkpoint_dir, 'train_log.txt')
    log_file = open(path_file, 'w')
    log_file.write(f'params: {params}\n')
    
    log_file.close()

    start_epoch=0
    
    if params.check_resume=='True':
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, os.path.join(params.checkpoint_dir,'checkpoint_epoch_'+ str(params.check_epoch) +'.pt'))
        print(f'Model loaded from epoch {start_epoch}')
    else: print('从头开始训练！')

    for epoch in range(start_epoch, stop_epoch):
        log_file = open(path_file, 'a')
        start = time.time()
        model.train()
        print_freq = 200
        avg_loss = 0
        avg_acc = 0
        start_time = time.time()
        tqdm_gen = tqdm.tqdm(base_loader, bar_format="{l_bar}%s{bar}%s{r_bar}" % (color_code, Fore.RESET))

        for i, (x, y) in enumerate(tqdm_gen):
            x = Variable(x.cuda())
            y = Variable(y.cuda())

            with torch.no_grad(): 
                scores_t_l, scores_t_r = model_t(x)
                scores_t_l_soft = torch.softmax(scores_t_l, dim=1)
                scores_t_r_soft = torch.softmax(scores_t_r, dim=1)
                
            scores_l, scores_r = model(x)
            scores_l_soft = torch.softmax(scores_l, dim=1)
            scores_r_soft = torch.softmax(scores_r, dim=1)
            scores = (scores_l_soft + scores_r_soft) / 2   
            
            loss_cls = loss_fn(scores, y)
            pred = scores.data.max(1)[1]
            train_acc = pred.eq(y.data.view_as(pred)).sum()
 
            loss_div = loss_div_fn(scores_l, scores_t_l) + loss_div_fn(scores_r, scores_t_r)
            loss = loss_cls * 0.5 + loss_div * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            avg_acc = avg_acc + train_acc.item()

            if i % print_freq == 0:
                print('Epoch {:d}/{:d} | Batch {:d} | Acc {:.3f} | Loss {:f} | Time {:.2f}'.format(epoch, stop_epoch, len(base_loader), avg_acc, avg_loss/float(i + 1), time.time()-start_time))
            start_time = time.time()


        model.eval()
        if params.val == 'last':
            valObj, acc = model.meta_test_loop(val_loader)
            print(Fore.RED +"val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
                
            log_file.write(f'epoch: {epoch}, val loss: {valObj}, val acc: {acc}\n')
                        
            trlog['val_loss'].append(valObj)
            trlog['val_acc'].append(acc)
            if acc > trlog['max_acc']:
                print(Fore.BLUE +"best model! save...")
                trlog['max_acc'] = acc
                trlog['max_acc_epoch'] = epoch
                max_acc, max_epoch = acc, epoch
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                log_file.write(f'          best model! save...\n')
                
            log_file.write(f'          current best acc: {max_acc}, best acc epoch: {max_epoch}\n')
            print(Fore.RED +"model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))


        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        
        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        if params.optimizer == 'SGD':
            lr_scheduler.step()
        elif params.optimizer == 'Adam':
            optimizer.step()
            lr_scheduler.step()
            
        print(Fore.RED +'Current Distill_lr:',  display_learning_rate(optimizer))
        print(Fore.RED +"This epoch use %.2f minutes" % ((time.time() - start) / 60))
        log_file.write(f'          Current Pre_lr: { display_learning_rate(optimizer)}. Epoch Time-consuming: {(time.time() - start) / 60}\n')
        
        torch.cuda.empty_cache()
        gc.collect()
        log_file.close()

    return model


if __name__ == '__main__':
    option_dataset=['cub_cropped','aircraft_fs','flowers_fs','stanford_car']
    option_models=['Le_Vision_EConv4', 'Re_Vision_EConv4', 'Le_Vision_EResnet12', 'Re_Vision_EResnet12', 
                   'Le_Vision_EConv4_narrow','Re_Vision_EConv4_narrow','Le_Vision_EConv4_middle','Re_Vision_EConv4_middle',
                   'Le_Vision_EResnet12_narrow','Re_Vision_EResnet12_narrow','Le_Vision_EResnet12_middle','Re_Vision_EResnet12_middle',
                   'Le_Vision_EResnet12_split_vision','Re_Vision_EResnet12_split_vision','Le_Vision_EResnet12_split_egcm',
                   'Re_Vision_EResnet12_split_egcm','Le_Vision_EResnet12_split_vision_egcm','Re_Vision_EResnet12_split_vision_egcm']
    option_methods=['BinoHeD']
    vision_methods=['pure', 'disparity']
    FLLS_choice=['shared', 'independent']
    AELS_choice=['shared', 'independent']
    DRS_choice=['GCM', 'GRM', 'SVD']
    
    ################################
    # 0 超参数
    ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--batch_size', default=64, type=int, help='pre-training batch size')
    parser.add_argument('--pre_lr', type=float, default=0.05, help='initial learning rate of the backbone')

    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'], help='Collaborative Identification Mechanism(CIM)')
    parser.add_argument('--distill_lr', type=float, default=0.05, help='initial learning rate of the backbone')

    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--t_lr', type=float, default=0.05, help='initial learning rate uesd for the temperature of bdc module')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[80, 120], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=200, type=int, help='stopping epoch')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')

    parser.add_argument('--dataset', default='miniimagenet_middle', choices=option_dataset)
    parser.add_argument('--val_dataset', default='miniimagenet_middle', choices=option_dataset)

    parser.add_argument('--data_path', type=str, help='dataset path')
    parser.add_argument('--val_data_path', type=str, help='validation dataset path')
    parser.add_argument('--model_l', default='ResNet12', choices=option_models)
    parser.add_argument('--model_r', default='ResNet12', choices=option_models)

    parser.add_argument('--method', default='stl_deepbdc', choices=option_methods)
    parser.add_argument('--vision_method', default='pure', choices=vision_methods)

    parser.add_argument('--val', default='meta', choices=['meta', 'last'], help='validation method')
    parser.add_argument('--val_n_episode', default=1000, type=int, help='number of episodes in meta validation')
    parser.add_argument('--val_n_way', default=5, type=int, help='number of  classes to classify in meta validation')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support during meta validation')
    parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

    parser.add_argument('--extra_dir', default='/distill_train/', help='recording additional information')
    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in training')
    parser.add_argument('--save_freq', default=50, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    parser.add_argument('--teacher_path', default='', help='teacher model .tar file path')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('--FLLS', default='shared', choices=FLLS_choice, help='learning style of BSFE')
    parser.add_argument('--AELS', default='independent', choices=AELS_choice, help='learning style of Singular HeM')
    
    parser.add_argument('--DRS', default='GCM', choices=DRS_choice, help='Dimention Reduction Style(DRS)')
    parser.add_argument('--CIM', default='ARS', choices=['WAS', 'ARS'], help='Collaborative Identification Mechanism(CIM)')
    
    # 断点保护：
    parser.add_argument('--check_epoch', default=0, type=int, help='Collaborative Identification Mechanism(CIM)')
    parser.add_argument("--check_resume", type=str, default='False', choices=['False', 'True'])  # Input method

    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')
    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')

    ################################
    # DistributedDataParallel
    ################################
    parser.add_argument('--amp_opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')


    params = parser.parse_args()

    num_gpu = set_gpu(params)
    set_seed(params.seed)


    ################################
    # 1 加载数据集
    ################################
    params.val_dataset = params.dataset
    params.val_data_path = params.data_path
    if params.val == 'last': val_file = None
    elif params.val == 'meta': val_file = 'val'

    # 设置数据集的类别数量：
    if params.dataset == 'cub_cropped':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 100
    elif params.dataset == 'aircraft_fs':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 50
    elif params.dataset == 'flowers_fs':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 71
    elif params.dataset == 'stanford_car':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 130
    else:
        ValueError('dataset error')

    base_datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size, dataset=params.dataset)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    if params.val == 'last':
        test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(params.val_data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, dataset=params.val_dataset, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    else:
        val_loader = None

    ################################
    # 2 初始化网络模型
    ################################
    model = Bino_BaselineTrain(params, model_dict[params.model_l], model_dict[params.model_r], params.num_classes)    # Student
    model_t = Bino_BaselineTrain(params, model_dict[params.model_l], model_dict[params.model_r], params.num_classes)  # Teacher

    model = model.cuda()
    model_t = model_t.cuda()

    if params.FLLS=='shared': FLLS='share'
    else: FLLS='split'
    if params.AELS=='shared': AELS='share'
    else: AELS='split'
    
    if params.model_l == 'Le_Vision_EConv4' and params.model_r == 'Re_Vision_EConv4':
        params.checkpoint_dir = './checkpoints/%s/Bino_Vision_EConv4_%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM)
    elif params.model_l == 'Le_Vision_EResnet12' and params.model_r == 'Re_Vision_EResnet12':
        params.checkpoint_dir = './checkpoints/%s/Bino_Vision_EResnet12_%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM) # 依据dataset,model和method，来初始化BaselineTrain。

    elif params.model_l == 'Le_Vision_EConv4_narrow' and params.model_r == 'Re_Vision_EConv4_narrow':
        params.checkpoint_dir = './checkpoints/%s/Bino_Vision_EConv4_narrow_%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM) # 依据dataset,model和method，来初始化BaselineTrain。
    elif params.model_l == 'Le_Vision_EConv4_middle' and params.model_r == 'Re_Vision_EConv4_middle':
        params.checkpoint_dir = './checkpoints/%s/Bino_Vision_EConv4_middle_%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM) # 依据dataset,model和method，来初始化BaselineTrain。
    
    elif params.model_l == 'Le_Vision_EResnet12_narrow' and params.model_r == 'Re_Vision_EResnet12_narrow':
        params.checkpoint_dir = './checkpoints/%s/Bino_Vision_EResnet12_narrow_%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM) # 依据dataset,model和method，来初始化BaselineTrain。
    elif params.model_l == 'Le_Vision_EResnet12_middle' and params.model_r == 'Re_Vision_EResnet12_middle':
        params.checkpoint_dir = './checkpoints/%s/Bino_Vision_EResnet12_middle_%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM) # 依据dataset,model和method，来初始化BaselineTrain。
    
    elif params.model_l == 'Le_Vision_EResnet12_split_vision' and params.model_r == 'Re_Vision_EResnet12_split_vision':
        params.checkpoint_dir = './checkpoints/%s/Bino_Vision_EResnet12_split_vision_%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM) # 依据dataset,model和method，来初始化BaselineTrain。
    elif params.model_l == 'Le_Vision_EResnet12_split_egcm' and params.model_r == 'Re_Vision_EResnet12_split_egcm':
        params.checkpoint_dir = './checkpoints/%s/Bino_Vision_EResnet12_split_egcm_%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM) # 依据dataset,model和method，来初始化BaselineTrain。
    elif params.model_l == 'Le_Vision_EResnet12_split_vision_egcm' and params.model_r == 'Re_Vision_EResnet12_split_vision_egcm':
        params.checkpoint_dir = './checkpoints/%s/Bino_Vision_EResnet12_split_vision_egcm_%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM) # 依据dataset,model和method，来初始化BaselineTrain。
    
    else:
        params.checkpoint_dir = './checkpoints/%s/%s_%s/%s_%s/FLLS_%s_AELS_%s_DRS_%s_CIM_%s' % (params.dataset, params.model_l, params.model_r, params.method, params.vision_method, FLLS, AELS, params.DRS, params.CIM) # 依据dataset,model和method，来初始化BaselineTrain。

    params.checkpoint_dir += '/distill_born%s_%s_%s_%s_%s_%s_%dway_%dshot' % (params.trial, params.optimizer, params.batch_size, float(params.pre_lr), params.val_n_episode, params.epoch, params.val_n_way, params.n_shot)
    
    params.checkpoint_dir += params.extra_dir

    if not os.path.isdir(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
    print(params)
    print(model)

    modelfile = os.path.join(params.teacher_path)
    tmp = torch.load(modelfile)
    state = tmp['state']
    model_t.load_state_dict(state)
    print(params)

    model = train_distill(params, base_loader, val_loader, model, model_t, params.epoch)
