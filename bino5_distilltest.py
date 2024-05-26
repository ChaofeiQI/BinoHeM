import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os, argparse, tqdm
from colorama import init, Fore
init()  # Init Colorama
color_code = Fore.BLUE
try: from apex import amp
except Exception:
    amp = None
    print('WARNING: could not import pygcransac')
    pass
from data.datamgr import SetDataManager
from utils import *
from methods.bino2_singular_HeM import Bino_Singular_HeM

def weighted_average(results1, results2, weight1=0.5, weight2=0.5):
    combined_results = (results1 * weight1 + results2 * weight2) / (weight1 + weight2)
    combined_results = torch.argmax(combined_results, dim=1)
    return combined_results 

def average_rank(results1, results2):
    rank_results1 = torch.softmax(results1, dim=1)
    rank_results2 = torch.softmax(results2, dim=1)
    combined_rank = (rank_results1 + rank_results2) / 2
    combined_results = torch.argmax(combined_rank, dim=1)
    return combined_results

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
    parser.add_argument('--image_size', default=84, type=int, choices=[84, 224])
    parser.add_argument('--dataset', default='miniimagenet_middle', choices=option_dataset)
    parser.add_argument('--data_path', type=str)
    
    parser.add_argument('--pre_batch_size', default=64, type=int, help='pre-training batch size')
    parser.add_argument('--pre_num_episode', default=1000, type=int, help='number of episodes in meta validation')
    parser.add_argument('--pre_optimizer', default='SGD', choices=['SGD', 'Adam'], help='Collaborative Identification Mechanism(CIM)')
    parser.add_argument('--pre_lr', type=float, default=0.05, help='initial learning rate of the backbone')
    parser.add_argument('--pre_epoch', default=200, type=int, help='stopping epoch')

    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'], help='Collaborative Identification Mechanism(CIM)')
    parser.add_argument('--distill_lr', type=float, default=0.05, help='initial learning rate of the backbone')

    parser.add_argument('--model_l', default='ResNet12', choices=option_models)
    parser.add_argument('--model_r', default='ResNet12', choices=option_models)

    parser.add_argument('--method', default='stl_deepbdc', choices=option_methods)
    parser.add_argument('--vision_method', default='pure', choices=vision_methods)

    parser.add_argument('--test_n_way', default=5, type=int, help='number of classes used for testing (validation)')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

    parser.add_argument('--test_n_episode', default=10000, type=int, help='number of episodes in test')
    parser.add_argument('--model_path', default='', help='meta-trained or pre-trained model .tar file path')
    parser.add_argument('--test_task_nums', default=5, type=int, help='test numbers')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')

    parser.add_argument('--penalty_C', default=0.1, type=float, help='logistic regression penalty parameter')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')
    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')

    parser.add_argument('--teacher_path', default='', help='teacher model .tar file path')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    
    parser.add_argument('--FLLS', default='shared', choices=FLLS_choice, help='learning style of BSFE')
    parser.add_argument('--AELS', default='independent', choices=AELS_choice, help='learning style of Singular HeM')
    
    parser.add_argument('--DRS', default='GCM', choices=DRS_choice, help='Dimention Reduction Style(DRS)')
    parser.add_argument('--CIM', default='ARS', choices=['WAS', 'ARS'], help='Collaborative Identification Mechanism(CIM)')

    ################################
    # DistributedDataParallel
    ################################
    parser.add_argument('--amp_opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')

    params = parser.parse_args()

    num_gpu = set_gpu(params)
    
    novel_file = 'test'

    novel_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    novel_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, dataset=params.dataset,  **novel_few_shot_params)
    novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)

    ################################
    # 2 初始化网络模型
    ################################
    model = Bino_Singular_HeM(params, model_dict[params.model_l], model_dict[params.model_r], **novel_few_shot_params)

    # model save path
    model = model.cuda()
    model.eval()

    print(params.model_path)
    model_file = os.path.join(params.model_path)
    model = load_model(model, model_file)

    print(params)

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

    params.checkpoint_dir += '/distill_born%s_%s_%s_%s_%s_%s_%dway_%dshot' % (params.trial, params.optimizer, params.pre_batch_size, float(params.pre_lr), params.pre_num_episode, params.pre_epoch, params.test_n_way, params.n_shot)
    params.checkpoint_dir += '/distill_test/' 

    if not os.path.isdir(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)

    ################################
    # 3 网络模型推理
    ################################
    path_file = os.path.join(params.checkpoint_dir, 'test_log.txt')
    log_file = open(path_file, 'w')
    log_file.write(f'params: {params}\n')
    log_file.close()

    iter_num = params.test_n_episode
    acc_all_task, std_all_task = [], []
    for _ in range(params.test_task_nums):
        log_file = open(path_file, 'a')
        acc_all = []
        test_start_time = time.time()
        tqdm_gen = tqdm.tqdm(novel_loader, bar_format="{l_bar}%s{bar}%s{r_bar}" % (color_code, Fore.RESET))
        for _, (x, _) in enumerate(tqdm_gen):
            
            with torch.no_grad():
                model.n_query = params.n_query

                scores_l, scores_r = model.set_test_forward(x, False)
                
                scores_l = torch.softmax(scores_l, dim=1)
                scores_r = torch.softmax(scores_r, dim=1)
                scores = (scores_l + scores_r) / 2                    

            if params.method in ['BinoHeD']:
                pred = scores.data.cpu().numpy().argmax(axis=1)
            else:
                pred = scores
            y = np.repeat(range(params.test_n_way), params.n_query)
            acc = np.mean(pred == y) * 100
            acc_all.append(acc)
            tqdm_gen.set_description(f'avg.acc:{(np.mean(acc_all)):.2f} (curr:{acc:.2f})')

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(Fore.RED +'%d Test Acc = %4.2f%% +- %4.2f%% (Time uses %.2f minutes)'
            % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num), (time.time() - test_start_time) / 60))

        log_file.write(f'iter_num: {iter_num}, test acc: {acc_mean}+-{1.96 * acc_std / np.sqrt(iter_num)}, \
                time-consuming: {(time.time()-test_start_time)/60}\n')

        acc_all_task.append(acc_all)
        std_all_task.append(1.96 * acc_std / np.sqrt(iter_num))
        
        log_file.close()

    log_file = open(path_file, 'a')
    acc_all_task_mean, acc_all_task_std = np.mean(acc_all_task), np.mean(std_all_task)

    print(Fore.RED +'%d test mean acc = %4.2f%% std = %4.2f%%' % (params.test_task_nums, acc_all_task_mean, acc_all_task_std))
    log_file.write(f'All tasks:{params.test_task_nums} , test mean acc:{acc_all_task_mean}, std:{acc_all_task_std}')
    
    log_file.close()