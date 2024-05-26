# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
from sqlite3 import paramstyle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
import tqdm, gc
from colorama import init, Fore
init() # Init Colorama
color_code = Fore.BLUE

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T=4.0):
        super(DistillKL, self).__init__()
        self.T = T
        
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss

class JS_Divergence_Loss(nn.Module):
    """JS divergence for distillation"""
    def __init__(self, T=4.0):
        super(JS_Divergence_Loss, self).__init__()
        self.kl_divergence = DistillKL(T)
        
    def forward(self, js_p, js_q):
        m = 0.5 * (js_p + js_q)
        return 0.5 * self.kl_divergence(js_p, m) + 0.5 * self.kl_divergence(js_q, m)

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

class Bino_BaselineTrain(nn.Module):
    def __init__(self, params, model_func1, model_func2, num_class):
        super(Bino_BaselineTrain, self).__init__()
        self.params = params
        self.num_class = num_class
        self.featurel = model_func1()
        self.featurer = model_func2()
        self.feat_dim = self.featurel.feat_dim[0]
        self.avgpool = nn.AdaptiveAvgPool2d(1)   
        self.classifier_l = nn.Linear(self.feat_dim, num_class)
        self.classifier_l.bias.data.fill_(0)
        self.classifier_r = nn.Linear(self.feat_dim, num_class)
        self.classifier_r.bias.data.fill_(0)
        self.loss_fn = nn.CrossEntropyLoss()
        self.disparity = JS_Divergence_Loss()

    def cal_sparse(self, sparse_tensor):
        non_zero_count = torch.nonzero(sparse_tensor).size(0)
        total_count = sparse_tensor.numel()
        sparsity = 1- non_zero_count/total_count
        return sparsity

    def js_divergence(self, A,B):
        assert A.dim() == 3 and B.dim() == 3
        A = F.softmax(A, dim=2)
        B = F.softmax(B, dim=2)
        M = 0.5 * (A+B)
        kl_divergence_A = torch.sum(A * torch.log(A / M), dim=2)
        kl_divergence_B = torch.sum(B * torch.log(B / M), dim=2)
        js_divergence = 0.5 * (kl_divergence_A + kl_divergence_B)
        return js_divergence

    def disc_hellinger_distance(self, qry, sup):
        qry_reshape = qry.view(qry.size(0), -1)
        sup_reshape = sup.view(sup.size(0), -1)
        qry_fea_expand = qry_reshape.unsqueeze(1).expand(qry_reshape.shape[0],sup_reshape.shape[0],-1)
        sup_fea_expand = sup_reshape.unsqueeze(0).expand(qry_reshape.shape[0],sup_reshape.shape[0],-1)
        qry_soft = F.softplus(qry_fea_expand)
        sup_soft = F.softplus(sup_fea_expand)
        M = (qry_soft + sup_soft) / 2
        return -1 / torch.sqrt(torch.tensor(2.0)) * torch.norm(torch.sqrt(qry_soft) - torch.sqrt(sup_soft), dim=-1)

    def svg_fq(self, data, drop_rate=0.0, num_singular='default'):
        new_shape = (data.shape[0]*data.shape[1],)+(data.shape[-2],-1)
        reshaped_data = data.view(new_shape)
        drop_out = nn.Dropout(drop_rate)
        U, s, V = torch.svd(reshaped_data)
        if num_singular == 'default':
            svg = s.reshape(data.shape[0],data.shape[1],-1)
        else:
            svg = s.reshape(data.shape[0],data.shape[1],-1)
            num_singular_values = num_singular
            svg[:,:,svg.shape[2]-1]=0
        svg = drop_out(svg)
        return svg

    def feature_forward(self, x): 
        out_l = self.featurel.forward(x) 
        out_r = self.featurer.forward(x) 
        out_l = self.avgpool(out_l).view(out_l.size(0), -1)
        out_r = self.avgpool(out_r).view(out_r.size(0), -1)
        return out_l, out_r

    def forward(self, x):
        x = Variable(x.cuda())                    
        out1, out2 = self.feature_forward(x)       
        scores_l = self.classifier_l.forward(out1)
        scores_r = self.classifier_r.forward(out2)
        return scores_l, scores_r
    
    def forward_loss(self, x, y):
        scores_l, scores_r = self.forward(x) 
        y = Variable(y.cuda())              
        if self.params.vision_method=='disparity':
            loss_disparity = self.disparity(scores_l, scores_r)
            return self.loss_fn(scores_l, y), scores_l, self.loss_fn(scores_r, y), scores_r, loss_disparity
        else:
            return self.loss_fn(scores_l, y), scores_l, self.loss_fn(scores_r, y), scores_r

    def train_loop(self, model, epoch, train_loader, optimizer):
        print_freq = 1
        avg_loss = 0
        total_correct = 0
        iter_num = len(train_loader)
        total = len(train_loader) * self.params.batch_size
        for i, (x, y) in enumerate(train_loader):
            y = Variable(y.cuda())
            if self.params.vision_method =="disparity":
                loss_l, output_l, loss_r, output_r, loss_disparity = self.forward_loss(x, y)
                loss = loss_l + loss_r + loss_disparity
            else:
                loss_l, output_l, loss_r, output_r = self.forward_loss(x, y)
                loss = loss_l + loss_r       
            if self.params.CIM =='WAS': pred = weighted_average(output_l, output_r, 0.5, 0.5)
            elif self.params.CIM =='ARS': pred = average_rank(output_l, output_r)
            total_correct += pred.eq(y.data.view_as(pred)).sum()
            optimizer.zero_grad()
            if self.params.amp_opt_level != "O0":
                if amp is not None:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else: loss.backward()
            else: loss.backward()
            optimizer.step() 
            avg_loss = avg_loss + loss.item()
            gc.collect()
            if i % print_freq == 0: print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, self.params.epoch, \
                                                                                         i, len(train_loader), avg_loss / float(i + 1)))
        return avg_loss / iter_num, float(total_correct) / total * 100
    
    def test_loop(self, val_loader):
        total_correct = 0
        avg_loss = 0.0
        total = len(val_loader) * self.params.batch_size
        tqdm_gen = tqdm.tqdm(val_loader, bar_format="{l_bar}%s{bar}%s{r_bar}" % (color_code, Fore.RESET))
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm_gen, 1):
                y = Variable(y.cuda())
                if self.params.vision_method =="disparity":
                    loss_l, output_l, loss_r, output_r, disparity = self.forward_loss(x, y)  # 函数8
                    loss = loss_l + loss_r + 2*disparity
                else: 
                    loss_l, output_l, loss_r, output_r = self.forward_loss(x, y)  # 函数8
                    loss = loss_l + loss_r
                if self.params.CIM =='WAS': 
                    pred = weighted_average(output_l, output_r, 0.5, 0.5)
                elif self.params.CIM =='ARS': 
                    pred = average_rank(output_l, output_r)
                    
                total_correct += pred.eq(y.data.view_as(pred)).sum()
                avg_loss = avg_loss + loss.item()
                gc.collect()
        avg_loss /= len(val_loader)
        acc = float(total_correct) / total
        return avg_loss, acc * 100

    def feature_val_forward(self, x): 
        out_l = self.featurel.forward(x)
        out_r = self.featurer.forward(x)

        if self.params.DRS == 'GCM': 
            out_l, out_r = out_l.mean(dim=-1), out_r.mean(dim=-1)
        elif self.params.DRS == 'GRM': 
            out_l, out_r = out_l.mean(dim=-2), out_r.mean(dim=-2)
        elif self.params.DRS == 'SVD': 
            out_l, out_r = self.svg_fq(out_l), self.svg_fq(out_r)

        return out_l, out_r

    def forward_meta_val(self, x):
        x = Variable(x.cuda())             
        x = x.contiguous().view(self.params.val_n_way * (self.params.n_shot + self.params.n_query), *x.size()[2:]) 
        out1, out2 = self.feature_val_forward(x)
        
        l_all = out1.view(self.params.val_n_way, self.params.n_shot + self.params.n_query, -1)
        l_support = l_all[:, :self.params.n_shot] 
        l_query = l_all[:, self.params.n_shot:]  
        
        l_proto = l_support.contiguous().view(self.params.val_n_way, self.params.n_shot, -1).mean(1)
        l_query = l_query.contiguous().view(self.params.val_n_way * self.params.n_query, -1)        
        r_all = out2.view(self.params.val_n_way, self.params.n_shot + self.params.n_query, -1)
        r_support = r_all[:, :self.params.n_shot] 
        r_query = r_all[:, self.params.n_shot:]  
        
        r_proto = r_support.contiguous().view(self.params.val_n_way, self.params.n_shot, -1).mean(1)
        r_query = r_query.contiguous().view(self.params.val_n_way * self.params.n_query, -1)        
        scores_l = self.disc_hellinger_distance(l_query, l_proto)
        scores_r = self.disc_hellinger_distance(r_query, r_proto)
        return scores_l, scores_r 

    def forward_meta_val_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.params.val_n_way), self.params.n_query))
        y_query = Variable(y_query.cuda())                                      
        y_label = np.repeat(range(self.params.val_n_way), self.params.n_query)          
        scores_l, scores_r = self.forward_meta_val(x) 
        if self.params.CIM =='WAS': 
            topk_labels = weighted_average(scores_l, scores_r, 0.5, 0.5)
        elif self.params.CIM =='ARS': 
            topk_labels = average_rank(scores_l, scores_r)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_label)
        
        return float(top1_correct), len(y_label), self.loss_fn(scores_l, y_query)+ self.loss_fn(scores_r, y_query), scores_l, scores_r
    
    def meta_test_loop(self, test_loader):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                correct_this, count_this, loss, _ , _ = self.forward_meta_val_loss(x) # 函数5
                acc_all.append(correct_this / count_this * 100)
                avg_loss = avg_loss + loss.item()
        acc_all = np.asarray(acc_all)
        acc_mean, acc_std = np.mean(acc_all), np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        
        return avg_loss/iter_num, acc_mean
 
class Bino_MetaTemplate(nn.Module):
    def __init__(self, params, model_func1, model_func2, n_way, n_support, change_way=True):
        super(Bino_MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = params.n_query 
        self.featurel = model_func1()
        self.featurer = model_func2()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.change_way = change_way
        self.params = params
        
    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)
    
    def cal_sparse(self, sparse_tensor):
        non_zero_count = torch.nonzero(sparse_tensor).size(0)
        total_count = sparse_tensor.numel()
        sparsity = 1- non_zero_count/total_count
        return sparsity
    
    def js_divergence(self, A,B):
        assert A.dim() == 3 and B.dim() == 3
        A = F.softmax(A, dim=2)
        B = F.softmax(B, dim=2)
        M = 0.5 * (A+B)
        kl_divergence_A = torch.sum(A * torch.log(A / M), dim=2)
        kl_divergence_B = torch.sum(B * torch.log(B / M), dim=2)
        js_divergence = 0.5 * (kl_divergence_A + kl_divergence_B)
        return js_divergence
    
    def svg_fq(self, data, drop_rate=0.0, num_singular='default'):
        new_shape = (data.shape[0]*data.shape[1],)+(data.shape[-2],-1)
        reshaped_data = data.view(new_shape)
        drop_out = nn.Dropout(drop_rate)
        U, s, V = torch.svd(reshaped_data)
        if num_singular == 'default':
            svg = s.reshape(data.shape[0],data.shape[1],-1)
        else:
            svg = s.reshape(data.shape[0],data.shape[1],-1)
            num_singular_values = num_singular
            svg[:,:,svg.shape[2]-1]=0
        svg = drop_out(svg)
        return svg
    
    def forward(self, x):
        out_l = self.featurel.forward(x)
        out_r = self.featurer.forward(x)
        return out_l, out_r
    def feature_forward(self, x, y): 
        if self.params.DRS == 'GCM': 
            out_l, out_r = x.mean(dim=-1), y.mean(dim=-1)
        elif self.params.DRS == 'GRM':
            out_l, out_r = x.mean(dim=-2), y.mean(dim=-2)
        elif self.params.DRS == 'SVD':
            out_l, out_r = self.svg_fq(x), self.svg_fq(y)
        return out_l, out_r
    
    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature: z_all = x  
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            x_l = self.featurel.forward(x)
            x_r = self.featurer.forward(x)
            z_all_l, z_all_r = self.feature_forward(x_l,x_r)
            z_all_l = z_all_l.view(self.n_way, self.n_support + self.n_query, -1)
            z_all_r = z_all_r.view(self.n_way, self.n_support + self.n_query, -1)
        z_support_l, z_support_r = z_all_l[:, :self.n_support], z_all_r[:, :self.n_support]
        z_query_l, z_query_r = z_all_l[:, self.n_support:], z_all_r[:, self.n_support:]
        return z_support_l, z_query_l, z_support_r, z_query_r
 
    @abstractmethod
    def set_forward(self, x, is_feature): pass
    
    @abstractmethod
    def set_forward_loss(self, x): pass

    def train_loop(self, model, epoch, train_loader, optimizer):
        print_freq = 1
        avg_loss = 0
        acc_all = []
        iter_num = len(train_loader)
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way: self.n_way = x.size(0)
            if self.params.vision_method=='disparity': 
                correct_this, count_this, loss, _, _, loss_disparity = self.set_forward_loss(x)
                avg_loss = avg_loss + loss.item() + loss_disparity.item()
            else:
                correct_this, count_this, loss, _, _ = self.set_forward_loss(x)
                avg_loss = avg_loss + loss.item()    
            acc_all.append(correct_this / count_this * 100)
            optimizer.zero_grad()
            if self.params.amp_opt_level != "O0":
                if amp is not None:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else: loss.backward()
            else: loss.backward()
            optimizer.step() 
            if i % print_freq == 0: 
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, self.params.epoch, i, len(train_loader), avg_loss / float(i + 1)))
            gc.collect()
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        return avg_loss / iter_num, acc_mean

    @abstractmethod
    def set_val(self, x, is_feature): pass
    
    @abstractmethod
    def set_val_loss(self, x): pass

    def test_loop(self, test_loader, record=None):
        acc_all = []
        avg_loss = 0
        iter_num = len(test_loader)
        with torch.no_grad():
            tqdm_gen = tqdm.tqdm(test_loader, bar_format="{l_bar}%s{bar}%s{r_bar}" % (color_code, Fore.RESET))
            for i, (x, _) in enumerate(tqdm_gen, 1):
                self.n_query = x.size(1) - self.n_support
                if self.change_way: self.n_way = x.size(0)                
                if self.params.vision_method=='disparity':
                   correct_this, count_this, loss, _, _, loss_disparity = self.set_val_loss(x)
                   avg_loss = avg_loss + loss.item() + loss_disparity.item()
                else:
                   correct_this, count_this, loss, _, _ = self.set_val_loss(x)
                   avg_loss = avg_loss + loss.item()
                acc_all.append(correct_this / count_this * 100)
                gc.collect()
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        torch.cuda.empty_cache()
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        return avg_loss / iter_num, acc_mean

    @abstractmethod
    def set_test_forward(self, x, is_feature): pass