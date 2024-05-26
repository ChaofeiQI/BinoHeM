# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from .bino1_template import Bino_MetaTemplate

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

class Bino_Singular_HeM(Bino_MetaTemplate):
    def __init__(self, params, model_func_l, model_func_r, n_way, n_support):
        super(Bino_Singular_HeM, self).__init__(params, model_func_l, model_func_r,n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()  
        self.disparity = JS_Divergence_Loss() 

    def disc_hellinger_distance(self, qry, sup):
        qry_reshape = qry.view(qry.size(0), -1)
        sup_reshape = sup.view(sup.size(0), -1)
        qry_fea_expand = qry_reshape.unsqueeze(1).expand(qry_reshape.shape[0],sup_reshape.shape[0],-1)
        sup_fea_expand = sup_reshape.unsqueeze(0).expand(qry_reshape.shape[0],sup_reshape.shape[0],-1)
        qry_soft = F.softplus(qry_fea_expand)
        sup_soft = F.softplus(sup_fea_expand)
        M = (qry_soft + sup_soft) / 2
        return -1 / torch.sqrt(torch.tensor(2.0)) * torch.norm(torch.sqrt(qry_soft) - torch.sqrt(sup_soft), dim=-1)

    def set_forward(self, x, is_feature=False):
        z_support_l, z_query_l, z_support_r, z_query_r = self.parse_feature(x, is_feature)
        z_proto_l = z_support_l.contiguous().view(self.n_way, self.n_support, -1).mean(1) 
        z_query_l = z_query_l.contiguous().view(self.n_way * self.n_query, -1)            
        z_proto_r = z_support_r.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query_r = z_query_r.contiguous().view(self.n_way * self.n_query, -1)
            
        metric_method = self.params.method
        if metric_method == 'BinoHeD':
            s_time = time.time()
            scores_l = self.disc_hellinger_distance(z_query_l, z_proto_l)
            scores_r = self.disc_hellinger_distance(z_query_r, z_proto_r)
            f_time = time.time()
            period = f_time - s_time

        return scores_l, scores_r

    def set_forward_loss(self, x):

        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        
        scores_l, scores_r = self.set_forward(x) 

        if self.params.CIM =='WAS': topk_labels = weighted_average(scores_l, scores_r, 0.5, 0.5)
        elif self.params.CIM =='ARS': topk_labels = average_rank(scores_l, scores_r)

        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_label)
        
        if self.params.vision_method=='disparity':
            loss_disparity = self.disparity(scores_l, scores_r)
            return float(top1_correct), len(y_label), self.loss_fn(scores_l, y_query)+ self.loss_fn(scores_r, y_query), scores_l, scores_r, loss_disparity
        else:
            return float(top1_correct), len(y_label), self.loss_fn(scores_l, y_query)+ self.loss_fn(scores_r, y_query), scores_l, scores_r

    def set_val_forword(self, x, is_feature=False):
        z_support_l, z_query_l, z_support_r, z_query_r = self.parse_feature(x, is_feature)  
        z_proto_l = z_support_l.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query_l = z_query_l.contiguous().view(self.n_way * self.n_query, -1)            
        z_proto_r = z_support_r.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query_r = z_query_r.contiguous().view(self.n_way * self.n_query, -1)           

        metric_method = self.params.method
        if metric_method == 'BinoHeD':
            s_time = time.time()
            scores_l = self.disc_hellinger_distance(z_query_l, z_proto_l)
            scores_r = self.disc_hellinger_distance(z_query_r, z_proto_r)
            f_time = time.time()
            period = f_time - s_time
        return scores_l, scores_r

    def set_val_loss(self, x):

        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        
        scores_l, scores_r = self.set_val_forword(x)

        if self.params.CIM =='WAS': topk_labels = weighted_average(scores_l, scores_r, 0.5, 0.5)
        
        elif self.params.CIM =='ARS': topk_labels = average_rank(scores_l, scores_r)

        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_label)
        
        if self.params.vision_method=='disparity': 
            loss_disparity = self.disparity(scores_l, scores_r)
            return float(top1_correct), len(y_label), self.loss_fn(scores_l, y_query)+self.loss_fn(scores_r, y_query), scores_l, scores_r, loss_disparity
        else:
            return float(top1_correct), len(y_label), self.loss_fn(scores_l, y_query)+self.loss_fn(scores_r, y_query), scores_l, scores_r

    def set_test_forward(self, x, is_feature=False):
        z_support_l, z_query_l, z_support_r, z_query_r = self.parse_feature(x, is_feature)
        z_proto_l = z_support_l.contiguous().view(self.n_way, self.n_support, -1).mean(1) 
        z_query_l = z_query_l.contiguous().view(self.n_way * self.n_query, -1)           
        z_proto_r = z_support_r.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query_r = z_query_r.contiguous().view(self.n_way * self.n_query, -1)
                
        metric_method = self.params.method 
        if metric_method == 'BinoHeD':
            s_time = time.time()
            scores_l = self.disc_hellinger_distance(z_query_l, z_proto_l)
            scores_r = self.disc_hellinger_distance(z_query_r, z_proto_r)
            f_time = time.time()
            period = f_time - s_time
            
        return scores_l, scores_r
