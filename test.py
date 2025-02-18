import os
import re
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

# from .utils import Expert, CKA, info_abundance, Projector
# from  modeling.backbone.base_model import BaseKD4Rec, BaseKD4CTR

# class DCD_optim(BaseKD4Rec):
#     def __init__(self, args, teacher, student):
#         super().__init__(args, teacher, student)
#         self.K = args.dcd_K
#         self.T = args.dcd_T
#         self.L = args.dcd_L
#         self.mxK = args.dcd_mxK
#         self.ablation = args.ablation
#         self.tau = args.dcd_tau
#         self.T_topk = self.get_topk_dict()
#         self.T_rank = torch.arange(self.mxK).repeat(self.num_users, 1).cuda() # 在教师视角T_topk里元素的rk就是1-n的正序排序

#         # For uninteresting item
#         self.mask = torch.ones((self.num_users, self.num_items))
#         train_pairs = self.dataset.train_pairs
#         self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
#         for user in range(self.num_users):
#             self.mask[user, self.T_topk[user]] = 0 # 把每个用户top mxk的interesting item以及交互过的item都mask掉,那么它们之后被采样的概率就是0了，剩余item的值都是1，会被等概率采样
#         self.mask.requires_grad = False

#     def get_topk_dict(self):
#         print('Generating Top-K dict...')
#         with torch.no_grad():
#             inter_mat = self.teacher.get_all_ratings() # usr-item score matrix
#             train_pairs = self.dataset.train_pairs
#             # remove true interactions from topk_dict
#             inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
#             _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
#         return topk_dict # top_k的索引
    
#     def get_samples(self, batch_user):
#         interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
#         uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
#         return interesting_samples, uninteresting_samples
 
#     def do_something_in_each_epoch(self, epoch):
#         with torch.no_grad():
#             S_pred = self.student.get_all_ratings() # user_num X item_num
#             S_topk = torch.argsort(S_pred, descending=True, dim=-1) # user_num X item_num, 返回降序排序后每一个位置对应的原item的idx
#             S_rank = torch.argsort(S_topk, dim=-1) # 返回
#             S_rank = S_rank[torch.arange(len(S_rank)).unsqueeze(-1), self.T_topk]
#             diff = abs(S_rank - self.T_rank)
#             rank_diff = torch.maximum(torch.tanh(torch.maximum(diff / self.T, torch.tensor(0.))), torch.tensor(1e-5))

#             # sampling_interesting
#             interesting_idx = torch.multinomial(rank_diff, self.K, replacement=False) # mxK里面采样k个
#             self.interesting_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), interesting_idx]

#             # sampling_uninteresting
#             m1 = self.mask[: self.num_users // 2, :].cuda()
#             tmp1 = torch.multinomial(m1, self.L, replacement=False)
#             del m1

#             m2 = self.mask[self.num_users // 2 : ,:].cuda()
#             tmp2 = torch.multinomial(m2, self.L, replacement=False)
#             del m2

#             self.uninteresting_items = torch.cat([tmp1, tmp2], 0)

    
#     def relaxed_ranking_loss(self, S1, S2):
#         S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
#         S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))

#         above = S1.sum(1, keepdims=True)

#         below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
#         below2 = S2.exp().sum(1, keepdims=True)

#         below = (below1 + below2).log().sum(1, keepdims=True)
        
#         return -(above - below).sum()
    
#     def ce_loss(self, logit_T, logit_S):
#         prob_T = torch.softmax(logit_T / self.tau, dim=-1)
#         loss = F.cross_entropy(logit_S / self.tau, prob_T, reduction='sum')
#         return loss

#     def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
#         users = batch_user.unique()
#         interesting_items, uninteresting_items = self.get_samples(users)
#         interesting_items = interesting_items.type(torch.LongTensor).cuda()
#         uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()

#         interesting_prediction = self.student.forward_multi_items(users, interesting_items)
#         uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)

#         if self.ablation:
#             interesting_prediction_T = self.teacher.forward_multi_items(users, interesting_items)
#             uninteresting_prediction_T = self.teacher.forward_multi_items(users, uninteresting_items)
#             prediction_T = torch.concat([interesting_prediction_T, uninteresting_prediction_T], dim=-1)
#             prediction_S = torch.concat([interesting_prediction, uninteresting_prediction], dim=-1)
#             loss = self.ce_loss(prediction_T, prediction_S)
#         else:
#             loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)

#         return loss
    
x = torch.randn(3, 4)
print(x)
indices = torch.tensor([[0, 2], [0, 1]])
torch.index_select(x, 0, indices)