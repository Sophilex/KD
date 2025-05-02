import os
import re
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from .utils import Expert, CKA, info_abundance, Projector
from .base_model import BaseKD4Rec, BaseKD4CTR


class Scratch(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()

        self.args = args
        self.backbone = backbone
        self.training = True

    def get_ratings(self, param):
        if self.args.task == "ctr":
            return self.backbone(param)
        else:
            return self.backbone.get_ratings(param)

    def do_something_in_each_epoch(self, epoch):
        return
    
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def get_params_to_update(self):
        return [{"params": self.backbone.parameters(), 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def forward(self, *params):
        if self.args.task == "ctr":
            data, labels = params
            output = self.backbone(data)
            base_loss = self.backbone.get_loss(output, labels)
        else:
            output = self.backbone(*params)
            base_loss = self.backbone.get_loss(output)
        loss = base_loss
        return loss, base_loss.detach(), torch.tensor(0.)

    @property
    def param_to_save(self):
        return self.backbone.state_dict()


class RD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.mu = args.rd_mu
        self.topk = args.rd_topk
        self.T = args.rd_T
        self.dynamic_sample_num = args.rd_dynamic_sample
        self.start_epoch = args.rd_start_epoch
        
        self.rank_aware = False
        self.RANK = None
        self.epoch = 0
        self._weight_renormalize = True

        self._generateTopK()
        self._static_weights = self._generateStaticWeights()

    def Sample_neg(self, dns_k):
        """python implementation for 'UniformSample_DNS'
        """
        S = []
        BinForUser = np.zeros(shape=(self.num_items, )).astype("int")
        for user in range(self.num_users):
            posForUser = list(self.dataset.train_dict[user])
            if len(posForUser) == 0:
                continue
            BinForUser[:] = 0
            BinForUser[posForUser] = 1
            NEGforUser = np.where(BinForUser == 0)[0]
            negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
            negitems = NEGforUser[negindex]
            add_pair = [*negitems]
            S.append(add_pair)
        return S

    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        self.dynamic_samples = self.Sample_neg(self.dynamic_sample_num)
        self.dynamic_samples = torch.Tensor(self.dynamic_samples).long().cuda()

    def _generateStaticWeights(self):
        w = torch.arange(1, self.topk + 1).float()
        w = torch.exp(-w / self.T)
        return (w / w.sum()).unsqueeze(0)

    def _generateTopK(self):
        if self.RANK is None:
            with torch.no_grad():
                self.RANK = torch.zeros((self.num_users, self.topk)).cuda()
                scores = self.teacher.get_all_ratings()
                train_pairs = self.dataset.train_pairs
                scores[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
                self.RANK = torch.topk(scores, self.topk)[1]

    def _weights(self, S_score_in_T, epoch, dynamic_scores):
        batch = S_score_in_T.shape[0]
        if epoch < self.start_epoch:
            return self._static_weights.repeat((batch, 1)).cuda()
        with torch.no_grad():
            static_weights = self._static_weights.repeat((batch, 1))
            # ---
            topk = S_score_in_T.shape[-1]
            num_dynamic = dynamic_scores.shape[-1]
            m_items = self.num_items
            dynamic_weights = torch.zeros(batch, topk)
            for col in range(topk):
                col_prediction = S_score_in_T[:, col].unsqueeze(1)
                num_smaller = torch.sum(col_prediction < dynamic_scores, dim=1).float()
                # print(num_smaller.shape)
                relative_rank = num_smaller / num_dynamic
                appro_rank = torch.floor((m_items - 1) * relative_rank) + 1

                dynamic = torch.tanh(self.mu * (appro_rank - col))
                dynamic = torch.clamp(dynamic, min=0.)

                dynamic_weights[:, col] = dynamic.squeeze()
            if self._weight_renormalize:
                return F.normalize(static_weights * dynamic_weights,
                                   p=1,
                                   dim=1).cuda()
            else:
                return (static_weights * dynamic_weights).cuda()

    def get_loss(self, batch_users, batch_pos_item, batch_neg_item):
        dynamic_samples = self.dynamic_samples[batch_users]
        dynamic_scores = self.student.forward_multi_items(batch_users, dynamic_samples).detach()
        topk_teacher = self.RANK[batch_users]

        S_score_in_T = self.student.forward_multi_items(batch_users, topk_teacher)
        weights = self._weights(S_score_in_T.detach(), self.epoch, dynamic_scores)
        
        RD_loss = -(weights * torch.log(torch.sigmoid(S_score_in_T)))
        
        RD_loss = RD_loss.sum(1)
        RD_loss = RD_loss.sum()

        return  RD_loss
    

class CD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.sample_num = args.cd_sample_num
        self.strategy = args.cd_strategy
        self.T = args.cd_T
        self.n_distill = args.cd_n_distill
        self.t1 = args.cd_t1
        self.t2 = args.cd_t2
        
        ranking_list = torch.exp(-torch.arange(1, self.sample_num + 1).float() / self.sample_num / self.T)
        self.ranking_mat = torch.stack([ranking_list] * self.num_users, 0)
        self.ranking_mat.requires_grad = False
        if self.strategy == "random":
            self.MODEL = None
        elif self.strategy == "student guide":
            self.MODEL = self.student
        elif self.strategy == "teacher guide":
            self.MODEL = self.teacher
        else:
            raise TypeError("CD support [random, student guide, teacher guide], " \
                            f"But got {self.strategy}")
        self.get_rank_sample(self.MODEL)
    
    def do_something_in_each_epoch(self, epoch):
        if self.strategy == "student guide":
            self.get_rank_sample(self.MODEL)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.num_items, (batch_size, self.n_distill))
        return torch.from_numpy(samples).long().cuda()

    def get_rank_sample(self, MODEL):
        if MODEL is None:
            self.rank_samples =  self.random_sample(self.num_users)
            return
        self.rank_samples = torch.zeros(self.num_users, self.n_distill)
        with torch.no_grad():
            scores = MODEL.get_all_ratings()
            train_pairs = self.dataset.train_pairs
            scores[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            rank_scores, rank_items = torch.topk(scores, self.sample_num)

            for user in range(self.num_users):
                ranking_list = self.ranking_mat[user]
                rating = rank_scores[user]
                negitems = rank_items[user]
                sampled_items = set()
                while True:
                    samples = torch.multinomial(ranking_list, 2, replacement=True)
                    if rating[samples[0]] > rating[samples[1]]:
                        sampled_items.add(negitems[samples[0]])
                    else:
                        sampled_items.add(negitems[samples[1]])
                    if len(sampled_items) >= self.n_distill:
                        break
                self.rank_samples[user] = torch.Tensor(list(sampled_items))
        self.rank_samples = self.rank_samples.cuda().long()


    def get_loss(self, batch_users, batch_pos_item, batch_neg_item):
        random_samples = self.rank_samples[batch_users, :]
        samples_scores_T = self.teacher.forward_multi_items(batch_users, random_samples)
        samples_scores_S = self.student.forward_multi_items(batch_users, random_samples)
        weights = torch.sigmoid((samples_scores_T + self.t2) / self.t1)
        inner = torch.sigmoid(samples_scores_S)
        CD_loss = -(weights * torch.log(inner + 1e-10) + (1 - weights) * torch.log(1 - inner + 1e-10))

        CD_loss = CD_loss.sum(1).sum()
        return CD_loss


class DE(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.max_epoch = args.epochs
        self.end_T = args.de_end_T
        self.anneal_size = args.de_anneal_size
        self.num_experts = args.de_num_experts
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.current_T = self.end_T * self.anneal_size

        expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim) // 2, self.teacher_dim]
        # expert_dims = [self.student_dim*2, (self.teacher_dim + self.student_dim), self.teacher_dim*2]
        self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

        self.user_selection_net = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts), nn.Softmax(dim=1))
        self.item_selection_net = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts), nn.Softmax(dim=1))

        self.sm = nn.Softmax(dim=1)

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def do_something_in_each_epoch(self, epoch):
        self.current_T = self.end_T * self.anneal_size * ((1. / self.anneal_size) ** (epoch / self.max_epoch))
        self.current_T = max(self.current_T, self.end_T)


    def get_DE_loss(self, batch_entity, is_user=True):
        if is_user:
            s = self.student.get_user_embedding(batch_entity)
            t = self.teacher.get_user_embedding(batch_entity)

            experts = self.user_experts
            selection_net = self.user_selection_net
        else:
            s = self.student.get_item_embedding(batch_entity)
            t = self.teacher.get_item_embedding(batch_entity)
            
            experts = self.item_experts
            selection_net = self.item_selection_net
        
        selection_dist = selection_net(t) 			# batch_size x num_experts
        
        if self.num_experts == 1:
            selection_result = 1.
        else:
            # Expert Selection
            g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).cuda()
            eps = 1e-10 										# for numerical stability
            selection_dist = selection_dist + eps
            selection_dist = self.sm((selection_dist.log() + g) / self.current_T)

            selection_dist = torch.unsqueeze(selection_dist, 1)					# batch_size x 1 x num_experts
            selection_result = selection_dist.repeat(1, self.teacher_dim, 1)			# batch_size x teacher_dims x num_experts

        expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)] 		# s -> t
        expert_outputs = torch.cat(expert_outputs, -1)							# batch_size x teacher_dims x num_experts
        # print(f"selection_result, {selection_result.shape}")
        # print(f"expert_outputs, {expert_outputs.shape}")
        expert_outputs = expert_outputs * selection_result						# batch_size x teacher_dims x num_experts
        expert_outputs = expert_outputs.sum(2)								# batch_size x teacher_dims	

        DE_loss = ((t - expert_outputs) ** 2).sum(-1).sum()

        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), is_user=True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), is_user=False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), is_user=False)
        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss

class DE_try(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.max_epoch = args.epochs
        self.end_T = args.de_end_T
        self.anneal_size = args.de_anneal_size
        self.num_experts = args.de_num_experts
        
        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        self.current_T = self.end_T * self.anneal_size

        expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim) // 2, self.teacher_dim]
        # expert_dims = [self.student_dim*2, (self.teacher_dim + self.student_dim), self.teacher_dim*2]
        self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

        self.user_selection_net = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts))
        self.item_selection_net = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts))

        self.sm = nn.Softmax(dim=1)

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def do_something_in_each_epoch(self, epoch):
        self.current_T = self.end_T * self.anneal_size * ((1. / self.anneal_size) ** (epoch / self.max_epoch))
        self.current_T = max(self.current_T, self.end_T)


    def get_DE_loss(self, batch_entity, is_user=True):
        if is_user:
            s = self.student.get_user_embedding(batch_entity)
            t = self.teacher.get_user_embedding(batch_entity)

            experts = self.user_experts
            selection_net = self.user_selection_net
        else:
            s = self.student.get_item_embedding(batch_entity)
            t = self.teacher.get_item_embedding(batch_entity)
            
            experts = self.item_experts
            selection_net = self.item_selection_net
        
        selection_dist = selection_net(t) 			# batch_size x num_experts
        _, selection_dist = torch.topk(selection_dist, 1, dim=1)
        selection_dist = F.one_hot(selection_dist, self.num_experts)	# batch_size x num_experts

        selection_dist = torch.unsqueeze(selection_dist, 1)					# batch_size x 1 x num_experts
        selection_result = selection_dist.repeat(1, self.teacher_dim, 1)			# batch_size x teacher_dims x num_experts

        expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)] 		# s -> t
        expert_outputs = torch.cat(expert_outputs, -1)							# batch_size x teacher_dims x num_experts
        # print(f"selection_result, {selection_result.shape}")
        # print(f"expert_outputs, {expert_outputs.shape}")
        expert_outputs = expert_outputs * selection_result						# batch_size x teacher_dims x num_experts
        expert_outputs = expert_outputs.sum(2)								# batch_size x teacher_dims	

        DE_loss = ((t - expert_outputs) ** 2).sum(-1).sum()

        return DE_loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        DE_loss_user = self.get_DE_loss(batch_user.unique(), is_user=True)
        DE_loss_pos = self.get_DE_loss(batch_pos_item.unique(), is_user=False)
        DE_loss_neg = self.get_DE_loss(batch_neg_item.unique(), is_user=False)
        DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg) * 0.5
        return DE_loss

class RRDUnselected(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        
        self.K = args.rrd_K
        self.L = args.rrd_L
        self.T = args.rrd_T
        self.mxK = args.rrd_mxK
        self.unselected = args.rrd_unselected
        self.cover = args.cover
        self.neg = args.neg

        # For interesting item
        self.get_topk_dict()
        ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
        self.ranking_mat = ranking_list.repeat(self.num_users, 1) # 对每一个用户生成一个固定的interesting样本采样概率列表

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items))
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        if self.cover == 1:
            for user in range(self.num_users):
                self.mask[user, self.topk_dict[user]] = 0 # 把每个用户top mxk的interesting item以及交互过的item都mask掉,那么它们之后被采样的概率就是0了，剩余item的值都是1，会被等概率采样
        self.mask.requires_grad = False



    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # teacher得到的user-item分数s矩阵
            train_pairs = self.dataset.train_pairs # user-item交互对list
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, self.topk_dict = torch.topk(inter_mat, self.mxK, dim=-1) # self.num_users X self.mxK， 去掉了已经交互过的user-item对, 返回每行topmaxK的idx
    
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
        self.potential_interesting_items = torch.index_select(self.topk_dict, 0, batch_user)

        return interesting_samples, uninteresting_samples

    # epoch 마다
    def do_something_in_each_epoch(self, epoch):
        # 得到interesting items 以及uninteresting items的索引
        with torch.no_grad():
            # interesting items
            self.interesting_items = torch.zeros((self.num_users, self.K)) # 初始化矩阵

            # sampling
            while True:
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False) # 不会采样重复的元素，输出的是索引矩阵
                if (samples > self.mxK).sum() == 0: # 保证采样的都是前self.maxK的元素
                    break

            samples = samples.sort(dim=1)[0] # samples会返回 排序后的matrix以及原本元素的索引,这里只取第一个返回

            for user in range(self.num_users):
                self.interesting_items[user] = self.topk_dict[user][samples[user]]

            self.interesting_items = self.interesting_items.cuda()

            # uninteresting items
            m1 = self.mask[: self.num_users // 2, :].cuda()
            tmp1 = torch.multinomial(m1, self.L, replacement=False)
            del m1

            m2 = self.mask[self.num_users // 2 : ,:].cuda()
            tmp2 = torch.multinomial(m2, self.L, replacement=False)
            del m2

            self.uninteresting_items = torch.cat([tmp1, tmp2], 0)
    
    def relaxed_ranking_loss(self, S1, S2, S3):
        
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))     # 之后要做exp操作，这里做一个截断，防止数据爆炸导致的一系列问题
        S3 = torch.minimum(S3, torch.tensor(80., device=S3.device))

        unselected_below = S3.exp().sum(1, keepdims=True) - S1.exp().sum(1, keepdims=True)
        unselected_below = torch.maximum(unselected_below, torch.tensor(0., device=unselected_below.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() o finteresting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + self.neg * below2 + self.unselected * unselected_below).log().sum(1, keepdims=True)
        
        return -(above - below).sum()


    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        interesting_items, uninteresting_items = self.get_samples(users)


        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()
        self.potential_interesting_items = self.potential_interesting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)
        self.potential_interesting_prediction = self.student.forward_multi_items(users, self.potential_interesting_items)

        URRD_loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction, self.potential_interesting_prediction)

        return URRD_loss

class RRD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "rrd"
        self.K = args.rrd_K
        self.L = args.rrd_L
        self.T = args.rrd_T
        self.mxK = args.rrd_mxK
        self.neg = args.neg_x

        # For interesting item
        self.get_topk_dict()
        ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
        self.ranking_mat = ranking_list.repeat(self.num_users, 1)

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items))
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.topk_dict[user]] = 0
        self.mask.requires_grad = False

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings()
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            self.top_score, self.topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
    
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)

        return interesting_samples, uninteresting_samples

    # epoch 마다
    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            # interesting items
            self.interesting_items = torch.zeros((self.num_users, self.K))

            # sampling
            while True:
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False)
                if (samples > self.mxK).sum() == 0:
                    break
            
            samples = samples.sort(dim=1)[0]

            for user in range(self.num_users):
                self.interesting_items[user] = self.topk_dict[user][samples[user]]

            self.interesting_items = self.interesting_items.cuda()

            # uninteresting items
            m1 = self.mask[: self.num_users // 2, :].cuda()
            tmp1 = torch.multinomial(m1, self.L, replacement=False)
            del m1

            m2 = self.mask[self.num_users // 2 : ,:].cuda()
            tmp2 = torch.multinomial(m2, self.L, replacement=False)
            del m2

            self.uninteresting_items = torch.cat([tmp1, tmp2], 0)
    
    def relaxed_ranking_loss(self, S1, S2):
        
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + self.neg * below2).log().sum(1, keepdims=True)
        
        return -(above - below).sum()

    def get_loss(self, *params):
        batch_user = params[0]
        users = batch_user.unique()
        interesting_items, uninteresting_items = self.get_samples(users)
        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)

        URRD_loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)

        return URRD_loss



class ISp(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "rrd"
        self.K = args.rrd_K # 采样个数
        self.softmax_T = args.rrd_softmax_T
        self.isp_T = args.isp_T

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            self.selected_items = torch.zeros((self.num_users, self.K)).cuda()
            inter_mat = self.teacher.get_all_ratings()

            num_parts = 256
            chunk_size = self.num_users // num_parts

            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != num_parts - 1 else self.num_users
                
                inter_mat_slice = inter_mat[start_idx:end_idx].cuda()

                p_prob = F.softmax(inter_mat_slice / self.softmax_T, dim=-1).cuda()
                q_prob = p_prob ** self.isp_T
                sample_prob = p_prob / q_prob # importance sampling
                
                selected_slice = torch.multinomial(sample_prob, self.K, replacement=False).cuda()
                
                sampled_values = torch.gather(sample_prob, 1, selected_slice)
                _, sorted_indices = torch.sort(sampled_values, dim=1, descending=True)
                # 用排序后的索引重新排列 selected_slice
                selected_slice = torch.gather(selected_slice, 1, sorted_indices)

                self.selected_items[start_idx:end_idx] = selected_slice
                # sample_result_value = torch.gather(inter_mat_slice, 1, selected_slice).cpu().numpy()

                # with open(f"sample_result.txt", "a") as f:
                #     row = sample_result_value[0]
                #     f.write("\t".join(map(str, row)) + "\n")
    
    def relaxed_ranking_loss(self, S):
        
        S = torch.minimum(S, torch.tensor(80., device=S.device))     # This may help

        above = S.sum(1, keepdims=True)

        below = S.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf

        below = below.log().sum(1, keepdims=True)
        
        return -(above - below).sum()

    def get_loss(self, *params):
        batch_user = params[0]
        users = batch_user.unique()
        selected_items = torch.index_select(self.selected_items, 0, users)

        selected_items = selected_items.type(torch.LongTensor).cuda()

        selected_prediction = self.student.forward_multi_items(users, selected_items)

        URRD_loss = self.relaxed_ranking_loss(selected_prediction)

        return URRD_loss



class RRDVar(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        
        self.K = args.rrd_K
        self.L = args.rrd_L
        self.T = args.rrd_T
        self.mxK = args.rrd_mxK
        self.unselected = args.rrd_unselected
        self.neg = args.rrd_neg

        # For interesting item
        self.get_topk_dict()
        ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
        self.ranking_mat = ranking_list.repeat(self.num_users, 1) # 对每一个用户生成一个固定的interesting样本采样概率列表

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items), dtype=torch.float)
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.topk_dict[user]] = 0 # 把每个用户top mxk的interesting item以及交互过的item都mask掉,那么它们之后被采样的概率就是0了，剩余item的值都是1，会被等概率采样
        self.mask.requires_grad = False

    def set_model_variance(self, model_variance):
        self.model_variance = model_variance.cpu()
        self.model_variance += 1e-8

    def get_topk_dict(self):

        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # teacher得到的user-item分数s矩阵
            train_pairs = self.dataset.train_pairs # user-item交互对list
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, self.topk_dict = torch.topk(inter_mat, self.mxK, dim=-1) # self.num_users X self.mxK， 去掉了已经交互过的user-item对, 返回每行topmaxK的idx
    
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
        self.potential_interesting_items = torch.index_select(self.topk_dict, 0, batch_user)

        return interesting_samples, uninteresting_samples

    # epoch 마다
    def do_something_in_each_epoch(self, epoch):
        # 得到interesting items 以及uninteresting items的索引
        with torch.no_grad():
            # interesting items
            self.interesting_items = torch.zeros((self.num_users, self.K)) # 初始化矩阵

            # sampling
            while True:
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False) # 不会采样重复的元素，输出的是索引矩阵
                if (samples > self.mxK).sum() == 0: # 保证采样的都是前self.maxK的元素
                    break

            samples = samples.sort(dim=1)[0] # samples会返回 排序后的matrix以及原本元素的索引,这里只取第一个返回

            for user in range(self.num_users):
                self.interesting_items[user] = self.topk_dict[user][samples[user]]

            self.interesting_items = self.interesting_items.cuda()
            

            # uninteresting items

            mask_mat = self.mask * self.model_variance

            m1 = mask_mat[: self.num_users // 2, :].cuda()
            tmp1 = torch.multinomial(m1, self.L, replacement=False)
            del m1

            m2 = mask_mat[self.num_users // 2 : ,:].cuda()
            tmp2 = torch.multinomial(m2, self.L, replacement=False)
            del m2

            self.uninteresting_items = torch.cat([tmp1, tmp2], 0)
    
    def relaxed_ranking_loss(self, S1, S2, S3):
        
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))     # 之后要做exp操作，这里做一个截断，防止数据爆炸导致的一系列问题
        S3 = torch.minimum(S3, torch.tensor(80., device=S3.device))

        unselected_below = S3.exp().sum(1, keepdims=True) - S1.exp().sum(1, keepdims=True)
        unselected_below = torch.maximum(unselected_below, torch.tensor(0., device=unselected_below.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() o finteresting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + self.neg * below2 + self.unselected * unselected_below).log().sum(1, keepdims=True)
        
        return -(above - below).sum()


    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        interesting_items, uninteresting_items = self.get_samples(users)


        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()
        self.potential_interesting_items = self.potential_interesting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)
        self.potential_interesting_prediction = self.student.forward_multi_items(users, self.potential_interesting_items)

        URRD_loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction, self.potential_interesting_prediction)

        return URRD_loss

class RRDVK(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        
        self.K = args.rrd_K
        self.L = args.rrd_L # 每次采样以及保留的负样本个数
        self.T = args.rrd_T
        self.extra = args.rrd_extra # 每calu_len轮之后，额外添加要求采样的样本个数
        self.mxK = args.rrd_mxK
        self.unselected = args.rrd_unselected
        self.neg = args.rrd_neg
        self.calu_len = args.calu_len
        self.mode = args.mode
        self.alpha = args.alpha
        self.neg_T = args.neg_T
        self.sample_type_for_extra = args.sample_type_for_extra
        self.T_for_extra = args.T_for_extra
        self.mx_T = args.mx_T
        # self.cover = args.cover

        # For interesting item
        self.get_topk_dict()
        ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
        self.ranking_mat = ranking_list.repeat(self.num_users, 1) # 对每一个用户生成一个固定的interesting样本采样概率列表

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items), dtype=torch.float) # 没事不要把这么大的张量直接放进gpu...
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.topk_dict[user]] = 0 # 把每个用户top mxk的interesting item以及交互过的item都mask掉,那么它们之后被采样的概率就是0了，剩余item的值都是1，会被等概率采样
        self.mask.requires_grad = False
    
    def item_idx_init(self):
        # return initial item_idx for further calculation of model_variance
        m1 = self.mask[: self.num_users // 2, :].cuda()
        tmp1 = torch.multinomial(m1, self.L + self.extra, replacement=False)
        del m1

        m2 = self.mask[self.num_users // 2 : ,:].cuda()
        tmp2 = torch.multinomial(m2, self.L + self.extra, replacement=False)
        del m2
        return torch.cat([tmp1, tmp2], 0)
    
    def set_model_variance(self, model_variance, item_idx):
        self.model_variance = model_variance # user_num X (rrd_L + rrd_extra)
        # print(f"Set model_variance - min: {self.model_variance.min()}, max: {self.model_variance.max()}")
        self.item_idx = item_idx
        self.model_variance = self.model_variance + 1e-6

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # teacher得到的user-item分数s矩阵
            train_pairs = self.dataset.train_pairs # user-item交互对list
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, self.topk_dict = torch.topk(inter_mat, self.mxK, dim=-1) # self.num_users X self.mxK， 去掉了已经交互过的user-item对, 返回每行topmaxK的idx
    
    def get_samples(self, batch_user):
        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
        self.potential_interesting_items = torch.index_select(self.topk_dict, 0, batch_user)

        return interesting_samples, uninteresting_samples

    # epoch 마다
    def do_something_in_each_epoch(self, epoch):
        # 得到interesting items 以及uninteresting items的索引
        with torch.no_grad():

            # interesting items
            self.interesting_items = torch.zeros((self.num_users, self.K)) # 初始化矩阵

            # sampling
            while True:
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False) # 不会采样重复的元素，输出的是索引矩阵
                if (samples > self.mxK).sum() == 0: # 保证采样的都是前self.maxK的元素
                    break

            samples = samples.sort(dim=1)[0] # samples会返回 排序后的matrix以及原本元素的索引,这里只取第一个返回

            for user in range(self.num_users):
                self.interesting_items[user] = self.topk_dict[user][samples[user]]

            self.interesting_items = self.interesting_items.cuda()
            

            # uninteresting items
            num_parts = 256
            chunk_size = self.num_users // num_parts
            all_tmp = []

            # different sample mode
            pred_lst = []
            for start_idx in range(0, self.num_users, num_parts):
                end_idx = min(start_idx + num_parts, self.num_users)
                batch_user = torch.arange(start_idx, end_idx)
                if self.mode == "val_diff_0":
                    print("val_diff_0")
                    # absolute difference between scores
                    T_inter_mat = self.teacher.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                    S_inter_mat = self.student.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                    pred_lst.append(torch.abs(T_inter_mat - S_inter_mat))
                    del T_inter_mat, S_inter_mat
                elif self.mode == "val_diff_1":
                    print("val_diff_1")
                    # absolute difference between idx
                    T_inter_mat = self.teacher.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                    S_inter_mat = self.student.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                    T_inter_mat = torch.argsort(T_inter_mat, dim=1)
                    S_inter_mat = torch.argsort(S_inter_mat, dim=1)
                    T_inter_mat = torch.argsort(T_inter_mat, dim=1)
                    S_inter_mat = torch.argsort(S_inter_mat, dim=1)
                    pred_lst.append(torch.abs(T_inter_mat - S_inter_mat))
                    del T_inter_mat, S_inter_mat
                elif self.mode == "val_diff_2":
                    print("val_diff_2")
                    # difference between idx
                    T_inter_mat = self.teacher.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                    S_inter_mat = self.student.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                    T_inter_mat = torch.argsort(T_inter_mat, dim=1, descending=True)    
                    S_inter_mat = torch.argsort(S_inter_mat, dim=1, descending=True)
                    T_inter_mat = torch.argsort(T_inter_mat, dim=1)
                    S_inter_mat = torch.argsort(S_inter_mat, dim=1)
                    pred_lst.append(T_inter_mat - S_inter_mat) # 学生给的排名越相对激进越容易被采样
                    del T_inter_mat, S_inter_mat
                elif self.mode == "val_S":
                    print("val_S")
                    S_inter_mat = self.student.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                    pred_lst.append(S_inter_mat)
                    del S_inter_mat
                elif self.mode == "val_T":
                    print("val_T")
                    T_inter_mat = self.teacher.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                    pred_lst.append(T_inter_mat)
                    del T_inter_mat
                else:
                    pred_lst.append(torch.zeros((batch_user.shape[0], self.item_idx.shape[1]), dtype=torch.float).cuda())

            distill_info = torch.cat(pred_lst, 0).cpu() # differencr between T,S if mode = "val_diff", else T's pred
            del pred_lst


            alpha = self.alpha * min(epoch * 1.0 / self.mx_T, 1)
            print(alpha)
            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != num_parts - 1 else self.num_users
                
                var_slice = self.model_variance[start_idx:end_idx, :].cuda()
                distill_part = distill_info[start_idx:end_idx, :].cuda()
                # print(f"dis: {distill_part.min()}, {distill_part.max()}, var: {var_slice.min()}, {var_slice.max()}")
                combine_part = alpha * var_slice + distill_part
                combine_part = torch.clamp(combine_part / self.neg_T, max=80.0)
                combine_part = torch.exp(combine_part)
                tmp_part = torch.multinomial(combine_part, self.L, replacement=False)
                # sampl neg items with distill_info and model_variance, control the rate between the with alpha
                idx_part = self.item_idx[start_idx:end_idx, :].cuda()
                all_tmp.append(torch.gather(idx_part, 1, tmp_part))
                
                del tmp_part

            self.uninteresting_items = torch.cat(all_tmp, 0)

            # extra items
            # with different types for sampling extra items: random_with_uninteresting random_regardless_uninteresting T_val_with_uninteresting
            all_tmp = []
            if self.sample_type_for_extra != "random_regardless_uninteresting": # 不考虑是否混入这一轮采样过的负样本
                self.mask[torch.arange(self.uninteresting_items.size(0)).unsqueeze(-1), self.uninteresting_items] = 0

            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != num_parts - 1 else self.num_users
                batch_user = torch.arange(start_idx, end_idx)
                m_part = self.mask[start_idx:end_idx, :].cuda() # batch_size X num_items
                if self.sample_type_for_extra == "random_with_uninteresting":
                    batch_extra_items = torch.multinomial(m_part, self.extra, replacement=False)
                    batch_uninteresting = self.uninteresting_items[start_idx:end_idx, :].cuda() 
                    tmp_part = torch.cat([batch_extra_items, batch_uninteresting], 1)

                elif self.sample_type_for_extra == "random_regardless_uninteresting":
                    tmp_part = torch.multinomial(m_part, self.extra + self.L, replacement=False)

                else:
                    # sample_type_for_extra == T_val_with_uninteresting
                    T_val = self.teacher.get_ratings(batch_user) # batch_size X num_items
                    batch_extra_items = torch.multinomial(m_part, self.extra * 2, replacement=False)
                    batch_uninteresting = self.uninteresting_items[start_idx:end_idx, :].cuda() 
                    all_items = torch.cat([batch_extra_items, batch_uninteresting], 1) # batch_size X (self.extra*2 + self.L)

                    m_part[torch.arange(batch_extra_items.size(0)).unsqueeze(-1), batch_uninteresting] = 1
                    masked_T_val = T_val * m_part 
                    m_part[torch.arange(batch_extra_items.size(0)).unsqueeze(-1), batch_uninteresting] = 0
                    batch_score = masked_T_val.gather(1, all_items) # generate value for all_items mentioned above

                    batch_score = torch.clamp(batch_score / self.T_for_extra, max=80.0)
                    batch_score = torch.exp(batch_score)

                    selected_indices = torch.multinomial(batch_score, self.extra + self.L, replacement=False) # batch_size X (self.extra + self.L)
                    tmp_part = all_items.gather(1, selected_indices) # return the idx of selected_indices

                all_tmp.append(tmp_part)
                del m_part, tmp_part
            if self.sample_type_for_extra != "random_regardless_uninteresting":
                self.mask[torch.arange(self.uninteresting_items.size(0)).unsqueeze(-1), self.uninteresting_items] = 1
            self.candidate_items = torch.cat(all_tmp, 0)
            del all_tmp

    def reset_item(self):
        return self.candidate_items
    
    def get_un_items(self):
        return self.uninteresting_items

    def relaxed_ranking_loss(self, S1, S2, S3):
        
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))     # 之后要做exp操作，这里做一个截断，防止数据爆炸导致的一系列问题
        S3 = torch.minimum(S3, torch.tensor(80., device=S3.device))

        unselected_below = S3.exp().sum(1, keepdims=True) - S1.exp().sum(1, keepdims=True)
        unselected_below = torch.maximum(unselected_below, torch.tensor(0., device=unselected_below.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + self.neg * below2 + self.unselected * unselected_below).log().sum(1, keepdims=True)
        
        return -(above - below).sum()


    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        interesting_items, uninteresting_items = self.get_samples(users)

        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()
        self.potential_interesting_items = self.potential_interesting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)
        self.potential_interesting_prediction = self.student.forward_multi_items(users, self.potential_interesting_items)

        URRD_loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction, self.potential_interesting_prediction)

        return URRD_loss
    
class RRDVK2(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        
        self.K = args.rrd_K
        self.L = args.rrd_L # 每次采样以及保留的负样本个数
        self.T = args.rrd_T
        self.extra = args.rrd_extra # 每calu_len轮之后，额外添加要求采样的样本个数
        self.mxK = args.rrd_mxK
        self.unselected = args.rrd_unselected
        self.neg = args.rrd_neg
        self.calu_len = args.calu_len
        self.mode = args.mode
        self.alpha = args.alpha
        self.neg_T = args.neg_T

        # For interesting item
        self.get_topk_dict()
        ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
        self.ranking_mat = ranking_list.repeat(self.num_users, 1) # 对每一个用户生成一个固定的interesting样本采样概率列表

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items), dtype=torch.float)
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.topk_dict[user]] = 0 # 把每个用户top mxk的interesting item以及交互过的item都mask掉,那么它们之后被采样的概率就是0了，剩余item的值都是1，会被等概率采样
        self.mask.requires_grad = False
    
    def item_idx_init(self):
        # return initial item_idx for further calculation of model_variance
        m1 = self.mask[: self.num_users // 2, :].cuda()
        tmp1 = torch.multinomial(m1, self.L + self.extra, replacement=False)
        del m1

        m2 = self.mask[self.num_users // 2 : ,:].cuda()
        tmp2 = torch.multinomial(m2, self.L + self.extra, replacement=False)
        del m2
        return torch.cat([tmp1, tmp2], 0)
    
    def set_model_variance(self, model_variance, item_idx):
        self.model_variance = model_variance # user_num X (rrd_L + rrd_extra)
        # print(f"Set model_variance - min: {self.model_variance.min()}, max: {self.model_variance.max()}")
        self.item_idx = item_idx
        self.model_variance = self.model_variance + 1e-6


    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # teacher得到的user-item分数s矩阵
            train_pairs = self.dataset.train_pairs # user-item交互对list
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, self.topk_dict = torch.topk(inter_mat, self.mxK, dim=-1) # self.num_users X self.mxK， 去掉了已经交互过的user-item对, 返回每行topmaxK的idx
    
    def get_samples(self, batch_user):
        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
        self.potential_interesting_items = torch.index_select(self.topk_dict, 0, batch_user)

        return interesting_samples, uninteresting_samples

    # epoch 마다
    def do_something_in_each_epoch(self, epoch):
        # 得到interesting items 以及uninteresting items的索引
        with torch.no_grad():
            # interesting items
            self.interesting_items = torch.zeros((self.num_users, self.K)) # 初始化矩阵

            # sampling
            while True:
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False) # 不会采样重复的元素，输出的是索引矩阵
                if (samples > self.mxK).sum() == 0: # 保证采样的都是前self.maxK的元素
                    break

            samples = samples.sort(dim=1)[0] # samples会返回 排序后的matrix以及原本元素的索引,这里只取第一个返回

            for user in range(self.num_users):
                self.interesting_items[user] = self.topk_dict[user][samples[user]]

            self.interesting_items = self.interesting_items.cuda()
            

            # uninteresting items
            num_parts = 256
            chunk_size = self.num_users // num_parts
            all_tmp = []

            # different sample mode
            targt = None
            if self.mode.lower == "val_diff":
                # add val information between S and T
                T_pred = self.teacher.get_user_item_ratings(torch.arange(self.num_users), self.item_idx)
                S_pred = self.student.get_user_item_ratings(torch.arange(self.num_users), self.item_idx)
                targt = (T_pred - S_pred) * self.mask
                del T_pred, S_pred
            elif self.mode.lower == "val_T":
                # add Teacher's val information
                T_pred = self.teacher.get_user_item_ratings(torch.arange(self.num_users), self.item_idx)
                targt = T_pred * self.mask
                del T_pred
            else: 
                # add no val informatio
                targt = torch.zeros_like(self.model_variance)


            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != num_parts - 1 else self.num_users
                var_slice = self.model_variance[start_idx:end_idx, :].cuda() / self.neg_T
                clamped_var = torch.clamp(var_slice, max=80.0)
                m_part = torch.exp(clamped_var).cuda()
                # m_part = torch.exp(torch.min(self.model_variance[start_idx:end_idx, :].cuda()/self.neg_T, 80.0)).cuda()
                # print(m_part.min(), m_part.max())
                # m_part = self.model_variance[start_idx:end_idx, :].cuda()
                tar_part = targt[start_idx:end_idx, :].cuda().softmax(dim=1)
                if self.mode == "val_diff" or self.mode == "val_T":
                    tmp_part = torch.multinomial(tar_part + self.alpha * m_part, self.L, replacement=False)
                else:
                    tmp_part = torch.multinomial(m_part, self.L, replacement=False)
                all_tmp.append(tmp_part)
                del m_part, tmp_part

            self.uninteresting_items = torch.cat(all_tmp, 0)

            # extra items
            all_tmp = []
            self.mask[torch.arange(self.uninteresting_items.size(0)).unsqueeze(-1), self.uninteresting_items] = 0

            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != num_parts - 1 else self.num_users
                
                m_part = self.mask[start_idx:end_idx, :].cuda()
                tmp_part = torch.multinomial(m_part, self.extra, replacement=False)

                all_tmp.append(tmp_part)
                del m_part, tmp_part
            self.mask[torch.arange(self.uninteresting_items.size(0)).unsqueeze(-1), self.uninteresting_items] = 1
            self.extra_items = torch.cat(all_tmp, 0)
            del all_tmp

    def reset_item(self):
        # print(f"visual extra_items shape: {self.extra_items.dtype}, uninteresting_items shape: {self.uninteresting_items.shape}")
        return torch.cat([self.uninteresting_items, self.extra_items], dim = 1) # user_num X (self.L + self.extra)

    def get_un_items(self):
        return self.uninteresting_items

    def relaxed_ranking_loss(self, S1, S2, S3):
        
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))     # 之后要做exp操作，这里做一个截断，防止数据爆炸导致的一系列问题
        S3 = torch.minimum(S3, torch.tensor(80., device=S3.device))

        unselected_below = S3.exp().sum(1, keepdims=True) - S1.exp().sum(1, keepdims=True)
        unselected_below = torch.maximum(unselected_below, torch.tensor(0., device=unselected_below.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + self.neg * below2 + self.unselected * unselected_below).log().sum(1, keepdims=True)
        
        return -(above - below).sum()


    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        interesting_items, uninteresting_items = self.get_samples(users)

        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()
        self.potential_interesting_items = self.potential_interesting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)
        self.potential_interesting_prediction = self.student.forward_multi_items(users, self.potential_interesting_items)

        URRD_loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction, self.potential_interesting_prediction)

        return URRD_loss

class RRDvkNoCon(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        
        self.K = args.rrd_K
        self.L = args.rrd_L # 每次采样以及保留的负样本个数
        self.T = args.rrd_T
        self.extra = args.rrd_extra # 每calu_len轮之后，额外添加要求采样的样本个数
        self.mxK = args.rrd_mxK
        self.unselected = args.rrd_unselected
        self.neg = args.rrd_neg
        self.calu_len = args.calu_len
        self.mode = args.mode
        self.alpha = args.alpha
        self.neg_T = args.neg_T

        # For interesting item
        self.get_topk_dict()
        ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
        self.ranking_mat = ranking_list.repeat(self.num_users, 1) # 对每一个用户生成一个固定的interesting样本采样概率列表

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items), dtype=torch.float) # 没事不要把这么大的张量直接放进gpu...
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.topk_dict[user]] = 0 # 把每个用户top mxk的interesting item以及交互过的item都mask掉,那么它们之后被采样的概率就是0了，剩余item的值都是1，会被等概率采样
        self.mask.requires_grad = False
    
    def item_idx_init(self):
        # return initial item_idx for further calculation of model_variance
        m1 = self.mask[: self.num_users // 2, :].cuda()
        tmp1 = torch.multinomial(m1, self.L + self.extra, replacement=False)
        del m1

        m2 = self.mask[self.num_users // 2 : ,:].cuda()
        tmp2 = torch.multinomial(m2, self.L + self.extra, replacement=False)
        del m2
        return torch.cat([tmp1, tmp2], 0)
    
    def set_model_variance(self, model_variance, item_idx):
        self.model_variance = model_variance # user_num X (rrd_L + rrd_extra)
        # print(f"Set model_variance - min: {self.model_variance.min()}, max: {self.model_variance.max()}")
        self.item_idx = item_idx

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # teacher得到的user-item分数s矩阵
            train_pairs = self.dataset.train_pairs # user-item交互对list
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, self.topk_dict = torch.topk(inter_mat, self.mxK, dim=-1) # self.num_users X self.mxK， 去掉了已经交互过的user-item对, 返回每行topmaxK的idx
    
    def get_samples(self, batch_user):
        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
        self.potential_interesting_items = torch.index_select(self.topk_dict, 0, batch_user)

        return interesting_samples, uninteresting_samples

    # epoch 마다
    def do_something_in_each_epoch(self, epoch):
        # 得到interesting items 以及uninteresting items的索引
        with torch.no_grad():
            # interesting items
            self.interesting_items = torch.zeros((self.num_users, self.K)) # 初始化矩阵

            # sampling
            while True:
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False) # 不会采样重复的元素，输出的是索引矩阵
                if (samples > self.mxK).sum() == 0: # 保证采样的都是前self.maxK的元素
                    break

            samples = samples.sort(dim=1)[0] # samples会返回 排序后的matrix以及原本元素的索引,这里只取第一个返回

            for user in range(self.num_users):
                self.interesting_items[user] = self.topk_dict[user][samples[user]]

            self.interesting_items = self.interesting_items.cuda()
            

            # uninteresting items
            num_parts = 256
            chunk_size = self.num_users // num_parts
            all_tmp = []

            # different sample mode
            targt = None
            if self.mode.lower == "val_diff":
                # add val information between S and T
                T_pred = self.teacher.get_user_item_ratings(torch.arange(self.num_users), self.item_idx)
                S_pred = self.student.get_user_item_ratings(torch.arange(self.num_users), self.item_idx)
                targt = (T_pred - S_pred) * self.mask
                del T_pred, S_pred
            elif self.mode.lower == "val_T":
                # add Teacher's val information
                T_pred = self.teacher.get_user_item_ratings(torch.arange(self.num_users), self.item_idx)
                targt = T_pred * self.mask
                del T_pred
            else: 
                # add no val informatio
                targt = torch.zeros_like(self.model_variance)


            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != num_parts - 1 else self.num_users
                var_slice = self.model_variance[start_idx:end_idx, :].cuda() / self.neg_T
                clamped_var = torch.clamp(var_slice, max=80.0)
                print(var_slice.min(), var_slice.max())
                m_part = torch.exp(clamped_var).cuda()
                # m_part = torch.exp(torch.min(self.model_variance[start_idx:end_idx, :].cuda()/self.neg_T, 80.0)).cuda()
                # print(m_part.min(), m_part.max())
                # m_part = self.model_variance[start_idx:end_idx, :].cuda()
                tar_part = targt[start_idx:end_idx, :].cuda().softmax(dim=1)
                if self.mode == "val_diff" or self.mode == "val_T":
                    tmp_part = torch.multinomial(tar_part + self.alpha * m_part, self.L, replacement=False)
                else:
                    tmp_part = torch.multinomial(m_part, self.L, replacement=False)
                idx_part = self.item_idx[start_idx:end_idx, :].cuda()
                all_tmp.append(torch.gather(idx_part, 1, tmp_part))
                del m_part, tmp_part

            self.uninteresting_items = torch.cat(all_tmp, 0)

            # extra items
            all_tmp = []
            # self.mask[torch.arange(self.uninteresting_items.size(0)).unsqueeze(-1), self.uninteresting_items] = 0

            for i in range(num_parts):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != num_parts - 1 else self.num_users
                
                m_part = self.mask[start_idx:end_idx, :].cuda()
                tmp_part = torch.multinomial(m_part, self.extra + self.L, replacement=False)

                all_tmp.append(tmp_part)
                del m_part, tmp_part
            # self.mask[torch.arange(self.uninteresting_items.size(0)).unsqueeze(-1), self.uninteresting_items] = 1
            self.extra_items = torch.cat(all_tmp, 0)
            
            del all_tmp

    def reset_item(self):
        # print(f"visual extra_items shape: {self.extra_items.dtype}, uninteresting_items shape: {self.uninteresting_items.shape}")
        return self.extra_items # user_num X (self.L + self.extra) 随机采样
    def get_un_items(self):
        sort_items = torch.sort(self.uninteresting_items[0])[0]
        sorted_items_str = " ".join(map(str, sort_items.tolist()))
        with open("output.txt", "a") as f:
            f.write(sorted_items_str + "\n")
        return self.uninteresting_items

    def relaxed_ranking_loss(self, S1, S2, S3):
        
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))     # 之后要做exp操作，这里做一个截断，防止数据爆炸导致的一系列问题
        S3 = torch.minimum(S3, torch.tensor(80., device=S3.device))

        unselected_below = S3.exp().sum(1, keepdims=True) - S1.exp().sum(1, keepdims=True)
        unselected_below = torch.maximum(unselected_below, torch.tensor(0., device=unselected_below.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + self.neg * below2 + self.unselected * unselected_below).log().sum(1, keepdims=True)
        
        return -(above - below).sum()


    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        interesting_items, uninteresting_items = self.get_samples(users)

        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()
        self.potential_interesting_items = self.potential_interesting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)
        self.potential_interesting_prediction = self.student.forward_multi_items(users, self.potential_interesting_items)

        URRD_loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction, self.potential_interesting_prediction)

        return URRD_loss

class DCD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.K = args.dcd_K
        self.T = args.dcd_T
        self.mxK = args.dcd_mxK
        self.ablation = args.ablation
        self.tau = args.dcd_tau
        self.negx = args.neg_x
        self.T_topk = self.get_topk_dict()
        self.T_rank = torch.arange(self.mxK).repeat(self.num_users, 1).cuda() # 在教师视角T_topk里元素的rk就是1-n的正序排序

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # usr-item score matrix
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict # top_k的索引
    
    def get_samples(self, batch_user):
        underestimated_samples = torch.index_select(self.underestimated_items, 0, batch_user)
        overestimated_samples = torch.index_select(self.overestimated_items, 0, batch_user)
        return underestimated_samples, overestimated_samples
 
    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            
            num_parts = 256
            chunk_size = self.num_users // num_parts
            S_pred = self.student.get_all_ratings() # user_num X item_num
            rank_diff = torch.zeros((self.num_users, self.mxK), dtype=torch.float).cuda()
            rank_diff_inv = torch.zeros((self.num_users, self.mxK), dtype=torch.float).cuda()

            for start_idx in range(0, self.num_users, num_parts):
                end_idx = min(start_idx + num_parts, self.num_users)
                batch_user = torch.arange(start_idx, end_idx)
                S_pred_slice = S_pred[batch_user, :].cuda() # batch_size X item_num

                S_topk = torch.argsort(S_pred_slice, descending=True, dim=-1) # user_num X item_num, 返回降序排序后每一个位置对应的原item的idx
                S_rank = torch.argsort(S_topk, dim=-1) # 返回
                S_rank = S_rank[torch.arange(len(S_rank)).unsqueeze(-1), self.T_topk[start_idx:end_idx].cuda()] # user_num X item_num, 取出每个用户的topk的rank

                diff = S_rank - self.T_rank[start_idx:end_idx]
                rank_diff[start_idx:end_idx] = torch.maximum(torch.tanh(torch.maximum(diff / self.T, torch.tensor(0.))), torch.tensor(1e-5))
                diff_inv = self.T_rank[start_idx:end_idx] - S_rank
                rank_diff_inv[start_idx:end_idx] = torch.maximum(torch.tanh(torch.maximum(diff_inv / self.T, torch.tensor(0.))), torch.tensor(1e-5))

            # sampling
            underestimated_idx = torch.multinomial(rank_diff, self.K, replacement=False)
            self.underestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), underestimated_idx]
            overestimated_idx = torch.multinomial(rank_diff_inv, self.K, replacement=False)
            self.overestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), overestimated_idx]
    
    def relaxed_ranking_loss(self, S1, S2):
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + self.negx*below2).log().sum(1, keepdims=True)
        
        return -(above - below).sum()
    
    def ce_loss(self, logit_T, logit_S):
        prob_T = torch.softmax(logit_T / self.tau, dim=-1)
        loss = F.cross_entropy(logit_S / self.tau, prob_T, reduction='sum')
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        underestimated_items, overestimated_items = self.get_samples(users)
        underestimated_items = underestimated_items.type(torch.LongTensor).cuda()
        overestimated_items = overestimated_items.type(torch.LongTensor).cuda()

        underestimated_prediction = self.student.forward_multi_items(users, underestimated_items)
        overestimated_prediction = self.student.forward_multi_items(users, overestimated_items)

        if self.ablation:
            underestimated_prediction_T = self.teacher.forward_multi_items(users, underestimated_items)
            overestimated_prediction_T = self.teacher.forward_multi_items(users, overestimated_items)
            prediction_T = torch.concat([underestimated_prediction_T, overestimated_prediction_T], dim=-1)
            prediction_S = torch.concat([underestimated_prediction, overestimated_prediction], dim=-1)
            loss = self.ce_loss(prediction_T, prediction_S)
        else:
            loss = self.relaxed_ranking_loss(underestimated_prediction, overestimated_prediction)

        return loss

class DCDVar(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.K = args.dcd_K
        self.T = args.dcd_T
        self.mxK = args.dcd_mxK
        self.L = args.dcd_L
        self.ablation = args.ablation
        self.tau = args.dcd_tau
        self.negx = args.dcd_negx
        self.T_topk = self.get_topk_dict()
        self.T_rank = torch.arange(self.mxK).repeat(self.num_users, 1).cuda() # 在教师视角T_topk里元素的rk就是1-n的正序排序
        self.var_tau = args.var_tau # 负采样中依据方差预测所占的权重
        
        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items), dtype=torch.float).cuda()
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.T_topk[user]] = 0 # 把每个用户top mxk的interesting item以及交互过的item都mask掉,那么它们之后被采样的概率就是0了，剩余item的值都是1，会被等概率采样
        self.mask.requires_grad = False
    
    
    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # usr-item score matrix
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict # top_k的索引
    
    def get_samples(self, batch_user):
        underestimated_samples = torch.index_select(self.underestimated_items, 0, batch_user)
        overestimated_samples = torch.index_select(self.overestimated_items, 0, batch_user)
        return underestimated_samples, overestimated_samples
    
    def set_model_variance(self, model_variance):
        self.model_variance = model_variance
        self.model_variance += 1e-8

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            S_pred = self.student.get_all_ratings() # user_num X item_num
            S_topk = torch.argsort(S_pred, descending=True, dim=-1) # user_num X item_num, 返回降序排序后每一个位置对应的原item的idx
            S_rank = torch.argsort(S_topk, dim=-1) # 返回
            S_rank = S_rank[torch.arange(len(S_rank)).unsqueeze(-1), self.T_topk]
            diff = S_rank - self.T_rank
            rank_diff = torch.maximum(torch.tanh(torch.maximum(diff / self.T, torch.tensor(0.))), torch.tensor(1e-5))
            
            
            # diff_inv = self.T_rank - S_rank
            # rank_diff_inv = torch.maximum(torch.tanh(torch.maximum(diff_inv / self.T, torch.tensor(0.))), torch.tensor(1e-5))

            # sampling
            underestimated_idx = torch.multinomial(rank_diff, self.K, replacement=False)
            self.underestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), underestimated_idx]
            # overestimated_idx = torch.multinomial(rank_diff_inv, self.K, replacement=False)
            # self.overestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), overestimated_idx]

            mask_mat = self.mask * self.model_variance
            m1 = mask_mat[: self.num_users // 2, :].cuda()
            tmp1 = torch.multinomial(m1, self.L, replacement=False)
            del m1

            m2 = mask_mat[self.num_users // 2 : ,:].cuda()
            tmp2 = torch.multinomial(m2, self.L, replacement=False)
            del m2

            self.overestimated_items = torch.cat([tmp1, tmp2], 0)


    def relaxed_ranking_loss(self, S1, S2):
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + self.negx*below2).log().sum(1, keepdims=True)
        
        return -(above - below).sum()
    
    def ce_loss(self, logit_T, logit_S):
        prob_T = torch.softmax(logit_T / self.tau, dim=-1)
        loss = F.cross_entropy(logit_S / self.tau, prob_T, reduction='sum')
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        underestimated_items, overestimated_items = self.get_samples(users)
        underestimated_items = underestimated_items.type(torch.LongTensor).cuda()
        overestimated_items = overestimated_items.type(torch.LongTensor).cuda()

        underestimated_prediction = self.student.forward_multi_items(users, underestimated_items)
        overestimated_prediction = self.student.forward_multi_items(users, overestimated_items)

        if self.ablation:
            underestimated_prediction_T = self.teacher.forward_multi_items(users, underestimated_items)
            overestimated_prediction_T = self.teacher.forward_multi_items(users, overestimated_items)
            prediction_T = torch.concat([underestimated_prediction_T, overestimated_prediction_T], dim=-1)
            prediction_S = torch.concat([underestimated_prediction, overestimated_prediction], dim=-1)
            loss = self.ce_loss(prediction_T, prediction_S)
        else:
            loss = self.relaxed_ranking_loss(underestimated_prediction, overestimated_prediction)

        return loss

class MRRD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "mrrd"
        self.K = args.mrrd_K
        self.L = args.mrrd_L
        self.T = args.mrrd_T
        self.mxK = args.mrrd_mxK
        self.no_sort = args.no_sort
        self.beta = args.mrrd_beta      # weight of rest of topk predictions
        self.loss_type = args.loss_type
        self.sample_rank = args.sample_rank
        self.tau = args.mrrd_tau
        self.gamma = args.mrrd_gamma    # weight of uninteresting predictions
        self.test_generalization = args.mrrd_test_type

        # For interesting item
        # if self.loss_type in ["ce", "listnet"]:
        #     self.mxK = self.K
        # if self.test_generalization == 1 or self.test_generalization == 2:
        #     self.topk_scores, self.topk_dict = self.get_topk_dict(self.mxK)
        #     if self.test_generalization == 1:
        #         f_test_topk_dict = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_dict_train.pkl")
        #         f_test_topk_score = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_score_train.pkl")
        #     else:
        #         f_test_topk_dict = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_dict_test.pkl")
        #         f_test_topk_score = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_score_test.pkl")
        #     if not os.path.exists(f_test_topk_dict) or not os.path.exists(f_test_topk_score):
        #         self.test_topk_dict = {}
        #         self.test_topk_score = {}
        #         train_dict = self.dataset.train_dict
        #         for u in range(self.num_users):
        #             if self.test_generalization == 1:
        #                 if u not in train_dict:
        #                     continue
        #                 test_topk_dict = train_dict[u][:100].long().cuda()
        #             else:
        #                 if u not in valid_dict or u not in test_dict:
        #                     continue
        #                 test_topk_dict = torch.concat([valid_dict[u], test_dict[u]]).long().cuda()

        #             test_topk_score = self.teacher.forward_multi_items(torch.tensor([u]).long().cuda(), test_topk_dict.unsqueeze(0))[0]
        #             idx = torch.argsort(test_topk_score, descending=True)
        #             self.test_topk_dict[u] = test_topk_dict[idx]
        #             self.test_topk_score[u] = test_topk_score[idx]
        #         dump_pkls((self.test_topk_dict, f_test_topk_dict), (self.test_topk_score, f_test_topk_score))
        #     else:
        #         _, self.test_topk_dict, self.test_topk_score = load_pkls(f_test_topk_dict, f_test_topk_score)
        # elif self.test_generalization == 3 or self.test_generalization == 5:
        #     self.test_K = 100
        #     topk_scores, topk_dict = self.get_topk_dict(self.mxK + self.test_K)
        #     f_train_idx = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"train_idx_{self.mxK}_{self.test_K}.npy")
        #     f_test_idx = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_idx_{self.mxK}_{self.test_K}.npy")
        #     if not os.path.exists(f_train_idx) or os.path.exists(f_test_idx):
        #         train_idx = torch.zeros(self.num_users, self.mxK).long()
        #         test_idx = torch.zeros(self.num_users, self.test_K).long()
        #         for u in range(self.num_users):
        #             tr_idx, te_idx = torch.utils.data.random_split(torch.arange(self.mxK + self.test_K), [self.mxK, self.test_K])
        #             train_idx[u], test_idx[u] = torch.tensor(tr_idx).sort()[0].long(), torch.tensor(te_idx).sort()[0].long()
        #         os.makedirs(os.path.dirname(f_train_idx), exist_ok=True)
        #         os.makedirs(os.path.dirname(f_test_idx), exist_ok=True)
        #         np.save(f_train_idx, train_idx.cpu().numpy())
        #         np.save(f_test_idx, test_idx.cpu().numpy())
        #     else:
        #         train_idx = torch.from_numpy(np.load(f_train_idx)).long()
        #         test_idx = torch.from_numpy(np.load(f_test_idx)).long()
        #     self.topk_scores, self.topk_dict = torch.zeros(self.num_users, self.mxK).cuda(), torch.zeros(self.num_users, self.mxK).long().cuda()
        #     self.test_topk_score, self.test_topk_dict = torch.zeros(self.num_users, self.test_K).cuda(), torch.zeros(self.num_users, self.test_K).long().cuda()
        #     for u in range(self.num_users):
        #         self.topk_scores[u], self.topk_dict[u] = topk_scores[u][train_idx[u]], topk_dict[u][train_idx[u]]
        #         self.test_topk_score[u], self.test_topk_dict[u] = topk_scores[u][test_idx[u]], topk_dict[u][test_idx[u]]
        # elif self.test_generalization == 4:
        #     self.test_K = 100
        #     self.topk_scores, self.topk_dict = self.get_topk_dict(self.mxK)
        #     f_test_topk_dict = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_dict_{self.mxK}_{self.test_K}.pkl")
        #     f_test_topk_score = os.path.join(args.CRAFT_DIR, args.dataset, self.student.model_name, self.teacher.model_name, self.model_name, f"test_topk_score_{self.mxK}_{self.test_K}.pkl")
        #     if os.path.exists(f_test_topk_dict) and os.path.exists(f_test_topk_score):
        #         _, self.test_topk_dict, self.test_topk_score = load_pkls(f_test_topk_dict, f_test_topk_score)
        #     else:
        #         self.test_topk_dict = torch.zeros(self.num_users, self.test_K).long().cuda()
        #         self.test_topk_score = torch.zeros(self.num_users, self.test_K).long().cuda()
        #         indices = torch.multinomial(torch.ones_like(self.topk_scores), self.test_K, replacement=False).sort(-1)[0]
        #         for u in range(self.num_users):
        #             self.test_topk_dict[u] = self.topk_dict[u][indices[u]]
        #             self.test_topk_score[u] = self.topk_scores[u][indices[u]]
        #         dump_pkls((self.test_topk_dict, f_test_topk_dict), (self.test_topk_score, f_test_topk_score))
        # else:
        self.topk_scores, self.topk_dict = self.get_topk_dict(self.mxK)

        if self.sample_rank:
            ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
            self.ranking_mat = ranking_list.repeat(self.num_users, 1)
        else:
            self.ranking_mat = torch.exp(self.topk_scores / self.tau)

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items))
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.topk_dict[user]] = 0
        self.mask.requires_grad = False

    def get_topk_dict(self, mxK):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings()
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            topk_scores, topk_dict = torch.topk(inter_mat, mxK, dim=-1)
        return topk_scores.cuda(), topk_dict.cuda()
    
    def get_samples(self, batch_user):
        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
        return interesting_samples, uninteresting_samples

    @torch.no_grad()
    def generalization_error(self):
        gen_errors = []
        user_list = torch.arange(self.num_users).cuda()
        for _ in range(5):
            errs = []
            if self.test_generalization == 1 or self.test_generalization == 2:
                for u in range(self.num_users):
                    if u not in self.test_topk_dict:
                        continue
                    if self.sample_rank:
                        ranking_list = torch.exp(-(torch.arange(len(self.test_topk_dict[u])) + 1) / self.T)
                    else:
                        ranking_list = torch.exp(self.test_topk_score[u] / self.tau)
                    samples = torch.multinomial(ranking_list, len(self.test_topk_dict[u]), replacement=False)
                    interesting_items_u = self.test_topk_dict[u][samples]
                    S1 = self.student.forward_multi_items(torch.tensor([u]).long().cuda(), interesting_items_u.unsqueeze(0))
                    above = S1.sum(-1)
                    below = S1.flip(-1).exp().cumsum(-1).log().sum(-1)
                    loss = -(above - below)
                    errs.append(loss)
            elif self.test_generalization == 3 or self.test_generalization == 4:
                if self.sample_rank:
                    ranking_list = torch.exp(-(torch.arange(self.test_K) + 1) / self.T)
                    ranking_mat = ranking_list.repeat(self.num_users, 1)
                else:
                    ranking_mat = torch.exp(self.test_topk_score / self.tau)
                samples = torch.multinomial(ranking_mat, self.test_K, replacement=False)
                interesting_items = torch.zeros((self.num_users, self.test_K)).long().cuda()
                for u in range(self.num_users):
                    interesting_items[u] = self.test_topk_dict[u][samples[u]]
                bs = self.args.batch_size
                for i in range(math.ceil(self.num_users / bs)):
                    batch_user = user_list[bs * i: bs * (i + 1)]
                    interesting_items_u = torch.index_select(interesting_items, 0, batch_user)
                    S1 = self.student.forward_multi_items(batch_user, interesting_items_u)
                    above = S1.sum(-1)
                    below = S1.flip(-1).exp().cumsum(-1).log().sum(-1)
                    loss = -(above - below)
                    errs.append(loss)
            loss = torch.concat(errs).mean().item()
            gen_errors.append(loss)
        err =  sum(gen_errors) / len(gen_errors)
        return err
    
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        output = self.student(batch_user, batch_pos_item, batch_neg_item)
        base_loss = self.student.get_loss(output)
        kd_loss = self.get_loss(batch_user, batch_pos_item, batch_neg_item)
        if self.test_generalization > 0:
            loss = self.lmbda * kd_loss
        else:
            loss = base_loss + self.lmbda * kd_loss
        return loss, base_loss.detach(), kd_loss.detach()
    
    @torch.no_grad()
    def plot_statistics(self, epoch):
        ce_errs, sigma_errs = [], []
        user_list = torch.arange(self.num_users).cuda()
        for _ in range(5):
            ce_err, sigma_err = [], []
            bs = self.args.batch_size
            for i in range(math.ceil(self.num_users / bs)):
                batch_user = user_list[bs * i: bs * (i + 1)]
                K = self.test_K // 2
                randidx = torch.randperm(self.test_K)[:K]
                test_items = self.test_topk_dict[batch_user][:, randidx]
                T = self.test_topk_score[batch_user][:, randidx]
                S = self.student.forward_multi_items(batch_user, test_items)
                expT = torch.exp(T / self.tau)  # bs, K
                prob_T1 = expT.unsqueeze(-1) / torch.sum(expT, dim=-1, keepdim=True).unsqueeze(-1)    # bs, K, 1
                Z_T2 = expT.sum(-1, keepdim=True).unsqueeze(1).repeat(1, K, K)   # bs, K, K
                Z_T2 = Z_T2 - expT.unsqueeze(-1)
                # make diag of prob_T2 0
                prob_T2 = expT.unsqueeze(1) / Z_T2 # bs, K, K
                prob_T2 -= torch.diag_embed(torch.diagonal(prob_T2, dim1=1, dim2=2), dim1=1, dim2=2)
                prob_T = prob_T1 * prob_T2  # bs, K, K
                expS = torch.exp(S / self.tau)
                log_prob_S1 = torch.log(expS.unsqueeze(-1) / torch.sum(expS, dim=-1, keepdim=True).unsqueeze(-1)) # bs, K, 1
                Z_S2 = expS.sum(-1, keepdim=True).unsqueeze(1).repeat(1, K, K)   # bs, K, K
                Z_S2 = Z_S2 - expS.unsqueeze(-1)
                Z_S2 = torch.maximum(Z_S2, torch.tensor(1e-4))
                log_prob_S2 = torch.log(expS.unsqueeze(1) / Z_S2)  # bs, K, K
                log_prob_S = log_prob_S1 + log_prob_S2  # bs, K, K
                loss_all = -(prob_T * log_prob_S).sum(-1).sum(-1)   # bs

                prob_T = torch.softmax(T / self.tau, dim=-1)
                loss_ce = F.cross_entropy(S / self.tau, prob_T, reduction='none') # bs
                ce_err.append(loss_ce)
                sigma_err.append(loss_all - loss_ce)
            loss_ce = torch.concat(ce_err).mean().item()
            loss_sigma = torch.cat(sigma_err).mean().item()
            ce_errs.append(loss_ce)
            sigma_errs.append(loss_sigma)
        ce_errs, sigma_errs = np.array(ce_errs), np.array(sigma_errs)
        mlflow.log_metrics({"sigma_expectation_pow2":np.power(sigma_errs.mean(), 2), "ce_expectation_pow2":np.power(ce_errs.mean(), 2), "sigma_variance":np.var(sigma_errs, ddof=1), "cov_sigma_ce":np.cov(sigma_errs, ce_errs, ddof=1)[0, 1]}, step=epoch // 5)

    def do_something_in_each_epoch(self, epoch):
        # if 1 <= self.test_generalization <= 4:
        #     if epoch % 5 == 0:
        #         err = self.generalization_error()
        #         mlflow.log_metric("gen_error", err, step=epoch // 5)
        # elif self.test_generalization >=5:
        #     if epoch % 5 == 0:
        #         self.plot_statistics(epoch)
        
        with torch.no_grad():
            if self.loss_type == "rrd":
                # interesting items
                self.interesting_items = torch.zeros((self.num_users, self.K))

                # sampling
                samples = torch.multinomial(self.ranking_mat, self.K, replacement=False)
                
                if not self.no_sort:
                    samples = samples.sort(dim=1)[0]
                
                for user in range(self.num_users):
                    self.interesting_items[user] = self.topk_dict[user][samples[user]]

                self.interesting_items = self.interesting_items.cuda()

            # uninteresting items
            m1 = self.mask[: self.num_users // 2, :].cuda()
            tmp1 = torch.multinomial(m1, self.L, replacement=False)
            del m1

            m2 = self.mask[self.num_users // 2 : ,:].cuda()
            tmp2 = torch.multinomial(m2, self.L, replacement=False)
            del m2

            self.uninteresting_items = torch.cat([tmp1, tmp2], 0)

    def rrd_all_loss(self, S1, S2, Stop):
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))
        Stop = torch.minimum(Stop, torch.tensor(80., device=Stop.device))
        above = S1.sum(-1)
        below1 = S1.flip(-1).exp().cumsum(-1)    # exp() of interesting_prediction results in inf
        below3 = Stop.exp().sum(-1, keepdims=True) - S1.exp().sum(-1, keepdims=True)
        below3 = torch.maximum(below3, torch.tensor(0., device=below3.device))
        below2 = S2.exp().sum(-1, keepdims=True)
        below = (below1 + self.gamma * below2 + self.beta * below3).log().sum(-1)
        loss = -(above - below).sum()
        return loss
    
    def neg_loss(self, logit_S_itemT, logit_S_uninteresting):
        above = torch.log(logit_S_itemT.exp().sum(-1))
        below = torch.log(logit_S_itemT.exp().sum(-1) + torch.exp(logit_S_uninteresting).sum(-1))
        loss = -(above - below).sum()
        return loss
    
    def ce_loss(self, S, T):
        if self.sample_rank:
            ranking_list = -(torch.arange(self.mxK) + 1) / self.T
            ranking_mat = ranking_list.repeat(len(T), 1).cuda()
            prob_T = torch.softmax(ranking_mat, dim=-1)     # bs, mxK
        else:
            prob_T = torch.softmax(T / self.tau, dim=-1)
        loss = F.cross_entropy(S / self.tau, prob_T, reduction='sum')
        return loss

    def list2_loss(self, S, T):
        S = torch.minimum(S, torch.tensor(60., device=S.device))
        if self.sample_rank:
            ranking_list = -(torch.arange(self.mxK) + 1) / self.T
            ranking_mat = ranking_list.repeat(len(T), 1).cuda()
            expT = torch.exp(ranking_mat)   # bs, mxK
        else:
            expT = torch.exp(T / self.tau)  # bs, mxK
        prob_T1 = expT.unsqueeze(-1) / torch.sum(expT, dim=-1, keepdim=True).unsqueeze(-1)    # bs, mxK, 1
        Z_T2 = expT.sum(-1, keepdim=True).unsqueeze(1).repeat(1, self.mxK, self.mxK)   # bs, mxK, mxK
        Z_T2 = Z_T2 - expT.unsqueeze(-1)
        prob_T2 = expT.unsqueeze(1) / Z_T2 # bs, mxK, mxK
        # make diag of prob_T2 0
        prob_T2 -= torch.diag_embed(torch.diagonal(prob_T2, dim1=1, dim2=2), dim1=1, dim2=2)
        prob_T = prob_T1 * prob_T2  # bs, mxK, mxK
        expS = torch.exp(S / self.tau)
        log_prob_S1 = torch.log(expS.unsqueeze(-1) / torch.sum(expS, dim=-1, keepdim=True).unsqueeze(-1)) # bs, mxK, 1
        Z_S2 = expS.sum(-1, keepdim=True).unsqueeze(1).repeat(1, self.mxK, self.mxK)   # bs, mxK, mxK
        Z_S2 = Z_S2 - expS.unsqueeze(-1)
        Z_S2 = torch.maximum(Z_S2, torch.tensor(1e-4))
        log_prob_S2 = torch.log(expS.unsqueeze(1) / Z_S2)  # bs, mxK, mxK
        log_prob_S = log_prob_S1 + log_prob_S2  # bs, mxK, mxK
        loss = -(prob_T * log_prob_S).sum()
        return loss
    
    def ce_all_loss(self, S, T, S2):
        if self.loss_type == "listnet":
            loss = self.list2_loss(S, T)
        else:
            loss = self.ce_loss(S, T)
        if self.gamma > 0:
            loss += self.gamma * self.neg_loss(S, S2)
        return loss
    
    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        # if self.loss_type in ["ce", "listnet"]:
        #     uninteresting_items = torch.index_select(self.uninteresting_items, 0, users).type(torch.LongTensor).cuda()
        #     uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)
        #     topk_prediction_S = self.student.forward_multi_items(users, self.topk_dict[users])
        #     topk_prediction_T = self.topk_scores[users]
        #     loss = self.ce_all_loss(topk_prediction_S, topk_prediction_T, uninteresting_prediction)
        # else:
        interesting_items, uninteresting_items = self.get_samples(users)
        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)
        topk_prediction = self.student.forward_multi_items(users, self.topk_dict[users])

        loss = self.rrd_all_loss(interesting_prediction, uninteresting_prediction, topk_prediction)
        return loss



class DCDv2(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.K = args.dcd_K
        self.T = args.dcd_T
        self.mxK = args.dcd_mxK
        self.ablation = args.ablation
        self.tau = args.dcd_tau
        self.T_topk = self.get_topk_dict()
        self.T_rank = torch.arange(self.mxK).repeat(self.num_users, 1).cuda()  # 在教师视角T_topk里元素的rk就是1-n的正序排序

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings()  # usr-item score matrix
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict  # top_k的索引

    def get_samples(self, batch_user):
        underestimated_samples = torch.index_select(self.underestimated_items, 0, batch_user)
        overestimated_samples = torch.index_select(self.overestimated_items, 0, batch_user)
        return underestimated_samples, overestimated_samples

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            S_pred = self.student.get_all_ratings()  # user_num X item_num
            S_topk = torch.argsort(S_pred, descending=True, dim=-1)  # user_num X item_num, 返回降序排序后每一个位置对应的原item的idx
            S_rank = torch.argsort(S_topk, dim=-1)  # 返回
            S_rank = S_rank[torch.arange(len(S_rank)).unsqueeze(-1), self.T_topk]
            diff = S_rank - self.T_rank
            rank_diff = torch.maximum(torch.tanh(torch.maximum(diff / self.T, torch.tensor(0.))), torch.tensor(1e-5))
            diff_inv = self.T_rank - S_rank
            rank_diff_inv = torch.maximum(torch.tanh(torch.maximum(diff_inv / self.T, torch.tensor(0.))),
                                          torch.tensor(1e-5))

            # sampling
            underestimated_idx = torch.multinomial(rank_diff, self.K, replacement=False)
            self.underestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), underestimated_idx]
            overestimated_idx = torch.multinomial(rank_diff_inv, self.K, replacement=False)
            self.overestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), overestimated_idx]

    def relaxed_ranking_loss(self, S1, S2):
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))  # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)  # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + below2).log().sum(1, keepdims=True)

        return -(above - below).sum()

    def ce_loss(self, logit_T, logit_S):
        prob_T = torch.softmax(logit_T / self.tau, dim=-1)
        loss = F.cross_entropy(logit_S / self.tau, prob_T, reduction='sum')
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        underestimated_items, overestimated_items = self.get_samples(users)
        underestimated_items = underestimated_items.type(torch.LongTensor).cuda()
        overestimated_items = overestimated_items.type(torch.LongTensor).cuda()

        underestimated_prediction_T = self.teacher.forward_multi_items(users, underestimated_items)
        overestimated_prediction_T = self.teacher.forward_multi_items(users, overestimated_items)
        underestimated_prediction_T = torch.softmax(underestimated_prediction_T, dim=-1)
        overestimated_prediction_T = torch.softmax(overestimated_prediction_T, dim=-1)
        underestimated_prediction_T = torch.sum(underestimated_prediction_T * underestimated_prediction_T, dim=-1)
        overestimated_prediction_T = torch.sum(overestimated_prediction_T * overestimated_prediction_T, dim=-1)

        _, topk_dict_under = torch.topk(underestimated_prediction_T, int(self.K/2), dim=-1)
        _, topk_dict_over = torch.topk(overestimated_prediction_T, int(self.K/2), dim=-1)
        underestimated_items = torch.index_select(underestimated_items, 0, topk_dict_under)
        overestimated_items = torch.index_select(overestimated_items, 0, topk_dict_over)

        underestimated_prediction = self.student.forward_multi_items(users, underestimated_items)
        overestimated_prediction = self.student.forward_multi_items(users, overestimated_items)

        if self.ablation:
            underestimated_prediction_T = self.teacher.forward_multi_items(users, underestimated_items)
            overestimated_prediction_T = self.teacher.forward_multi_items(users, overestimated_items)
            prediction_T = torch.concat([underestimated_prediction_T, overestimated_prediction_T], dim=-1)
            prediction_S = torch.concat([underestimated_prediction, overestimated_prediction], dim=-1)
            loss = self.ce_loss(prediction_T, prediction_S)
        else:
            loss = self.relaxed_ranking_loss(underestimated_prediction, overestimated_prediction)

        return loss
class DCDoptim(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.K = args.dcd_K
        self.T = args.dcd_T
        self.L = args.dcd_L
        self.mxK = args.dcd_mxK
        self.ablation = args.ablation
        self.tau = args.dcd_tau
        self.T_topk = self.get_topk_dict()
        self.T_rank = torch.arange(self.mxK).repeat(self.num_users, 1).cuda() # 在教师视角T_topk里元素的rk就是1-n的正序排序

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items))
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.T_topk[user]] = 0 # 把每个用户top mxk的interesting item以及交互过的item都mask掉,那么它们之后被采样的概率就是0了，剩余item的值都是1，会被等概率采样
        self.mask.requires_grad = False

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # usr-item score matrix
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict # top_k的索引
    
    def get_samples(self, batch_user):
        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
        return interesting_samples, uninteresting_samples
 
    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            S_pred = self.student.get_all_ratings() # user_num X item_num
            S_topk = torch.argsort(S_pred, descending=True, dim=-1) # user_num X item_num, 返回降序排序后每一个位置对应的原item的idx
            S_rank = torch.argsort(S_topk, dim=-1) # 返回
            S_rank = S_rank[torch.arange(len(S_rank)).unsqueeze(-1), self.T_topk]
            diff = abs(S_rank - self.T_rank)
            rank_diff = torch.maximum(torch.tanh(torch.maximum(diff / self.T, torch.tensor(0.))), torch.tensor(1e-5))

            # sampling_interesting
            interesting_idx = torch.multinomial(rank_diff, self.K, replacement=False) # mxK里面采样k个
            self.interesting_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), interesting_idx]

            # sampling_uninteresting
            m1 = self.mask[: self.num_users // 2, :].cuda()
            tmp1 = torch.multinomial(m1, self.L, replacement=False)
            del m1

            m2 = self.mask[self.num_users // 2 : ,:].cuda()
            tmp2 = torch.multinomial(m2, self.L, replacement=False)
            del m2

            self.uninteresting_items = torch.cat([tmp1, tmp2], 0)

    
    def relaxed_ranking_loss(self, S1, S2):
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + below2).log().sum(1, keepdims=True)
        
        return -(above - below).sum()
    
    def ce_loss(self, logit_T, logit_S):
        prob_T = torch.softmax(logit_T / self.tau, dim=-1)
        loss = F.cross_entropy(logit_S / self.tau, prob_T, reduction='sum')
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        interesting_items, uninteresting_items = self.get_samples(users)
        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)

        if self.ablation:
            interesting_prediction_T = self.teacher.forward_multi_items(users, interesting_items)
            uninteresting_prediction_T = self.teacher.forward_multi_items(users, uninteresting_items)
            prediction_T = torch.concat([interesting_prediction_T, uninteresting_prediction_T], dim=-1)
            prediction_S = torch.concat([interesting_prediction, uninteresting_prediction], dim=-1)
            loss = self.ce_loss(prediction_T, prediction_S)
        else:
            loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)

        return loss

class DCDoptim2(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.K = args.dcd_K
        self.T = args.dcd_T
        self.L = args.dcd_L
        self.mxK = args.dcd_mxK
        self.ablation = args.ablation
        self.tau = args.dcd_tau
        self.T_topk = self.get_topk_dict()
        self.T_rank = torch.arange(self.mxK).repeat(self.num_users, 1).cuda() # 在教师视角T_topk里元素的rk就是1-n的正序排序

        # For uninteresting item
        self.mask = torch.ones((self.num_users, self.num_items))
        train_pairs = self.dataset.train_pairs
        self.mask[train_pairs[:, 0], train_pairs[:, 1]] = 0
        for user in range(self.num_users):
            self.mask[user, self.T_topk[user]] = 0 # 把每个用户top mxk的interesting item以及交互过的item都mask掉,那么它们之后被采样的概率就是0了，剩余item的值都是1，会被等概率采样
        self.mask.requires_grad = False

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # usr-item score matrix
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict # top_k的索引
    
    def get_samples(self, batch_user):
        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)
        return interesting_samples, uninteresting_samples
 
    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            S_pred = self.student.get_all_ratings() # user_num X item_num
            S_topk = torch.argsort(S_pred, descending=True, dim=-1) # user_num X item_num, 返回降序排序后每一个位置对应的原item的idx
            S_rank = torch.argsort(S_topk, dim=-1) # 返回
            S_rank = S_rank[torch.arange(len(S_rank)).unsqueeze(-1), self.T_topk]
            diff = S_rank - self.T_rank
            rank_diff = torch.maximum(torch.tanh(torch.maximum(diff / self.T, torch.tensor(0.))), torch.tensor(1e-5))

            # sampling_interesting
            interesting_idx = torch.multinomial(rank_diff, self.K, replacement=False) # mxK里面采样k个
            self.interesting_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), interesting_idx]

            # sampling_uninteresting
            m1 = self.mask[: self.num_users // 2, :].cuda()
            tmp1 = torch.multinomial(m1, self.L, replacement=False)
            del m1

            m2 = self.mask[self.num_users // 2 : ,:].cuda()
            tmp2 = torch.multinomial(m2, self.L, replacement=False)
            del m2

            self.uninteresting_items = torch.cat([tmp1, tmp2], 0)

    
    def relaxed_ranking_loss(self, S1, S2):
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + below2).log().sum(1, keepdims=True)
        
        return -(above - below).sum()
    
    def ce_loss(self, logit_T, logit_S):
        prob_T = torch.softmax(logit_T / self.tau, dim=-1)
        loss = F.cross_entropy(logit_S / self.tau, prob_T, reduction='sum')
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        interesting_items, uninteresting_items = self.get_samples(users)
        interesting_items = interesting_items.type(torch.LongTensor).cuda()
        uninteresting_items = uninteresting_items.type(torch.LongTensor).cuda()

        interesting_prediction = self.student.forward_multi_items(users, interesting_items)
        uninteresting_prediction = self.student.forward_multi_items(users, uninteresting_items)

        if self.ablation:
            interesting_prediction_T = self.teacher.forward_multi_items(users, interesting_items)
            uninteresting_prediction_T = self.teacher.forward_multi_items(users, uninteresting_items)
            prediction_T = torch.concat([interesting_prediction_T, uninteresting_prediction_T], dim=-1)
            prediction_S = torch.concat([interesting_prediction, uninteresting_prediction], dim=-1)
            loss = self.ce_loss(prediction_T, prediction_S)
        else:
            loss = self.relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)

        return loss

class DCDoptim3(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.K = args.dcd_K
        self.T = args.dcd_T
        self.mxK = args.dcd_mxK
        self.a = args.dcd_a
        self.ablation = args.ablation
        self.tau = args.dcd_tau
        self.T_topk = self.get_topk_dict()
        self.T_rank = torch.arange(self.mxK).repeat(self.num_users, 1).cuda() # 在教师视角T_topk里元素的rk就是1-n的正序排序

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # usr-item score matrix
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict # top_k的索引
    
    def get_samples(self, batch_user):
        underestimated_samples = torch.index_select(self.underestimated_items, 0, batch_user)
        overestimated_samples = torch.index_select(self.overestimated_items, 0, batch_user)
        return underestimated_samples, overestimated_samples
 
    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            S_pred = self.student.get_all_ratings() # user_num X item_num
            S_topk = torch.argsort(S_pred, descending=True, dim=-1) # user_num X item_num, 返回降序排序后每一个位置对应的原item的idx
            S_rank = torch.argsort(S_topk, dim=-1) # 返回
            S_rank = S_rank[torch.arange(len(S_rank)).unsqueeze(-1), self.T_topk]
            diff = S_rank - self.T_rank
            rank_diff = torch.maximum(torch.tanh(torch.maximum(diff / self.T, torch.tensor(0.))), torch.tensor(1e-5))
            diff_inv = self.T_rank - S_rank
            rank_diff_inv = torch.maximum(torch.tanh(torch.maximum(diff_inv / self.T, torch.tensor(0.))), torch.tensor(1e-5))

            # sampling
            underestimated_idx = torch.multinomial(rank_diff, self.K, replacement=False)
            self.underestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), underestimated_idx]
            overestimated_idx = torch.multinomial(rank_diff_inv, self.K, replacement=False)
            self.overestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), overestimated_idx]
    
    def relaxed_ranking_loss(self, S1, S2):
        S1 = torch.minimum(self.a * S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(self.a * S2, torch.tensor(80., device=S2.device))

        above = S1.sum(1, keepdims=True)

        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        below = (below1 + below2).log().sum(1, keepdims=True)
        
        return -(above - below).sum()
    
    def ce_loss(self, logit_T, logit_S):
        prob_T = torch.softmax(logit_T / self.tau, dim=-1)
        loss = F.cross_entropy(logit_S / self.tau, prob_T, reduction='sum')
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        underestimated_items, overestimated_items = self.get_samples(users)
        underestimated_items = underestimated_items.type(torch.LongTensor).cuda()
        overestimated_items = overestimated_items.type(torch.LongTensor).cuda()

        underestimated_prediction = self.student.forward_multi_items(users, underestimated_items)
        overestimated_prediction = self.student.forward_multi_items(users, overestimated_items)

        if self.ablation:
            underestimated_prediction_T = self.teacher.forward_multi_items(users, underestimated_items)
            overestimated_prediction_T = self.teacher.forward_multi_items(users, overestimated_items)
            prediction_T = torch.concat([underestimated_prediction_T, overestimated_prediction_T], dim=-1)
            prediction_S = torch.concat([underestimated_prediction, overestimated_prediction], dim=-1)
            loss = self.ce_loss(prediction_T, prediction_S)
        else:
            loss = self.relaxed_ranking_loss(underestimated_prediction, overestimated_prediction)

        return loss
    

class DCDoptim4(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.K = args.dcd_K
        self.T = args.dcd_T
        self.mxK = args.dcd_mxK
        self.ablation = args.ablation
        self.tau = args.dcd_tau
        self.x = args.dcd_x
        self.y = args.dcd_y
        self.T_topk = self.get_topk_dict()
        self.T_rank = torch.arange(self.mxK).repeat(self.num_users, 1).cuda() # 在教师视角T_topk里元素的rk就是1-n的正序排序

    def get_topk_dict(self):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = self.teacher.get_all_ratings() # usr-item score matrix
            train_pairs = self.dataset.train_pairs
            # remove true interactions from topk_dict
            inter_mat[train_pairs[:, 0], train_pairs[:, 1]] = -1e6
            _, topk_dict = torch.topk(inter_mat, self.mxK, dim=-1)
        return topk_dict # top_k的索引
    
    def get_samples(self, batch_user):
        underestimated_samples = torch.index_select(self.underestimated_items, 0, batch_user)
        overestimated_samples = torch.index_select(self.overestimated_items, 0, batch_user)
        return underestimated_samples, overestimated_samples
 
    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            S_pred = self.student.get_all_ratings() # user_num X item_num
            S_topk = torch.argsort(S_pred, descending=True, dim=-1) # user_num X item_num, 返回降序排序后每一个位置对应的原item的idx
            S_rank = torch.argsort(S_topk, dim=-1) # 返回
            S_rank = S_rank[torch.arange(len(S_rank)).unsqueeze(-1), self.T_topk]
            diff = S_rank - self.T_rank
            rank_diff = torch.maximum(torch.tanh(torch.maximum(diff / self.T, torch.tensor(0.))), torch.tensor(1e-5))
            diff_inv = self.T_rank - S_rank
            rank_diff_inv = torch.maximum(torch.tanh(torch.maximum(diff_inv / self.T, torch.tensor(0.))), torch.tensor(1e-5))

            # sampling
            underestimated_idx = torch.multinomial(rank_diff, self.K, replacement=False)
            self.underestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), underestimated_idx]
            overestimated_idx = torch.multinomial(rank_diff_inv, self.K, replacement=False)
            self.overestimated_items = self.T_topk[torch.arange(self.num_users).unsqueeze(-1), overestimated_idx]
    
    def relaxed_ranking_loss(self, S1, S2):
        S1 = torch.minimum(S1, torch.tensor(80., device=S1.device))     # This may help
        S2 = torch.minimum(S2, torch.tensor(80., device=S2.device))

        
        alsum = S1.exp().sum(1, keepdims=True) + S2.exp().sum(1, keepdims = True)
        alsum = alsum.log()
        subsum1 = S1.sum(1, keepdims = True)
    
        below1 = S1.flip(-1).exp().cumsum(1)    # exp() of interesting_prediction results in inf
        below2 = S2.exp().sum(1, keepdims=True)

        subsum2 = (below1 + below2).log().sum(1, keepdims=True)
        
        return -(subsum1 - self.x*subsum2 +(self.y)*alsum).sum()
    
    def ce_loss(self, logit_T, logit_S):
        prob_T = torch.softmax(logit_T / self.tau, dim=-1)
        loss = F.cross_entropy(logit_S / self.tau, prob_T, reduction='sum')
        return loss

    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        users = batch_user.unique()
        underestimated_items, overestimated_items = self.get_samples(users)
        underestimated_items = underestimated_items.type(torch.LongTensor).cuda()
        overestimated_items = overestimated_items.type(torch.LongTensor).cuda()

        underestimated_prediction = self.student.forward_multi_items(users, underestimated_items)
        overestimated_prediction = self.student.forward_multi_items(users, overestimated_items)

        if self.ablation:
            underestimated_prediction_T = self.teacher.forward_multi_items(users, underestimated_items)
            overestimated_prediction_T = self.teacher.forward_multi_items(users, overestimated_items)
            prediction_T = torch.concat([underestimated_prediction_T, overestimated_prediction_T], dim=-1)
            prediction_S = torch.concat([underestimated_prediction, overestimated_prediction], dim=-1)
            loss = self.ce_loss(prediction_T, prediction_S)
        else:
            # underestimated_items = underestimated_items.type(torch.FloatTensor).cuda()
            # overestimated_items = overestimated_items.type(torch.FloatTensor).cuda()

            loss = self.relaxed_ranking_loss(underestimated_prediction, overestimated_prediction)

        return loss

class HTD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.max_epoch = args.epochs
        self.alpha = args.htd_alpha
        self.num_experts = args.htd_num_experts
        self.choice = args.htd_choice
        self.T = args.htd_T

        self.student_dim = self.student.embedding_dim
        self.teacher_dim = self.teacher.embedding_dim

        # Group Assignment related parameters
        F_dims = [self.student_dim, (self.teacher_dim + self.student_dim) // 2, self.teacher_dim]

        self.user_f = nn.ModuleList([Expert(F_dims) for i in range(self.num_experts)])
        self.item_f = nn.ModuleList([Expert(F_dims) for i in range(self.num_experts)])

        self.user_v = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts), nn.Softmax(dim=1))
        self.item_v = nn.Sequential(nn.Linear(self.teacher_dim, self.num_experts), nn.Softmax(dim=1))

        self.sm = nn.Softmax(dim=1)

    def get_params_to_update(self):
        return [{"params": [param for param in self.parameters() if param.requires_grad], 'lr': self.args.lr, 'weight_decay': self.args.wd}]
    
    def do_something_in_each_epoch(self, epoch):
        self.T = 1.0 * ((1e-10 / 1.0) ** (epoch / self.max_epoch))
        self.T = max(self.T, 1e-10)

    def sim(self, A, B, is_inner=False):
        if not is_inner:
            denom_A = 1 / (A ** 2).sum(1, keepdim=True).sqrt()
            denom_B = 1 / (B.T ** 2).sum(0, keepdim=True).sqrt()

            sim_mat = torch.mm(A, B.T) * denom_A * denom_B
        else:
            sim_mat = torch.mm(A, B.T)
        return sim_mat

    def get_group_result(self, batch_entity, is_user=True):
        with torch.no_grad():
            if is_user:
                t = self.teacher.get_user_embedding(batch_entity)
                v = self.user_v
            else:
                t = self.teacher.get_item_embedding(batch_entity)	
                v = self.item_v

            z = v(t).max(-1)[1] 
            if not is_user:
                z = z + self.num_experts
                
            return z

    # For Adaptive Group Assignment
    def get_GA_loss(self, batch_entity, is_user=True):

        if is_user:
            s = self.student.get_user_embedding(batch_entity)													
            t = self.teacher.get_user_embedding(batch_entity)										

            f = self.user_f
            v = self.user_v
        else:
            s = self.student.get_item_embedding(batch_entity)													
            t = self.teacher.get_item_embedding(batch_entity)											
    
            f = self.item_f
            v = self.item_v

        alpha = v(t)
        g = torch.distributions.Gumbel(0, 1).sample(alpha.size()).cuda()
        alpha = alpha + 1e-10
        z = self.sm((alpha.log() + g) / self.T)

        z = torch.unsqueeze(z, 1)
        z = z.repeat(1, self.teacher_dim, 1)

        f_hat = [f[i](s).unsqueeze(-1) for i in range(self.num_experts)]
        f_hat = torch.cat(f_hat, -1)
        f_hat = f_hat * z
        f_hat = f_hat.sum(2)

        GA_loss = ((t - f_hat) ** 2).sum(-1).sum()

        return GA_loss
    
    def get_TD_loss(self, batch_user, batch_item):
        if self.choice == 'first':
            return self.get_TD_loss1(batch_user, batch_item)
        else:
            return self.get_TD_loss2(batch_user, batch_item)
                
    # Topology Distillation Loss (with Group(P,P))
    def get_TD_loss1(self, batch_user, batch_item):

        s = torch.cat([self.student.get_user_embedding(batch_user), self.student.get_item_embedding(batch_item)], 0)
        t = torch.cat([self.teacher.get_user_embedding(batch_user), self.teacher.get_item_embedding(batch_item)], 0)
        z = torch.cat([self.get_group_result(batch_user, is_user=True), self.get_group_result(batch_item, is_user=False)], 0)
        G_set = z.unique()
        Z = F.one_hot(z).float()	

        # Compute Prototype
        with torch.no_grad():
            tmp = Z.T
            tmp = tmp / (tmp.sum(1, keepdims=True) + 1e-10)
            P_s = tmp.mm(s)[G_set]
            P_t = tmp.mm(t)[G_set]

        # entity_level topology
        entity_mask = Z.mm(Z.T)        
        
        t_sim_tmp = self.sim(t, t) * entity_mask
        t_sim_dist = t_sim_tmp[t_sim_tmp > 0.]
        
        s_sim_dist = self.sim(s, s) * entity_mask    
        s_sim_dist = s_sim_dist[t_sim_tmp > 0.]
         
        # # Group_level topology
        t_proto_dist = self.sim(P_t, P_t).view(-1)
        s_proto_dist = self.sim(P_s, P_s).view(-1)

        total_loss = ((s_sim_dist - t_sim_dist) ** 2).sum() + ((s_proto_dist - t_proto_dist) ** 2).sum()

        return total_loss


    # Topology Distillation Loss (with Group(P,e))
    def get_TD_loss2(self, batch_user, batch_item):

        s = torch.cat([self.student.get_user_embedding(batch_user), self.student.get_item_embedding(batch_item)], 0)
        t = torch.cat([self.teacher.get_user_embedding(batch_user), self.teacher.get_item_embedding(batch_item)], 0)
        z = torch.cat([self.get_group_result(batch_user, is_user=True), self.get_group_result(batch_item, is_user=False)], 0)
        G_set = z.unique()
        Z = F.one_hot(z).float()

        # Compute Prototype
        with torch.no_grad():
            tmp = Z.T
            tmp = tmp / (tmp.sum(1, keepdims=True) + 1e-10)
            P_s = tmp.mm(s)[G_set]
            P_t = tmp.mm(t)[G_set]

        # entity_level topology
        entity_mask = Z.mm(Z.T)
        
        t_sim_tmp = self.sim(t, t) * entity_mask
        t_sim_dist = t_sim_tmp[t_sim_tmp > 0.]
        
        s_sim_dist = self.sim(s, s) * entity_mask    
        s_sim_dist = s_sim_dist[t_sim_tmp > 0.]
         
        # # Group_level topology 
        # t_proto_dist = (sim(P_t, t) * (1 - Z.T)[G_set]).view(-1)
        # s_proto_dist = (sim(P_s, s) * (1 - Z.T)[G_set]).view(-1)

        t_proto_dist = self.sim(P_t, t).view(-1)
        s_proto_dist = self.sim(P_s, s).view(-1)

        total_loss = ((s_sim_dist - t_sim_dist) ** 2).sum() + ((s_proto_dist - t_proto_dist) ** 2).sum()

        return total_loss
    
    def get_loss(self, batch_user, batch_pos_item, batch_neg_item):
        batch_neg_item = batch_neg_item.reshape(-1)
        ## Group Assignment
        GA_loss_user = self.get_GA_loss(batch_user.unique(), is_user=True)
        GA_loss_item = self.get_GA_loss(torch.cat([batch_pos_item, batch_neg_item], 0).unique(), is_user=False)
        GA_loss = GA_loss_user + GA_loss_item

        ## Topology Distillation
        TD_loss  = self.get_TD_loss(batch_user.unique(), torch.cat([batch_pos_item, batch_neg_item], 0).unique())
        HTD_loss = TD_loss * self.alpha + GA_loss * (1 - self.alpha)
        return HTD_loss


class UnKD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)

        self.group_num = args.unkd_group_num
        self.lamda = args.unkd_lamda
        self.n_distill = args.unkd_n_distill

        self.item_group, self.group_ratio, self.user_group_ratio, self.user_group_items = self.splitGroup()

        self.get_rank_negItems()
    
    def splitGroup(self):
        print('***begin group***')
        item_count = {}
        train_pairs = self.dataset.train_pairs
        train_users = train_pairs[:, 0]
        train_items = train_pairs[:, 1]
        listLen = len(train_pairs)
        count_sum = 0
        for i in range(listLen):
                k = train_items[i]
                if k not in item_count:
                    item_count[k] = 0
                item_count[k] += 1
                count_sum += 1
        count = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        group_aver = count_sum / self.group_num
        item_group = []
        temp_group = []
        temp_count = 0
        for l in range(len(count)):
            i, j = count[l][0], count[l][1]
            temp_group.append(i)
            temp_count += j
            if temp_count > group_aver:
                if len(temp_group) == 1:
                    item_group.append(temp_group)
                    temp_group = []
                    temp_count = 0
                    continue
                temp_group.remove(i)
                item_group.append(temp_group)
                temp_group = []
                temp_group.append(i)
                temp_count = j
        if len(temp_group) > 0:
            if temp_count > group_aver / 2:
                item_group.append(temp_group)
            else:
                if len(item_group) > 0:
                    item_group[-1].extend(temp_group)
                else:
                    item_group.append(temp_group)

        print('group_len')
        for i in range(len(item_group)):
            print(len(item_group[i]))
        cate_ratio = []
        temp = 0
        print('popualrity sum')
        for i in range(len(item_group)):
            tot = 0
            tot_n = 0
            for j in range(len(item_group[i])):
                tot += item_count[item_group[i][j]]
                tot_n += 1
            print(tot)
            cate_ratio.append(tot / tot_n)
        print(cate_ratio)
        maxP = max(cate_ratio)
        minP = min(cate_ratio)
        for i in range(len(cate_ratio)):
            cate_ratio[i] = (maxP + minP) - cate_ratio[i]
            temp += cate_ratio[i]
        for i in range(len(cate_ratio)):
            cate_ratio[i] = round((cate_ratio[i] / temp), 2)
        # cate_ratio.reverse()
        for i in range(len(cate_ratio)):
            if cate_ratio[i] < 0.1:
                cate_ratio[i] = 0.1
        print(cate_ratio)

        user_group_ratio=[[] for j in range(self.num_users)]
        user_group_items = [[] for j in range(self.num_users)]
        for i in range(self.num_users):
            user_group_items[i] = [[] for j in range(self.group_num)]
            user_group_ratio[i] = [0 for j in range(self.group_num)]
        for i in range(len(train_items)):
            for k in range(len(item_group)):
                if train_items[i] in item_group[k]:
                    user_group_ratio[train_users[i]][k] += 1
                    user_group_items[train_users[i]][k].append(train_items[i])
        print('***end group***')
        return item_group, cate_ratio, user_group_ratio, user_group_items

    def get_rank_negItems(self):
        all_ratio = 0.0
        for i in range(len(self.group_ratio)):
            self.group_ratio[i] = math.pow(self.group_ratio[i], self.lamda)
            all_ratio += self.group_ratio[i]
        for i in range(len(self.group_ratio)):
            self.group_ratio[i] = self.group_ratio[i] / all_ratio
        print(self.group_ratio)
        all_n = 0
        for i in self.group_ratio:
            all_n += round(i * self.n_distill)
        print(all_n)
        if all_n < self.n_distill:
            all_n = self.n_distill
        ranking_list = np.asarray([(1 + i) / 20 for i in range(1000)])
        ranking_list = torch.FloatTensor(ranking_list)
        ranking_list = torch.exp(-ranking_list)
        self.ranking_list = ranking_list
        self.ranking_list.requires_grad = False
        self.user_negitems = [list() for u in range(self.num_users)]

        self.pos_items = torch.zeros((self.num_users, all_n))
        self.neg_items = torch.zeros((self.num_users, all_n))
        self.item_tag = torch.zeros((self.num_users, all_n))
        for i in range(len(self.item_group)):
            cate_items = set(self.item_group[i])
            ratio = self.group_ratio[i]
            dis_n = math.ceil(self.n_distill * ratio)
            for user in range(self.num_users):
                crsNeg = set(list(self.dataset.train_dict[user]))
                neglist = list(cate_items - crsNeg)
                negItems = torch.LongTensor(neglist).cuda()
                rating = self.teacher.forward_multi_items(torch.tensor([user]).cuda(), negItems.reshape(1, -1), self.teacher).reshape((-1))
                n_rat = rating.sort(dim=-1, descending=True)[1]
                negItems = negItems[n_rat]
                self.user_negitems[user].append(negItems[:1000])

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            # interesting items
            pos_items = [list() for u in range(self.num_users)]
            neg_items = [list() for u in range(self.num_users)]
            item_tag = [list() for u in range(self.num_users)]
            all_n = 0
            for i in self.group_ratio:
                all_n += round(i * self.n_distill)

            for i in range(len(self.item_group)):
                ratio = self.group_ratio[i]
                dis_n = round(ratio * self.n_distill)
                if all_n < self.n_distill:
                    tm = self.n_distill - all_n
                    if i < tm:
                        dis_n += 1
                for user in range(self.num_users):
                    temp_ranklist = self.ranking_list.clone()
                    user_weight = self.user_group_ratio[user][i]
                    negItems = self.user_negitems[user][i]

                    while True:
                        k = 0
                        samples1 = torch.multinomial(temp_ranklist[:len(negItems)], dis_n, replacement=True)
                        samples2 = torch.multinomial(temp_ranklist[:len(negItems)], dis_n, replacement=True)
                        for l in range(len(samples1)):
                            if samples1[l] < samples2[l]:
                                k += 1
                            elif samples1[l] > samples2[l]:
                                k += 1
                                samples1[l], samples2[l] = samples2[l], samples1[l]
                        if k >= dis_n:
                            pos_items[user].extend(negItems[samples1])
                            neg_items[user].extend(negItems[samples2])
                            item_tag[user].extend([user_weight] * len(samples1))
                            break
            for user in range(self.num_users):
                self.pos_items[user] = torch.Tensor(pos_items[user])
                self.neg_items[user] = torch.Tensor(neg_items[user])
                self.item_tag[user] = torch.Tensor(item_tag[user])
        self.pos_items = self.pos_items.long().cuda()
        self.neg_items = self.neg_items.long().cuda()
        self.item_tag = self.item_tag.cuda()

    def get_loss(self, batch_users, batch_pos_item, batch_neg_item):
        pos_samples = self.pos_items[batch_users]
        neg_samples = self.neg_items[batch_users]
        weight_samples = self.item_tag[batch_users]
        pos_scores_S = self.student.forward_multi_items(batch_users, pos_samples)
        neg_scores_S = self.student.forward_multi_items(batch_users, neg_samples)
        mf_loss = torch.log(torch.sigmoid(pos_scores_S - neg_scores_S) + 1e-10)
        mf_loss = torch.mean(torch.neg(mf_loss), dim=-1)
        mf_loss = torch.sum(mf_loss)
        return mf_loss


class HetComp(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.num_ckpt = args.hetcomp_num_ckpt
        self.K = args.hetcomp_K
        self.p = args.hetcomp_p
        self.alpha = args.hetcomp_alpha

        self.perms, self.top_items, self.pos_items = self.construct_teacher_trajectory()
        self.user_mask = self.student.dataset.train_mat.cuda()
        self.v_result = 0
        self.last_max_idx = np.zeros(self.num_users)
        self.last_dist = None
        self.is_first = True

    def _cmp(self, f_name):
        if "BEST_EPOCH" in f_name:
            return math.inf
        pt = re.compile(r".*EPOCH_([\d\.\-\+e]+)\.pt")
        return eval(pt.search(f_name).group(1))
    
    @torch.no_grad()
    def _get_permutations(self, model):
        training = model.training
        model.eval()
        test_loader = data.DataLoader(model.user_list, batch_size=self.args.batch_size)
        topK_items = torch.zeros((model.num_users, self.K), dtype=torch.long)
        for batch_user in test_loader:
            score_mat = model.get_ratings(batch_user)
            for idx, user in enumerate(batch_user):
                pos = model.dataset.train_dict[user.item()]
                score_mat[idx, pos] = -1e10
            _, sorted_mat = torch.topk(score_mat, k=self.K, dim=1)
            topK_items[batch_user, :] = sorted_mat.detach().cpu()
        model.train(training)
        return topK_items

    def construct_teacher_trajectory(self):
        T_dir = os.path.join("checkpoints", self.args.dataset, self.args.backbone, f"scratch-{self.teacher.embedding_dim}")
        assert os.path.exists(T_dir), f"Teacher path {T_dir} doesn't exists."
        old_state = deepcopy(self.teacher.state_dict())
        f_ckpts = []
        for f in os.scandir(T_dir):
            f_ckpts.append(f.path)
        assert len(f_ckpts) >= self.num_ckpt, "Number of checkpoints must be no less than self.num_ckpt."
        f_ckpts = sorted(f_ckpts, key=lambda x: self._cmp(x))
        perms = []
        for f_ckpt in f_ckpts:
            self.teacher.load_state_dict(torch.load(f_ckpt))
            perm = self._get_permutations(self.teacher)
            perms.append(perm.cuda())
        self.teacher.load_state_dict(old_state)

        pos_dict = self.teacher.dataset.train_dict
        pos_K = min([len(p_items) for p_items in pos_dict.values()])
        pos_items = torch.zeros((self.num_users, pos_K), dtype=torch.long)
        for u, p_items in pos_dict.items():
            pos_items[u, :pos_K] = p_items[:pos_K]
        top_items = perms[0]
        return perms, top_items, pos_items.cuda()
    
    @torch.no_grad()
    def _get_NDCG_u(self, sorted_list, teacher_t_items, k):
        top_scores = np.asarray([np.exp(-t / 10) for t in range(k)])
        top_scores = (2 ** top_scores) - 1
        
        t_items = teacher_t_items[:k].cpu()

        denom = np.log2(np.arange(2, k + 2))
        dcg = np.sum((np.in1d(sorted_list[:k], list(t_items)) * top_scores) / denom)
        idcg = np.sum((top_scores / denom)[:k])
        return round(dcg / idcg, 4)

    def _DKC(self, sorted_mat, last_max_idx, last_dist, is_first, epoch):
        next_idx = last_max_idx[:] 
        if is_first:
            last_dist = np.ones_like(next_idx)
            for user in self.student.user_list:
                next_v = min(self.num_ckpt - 1, int(next_idx[user]) + 1)

                next_perm = self.perms[next_v][user]
                next_dist = 1. - self._get_NDCG_u(sorted_mat[user], next_perm, self.K)
                
                last_dist[user] = next_dist

            return next_idx, last_dist

        th = self.alpha * (0.995 ** (epoch // self.p))

        for user in self.student.user_list:
            if int(last_max_idx[user]) == self.num_ckpt - 1:
                continue

            next_v = min(self.num_ckpt - 1, int(next_idx[user]) + 1)
            next_next_v = min(self.num_ckpt - 1, int(next_idx[user]) + 2)
            
            next_perm = self.perms[next_v][user]
            next_next_perm = self.perms[next_next_v][user]
            
            next_dist = 1. - self._get_NDCG_u(sorted_mat[user], next_perm, self.K)
            
            if (last_dist[user] / next_dist > th) or (last_dist[user] / next_dist < 1):
                next_idx[user] += 1
                next_next_dist = 1. - self._get_NDCG_u(sorted_mat[user], next_next_perm, self.K)
                last_dist[user] = next_next_dist
        return next_idx, last_dist
    
    def do_something_in_each_epoch(self, epoch):
        ### DKC
        if epoch % self.p == 0 and epoch >= self.p and self.v_result < self.num_ckpt - 1:
            sorted_mat = self._get_permutations(self.student)
            if self.is_first == True:
                self.last_max_idx, self.last_dist = self._DKC(sorted_mat, self.last_max_idx, self.last_dist, True, epoch)
                self.is_first = False
            else:
                self.last_max_idx, self.last_dist = self._DKC(sorted_mat, self.last_max_idx, self.last_dist, False, epoch)
            for user in self.student.user_list:
                self.top_items[user] = self.perms[int(self.last_max_idx[user])][user]
            self.v_result = round(self.last_max_idx.mean(), 2)
        
    def overall_loss(self, batch_full_mat, pos_items, top_items, batch_user_mask):
        tops = torch.gather(batch_full_mat, 1, torch.cat([pos_items, top_items], -1))
        tops_els = (batch_full_mat.exp() * (1 - batch_user_mask)).sum(1, keepdims=True)
        els = tops_els - torch.gather(batch_full_mat, 1, top_items).exp().sum(1, keepdims=True)
        
        above = tops.view(-1, 1)
        below = torch.cat((pos_items.size(1) + top_items.size(1)) * [els], 1).view(-1, 1) + above.exp()
        below = torch.clamp(below, 1e-5).log()

        return -(above - below).sum()

    def rank_loss(self, batch_full_mat, pos_items, top_items, batch_user_mask):
        S_pos = torch.gather(batch_full_mat, 1, pos_items)
        S_top = torch.gather(batch_full_mat, 1, top_items[:, :top_items.size(1) // 2])

        below2 = (batch_full_mat.exp() * (1 - batch_user_mask)).sum(1, keepdims=True) - S_top.exp().sum(1, keepdims=True)
        
        above_pos = S_pos.sum(1, keepdims=True)
        above_top = S_top.sum(1, keepdims=True)
        
        below_pos = S_pos.flip(-1).exp().cumsum(1)
        below_top = S_top.flip(-1).exp().cumsum(1)
        
        below_pos = (torch.clamp(below_pos + below2, 1e-5)).log().sum(1, keepdims=True)        
        below_top = (torch.clamp(below_top + below2, 1e-5)).log().sum(1, keepdims=True)  

        pos_KD_loss = -(above_pos - below_pos).sum()

        S_top_sub = torch.gather(batch_full_mat, 1, top_items[:, :top_items.size(1) // 10])
        below2_sub = (batch_full_mat.exp() * (1-batch_user_mask)).sum(1, keepdims=True) - S_top_sub.exp().sum(1, keepdims=True)
        
        above_top_sub = S_top_sub.sum(1, keepdims=True)
        below_top_sub = S_top_sub.flip(-1).exp().cumsum(1)
        below_top_sub = (torch.clamp(below_top_sub + below2_sub, 1e-5)).log().sum(1, keepdims=True)  

        top_KD_loss = - (above_top - below_top).sum() - (above_top_sub - below_top_sub).sum()

        return  pos_KD_loss + top_KD_loss / 2

    def get_loss(self, batch_users):
        batch_full_mat = torch.clamp(self.student.get_ratings(batch_users), min=-40, max=40)
        batch_user_mask = torch.index_select(self.user_mask, 0, batch_users).to_dense()
        t_items = torch.index_select(self.top_items, 0, batch_users)
        p_items = torch.index_select(self.pos_items, 0, batch_users)
        if self.v_result < self.num_ckpt - 1:
            KD_loss = self.overall_loss(batch_full_mat, p_items, t_items, batch_user_mask) * 2.
        else:
            KD_loss = self.rank_loss(batch_full_mat, p_items, t_items, batch_user_mask)
        return KD_loss

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        kd_loss = self.get_loss(batch_user)
        loss = self.lmbda * kd_loss
        return loss


class WarmUp(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "warmup"
        self.freeze = args.warmup_freeze
        self.student.embedding_layer = deepcopy(self.teacher.embedding_layer)
        for param in self.student.embedding_layer.parameters():
                param.requires_grad = (not self.freeze)
    
    def get_loss(self, data, label):
        return torch.tensor(0.).cuda()


class FitNet(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "fitnet"
        self.layer = args.fitnet_layer
        self.verbose = args.verbose
        if self.layer == "embedding":
            self.projector = nn.Linear(self.student.embedding_layer_dim, self.teacher.embedding_layer_dim)
        elif self.layer == "penultimate":
            if isinstance(self.teacher._penultimate_dim, int):
                # For one-stream models
                teacher_penultimate_dim = self.teacher._penultimate_dim
                student_penultimate_dim = self.student._penultimate_dim
                self.projector = Projector(student_penultimate_dim, teacher_penultimate_dim, 1, norm=False, dropout_rate=0., shallow=False)
            else:
                # For two-stream models
                cross_dim, deep_dim = self.teacher._penultimate_dim
                student_penultimate_dim = self.student._penultimate_dim
                self.projector_cross = nn.Linear(student_penultimate_dim, cross_dim)
                self.projector_deep = nn.Linear(student_penultimate_dim, deep_dim)
        elif self.layer == "none":
            pass
        else:
            raise ValueError
        
    def do_something_in_each_epoch(self, epoch):
        self.epoch = epoch
        if self.verbose:
            if epoch > 0 and not self.cka is None:
                print(self.cka)
            self.cka = None
            self.cnt = 0
    
    def get_loss(self, data, label):
        if self.layer == "embedding":
            S_emb = self.student.forward_embed(data)
            S_emb = S_emb.reshape(S_emb.shape[0], -1)
            T_emb = self.teacher.forward_embed(data)
            T_emb = T_emb.reshape(T_emb.shape[0], -1)
            S_emb_proj = self.projector(S_emb)
            loss = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
        elif self.layer == "penultimate":
            S_emb = self.student.forward_penultimate(data)
            T_emb = self.teacher.forward_penultimate(data)
            if isinstance(self.teacher._penultimate_dim, int):
                S_emb_proj = self.projector(S_emb)
                loss = (T_emb.detach() - S_emb_proj).pow(2).sum(-1).mean()
            else:
                S_emb_cross = self.projector_cross(S_emb)
                S_emb_deep = self.projector_deep(S_emb)
                T_emb_cross, T_emb_deep = T_emb
                loss = (T_emb_cross.detach() - S_emb_cross).pow(2).sum(-1).mean() * 0.5 + (T_emb_deep.detach() - S_emb_deep).pow(2).sum(-1).mean() * 0.5
        elif self.layer == "none":
            loss = torch.tensor(0.).cuda()
        else: raise ValueError

        if self.verbose and self.cnt < 5:
            # # calculate CKA
            # with torch.no_grad():
            #     S_embs = self.student.forward_all_feature(data)
            #     T_embs = self.teacher.forward_all_feature(data)
            #     CKA_mat = np.zeros((len(T_embs), len(S_embs)))
            #     for id_T, T_emb in enumerate(T_embs):
            #         for id_S, S_emb in enumerate(S_embs):
            #             CKA_mat[id_T, id_S] = CKA(T_emb, S_emb).item()
            #     if self.cka is None:
            #         self.cka = CKA_mat
            #     else:
            #         self.cka = (self.cka * self.cnt + CKA_mat) / (self.cnt + 1)
            #         self.cnt += 1

            # calculate information abundance
            with torch.no_grad():
                info_S = info_abundance(S_emb)
                info_T = info_abundance(T_emb)
                info_S_proj = info_abundance(S_emb_proj)
                print(f"infoS:{info_S:.2f}, infoT:{info_T:.2f}, infoS_proj:{info_S_proj:.2f}")
                self.cnt += 1
        return loss


class RKD(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "rkd"
        self.K = args.rkd_K

    def get_loss(self, data, label):
        S_emb = self.student.forward_penultimate(data)[:self.K]
        T_emb = self.teacher.forward_penultimate(data)[:self.K]
        S_mat = S_emb.mm(S_emb.T)
        T_mat = T_emb.mm(T_emb.T)
        return (S_mat - T_mat).pow(2).mean()


class BCED(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "bced"

    def get_loss(self, feature, label):
        logit_S = self.student(feature)
        logit_T = self.teacher(feature)
        y_T = torch.sigmoid(logit_T)
        loss = F.binary_cross_entropy_with_logits(logit_S, y_T.float())
        return loss


class CLID(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "clid"

    def get_loss(self, feature, label):
        logit_S = self.student(feature).squeeze(1)
        logit_T = self.teacher(feature).squeeze(1)
        y_T = torch.sigmoid(logit_T)
        y_S = torch.sigmoid(logit_S)
        y_T = y_T / y_T.sum()
        y_S = y_S / y_S.sum()
        loss = F.binary_cross_entropy(y_S, y_T)
        return loss


class OFA(BaseKD4CTR):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "ofa"
        self.beta = args.ofa_beta
        self.layer_dims = self.student._all_layer_dims
        self.projectors = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        ) for dim in self.layer_dims])

    def get_loss(self, feature, label):
        logit_S = self.student(feature)
        logit_T = self.teacher(feature)
        y_T = torch.sigmoid(logit_T)
        loss_kd = F.binary_cross_entropy_with_logits(logit_S, y_T)
        loss_ofa = 0.
        features = self.student.forward_all_feature(feature)
        for idx, h in enumerate(features):
            logit_h = self.projectors[idx](h)
            loss_ofa += F.binary_cross_entropy_with_logits(logit_h, y_T)
        loss = loss_kd + loss_ofa * self.beta
        return loss
