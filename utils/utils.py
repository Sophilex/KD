import os
import sys
from datetime import date, datetime
import random
import yaml
import pyro
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_np(x):
    return x.detach().data.cpu().numpy()


def load_yaml(path):
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


def seed_all(seed:int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def avg_dict(eval_dicts, final_dict=None):
    if final_dict is None:
        final_dict = {}
    flg_dict = eval_dicts[0]
    for k in flg_dict:
        if isinstance(flg_dict[k], dict):
            final_dict[k] = avg_dict([eval_dict[k] for eval_dict in eval_dicts])
        else:
            final_dict[k] = 0
            for eval_dict in eval_dicts:
                final_dict[k] += eval_dict[k]
            final_dict[k] /= len(eval_dicts)
    return final_dict


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class Logger:
    def __init__(self, args, no_log, is_ans = False):
        if is_ans == False:
            self.log_path = os.path.join(args.LOG_DIR, args.dataset, args.backbone, args.model + ('_' if args.suffix != '' else '') + args.suffix + '.log')
        else:
            self.log_path = os.path.join(args.SHORT_LOG_DIR, args.dataset, args.backbone, args.model + ('_' if args.suffix != '' else '') + args.suffix + '.log')
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.no_log = no_log

    def log(self, content='', pre=True, end='\n'):
        string = str(content)
        if len(string) == 0:
            pre = False
        if pre:
            today = date.today()
            today_date = today.strftime("%m/%d/%Y")
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            string = today_date + "," + current_time + ": " + string
        string = string + end

        if not self.no_log:
            with open(self.log_path, 'a') as logf:
                logf.write(string)

        sys.stdout.write(string)
        sys.stdout.flush()
    
    def log_args(self, args, text="ARGUMENTS"):
        ans = ""
        self.log('-' * 40 + text + '-' * 40, pre=False)
        for arg in vars(args):
            ans += '{:40} {}'.format(arg, getattr(args, arg)) + '\n'
            self.log('{:40} {}'.format(arg, getattr(args, arg)), pre=False)
        self.log('-' * 40 + text + '-' * 40, pre=False)


class Drawer:
    def __init__(self, args, mxK, path):
        self.mxK = mxK
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    
    def gt_CCDF4negs(self, model, train_loader, valid_dataset, test_dataset):
        """
        for particular user, calculate the CCDF for negative items
        """
        train_dict = train_loader.dataset.train_dict
        valid_dict = valid_dataset.inter_dict
        test_dict = test_dataset.inter_dict
        num_users = train_loader.dataset.num_users
        num_items = train_loader.dataset.num_items

        with torch.no_grad():

            test_loader = data.DataLoader(list(test_dict.keys()), batch_size=train_loader.batch_size)
            adjusted_mxK = self.mxK
            topK_neg_items = torch.zeros((num_users, num_items - adjusted_mxK), dtype=torch.float).cuda()

            for batch_user in test_loader:
                inter_mat = model.teacher.get_ratings(batch_user) # 负样本由教师定义
                # np.savetxt("inter_mat.txt", np.sort(inter_mat[0].cpu().numpy()))
                # for idx, user in enumerate(batch_user):
                #     pos = train_dict[user.item()]
                #     inter_mat[idx, pos] = 1e8
                topk_value, topk_dict = torch.topk(-inter_mat, num_items - adjusted_mxK, dim=-1) 
                # np.savetxt("topk_value.txt", np.sort(-topk_value[0].cpu().numpy()))
                topK_neg_items[batch_user] = -topk_value
  
        self.topk_value = torch.softmax(topK_neg_items, dim=-1).mean(dim=0)
        self.topk_value = self.topk_value.cpu().numpy()
        # if np.all(np.diff(self.topk_value) >= 0):  # 检查是否已经按升序排序
        #     print("Array is already sorted in ascending order")
        prob = self.topk_value
        # np.savetxt("prob.txt", self.topk_value)
        ccdf = (len(prob) - np.arange(1, len(prob) + 1)) / len(prob)
        return prob, ccdf
    
    def plot_CCDF4negs(self, model, train_loader, valid_dataset, test_dataset, filename):
        prob, ccdf = self.gt_CCDF4negs(model, train_loader, valid_dataset, test_dataset)

        plt.plot(prob, ccdf)
        plt.margins(x=0, y=0)  # 同时消除x轴和y轴的空隙

        final_path = os.path.join(self.path, filename)
        plt.savefig(final_path)


class Var_calc:
    def __init__(self, args, data_loader):
        self.num_users = data_loader.dataset.num_users
        self.num_items = data_loader.dataset.num_items
        # print(f"num_users: {num_users}, num_items: {num_items}")
        self.rating_history = torch.zeros((self.num_users, self.num_items), dtype=torch.float64).cuda()

        self.rating_square = torch.zeros((self.num_users, self.num_items), dtype=torch.float64).cuda()
        self.rating_variance = torch.zeros((self.num_users, self.num_items), dtype=torch.float64).cuda()
        # self.count = torch.zeros((self.num_users, self.num_items), dtype = torch.int).cuda()
        self.args = args
        self.data_loader = data_loader
        self.batch_size = 256



    def update_ratings(self, model):
        with torch.no_grad():
            for start_idx in range(0, self.num_users, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.num_users)
                batch_user = torch.arange(start_idx, end_idx)
                inter_mat = model.get_ratings(batch_user).cuda()
                self.rating_history[batch_user, :] += inter_mat
                # self.count[batch_user, :] += 1
                self.rating_square[batch_user, :] += inter_mat ** 2
                # if 264 in batch_user:
                #     print(f"inter_mat: {inter_mat[264-start_idx, 11684]}")
                #     print(f"count: {self.rating_history[264, 11684], self.rating_square[264, 11684]}")
            # print(f"count: {self.count.min()}")
            # for idx, (batch_user, batch_pos_item, batch_neg_item) in enumerate(self.data_loader):
            #     batch_user = batch_user.cuda()

            #     # print(f"batch_user dtype: {batch_user.dtype}, min: {batch_user.min()}, max: {batch_user.max()}, any NaN: {torch.isnan(batch_user).any()}")
            #     batch_pos_item = batch_pos_item.cuda()
            #     batch_neg_item = batch_neg_item.cuda()

            #     inter_mat = model.get_ratings(batch_user).cuda()
            #     # inter_mat = torch.round(inter_mat * self.round_num) / self.round_num

            #     self.rating_history[batch_user, :] += inter_mat
            #     self.rating_square[batch_user, :] += inter_mat ** 2
            #     self.count[batch_user, :] += 1
            # # print(f"max rating: {self.rating_history.max()}, min rating: {self.rating_history.min()}")
            # # print(f"max rating square: {self.rating_square.max()}, min rating square: {self.rating_square.min()}")

    def update_rating_variance(self, model, epoch):
        self.update_ratings(model)

        # self.rating_history = torch.round(self.rating_history * self.round_num) / self.round_num
        # self.rating_square = torch.round(self.rating_square * self.round_num) / self.round_num

        # print(f"max rating: {self.rating_history.max()}, min rating: {self.rating_history.min()}")
        # print(f"max rating square: {self.rating_square.max()}, min rating square: {self.rating_square.min()}")
        # 后续改掉，增加了很多没必要的显存
        mean = self.rating_history / (epoch + 1)
        mean_square = self.rating_square / (epoch + 1)
        self.rating_variance = mean_square - mean ** 2 + 1e-5 # 减少精度影响
        # print(f"mean: {mean[264, 11684]}, mean_square: {mean_square[264, 11684]}, epoch: {epoch + 1}")
        

        # min_variance_index_flat = np.argmin(self.rating_variance.cpu().numpy())
        # min_variance_index = np.unravel_index(min_variance_index_flat, self.rating_variance.shape)
        # print(f"min variance index: {min_variance_index}")

        # min_variance_index_flat = np.argmin(self.rating_variance.cpu().numpy())
        # min_variance_index = np.unravel_index(min_variance_index_flat, self.rating_variance.shape)
        # min_variance_value = self.rating_variance[min_variance_index]  # 获取对应的值
        # print(f"Min variance index: {min_variance_index}, Value: {min_variance_value.item()}, rating_history: {mean[min_variance_index]}, rating_square: {mean_square[min_variance_index]}")



        # print(f"updating variance... - min: {self.rating_variance.min()}, max: {self.rating_variance.max()}")
        # self.rating_variance = self.rating_square / (epoch + 1) - ( self.rating_history / (epoch + 1) ) ** 2 
        # print(f"updated variance - min: {self.rating_variance.min()}, max: {self.rating_variance.max()}")

    def get_rating_variance(self):
        
        return self.rating_variance

class Var_calcer:
    def __init__(self, args, data_loader, mode):
        self.num_users = data_loader.dataset.num_users
        self.num_items = data_loader.dataset.num_items
        self.rating_history = torch.zeros((self.num_users, args.rrd_L + args.rrd_extra), dtype=torch.float32).cuda()
        self.item_idx = torch.zeros((self.num_users, args.rrd_L + args.rrd_extra), dtype=torch.int64).cuda()
        self.rating_square = torch.zeros((self.num_users, args.rrd_L + args.rrd_extra), dtype=torch.float32).cuda()
        self.rating_variance = torch.zeros((self.num_users, args.rrd_L + args.rrd_extra), dtype=torch.float32).cuda()
        self.args = args
        self.data_loader = data_loader
        self.batch_size = 256
        self.calu_len = args.calu_len
        self.mode = mode # 采样方式



    def update_ratings(self, model):
        with torch.no_grad():
            for start_idx in range(0, self.num_users, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.num_users)
                batch_user = torch.arange(start_idx, end_idx)
                inter_mat = model.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                self.rating_history[batch_user, :] += inter_mat
                self.rating_square[batch_user, :] += inter_mat ** 2

                del inter_mat
    
    def reset(self, item_idx):
        self.item_idx = item_idx.cuda()
        self.rating_square.zero_() # 原地置0
        self.rating_history.zero_()


    def update_rating_variance(self, model, epoch):
        self.update_ratings(model)
        # 这里可以时间换内存 需要时修改
        mean = self.rating_history / (epoch + 1)
        mean_square = self.rating_square / (epoch + 1)
        self.rating_variance = mean_square - mean ** 2
        # min_variance_index_flat = np.argmin(self.rating_variance.cpu().numpy())
        # min_variance_index = np.unravel_index(min_variance_index_flat, self.rating_variance.shape)
        # min_variance_value = self.rating_variance[min_variance_index]  # 获取对应的值
        # print(f"Min variance index: {min_variance_index}, Value: {min_variance_value.item()}, rating_history: {self.rating_history[min_variance_index]}, rating_square: {self.rating_square[min_variance_index]}")


    
    def check(self):
        min_variance_index_flat = np.argmin(self.rating_variance.cpu().numpy())
        min_variance_index = np.unravel_index(min_variance_index_flat, self.rating_variance.shape)
        min_variance_value = self.rating_variance[min_variance_index]  # 获取对应的值
        print(f"Min variance index: {min_variance_index}, Value: {min_variance_value.item()}, rating_history: {self.rating_history[min_variance_index]}, rating_square: {self.rating_square[min_variance_index]}")

    def get_rating_variance(self):
        return self.rating_variance, self.item_idx

