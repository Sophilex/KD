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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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
        self.is_ans = is_ans
    
    def prelog(self):
        self.log("", pre = False)
        self.log("", pre = False)
        self.log(" ")
        self.log('-' * 40 + "Start Training" + '-' * 40, pre=False)
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
        if not self.is_ans:
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
    def __init__(self, args, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.batch_size = 256
        self.type = args.draw_type

        #  多轮画图
        self.x_axis_lst = []
        self.label_lst = []
        self.y_axis_lst = []

    
    def gt_CCDF4negs(self, model, train_loader, valid_dataset, test_dataset, mxK):
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
            adjusted_mxK = mxK
            topK_neg_items = torch.zeros((num_users, num_items - adjusted_mxK), dtype=torch.float).cuda()

            for batch_user in test_loader:
                inter_mat = model.get_ratings(batch_user)
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
    
    def plot_CCDF4negs(self, model, train_loader, valid_dataset, test_dataset, label, mxK):
        prob, ccdf = self.gt_CCDF4negs(model, train_loader, valid_dataset, test_dataset, mxK)
        # plt.plot(prob, ccdf, label=label)
        # plt.savefig(os.path.join(self.path, label))
        self.add(prob, ccdf, label)

    def gt_CDF4negs(self, rating_variance):
        """
        Calculate the average rating variance across users and its CDF.

        Args:
            rating_variance: A 2D matrix of shape (num_users, num_items),
                            where rating_variance[i, j] is the predicted variance
                            of user i for item j.
        Returns:
            avg_rating_variance: The average rating variance across users.
            cdf: The CDF of the average rating variance.
        """
        num_users, num_items = rating_variance.shape

        rating_variance = torch.softmax(rating_variance, dim = -1).mean(dim = 0) # 1 X num_items
        rating_variance = rating_variance.cpu().numpy() 
        sorted_variance = np.sort(rating_variance)

        cdf = np.arange(1, num_items + 1) / (num_items + 1)
        return sorted_variance, cdf

    
    def plot_CDF4negs(self, rating_variance, label):
        prob, cdf = self.gt_CDF4negs(rating_variance)
        self.add(prob, cdf, label)

    
    def add(self, x_axis, y_axis, label):
        self.x_axis_lst.append(x_axis)
        self.y_axis_lst.append(y_axis)
        self.label_lst.append(label)
    

    def plot_all_subfig(self, filename, x_name, y_name, savetype):
        # 找到所有曲线的最小和最大 x 值
        all_x = [x for lst in self.x_axis_lst for x in lst]
        x_min, x_max = min(all_x), max(all_x)

        fig, ax = plt.subplots(figsize=(8, 6))  # 创建主图

        # 主图绘制
        for i in range(len(self.x_axis_lst)):
            ax.plot(self.x_axis_lst[i], self.y_axis_lst[i], label=self.label_lst[i])

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.legend(loc="upper right")  # 避免挡住子图
        ax.grid(True)
        ax.margins(x=0, y=0)

        # 确定放大区域 (左下角拐角处)
        zoom_x_min, zoom_x_max = 0.00003, 0.0002
        zoom_y_min, zoom_y_max = 0, 0.2

        # 创建子图 (放在中间，靠右位置)
        ax_inset = inset_axes(ax, width="35%", height="30%", loc="center right")

        # 子图绘制
        for i in range(len(self.x_axis_lst)):
            ax_inset.plot(self.x_axis_lst[i], self.y_axis_lst[i])

        ax_inset.set_xlim(zoom_x_min, zoom_x_max)
        ax_inset.set_ylim(zoom_y_min, zoom_y_max)  # 自动匹配 y 轴
        ax_inset.tick_params(axis='both', which='both', length=0)  # 隐藏刻度
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        # 添加放大框 & 虚线连接
        mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="black", linestyle="--", lw=1)

        # 保存图片
        final_path = os.path.join(self.path, filename)
        plt.savefig(final_path, format=savetype, bbox_inches='tight')
        plt.close()

    
    def plot_all(self, filename, x_name, y_name, savetype):
        print(len(self.x_axis_lst))
        # 找到所有曲线的最小和最大 x 值
        all_x = [x for lst in self.x_axis_lst for x in lst]
        x_min, x_max = min(all_x), max(all_x)
        
        for i in range(len(self.x_axis_lst)):
            # 将所有的 x 轴限制在统一范围内
            plt.plot(self.x_axis_lst[i], self.y_axis_lst[i], label=self.label_lst[i])

        plt.xlim(x_min, x_max)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.legend()
        plt.grid(True)
        plt.margins(x=0, y=0)
        final_path = os.path.join(self.path, filename)
        plt.savefig(final_path, format= savetype)
    
    def plot_sample(self, x, num_users, num_items, filename, x_name, y_name):
        nonzero_samples = [x[i][x[i] > 0] for i in range(num_users)]  # 取出每个用户非零项
        max_count = x.max().item()  # 找到最大采样次数，决定分布范围
        all_distributions = torch.zeros((num_users, max_count + 1)).cpu()  # 记录每个用户的采样分布

        for i in range(num_users):
            if len(nonzero_samples[i]) > 0:
                values, counts = torch.unique(nonzero_samples[i], return_counts=True)  # 统计每个采样次数的频率
                values, counts = values.cpu(), counts.cpu()
                all_distributions[i, values] = counts.float() / counts.sum()  # 归一化为概率分布
        
        avg_distribution = all_distributions.mean(dim=0)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_count + 1), avg_distribution.numpy()[1:], marker='o', linestyle='-')
        plt.xlabel("sample_num")
        plt.ylabel("probability")
        plt.grid(True)
        plt.show()
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
        self.cur_epoch = 0
        self.rescale = 10


    def update_ratings(self, model):
        with torch.no_grad():
            for start_idx in range(0, self.num_users, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.num_users)
                batch_user = torch.arange(start_idx, end_idx)
                inter_mat = model.get_user_item_ratings(batch_user, self.item_idx[batch_user]).cuda() # 每个user对应的item的分数
                inter_mat *= self.rescale
                self.rating_history[batch_user, :] += inter_mat
                self.rating_square[batch_user, :] += inter_mat ** 2

                del inter_mat
    
    def reset(self, item_idx):
        self.item_idx = item_idx.cuda()
        self.rating_square.zero_() # 原地置0
        self.rating_history.zero_()
        self.cur_epoch = 0 # cur_epoch是相对的，及时更新


    def update_rating_variance(self, model):
        self.update_ratings(model)
        self.cur_epoch += 1
        # print("epoch: {}".format(self.cur_epoch))
        # 这里可以时间换内存 需要时修改
        self.rating_history /= self.cur_epoch
        self.rating_variance = self.rating_square / self.cur_epoch
        self.rating_variance -= self.rating_history ** 2
        self.rating_variance = torch.clamp(self.rating_variance, min=0.0)  # 负值截断
        self.rating_history *= self.cur_epoch # 还原
        # self.check()
    
    def check(self):
        min_variance_index_flat = np.argmin(self.rating_variance.cpu().numpy())
        min_variance_index = np.unravel_index(min_variance_index_flat, self.rating_variance.shape)
        min_variance_value = self.rating_variance[min_variance_index]  # 获取对应的值
        print(f"Min variance index: {min_variance_index}, Value: {min_variance_value.item()}, rating_history: {self.rating_history[min_variance_index]}, rating_square: {self.rating_square[min_variance_index]}")

    def get_rating_variance(self):
        return self.rating_variance.detach(), self.item_idx

    # def get_rating_variance(self):
    #     epoch_plus_1 = self.current_epoch + 1
    #     var = torch.zeros_like(self.rating_history, dtype=torch.float16)  # 半精度存储结果
    #     # 分批次计算方差，避免峰值显存
    #     for start in range(0, self.num_users, self.batch_size):
    #         end = min(start + self.batch_size, self.num_users)
    #         hist = self.rating_history[start:end].float()  # 转float32计算避免精度丢失
    #         sq = self.rating_square[start:end].float()
    #         mean = hist / epoch_plus_1
    #         batch_var = (sq / epoch_plus_1) - (mean ** 2)
    #         var[start:end] = batch_var.half()  # 转回半精度存储
    #     return var, self.item_idx

