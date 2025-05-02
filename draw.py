import os
import re
import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt


# x = '''| 0.03506 | 0.03891 | 0.04076 | 0.04092 | 0.03970 | 0.03951 | 0.03979 | 0.04033 |
# | 0.01604 | 0.01703 | 0.01732 | 0.01735 | 0.01689 | 0.01677 | 0.01674 | 0.01703 |
# | 0.02054 | 0.03166 | 0.03421 | 0.03225 | 0.03232 | 0.03485 | 0.03287 | 0.03514 |
# | 0.01037 | 0.01433 | 0.01534 | 0.01492 | 0.01492 | 0.01517 | 0.01483 | 0.01510 |
# '''
# x = x.replace("|", "").replace("  ", " ").replace(" ",", ")
# print(x)

# y = [[0.02941, 0.03529, 0.03703, 0.03875, 0.03957, 0.03982, 0.03841], 
# [0.01239, 0.01653, 0.01765, 0.01776, 0.01769, 0.01762, 0.01744], 
# [0.02747, 0.03218, 0.03563, 0.03654, 0.02220, 0.02356, 0.01654], 
# [0.01146, 0.01459, 0.01558, 0.01587, 0.00988, 0.00962, 0.00755]]
labels = ["rrd_R@20", "rrd_N@20", "dcd_R@20", "dcd_N@20"]

# # 样式配置
# colors = ['blue', 'blue', 'red', 'red']
# linestyles = ['-', '--', '-', '--']

# # 创建图形
# plt.figure(figsize=(8, 5))

# # 绘制每条曲线
# for i in range(4):
#     plt.plot(x, y[i], color=colors[i], linestyle=linestyles[i], label=labels[i])

# # 添加标签、图例
# plt.xlabel("neg_x")
# plt.ylabel("Performance")
# plt.title("LightGCN")
# plt.legend()
# plt.grid(False)
# plt.tight_layout()
# plt.show()
# plt.savefig("1.svg", format='svg')

y = [[0.03506, 0.03891, 0.04076, 0.04092, 0.03970, 0.03951, 0.03979, 0.04033], 
[0.01604, 0.01703, 0.01732, 0.01735, 0.01689, 0.01677, 0.01674, 0.01703], 
[0.02054, 0.03166, 0.03421, 0.03225, 0.03232, 0.03485, 0.03287, 0.03514], 
[0.01037, 0.01433, 0.01534, 0.01492, 0.01492, 0.01517, 0.01483, 0.01510]]
x = [0, 0.1, 0.5, 1, 2, 3, 5, 10]
# 样式配置
colors = ['blue', 'blue', 'red', 'red']
linestyles = ['-', '--', '-', '--']

# 创建图形
plt.figure(figsize=(8, 5))

# 绘制每条曲线
for i in range(4):
    plt.plot(x, y[i], color=colors[i], linestyle=linestyles[i], label=labels[i])

# 添加标签、图例
plt.xlabel("neg_x")
plt.ylabel("Performance")
plt.title("BPR")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()
plt.savefig("2.svg", format='svg')