import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
import glob
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.models.zoo.amppo import AMPPO
from tensordict import TensorDict
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance

from examples.PO.train_preference import PreferenceDataset, preference_collate_fn
from examples.PO.collect_feedback import serialize_tensordict

# 设置matplotlib中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 配置
class Config:
    learning_rate = 1e-5 # DPO 微调通常使用更小的学习率
    max_epochs = 20
    batch_size = 8 # 根据显存调整
    beta = 0.5 # DPO的超参数
    num_loc = 50

if __name__ == "__main__":
    config = Config()

    print("\n=== 重新加载action和reward ===")
    results_dir = "dpo_results"

    with open(os.path.join(results_dir, "dpo_comparison_result.json"), "r") as f:
        data = json.load(f)

    with open(os.path.join(results_dir, "dpo_comparison_td_init.json"), "r") as f:
        td_init = TensorDict(json.load(f))
        td_init = td_init.to(device)
        td_init.batch_size = [10]

    env = CVRPEnv(generator_params=dict(num_loc=config.num_loc))
    for idx, item in enumerate(data):
        original_out = item["original_model_out"]
        dpo_out = item["dpo_model_output"]
        original_reward = item["original_model_reward"]
        dpo_reward = item["dpo_model_reward"]

        # dpo_out 应该是一个 action 序列（list），直接转 tensor
        dpo_actions = torch.tensor(dpo_out, dtype=torch.long, device=device)

        # 从批量数据中获取与当前结果对应的地图（locations）
        instance_locs = td_init[idx]["locs"]

        # 1. 根据action序列，获取客户点的坐标
        locs_ordered = gather_by_index(instance_locs, dpo_actions, dim=0)

        # 2. 将depot（仓库，即第0个点）拼接到路径开头
        depot = instance_locs[0:1]
        full_tour_locs = torch.cat([depot, locs_ordered], dim=0)

        # 3. 使用 get_tour_length 计算总路径长度
        # 需要为函数增加一个临时的batch维度 (unsqueeze)，计算后再移除 (squeeze)
        tour_length = get_tour_length(full_tour_locs.unsqueeze(0))
        new_dpo_reward = -tour_length.squeeze()

        print(f"original_reward: {original_reward}, dpo_reward: {dpo_reward}, new_dpo_reward: {new_dpo_reward.item():.4f}")


        