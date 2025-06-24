"""
比较原始模型和DPO微调后的模型
"""

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

# 设置matplotlib中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def serialize_tensordict(td: TensorDict) -> dict:
    """一个辅助函数，用于将TensorDict转换成可序列化的字典。"""
    serializable_td = {}
    # 遍历TensorDict中的每一个键和张量
    for key, tensor in td.items():
        # 将张量转换为列表
        serializable_td[key] = tensor.tolist()
    return serializable_td

def preference_collate_fn(batch):
    """
    处理可变长度序列的自定义collate函数
    """
    if not batch:
        return None, None, None, None

    # 1. 分离数据
    states, actions1, actions2, preferences = zip(*[
        (item['state'], item['action1'], item['action2'], item['preference']) 
        for item in batch
    ])

    # 2. 堆叠状态 (TensorDict可以被stack)
    # 注意：确保所有TensorDict实例具有相同的keys和batch_size=[]
    states_batch = torch.stack(states)

    # 3. 填充动作序列
    # batch_first=True让输出的维度是 (batch_size, sequence_length)
    actions1_padded = pad_sequence(actions1, batch_first=True, padding_value=0)
    actions2_padded = pad_sequence(actions2, batch_first=True, padding_value=0)

    # 4. 堆叠偏好
    preferences_batch = torch.stack(preferences)

    return states_batch, actions1_padded, actions2_padded, preferences_batch

class PreferenceDataset(Dataset):
    """偏好数据集"""
    
    def __init__(self, feedback_files):
        self.preferences = []
        self.load_feedback_data(feedback_files)
    
    def load_feedback_data(self, feedback_files):
        """加载反馈数据"""
        for file_path in feedback_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                tensor_data = {}
                state_data = {}
                if item["preference"] is not None:
                    for key, value in item["state"].items():
                        tensor_value = torch.tensor(value)
                        state_data[key] = tensor_value
                    tensor_data['state'] = TensorDict(state_data)
                    tensor_data["action1"] = torch.tensor(item["action1"])
                    tensor_data["action2"] = torch.tensor(item["action2"])
                    tensor_data["preference"] = torch.tensor([item["preference"]])
                    tensor_data["reward1"] = torch.tensor([item["reward1"]])
                    tensor_data["reward2"] = torch.tensor([item["reward2"]])

                    self.preferences.append(tensor_data)
        
        print(f"加载了 {len(self.preferences)} 个有效偏好对")
    
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        return self.preferences[idx]

def test_2_models(env, td_init, results_dir):
    # 原始模型(greedy)输出
    # 我们需要重新加载原始模型，因为policy_model已经被微调了
    original_model = AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device)
    original_model.eval()
    original_out = original_model(td_init.clone(), phase="test", decode_type="sampling")

    # 加载DPO微调后的模型
    policy_model = AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt")
    policy_model.load_state_dict(torch.load("./dpo_results/model/dpo_finetuned_model-ppo50-d2000.pth", map_location="cpu"))
    policy_model = policy_model.to(device)
    policy_model.eval()
    
    # DPO微调后模型输出
    dpo_out = policy_model(td_init.clone(), phase="test", decode_type="sampling")

    print("\n=== 测试对比 ===")
    print(f"原始模型 (Greedy) 平均奖励: {original_out['reward'].mean().item():.4f}")
    print(f"DPO 微调模型 (Greedy) 平均奖励: {dpo_out['reward'].mean().item():.4f}")
    
    # 保存结果
    for i in range(td_init.batch_size[0]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 原始模型结果
        env.render(td_init[i], original_out["actions"][i].cpu(), ax=ax1)
        ax1.set_title(f"原始模型\nTour Length: {-original_out['reward'][i].item():.2f}")
        
        # DPO模型结果
        env.render(td_init[i], dpo_out["actions"][i].cpu(), ax=ax2)
        ax2.set_title(f"DPO 微调模型\nTour Length: {-dpo_out['reward'][i].item():.2f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir+"/dpo_results_picture", f"dpo_comparison_result_{i}.png"), dpi=300)
        plt.close()

    with open(os.path.join(results_dir, "dpo_comparison_result.json"), "w") as f:
        serializable_data = []
        for i in range(td_init.batch_size[0]):
            serializable_item = {
                "state": serialize_tensordict(td_init[i]),
                "original_model_out": original_out["actions"][i].cpu().tolist(),
                "original_model_reward": original_out["reward"][i].item(),
                "dpo_model_output": dpo_out["actions"][i].cpu().tolist(),
                "dpo_model_reward": dpo_out["reward"][i].item()
            }
            serializable_data.append(serializable_item)
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(results_dir, "dpo_comparison_td_init.json"), "w") as f:
        json.dump(serialize_tensordict(td_init), f, indent=2, ensure_ascii=False)
    
    print(f"对比结果图已保存到 {results_dir} 目录")
    print("\n=== 流程结束 ===")

def test_4_models(env, td_init, results_dir, config):
    # 原始模型(greedy)输出
    # 我们需要重新加载原始模型，因为policy_model已经被微调了
    original_model = AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device)
    original_model.eval()
    original_out = original_model(td_init.clone(), phase="test", decode_type=config.test_approch)

    # 加载DPO微调后的模型
    policy_model_10 = AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device)
    policy_model_10.load_state_dict(torch.load("./dpo_results/model/dpo_finetuned_model-ppo10-d2000.pth", map_location="cpu"))
    policy_model_10.eval()

    policy_model_25 = AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device)
    policy_model_25.load_state_dict(torch.load("./dpo_results/model/dpo_finetuned_model-ppo25-d2000.pth", map_location="cpu"))
    policy_model_25.eval()

    policy_model_50 = AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device)
    policy_model_50.load_state_dict(torch.load("./dpo_results/model/dpo_finetuned_model-ppo50-d2000.pth", map_location="cpu"))
    policy_model_50.eval()

    # DPO微调后模型输出
    out_10 = policy_model_10(td_init.clone(), phase="test", decode_type=config.test_approch)
    out_25 = policy_model_25(td_init.clone(), phase="test", decode_type=config.test_approch)
    out_50 = policy_model_50(td_init.clone(), phase="test", decode_type=config.test_approch)

    print("\n=== 测试对比 ===")
    print(f"原始模型 (Greedy) 平均奖励: {original_out['reward'].mean().item():.4f}")
    print(f"DPO 微调模型 10 epochs (Greedy) 平均奖励: {out_10['reward'].mean().item():.4f}")
    print(f"DPO 微调模型 25 epochs (Greedy) 平均奖励: {out_25['reward'].mean().item():.4f}")
    print(f"DPO 微调模型 50 epochs (Greedy) 平均奖励: {out_50['reward'].mean().item():.4f}")
    
    # 保存结果
    for i in range(td_init.batch_size[0]):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(32, 8))
        
        # 原始模型结果
        env.render(td_init[i], original_out["actions"][i].cpu(), ax=ax1)
        ax1.set_title(f"原始模型\nTour Length: {-original_out['reward'][i].item():.2f}")
        
        # DPO模型结果
        env.render(td_init[i], out_10["actions"][i].cpu(), ax=ax2)
        ax2.set_title(f"DPO 微调模型 10 epochs\nTour Length: {-out_10['reward'][i].item():.2f}")
        
        env.render(td_init[i], out_25["actions"][i].cpu(), ax=ax3)
        ax3.set_title(f"DPO 微调模型 25 epochs\nTour Length: {-out_25['reward'][i].item():.2f}")
        
        env.render(td_init[i], out_50["actions"][i].cpu(), ax=ax4)
        ax4.set_title(f"DPO 微调模型 50 epochs\nTour Length: {-out_50['reward'][i].item():.2f}")
        
        plt.tight_layout()
        picture_dir = os.path.join(results_dir, "result_pictures")
        if not os.path.exists(picture_dir):
            os.makedirs(picture_dir)
        plt.savefig(os.path.join(picture_dir, f"dpo_comparison_result_{i}.png"), dpi=300)
        plt.close()

    with open(os.path.join(results_dir, "result.json"), "w") as f:
        serializable_data = []
        for i in range(td_init.batch_size[0]):
            serializable_item = {
                "state": serialize_tensordict(td_init[i]),
                "original_model_out": original_out["actions"][i].cpu().tolist(),
                "original_model_reward": original_out["reward"][i].item(),
                "dpo_model_10_output": out_10["actions"][i].cpu().tolist(),
                "dpo_model_10_reward": out_10["reward"][i].item(),
                "dpo_model_25_output": out_25["actions"][i].cpu().tolist(),
                "dpo_model_25_reward": out_25["reward"][i].item(),
                "dpo_model_50_output": out_50["actions"][i].cpu().tolist(),
                "dpo_model_50_reward": out_50["reward"][i].item()
            }
            serializable_data.append(serializable_item)
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"对比结果图已保存到 {results_dir} 目录")
    print("\n=== 流程结束 ===")

def main():
    print("\n=== 测试DPO微调模型 ===")

    # 配置
    class Config:
        learning_rate = 1e-5 # DPO 微调通常使用更小的学习率
        max_epochs = 20
        batch_size = 16 # 根据显存调整
        beta = 0.5 # DPO的超参数
        num_loc = 100 # 100 initial
        capacity = 50 # 50 initial
        test_approch = "greddy"
        test_data_size = 100
    
    config = Config()

    results_dir = f"dpo_results_{config.test_data_size}_{config.test_approch}_numloc{config.num_loc}_capacity{config.capacity}"
    os.makedirs(results_dir, exist_ok=False)

    # --- 测试模型 ---
    env = CVRPEnv(generator_params=dict(num_loc=config.num_loc, capacity=config.capacity))
    td_test_data = env.generator(batch_size=[config.batch_size])
    td_init = env.reset(td_test_data.clone()).to(device)
    
    test_4_models(env, td_init, results_dir, config)

if __name__ == "__main__":
    main() 