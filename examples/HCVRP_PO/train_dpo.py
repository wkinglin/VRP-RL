"""
Direct Preference Optimization (DPO) Trainer
使用DPO方法直接微调策略模型
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

from parco.envs import HCVRPEnv
from parco.models import PARCORLModule
from tensordict import TensorDict

# 设置matplotlib中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preference_collate_fn(batch):
    """
    处理可变长度序列的自定义collate函数
    支持action为二维（多车辆）或一维（单车辆）情况，自动补齐到最大长度
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

    # 3. 处理action的维度
    # 支持action为(batch, n_vehicle, seq_len)或(batch, seq_len)
    def pad_action_list(action_list):
        # action_list: batch个tensor，每个tensor形状可能是(seq_len,)或(n_vehicle, seq_len)
        # 统一为三维：(batch, n_vehicle, seq_len)
        # 如果是二维，自动补一维
        padded = []
        max_vehicle = 1
        max_len = 1
        # 先统计最大车辆数和最大序列长度
        for a in action_list:
            if a.ndim == 1:
                n_vehicle = 1
                seq_len = a.shape[0]
            else:
                n_vehicle = a.shape[0]
                seq_len = a.shape[1]
            max_vehicle = max(max_vehicle, n_vehicle)
            max_len = max(max_len, seq_len)
        # 逐个补齐
        for a in action_list:
            if a.ndim == 1:
                # (seq_len,) -> (1, seq_len)
                a = a.unsqueeze(0)
            n_vehicle, seq_len = a.shape
            # 先pad seq_len
            if seq_len < max_len:
                pad_width = (0, max_len - seq_len)
                a = F.pad(a, pad_width, value=0)
            # 再pad n_vehicle
            if n_vehicle < max_vehicle:
                pad_width = (0, 0, 0, max_vehicle - n_vehicle)
                a = F.pad(a, pad_width, value=0)
            padded.append(a)
        # 堆叠
        return torch.stack(padded)  # (batch, n_vehicle, seq_len)

    actions1_padded = pad_action_list(actions1)
    actions2_padded = pad_action_list(actions2)

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

class DPOTrainer:
    """
    Direct Preference Optimization (DPO) 训练器
    """
    def __init__(self, policy_model: PARCORLModule, ref_model: PARCORLModule, config):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.max_epochs)
        self.history = {"losses": [], "accuracies": [], "log_probs_ratio": []}
        
        # 冻结参考模型的参数
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _get_log_probs(self, model: PARCORLModule, states_td: TensorDict, actions: torch.Tensor):
        """计算给定模型、状态和动作序列的对数概率"""
        # rl4co模型在评估模式下可以直接计算给定动作序列的对数似然
        # 需要确保模型处于评估模式
        model.eval()
        # forward pass with specified actions to get log-likelihood
        # 注意：我们需要确保模型内部使用我们提供的actions，而不是自己解码
        # AMPPO 的 forward 支持 `actions` 参数用于此目的
        output = model(states_td, phase="test", actions=actions, decode_type="greedy")
        return output['log_likelihood']

    def compute_loss(self, states_td: TensorDict, actions1: torch.Tensor, actions2: torch.Tensor, preferences: torch.Tensor):
        """
        计算DPO损失
        preferences=0: action2 is preferred
        preferences=1: action1 is preferred
        """
        # 为了代码清晰，我们定义 chosen 和 rejected
        # 当 preference=0, 2是chosen, 1是rejected
        # 当 preference=1, 1是chosen, 2是rejected
        
        # 根据preferences确定哪个是chosen，哪个是rejected
        is_action1_chosen = preferences.squeeze().bool()

        # 对齐actions1和actions2的最后维度
        if actions1.shape[-1] < actions2.shape[-1]:
            pad = (0, actions2.shape[-1] - actions1.shape[-1])
            actions1 = F.pad(actions1, pad, "constant", 0)
        elif actions1.shape[-1] > actions2.shape[-1]:
            pad = (0, actions1.shape[-1] - actions2.shape[-1])
            actions2 = F.pad(actions2, pad, "constant", 0)
        
        assert actions1.shape == actions2.shape, f"actions1.shape: {actions1.shape}, actions2.shape: {actions2.shape}"

         # 扩展is_action1_chosen到 (batch, 1, 1) 以便广播到actions的shape
        is_action1_chosen = is_action1_chosen.view(-1, 1, 1)
        actions_chosen = torch.where(is_action1_chosen, actions1, actions2)
        actions_rejected = torch.where(is_action1_chosen, actions2, actions1)

        # 计算对数概率
        # 注意: 需要为每个模型克隆 tensordict，因为模型可能会在内部修改它
        with torch.no_grad():
            ref_log_probs_chosen = self._get_log_probs(self.ref_model, states_td.clone(), actions_chosen)
            ref_log_probs_rejected = self._get_log_probs(self.ref_model, states_td.clone(), actions_rejected)

        # policy model 需要计算梯度
        self.policy_model.train() # 切换回训练模式
        policy_log_probs_chosen = self._get_log_probs(self.policy_model, states_td.clone(), actions_chosen)
        policy_log_probs_rejected = self._get_log_probs(self.policy_model, states_td.clone(), actions_rejected)

        # 计算DPO损失
        pi_log_ratio = policy_log_probs_chosen - policy_log_probs_rejected
        ref_log_ratio = ref_log_probs_chosen - ref_log_probs_rejected
        
        logits = pi_log_ratio - ref_log_ratio
        loss = -F.logsigmoid(self.config.beta * logits).mean()
        
        # 计算准确率 (策略模型更倾向于选择哪个)
        with torch.no_grad():
            accuracy = (pi_log_ratio > ref_log_ratio).float().mean()

        return loss, accuracy, pi_log_ratio.mean()

    def train_epoch(self, dataloader: DataLoader):
        """训练一个epoch"""
        self.policy_model.train()
        total_loss, total_accuracy, total_log_probs_ratio = 0, 0, 0
        num_batches = len(dataloader)
        if num_batches == 0: return {}
        
        for states_td, actions1, actions2, preferences in dataloader:
            if states_td is None: continue
            
            states_td, actions1, actions2, preferences = (
                states_td.to(device), actions1.to(device), 
                actions2.to(device), preferences.to(device)
            )
            
            loss, accuracy, log_probs_ratio = self.compute_loss(states_td, actions1, actions2, preferences)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_log_probs_ratio += log_probs_ratio.item()

        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "log_probs_ratio": total_log_probs_ratio / num_batches
        }

    def train(self, dataloader, epochs):
        print(f"开始DPO微调，共 {epochs} 个epoch")
        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader)
            if not metrics:
                print("数据加载器为空，跳过epoch")
                continue
            self.history["losses"].append(metrics["loss"])
            self.history["accuracies"].append(metrics["accuracy"])
            self.history["log_probs_ratio"].append(metrics["log_probs_ratio"])
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {metrics['loss']:.4f}, "
                  f"Accuracy: {metrics['accuracy']:.4f}, "
                  f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
        print("训练完成")
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(self.history["losses"])
        axes[0].set_title("DPO Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True)
        axes[1].plot(self.history["accuracies"])
        axes[1].set_title("DPO Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].grid(True)
        axes[2].plot(self.history["log_probs_ratio"])
        axes[2].set_title("Log Probs Ratio (Chosen/Rejected)"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Ratio"); axes[2].grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存: {save_path}")
        plt.show()

def main():
    """主函数"""
    print("=== DPO 微调器 ===")
    
    # 配置
    class Config:
        learning_rate = 1e-5 # DPO 微调通常使用更小的学习率
        max_epochs = 25
        batch_size = 8 # 根据显存调整
        beta = 0.5 # DPO的超参数
        num_loc = 50
    
    config = Config()
    
    # 查找反馈文件
    feedback_dir = "human_feedback"
    feedback_files = glob.glob(os.path.join(feedback_dir, "feedback_*.json"))
    if not feedback_files:
        print(f"在 {feedback_dir} 目录中没有找到反馈文件。请先运行 collect_feedback.py。")
        return
    print(f"找到 {len(feedback_files)} 个反馈文件")
    
    # 加载数据集
    dataset = PreferenceDataset(feedback_files)
    if len(dataset) == 0:
        print("没有有效的偏好数据，无法训练。")
        return
    # 注意：collate_fn是从train_preference.py导入的
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=preference_collate_fn)
    
    # 加载预训练的基础模型
    base_model = PARCORLModule.load_from_checkpoint("../hcvrp_models/hcvrp_rl_model.ckpt").to(device)
    # 创建参考模型 (冻结参数) 和策略模型 (待训练)
    ref_model = copy.deepcopy(base_model).to(device)
    policy_model = base_model # policy_model是我们将要微调的模型
    print("已加载预训练的基础模型，并创建了策略模型和参考模型")
    
    # 创建DPO训练器
    trainer = DPOTrainer(policy_model, ref_model, config)
    trainer.train(dataloader, config.max_epochs)
    trainer.plot_training_history("dpo_training_history.png")

    # --- 测试模型 ---
    print("\n=== 测试DPO微调模型 ===")
    env = HCVRPEnv(generator_params=dict(num_loc=config.num_loc))
    td_test_data = env.generator(batch_size=[3])
    td_init = env.reset(td_test_data.clone()).to(device)
    
    # 原始模型(greedy)输出
    # 我们需要重新加载原始模型，因为policy_model已经被微调了
    original_model = PARCORLModule.load_from_checkpoint("../hcvrp_models/hcvrp_rl_model.ckpt").to(device)
    original_out = original_model(td_init.clone(), phase="test", decode_type="greedy")
    
    # DPO微调后模型输出
    dpo_out = policy_model(td_init.clone(), phase="test", decode_type="greedy")
    
    print("\n=== 测试对比 ===")
    print(f"原始模型 (Greedy) 平均奖励: {original_out['reward'].mean().item():.4f}")
    print(f"DPO 微调模型 (Greedy) 平均奖励: {dpo_out['reward'].mean().item():.4f}")
    
    # 保存结果
    results_dir = "dpo_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "model"), exist_ok=True)
    
    torch.save(policy_model.state_dict(), os.path.join(results_dir, "model", "dpo_finetuned_model.pth"))
    print("\nDPO微调模型已保存")
    
    for i in range(td_init.batch_size[0]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 原始模型结果
        env.render(td_init[i], original_out["actions"][i].cpu(), ax=ax1)
        ax1.set_title(f"原始模型\nTour Length: {-original_out['reward'][i].item():.2f}")
        
        # DPO模型结果
        env.render(td_init[i], dpo_out["actions"][i].cpu(), ax=ax2)
        ax2.set_title(f"DPO 微调模型\nTour Length: {-dpo_out['reward'][i].item():.2f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"dpo_comparison_result_{i}.png"), dpi=300)
        plt.close()
    
    print(f"对比结果图已保存到 {results_dir} 目录")
    print("\n=== 流程结束 ===")

if __name__ == "__main__":
    main() 