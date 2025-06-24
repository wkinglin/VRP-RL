"""
偏好模型训练器
使用收集的人类反馈训练偏好奖励模型
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
import glob
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.models.zoo.am.encoder import AttentionModelEncoder # 导入正确的Encoder
from rl4co.models.zoo.amppo import AMPPO

from tensordict import TensorDict

# 设置matplotlib中文字体支持

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class AttentionRewardModel(nn.Module):
    """
    基于注意力的偏好奖励模型 (改进版)
    使用与策略模型类似的结构来理解CVRP问题的空间特性。
    """
    def __init__(self, embed_dim=128, num_heads=8, num_layers=3, num_actions=52):
        super().__init__()
        
        # 状态编码器 (使用 rl4co 的 AttentionModelEncoder)
        self.state_encoder1 = AttentionModelEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            normalization="batch"
        )
        # self.state_encoder2 = nn.LSTM(embed_dim, embed_dim, batch_first=True)

        # 动作编码器 (使用RNN来捕捉序列信息，优于简单的mean pooling)
        self.action_embedding = nn.Embedding(num_actions, embed_dim, padding_idx=0)
        # self.action_encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        
        # 使用transformer来编码序列
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True),
            num_layers=num_layers
        )

        # 奖励头
        # 输入: [graph_embedding, action_embedding]
        self.reward_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, state_td: TensorDict, actions: torch.Tensor):
        """
        计算给定状态和动作序列的偏好奖励
        :param state_td: TensorDict 格式的状态，包含 'locs', 'depot' 等
        :param actions: (batch_size, seq_len) 的动作序列
        :return: (batch_size, 1) 的奖励值
        """
        # 1. 编码状态
        # graph_embed 的形状是 (batch_size, seq_len, embed_dim)
        graph_embed, _ = self.state_encoder1(state_td)
        # _, (h_n, _) = self.state_encoder2(graph_embed)
        # graph_embed = h_n.squeeze(0)
        
        # 2. 编码动作
        action_embeds = self.action_embedding(actions.long()) # (batch, seq_len, embed_dim)
        # 使用LSTM的最后一个隐藏状态来总结整个动作序列
        # lstm_out 包含所有时间步的输出, (h_n, c_n) 是最后一个时间步的隐藏状态和细胞状态
        # _, (h_n, _) = self.action_encoder(action_embeds) # h_n 的形状: (num_layers, batch, embed_dim)
        # 我们使用最后一层的隐藏状态作为动作序列的向量表示
        # action_vec = h_n.squeeze(0) # (batch, embed_dim)

        # 把graph_embed中间的维度补齐到action_vec的seq_len
        # padding_needed = action_vec.size(1) - graph_embed.size(1)
        # if padding_needed > 0:
        #     paddings = (0, 0, 0, padding_needed)
        #     graph_embed = F.pad(graph_embed, paddings, "constant", 0)
        # elif padding_needed < 0:
        #     graph_embed = graph_embed[:, :action_vec.size(1), :]
        
        # 3. 组合并预测奖励
        # combined = torch.cat([graph_embed, action_vec], dim=-1) # (batch, 2*embed_dim)
        combined = torch.cat([graph_embed, action_embeds], dim=1) # (batch, seq_len, 2*embed_dim)

        # 使用transformer来编码动作序列
        combined = self.transformer_encoder(combined)
        # 对中间维度（seq_len）做池化，得到(batch, 2*embed_dim)的向量
        # 这里采用平均池化
        combined = combined.mean(dim=1)

        from ipdb import set_trace
        set_trace()

        reward = self.reward_head(combined)

        from ipdb import set_trace
        set_trace()
        return reward
    
class PreferenceTrainer:
    """偏好训练器 (适配新模型)"""
    def __init__(self, reward_model: AttentionRewardModel, config):
        self.reward_model = reward_model
        self.config = config
        self.optimizer = torch.optim.AdamW(reward_model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.max_epochs)
        self.history = {"losses": [], "accuracies": [], "rewards1": [], "rewards2": []}
    
    def compute_loss(self, states_td: TensorDict, actions1: torch.Tensor, actions2: torch.Tensor, preferences: torch.Tensor):
        """计算偏好损失"""
        # (batch, 1)
        rewards1 = self.reward_model(states_td, actions1)
        rewards2 = self.reward_model(states_td, actions2)

        # 计算偏好损失 (Bradley-Terry model)
        logits = (rewards1 - rewards2).squeeze() / self.config.temperature
        loss = F.binary_cross_entropy_with_logits(logits, preferences.squeeze().float())
        
        # 计算准确率
        with torch.no_grad():
            predictions = (logits > 0).float()
            accuracy = (predictions == preferences.squeeze()).float().mean()
        
        return loss, accuracy, rewards1.mean(), rewards2.mean()

    def train_epoch(self, dataloader: DataLoader):
        """训练一个epoch"""
        self.reward_model.train()
        total_loss, total_accuracy, total_rewards1, total_rewards2 = 0, 0, 0, 0
        num_batches = len(dataloader)
        if num_batches == 0: return {}
        
        for states_td, actions1, actions2, preferences in dataloader:
            if states_td is None: continue
            
            states_td, actions1, actions2, preferences = (
                states_td.to(device), actions1.to(device), 
                actions2.to(device), preferences.to(device)
            )
            
            loss, accuracy, reward1, reward2 = self.compute_loss(states_td, actions1, actions2, preferences)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_rewards1 += reward1.item()
            total_rewards2 += reward2.item()
        
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "reward1": total_rewards1 / num_batches,
            "reward2": total_rewards2 / num_batches
        }

    def train(self, dataloader, epochs):
        print(f"开始训练偏好模型，共 {epochs} 个epoch")
        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader)
            if not metrics:
                print("数据加载器为空，跳过epoch")
                continue
            self.history["losses"].append(metrics["loss"])
            self.history["accuracies"].append(metrics["accuracy"])
            self.history["rewards1"].append(metrics["reward1"])
            self.history["rewards2"].append(metrics["reward2"])
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {metrics['loss']:.4f}, "
                  f"Accuracy: {metrics['accuracy']:.4f}, "
                  f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
        print("训练完成")
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(self.history["losses"])
        axes[0, 0].set_title("Training Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss"); axes[0, 0].grid(True)
        axes[0, 1].plot(self.history["accuracies"])
        axes[0, 1].set_title("Preference Accuracy"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy"); axes[0, 1].grid(True)
        axes[1, 0].plot(self.history["rewards1"], label="Reward 1"); axes[1, 0].plot(self.history["rewards2"], label="Reward 2")
        axes[1, 0].set_title("Average Predicted Rewards"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Reward"); axes[1, 0].legend(); axes[1, 0].grid(True)
        reward_diff = [r1 - r2 for r1, r2 in zip(self.history["rewards1"], self.history["rewards2"])]
        axes[1, 1].plot(reward_diff); axes[1, 1].set_title("Reward Difference (R1 - R2)"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Difference"); axes[1, 1].grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存: {save_path}")
        plt.show()

class PreferenceOptimizedModel(nn.Module):
    """
    偏好优化模型 (改进版)
    通过"采样+重排序"策略，在推理时利用偏好奖励模型找到更好的解。
    """
    def __init__(self, base_model: AMPPO, reward_model: AttentionRewardModel, preference_weight=0.5, num_samples=64):
        super().__init__() # 必须先调用父类的构造函数
        self.base_model = base_model
        self.reward_model = reward_model
        self.preference_weight = preference_weight
        self.num_samples = num_samples # 用于采样的路径数量

    @torch.no_grad()
    def forward(self, td: TensorDict, phase="test", return_actions=True):
        """
        使用"采样+重排序"进行推理，找到优化后的路径。
        """
        self.base_model.eval()
        self.reward_model.eval()

        # 1. 多次采样生成候选路径
        # AMPPO/AttentionModel不直接支持num_samples，我们需要手动扩展td
        batch_size = td.batch_size[0]
        td_expanded = td.expand(self.num_samples, *td.batch_size).reshape(-1) # [B*N]

        # decode_type='sampling'
        base_output = self.base_model(td_expanded, phase=phase, decode_type="sampling", return_actions=return_actions)
        
        # base_output["reward"] shape: (batch_size * num_samples)
        # base_output["actions"] shape: (batch_size * num_samples, seq_len)
        
        # 2. 计算每条候选路径的偏好奖励
        preference_rewards = self.reward_model(td_expanded, base_output["actions"]).squeeze(-1) # (B*N)
        
        # 3. 归一化两种奖励，防止尺度差异过大
        # Reshape rewards to (num_samples, batch_size) for per-instance normalization
        base_rewards_rs = base_output["reward"].view(self.num_samples, batch_size)
        pref_rewards_rs = preference_rewards.view(self.num_samples, batch_size)

        # Normalize over samples for each instance in the batch
        base_rewards_norm = (base_rewards_rs - base_rewards_rs.mean(0)) / (base_rewards_rs.std(0) + 1e-8)
        pref_rewards_norm = (pref_rewards_rs - pref_rewards_rs.mean(0)) / (pref_rewards_rs.std(0) + 1e-8)

        # 4. 计算组合奖励
        # 注意: base_reward是成本(越小越好)，pref_reward是奖励(越大越好)，需要统一方向
        # 我们将base_reward取反，变成最大化问题
        combined_rewards = (1 - self.preference_weight) * (-base_rewards_norm) + self.preference_weight * pref_rewards_norm

        # 5. 为每个问题实例选择最佳路径
        best_indices = torch.argmax(combined_rewards, dim=0) # (batch_size)

        # 6. 提取最佳结果
        # We need to map the flat best_indices back to the original expanded tensor
        # E.g., for batch item `i`, the best sample is `best_indices[i]`. Its flat index is `best_indices[i] * batch_size + i`.
        # A simpler way is to gather from the reshaped tensors.
        final_actions = base_output["actions"].view(self.num_samples, batch_size, -1).gather(
            0, best_indices.view(1, batch_size, 1).expand(1, batch_size, base_output["actions"].size(-1))
        ).squeeze(0)
        
        final_base_reward = base_rewards_rs.gather(0, best_indices.view(1, -1)).squeeze(0)
        final_pref_reward = pref_rewards_rs.gather(0, best_indices.view(1, -1)).squeeze(0)
        final_combined_reward = combined_rewards.gather(0, best_indices.view(1, -1)).squeeze(0)

        return { 
            "actions": final_actions,
            "reward": final_base_reward, # 返回原始奖励以作对比
            "preference_reward": final_pref_reward,
            "combined_score": final_combined_reward
        }

def main():
    """主函数"""
    print("=== 偏好模型训练器 ===")
    
    # 配置
    class Config:
        learning_rate = 1e-4
        max_epochs = 50
        batch_size = 16
        temperature = 0.1
        hidden_dim = 256
        embed_dim = 128
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
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=preference_collate_fn)
    
    # 创建奖励模型 (使用新的 AttentionRewardModel)
    reward_model = AttentionRewardModel(
        embed_dim=config.embed_dim,
        num_heads=8,
        num_layers=3,
        num_actions=config.num_loc + 2 # num_loc + depot + padding_idx
    ).to(device)
    
    # 创建训练器
    trainer = PreferenceTrainer(reward_model, config)
    trainer.train(dataloader, config.max_epochs)
    trainer.plot_training_history("preference_training_history.png")

     # 创建环境用于测试
    env = CVRPEnv(generator_params=dict(num_loc=50, capacity=50))

    # 加载预训练的基础模型
    base_model = AMPPO.load_from_checkpoint("../model/cvrp10-sampling.ckpt").to(device)
    print("已加载预训练的基础模型")
    
    # 创建偏好优化模型 (使用新的实现)
    preference_model = PreferenceOptimizedModel(base_model, reward_model, preference_weight=0.5, num_samples=64)
    
    # 测试模型
    print("\n=== 测试偏好优化模型 ===")
    td_test_data = env.generator(batch_size=[3])
    td_init = env.reset(td_test_data.clone()).to(device)
    
    print("\n=== 测试对比 ===")
    # 原始模型(greedy)输出
    original_out = base_model(td_init.clone(), phase="test", decode_type="greedy")
    # 偏好优化模型输出
    preference_out = preference_model(td_init.clone(), phase="test")
    
    print(f"\n原始模型 (Greedy) 平均奖励: {original_out['reward'].mean().item():.4f}")
    print(f"偏好优化模型 (Rerank) 平均奖励: {preference_out['reward'].mean().item():.4f}")
    print(f"偏好优化模型平均偏好分数: {preference_out['preference_reward'].mean().item():.4f}")
    
    # 保存结果
    results_dir = "preference_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "model"), exist_ok=True)
    
    torch.save(reward_model.state_dict(), os.path.join(results_dir, "model", "preference_reward_model.pth"))
    print("\n偏好奖励模型已保存")
    
    for i in range(td_init.batch_size[0]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 原始模型结果
        env.render(td_init[i], original_out["actions"][i].cpu(), ax=ax1)
        ax1.set_title(f"原始模型\nTour Length: {-original_out['reward'][i].item():.2f}")
        
        # 偏好优化模型结果
        env.render(td_init[i], preference_out["actions"][i].cpu(), ax=ax2)
        ax2.set_title(f"偏好优化模型\nTour Length: {-preference_out['reward'][i].item():.2f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"comparison_result_{i}.png"), dpi=300)
        plt.close()
    
    print(f"对比结果图已保存到 {results_dir} 目录")
    print("\n=== 流程结束 ===")

if __name__ == "__main__":
    main() 