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
from tqdm import tqdm

from rl4co.envs.routing.cvrp.env import CVRPEnv
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

class DPOTrainer:
    """
    Direct Preference Optimization (DPO) 训练器
    """
    def __init__(self, policy_model: AMPPO, ref_model: AMPPO, env: CVRPEnv, config):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.env = env
        self.config = config
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.max_epochs)
        self.history = {"losses": [], "accuracies": [], "log_probs_ratio": [], "rewards": []}
        
        # 冻结参考模型的参数
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _get_log_probs(self, model: AMPPO, states_td: TensorDict, actions: torch.Tensor):
        """计算给定模型、状态和动作序列的对数概率"""
        # rl4co模型在评估模式下可以直接计算给定动作序列的对数似然
        # 需要确保模型处于评估模式
        model.eval()
        # forward pass with specified actions to get log-likelihood
        # 注意：我们需要确保模型内部使用我们提供的actions，而不是自己解码
        # AMPPO 的 forward 支持 `actions` 参数用于此目的
        output = model(states_td, phase="test", actions=actions, return_actions=True)
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

        actions_chosen = torch.where(is_action1_chosen.unsqueeze(-1), actions1, actions2)
        actions_rejected = torch.where(is_action1_chosen.unsqueeze(-1), actions2, actions1)

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

            # 更新tqdm的实时信息
            if hasattr(dataloader, 'set_postfix'):
                dataloader.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{accuracy.item():.4f}"
                })

        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "log_probs_ratio": total_log_probs_ratio / num_batches
        }

    def train(self, dataloader, dataloader_test, epochs):
        print(f"开始DPO微调，共 {epochs} 个epoch")
        for epoch in range(epochs):
            # 使用tqdm包装dataloader
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", total=len(dataloader))
            metrics = self.train_epoch(pbar)
            if not metrics:
                print("数据加载器为空，跳过epoch")
                continue
            self.history["losses"].append(metrics["loss"])
            self.history["accuracies"].append(metrics["accuracy"])
            self.history["log_probs_ratio"].append(metrics["log_probs_ratio"])
            self.scheduler.step()
            
            # 更新tqdm的后缀信息
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.4f}",
                'Acc': f"{metrics['accuracy']:.4f}",
                'LR': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })

            # 测试模型
            if epoch % 2 == 0:
                reward = self.test_model(self.env, dataloader_test)
                self.history["rewards"].append(reward)

        print("训练完成")
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].plot(self.history["losses"])
        axes[0].set_title("DPO Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True)
        axes[1].plot(self.history["accuracies"])
        axes[1].set_title("DPO Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].grid(True)
        axes[2].plot(self.history["log_probs_ratio"])
        axes[2].set_title("Log Probs Ratio (Chosen/Rejected)"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Ratio"); axes[2].grid(True)
        axes[3].plot(self.history["rewards"])
        axes[3].set_title("Test Reward"); axes[3].set_xlabel("Epoch"); axes[3].set_ylabel("Reward"); axes[3].grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存: {save_path}")
        plt.show()

    def test_model(self, env, dataloader_test):
        """测试模型"""
        self.policy_model.eval()
        rewards = []
        for batch_idx, batch in enumerate(dataloader_test):
            td_init = env.reset(batch.clone()).to(device)

            out = self.policy_model(td_init.clone(), phase="test", decode_type=self.config.test_approach)
            rewards.append(out["reward"].mean().item())
        print(f"测试平均奖励: {np.mean(rewards):.4f}")
        return np.mean(rewards)

def main():
    """主函数"""
    print("=== DPO 微调器 ===")
    
    # 配置
    class Config:
        learning_rate = 1e-5 # DPO 微调通常使用更小的学习率
        max_epochs = 20
        batch_size = 8 # 根据显存调整
        beta = 0.5 # DPO的超参数
        num_loc = 100
        capacity = 50
        test_data_size = 100    # 测试集总大小
        eval_batch_size = 50      # 评估时每个批次的大小
        test_approach = "sampling"
        save_dir = "dpo_results"
    
    config = Config()

    # 创建测试数据
    env = CVRPEnv(generator_params=dict(num_loc=config.num_loc, capacity=config.capacity))
    dataset_test = env.dataset([config.test_data_size])
    dataloader_test = DataLoader(dataset_test, batch_size=config.eval_batch_size, collate_fn=dataset_test.collate_fn)

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
    base_model = AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device)
    # 创建参考模型 (冻结参数) 和策略模型 (待训练)
    ref_model = copy.deepcopy(base_model).to(device)
    policy_model = base_model # policy_model是我们将要微调的模型
    print("已加载预训练的基础模型，并创建了策略模型和参考模型")
    
    # 创建DPO训练器
    save_file_name = f"dpo_finetuned_model-ppo{config.max_epochs}-d{len(dataset)}_new"
    trainer = DPOTrainer(policy_model, ref_model, env, config)
    trainer.train(dataloader, dataloader_test, config.max_epochs)
    trainer.plot_training_history(os.path.join("model", save_file_name+".png"))

    # 保存结果
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(os.path.join(config.save_dir, "model"), exist_ok=True)
    
    torch.save(policy_model.state_dict(), os.path.join("model", save_file_name+".pth"))
    print("\nDPO微调模型已保存")

if __name__ == "__main__":
    main() 