"""
人类反馈收集器
用于收集人类对CVRP解的偏好
"""

import torch
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from typing import List, Tuple
import numpy as np

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.amppo import AMPPO
from rl4co.models.zoo.symnco import SymNCO
from tensordict.tensordict import TensorDict

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

class HumanFeedbackCollector:
    """人类反馈收集器"""
    
    def __init__(self, env, save_dir="human_feedback"):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def collect_feedback(self, model, num_pairs=10):
        """收集人类反馈"""
        feedback_data = []
        
        for i in range(num_pairs):
            print(f"\n=== 收集反馈对 {i+1}/{num_pairs} ===")
            
            # 生成测试数据
            td_test_data = self.env.generator(batch_size=[1])
            td_init = self.env.reset(td_test_data.clone()).to(device)
    
            # 生成两个不同的解
            with torch.no_grad():
                # 使用贪婪解码
                out1 = model(td_init.clone(), phase="test", decode_type="sampling", return_actions=True)
                
                # 使用采样解码
                out2 = model(td_init.clone(), phase="test", decode_type="sampling", return_actions=True)
            
            # 保存两个解的图片
            self.save_comparison_images(td_init[0], out1["actions"][0], out2["actions"][0], i)
            
            # 获取人类偏好
            # preference = self.get_human_preference(i)
            if out1["reward"][0].item() - out2["reward"][0].item() > 0:
                preference = 1
            else:
                preference = 0
            
            # 存储反馈 - 确保数据在CPU上
            feedback_data.append({
                "pair_id": i,
                "state": td_init[0].cpu(),
                "action1": out1["actions"][0].cpu(),
                "action2": out2["actions"][0].cpu(),
                "reward1": out1["reward"][0].item(),
                "reward2": out2["reward"][0].item(),
                "preference": preference,
                "timestamp": datetime.now().isoformat()
            })
            
        # 保存反馈数据
        self.save_feedback_data(feedback_data)
        return feedback_data
    
    def save_comparison_images(self, td, action1, action2, pair_id):
        """保存对比图片"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 绘制第一个解
        self.env.render(td, action1.cpu(), ax=ax1)
        reward1 = -self.env._get_reward(td.unsqueeze(0), action1.unsqueeze(0)).item()
        ax1.set_title(f"解 1 (路线长度: {reward1:.2f})")
        
        # 绘制第二个解
        self.env.render(td, action2.cpu(), ax=ax2)
        reward2 = -self.env._get_reward(td.unsqueeze(0), action2.unsqueeze(0)).item()
        ax2.set_title(f"解 2 (路线长度: {reward2:.2f})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'./comparison/comparison_{pair_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"对比图片已保存: comparison_{pair_id}.png")
        print(f"解1路线长度: {reward1:.2f}")
        print(f"解2路线长度: {reward2:.2f}")
    
    def get_human_preference(self, pair_id):
        """获取人类偏好"""
        while True:
            try:
                choice = input(f"对于对比图片 {pair_id}，你更喜欢哪个解？(1/2/skip): ").strip().lower()
                if choice == '1':
                    return 1
                elif choice == '2':
                    return 0
                elif choice == 'skip':
                    return None
                else:
                    print("请输入 1, 2 或 skip")
            except KeyboardInterrupt:
                print("\n跳过这个对比")
                return None
    
    def save_feedback_data(self, feedback_data):
        """保存反馈数据"""
        filename = os.path.join(self.save_dir, f'feedback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        # 转换tensor为numpy数组以便JSON序列化
        serializable_data = []
        for item in feedback_data:
            serializable_item = {
                "pair_id": item["pair_id"],
                "state": serialize_tensordict(item["state"]),
                "action1": item["action1"].tolist(),
                "action2": item["action2"].tolist(),
                "reward1": item["reward1"],
                "reward2": item["reward2"],
                "preference": item["preference"],
                "timestamp": item["timestamp"]
            }
            serializable_data.append(serializable_item)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"反馈数据已保存: {filename}")

def main():
    """主函数"""
    print("=== 人类反馈收集器 ===")
    
    # 创建环境
    env = CVRPEnv(
        generator_params=dict(num_loc=50, capacity=50),
        num_samples={"train": 10000, "val": 1000, "test": 1000}
    )
    
    model_a = AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device)
    model_b = SymNCO.load_from_checkpoint("../model/cvrp50-symnco.ckpt", baseline="symnco").to(device)
    feedback_collector = HumanFeedbackCollector(env, save_dir="human_feedback")
    
    # 收集反馈
    print("\n开始收集人类反馈...")
    num_pairs = int(input("请输入要收集的反馈对数量 (默认10): ") or "10")
    feedback_data = feedback_collector.collect_feedback(model_a, num_pairs=num_pairs)
    
    # 统计反馈
    valid_feedback = [f for f in feedback_data if f["preference"] is not None]
    if valid_feedback:
        preferences = [f["preference"] for f in valid_feedback]
        print(f"\n=== 反馈统计 ===")
        print(f"总反馈对: {len(feedback_data)}")
        print(f"有效反馈: {len(valid_feedback)}")
        print(f"选择解1的次数: {sum(preferences)}")
        print(f"选择解2的次数: {len(preferences) - sum(preferences)}")
    
    print("\n反馈收集完成！")
    print("现在可以运行 train_preference.py 来训练偏好模型")

if __name__ == "__main__":
    main() 