import json
import os
import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from rl4co.envs.routing.cvrp.env import CVRPEnv

# 配置
class Config:
    num_loc = 100
    capacity = 50

config = Config()

# 创建一个临时的env实例用于渲染
env = CVRPEnv(generator_params=dict(num_loc=config.num_loc, capacity=config.capacity))

# 检查JSON文件是否存在
json_file_path = "example.json"
if not os.path.exists(json_file_path):
    print(f"错误: 未在 '{json_file_path}' 找到JSON文件。请确保文件名正确且文件存在。")
    exit()

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 创建保存图片的目录
picture_dir = "plotted_results"
os.makedirs(picture_dir, exist_ok=True)

print(f"开始从 '{json_file_path}' 生成对比图...")

# 遍历JSON中的每个样本并绘图
for i, item in enumerate(data):
    # 1. 从JSON数据重构TensorDict (将列表转为Tensor)
    state_data = {k: torch.tensor(v) for k, v in item["state"].items()}
    td_instance = TensorDict(state_data, batch_size=[])

    # 2. 从JSON获取actions和rewards
    original_actions = torch.tensor(item["Original_actions"])
    original_reward = item["Original_reward"]

    dpo10_actions = torch.tensor(item["DPO_10_epochs_actions"])
    dpo10_reward = item["DPO_10_epochs_reward"]

    dpo25_actions = torch.tensor(item["DPO_25_epochs_actions"])
    dpo25_reward = item["DPO_25_epochs_reward"]

    dpo50_actions = torch.tensor(item["DPO_50_epochs_actions"])
    dpo50_reward = item["DPO_50_epochs_reward"]

    # 3. 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 4. 绘制每个模型的结果
    # 原始模型
    env.render(td_instance, original_actions, ax=axes[0, 0])
    axes[0, 0].set_title(f"Original Model\nTour Length: {-original_reward:.2f}")
    
    # DPO 10 epochs
    env.render(td_instance, dpo10_actions, ax=axes[0, 1])
    axes[0, 1].set_title(f"Fine-tuned Model (10 Epochs) \nTour Length: {-dpo10_reward:.2f}")
    
    # DPO 25 epochs
    env.render(td_instance, dpo25_actions, ax=axes[1, 0])
    axes[1, 0].set_title(f"Fine-tuned Model (25 Epochs) \nTour Length: {-dpo25_reward:.2f}")
    
    # DPO 50 epochs
    env.render(td_instance, dpo50_actions, ax=axes[1, 1])
    axes[1, 1].set_title(f"Fine-tuned Model (50 Epochs) \nTour Length: {-dpo50_reward:.2f}")
    
    plt.tight_layout(pad=3.0)
    save_path = os.path.join(picture_dir, f"comparison_result_{i}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"已保存图像: {save_path}")

print(f"\n绘图完成！所有图像已保存到 '{picture_dir}' 目录。")