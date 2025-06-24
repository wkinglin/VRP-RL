"""
比较原始模型和DPO微调后的模型在整个测试集上的性能
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from typing import List

import torch

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.models.zoo.amppo import AMPPO
from tensordict import TensorDict
from torch.utils.data import DataLoader

# 设置matplotlib中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def serialize_tensordict(td: TensorDict) -> dict:
    """一个辅助函数，用于将TensorDict转换成可序列化的字典。"""
    serializable_td = {}
    for key, tensor in td.items():
        serializable_td[key] = tensor.cpu().tolist()
    return serializable_td

def run_evaluation(env, dataloader, results_dir, config):
    """
    在整个数据集上评估模型，并保存结果。
    """
    print("加载模型...")
    # 定义要评估的模型
    models = {
        "Original": AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device),
        "DPO_10_epochs": AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device),
        "DPO_25_epochs": AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device),
        "DPO_50_epochs": AMPPO.load_from_checkpoint("../model/cvrp50-sampling.ckpt").to(device),
    }

    # 加载DPO微调后的权重
    models["DPO_10_epochs"].load_state_dict(torch.load("./model/dpo_finetuned_model-ppo10-d2000.pth", map_location=device))
    models["DPO_25_epochs"].load_state_dict(torch.load("./model/dpo_finetuned_model-ppo25-d2000.pth", map_location=device))
    models["DPO_50_epochs"].load_state_dict(torch.load("./model/dpo_finetuned_model-ppo50-d2000.pth", map_location=device))

    # 设置所有模型为评估模式
    for model in models.values():
        model.eval()

    # 初始化统计变量
    total_rewards = {name: 0.0 for name in models.keys()}
    total_instances = 0
    example_results_for_json = []
    
    print("开始在测试集上进行评估...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            td_init = env.reset(batch.clone()).to(device)
            current_batch_size = batch.batch_size[0]
            total_instances += current_batch_size
            
            print(f"处理批次 {batch_idx + 1}/{len(dataloader)}...")

            batch_outputs = {}
            for name, model in models.items():
                out = model(td_init.clone(), phase="test", decode_type=config.test_approach)
                batch_outputs[name] = out
                total_rewards[name] += out['reward'].sum().item()

            # 仅保存第一个批次的样本作为示例
            if batch_idx == 0:
                num_samples_to_save = min(current_batch_size, config.num_images_to_save)
                for i in range(num_samples_to_save):
                    # 1. 保存对比图像
                    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 8))
                    axes = axes.flatten()
                    for ax_idx, (name, out) in enumerate(batch_outputs.items()):
                        env.render(td_init[i], out["actions"][i].cpu(), ax=axes[ax_idx])
                        axes[ax_idx].set_title(f"{name}\nTour Length: {-out['reward'][i].item():.2f}")
                    
                    plt.tight_layout()
                    picture_dir = os.path.join(results_dir, "result_pictures")
                    os.makedirs(picture_dir, exist_ok=True)
                    plt.savefig(os.path.join(picture_dir, f"comparison_sample_{i}.png"), dpi=150)
                    plt.close(fig)

                    # 2. 收集样本数据用于JSON保存
                    serializable_item = {
                        "sample_id": f"batch_{batch_idx}_instance_{i}",
                        "state": serialize_tensordict(td_init[i]),
                    }
                    for name, out in batch_outputs.items():
                        serializable_item[f"{name}_actions"] = out["actions"][i].cpu().tolist()
                        serializable_item[f"{name}_reward"] = out["reward"][i].item()
                    example_results_for_json.append(serializable_item)

    # 计算平均奖励
    avg_rewards = {name: reward / total_instances for name, reward in total_rewards.items()}

    print("\n" + "="*20 + " 评估结果 " + "="*20)
    print(f"在 {total_instances} 个测试样本上的平均奖励:")
    for name, avg_reward in avg_rewards.items():
        print(f"  - {name}: {avg_reward:.4f}")
    print("="*52 + "\n")

    # 保存汇总的JSON文件
    summary_data = {
        "evaluation_config": {k: v for k, v in vars(config).items() if not k.startswith('__')},
        "average_rewards": avg_rewards,
        "example_results": example_results_for_json
    }
    with open(os.path.join(results_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    print(f"评估结果和样本图像已保存到目录: {results_dir}")

def main():
    # 配置
    class Config:
        num_loc = 100
        capacity = 50
        test_approach = "greedy"  # 使用 'greedy' 或 'sampling'
        test_data_size = 100    # 测试集总大小
        eval_batch_size = 50      # 评估时每个批次的大小
        num_images_to_save = 50    # 保存多少个对比图像样本

    config = Config()
    
    print("=== 开始 DPO 模型评估 ===")
    
    results_dir = f"dpo_results/dpo_results_{config.test_data_size}_{config.test_approach}_numloc{config.num_loc}_capacity{config.capacity}"
    os.makedirs(results_dir, exist_ok=False)

    # 创建环境和测试数据加载器
    env = CVRPEnv(generator_params=dict(num_loc=config.num_loc, capacity=config.capacity))
    print(f"正在生成 {config.test_data_size} 个测试样本...")
    
    # 1.向 env.dataset 传递总样本数来创建完整数据集
    dataset = env.dataset([config.test_data_size])
    # 2. DataLoader 会自动将数据集切分成多个批次
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, collate_fn=dataset.collate_fn)

    # 运行评估
    run_evaluation(env, dataloader, results_dir, config)
    
    print("\n=== 评估流程结束 ===")

if __name__ == "__main__":
    main() 