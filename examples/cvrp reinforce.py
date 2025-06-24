import torch
from rl4co.utils.trainer import RL4COTrainer

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.models.zoo.symnco import SymNCO

import numpy as np
import matplotlib.pyplot as plt
import os
torch.set_printoptions(threshold=np.inf)

def save_route_image(env, td, actions, filename, dpi=300, figsize=(8, 8)):
    """
    保存路线图片到文件
    
    Args:
        env: 环境对象
        td: TensorDict数据
        actions: 动作序列
        filename: 保存的文件名
        dpi: 图片分辨率
        figsize: 图片大小
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 调用render函数
    env.render(td, actions, ax=ax)
    
    # 保存图片
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"图片已保存为: {filename}")
    
    # 关闭图形以释放内存
    plt.close()

def save_route_images_batch(env, td_batch, actions_batch, save_dir="output_images", 
                           prefix="route", formats=['png', 'pdf'], dpi=300, figsize=(8, 8)):
    """
    批量保存路线图片
    
    Args:
        env: 环境对象
        td_batch: TensorDict批次数据
        actions_batch: 动作批次数据
        save_dir: 保存目录
        prefix: 文件名前缀
        formats: 保存的图片格式列表
        dpi: 图片分辨率
        figsize: 图片大小
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(actions_batch)):
        # 为每种格式保存图片
        for fmt in formats:
            filename = os.path.join(save_dir, f'{prefix}_{i}.{fmt}')
            save_route_image(env, td_batch[i], actions_batch[i].cpu(), filename, dpi, figsize)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建标准CVRP环境
env = CVRPEnv(
    generator_params=dict(num_loc=100, capacity=50),
    num_samples={"train": 10000, "val": 1000, "test": 1000}
)

 # 创建SymNCO模型
model = SymNCO(
    env,
    baseline = "symnco",
    train_data_size=10000,     # 训练数据大小
    val_data_size=1000,        # 验证数据大小
    batch_size=64,             # 批量大小
    optimizer_kwargs={'lr': 1e-4, 'weight_decay': 0},
)

# 创建训练器
trainer = RL4COTrainer(
    max_epochs=50,            
    accelerator="gpu",        # 使用GPU加速
    devices=1,                # 使用1个GPU设备
    logger=None,
)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_path):
    # 创建模型并加载权重
    test_model = SymNCO.load_from_checkpoint(model_path, baseline="symnco").to(device)
    print(f"模型已加载: {model_path}")
    return test_model

def test_model(model, env, td_init, save_dir="output_images"):
    # 使用训练好的模型进行测试
    model = model.to(device)
    out = model(td_init_test.clone(), phase="test", decode_type="sample", return_actions=True)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 打印结果
    actions = out["actions"]
    print("平均路线长度: {:.2f}".format(-out['reward'].mean().item()))
    for i in range(min(3, len(actions))):
        print(f"路线 {i} 长度: {-out['reward'][i].item():.2f}")
        
        # 保存路线图片
        filename = os.path.join(save_dir, f'route_{i}.png')
        save_route_image(env, td_init[i], actions[i].cpu(), filename)


if __name__ == "__main__":
    is_train = True

    # 创建测试数据
    td_test_data = env.generator(batch_size=[3])
    td_init = env.reset(td_test_data.clone()).to(device)
    td_init_test = td_init.clone()

     # 训练模型
    test_model(model, env, td_init_test, save_dir="results/original")
    if is_train:
        trainer.fit(model)
        trainer.save_checkpoint("./model/cvrp50-symnco.ckpt")

    # 取消下面的注释来加载和运行已保存的模型
    loaded_model = load_model("./model/cvrp50-symnco.ckpt")

    # 测试模型并保存图片
    # print("测试加载的模型...")
    test_model(loaded_model, env, td_init_test, save_dir="results")
    
    # 也可以使用批量保存函数
    # print("使用批量保存函数...")
    # out = loaded_model(td_init_test.clone(), phase="test", decode_type="greedy", return_actions=True)
    # save_route_images_batch(env, td_init_test, out["actions"], 
    #                        save_dir="results/batch", 
    #                        prefix="cvrp_route", 
    #                        formats=['png', 'pdf'])
    
    print("--------------------------------")
    # test_model(loaded_model, env, td_init_test)
