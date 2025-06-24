# CVRP偏好优化系统使用指南

本系统实现了基于人类反馈的CVRP（容量约束车辆路径问题）偏好优化，通过学习人类偏好来改进路线规划算法。

## 🎯 系统概述

偏好优化系统包含以下核心组件：

1. **人类反馈收集器** (`collect_feedback.py`)
2. **偏好模型训练器** (`train_preference.py`)
3. **高级偏好学习系统** (`advanced_preference_learning.py`)

## 🚀 快速开始

### 步骤1: 收集人类反馈

```bash
python examples/collect_feedback.py
```

这个脚本会：
- 加载预训练的CVRP模型
- 生成多个CVRP问题的不同解
- 展示对比图片供你选择偏好
- 保存你的反馈到 `human_feedback` 目录

**使用说明：**
- 系统会生成两个不同的路线解
- 查看对比图片，选择你更喜欢的解（1或2）
- 可以输入 `skip` 跳过某个对比
- 建议收集10-20个反馈对以获得好的效果

### 步骤2: 训练偏好模型

```bash
python examples/train_preference.py
```

这个脚本会：
- 加载收集的人类反馈数据
- 训练偏好奖励模型
- 生成训练历史图表
- 测试偏好优化效果
- 保存结果到 `preference_results` 目录

## 📊 系统架构

### 核心组件

#### 1. 偏好数据集 (`PreferenceDataset`)
```python
class PreferenceDataset(Dataset):
    """存储人类反馈的偏好对"""
    def add_preference(self, state, action1, action2, preference)
```

#### 2. 偏好奖励模型 (`PreferenceRewardModel`)
```python
class PreferenceRewardModel(nn.Module):
    """学习从状态和动作预测奖励的神经网络"""
    def forward(self, state, action) -> reward
```

#### 3. 偏好训练器 (`PreferenceTrainer`)
```python
class PreferenceTrainer:
    """使用人类反馈训练偏好模型"""
    def train(self, dataloader, epochs)
    def plot_training_history(self, save_path)
```

#### 4. 偏好优化模型 (`PreferenceOptimizedModel`)
```python
class PreferenceOptimizedModel:
    """结合原始模型和偏好奖励的优化模型"""
    def forward(self, td, ...) -> optimized_output
```

## 🔧 配置参数

### 基础配置
```python
class Config:
    learning_rate = 1e-4      # 学习率
    max_epochs = 50           # 训练轮数
    batch_size = 16           # 批次大小
    temperature = 0.1         # 温度参数
    hidden_dim = 256          # 隐藏层维度
    embed_dim = 128           # 嵌入维度
```

### 偏好学习参数
- `preference_weight`: 偏好奖励的权重 (默认0.1)
- `temperature`: 偏好损失的温度参数 (默认0.1)
- `max_dataset_size`: 数据集最大大小 (默认10000)

## 📈 训练过程

### 1. 数据收集阶段
- 生成多样化的CVRP解
- 收集人类偏好反馈
- 保存反馈数据为JSON格式

### 2. 模型训练阶段
- 加载人类反馈数据
- 训练偏好奖励模型
- 监控训练指标（损失、准确率等）

### 3. 模型评估阶段
- 比较原始模型和偏好优化模型
- 生成对比结果图片
- 分析偏好优化效果

## 📁 输出文件结构

```
project/
├── human_feedback/              # 人类反馈数据
│   ├── comparison_0.png         # 对比图片
│   ├── comparison_1.png
│   └── feedback_20231201_143022.json
├── preference_results/          # 偏好优化结果
│   ├── preference_reward_model.pth
│   ├── comparison_result_0.png
│   └── comparison_result_1.png
└── preference_training_history.png  # 训练历史图
```

## 🎨 偏好选择指南

在收集人类反馈时，可以考虑以下因素：

### 路线质量指标
- **总距离**: 路线总长度越短越好
- **避免交叉**: 路线交叉越少越好
- **直观性**: 路线更符合人类直觉
- **平衡性**: 各车辆负载相对均衡

### 实际应用考虑
- **时间窗口**: 考虑客户的时间约束
- **优先级**: 重要客户优先服务
- **区域划分**: 按地理区域组织路线
- **车辆特性**: 考虑车辆类型和容量

## 🔍 结果分析

### 训练指标
- **损失曲线**: 训练损失随时间变化
- **准确率**: 偏好预测准确率
- **奖励分布**: 不同解的奖励分布
- **奖励差异**: 偏好解与非偏好解的奖励差异

### 效果评估
- **路线长度对比**: 优化前后的平均路线长度
- **视觉质量**: 路线的直观性和美观性
- **用户满意度**: 符合人类偏好的程度

## 🛠️ 高级功能

### 1. 多样化解生成
系统会生成多种不同的解：
- 贪婪解码解
- 采样解码解
- 添加噪声的解

### 2. 智能对比选择
- 自动选择最不同的解进行对比
- 计算解的多样性分数
- 确保对比的有效性

### 3. 训练历史可视化
- 损失曲线
- 准确率变化
- 奖励分布
- 奖励差异趋势

## 🚨 故障排除

### 常见问题

1. **没有找到反馈文件**
   ```
   解决方案: 先运行 collect_feedback.py 收集反馈
   ```

2. **数据集太小**
   ```
   解决方案: 收集更多的反馈对（建议10-20个）
   ```

3. **训练不收敛**
   ```
   解决方案: 
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量
   ```

4. **内存不足**
   ```
   解决方案:
   - 减小批次大小
   - 减少模型维度
   - 使用GPU加速
   ```

### 性能优化

1. **GPU加速**: 确保使用GPU进行训练
2. **数据预处理**: 批量处理提高效率
3. **模型缓存**: 保存中间结果避免重复计算

## 📚 技术细节

### 偏好学习算法

#### 1. 直接偏好优化 (DPO)
```python
# 计算偏好损失
logits = (rewards1 - rewards2) / temperature
loss = F.binary_cross_entropy_with_logits(logits, preferences)
```

#### 2. 奖励模型架构
- 状态编码器: 编码CVRP问题状态
- 动作编码器: 编码路线动作序列
- 奖励头: 预测偏好奖励

#### 3. 模型融合
```python
# 组合原始奖励和偏好奖励
combined_rewards = base_reward + preference_weight * preference_reward
```

## 🔮 扩展功能

### 1. 多目标优化
- 结合距离、时间、成本等多个目标
- 使用加权方法平衡不同目标

### 2. 在线学习
- 实时收集用户反馈
- 动态更新偏好模型

### 3. 个性化偏好
- 学习不同用户的偏好模式
- 提供个性化的路线建议

### 4. 解释性分析
- 分析偏好模型的学习模式
- 提供偏好决策的解释

## 📞 技术支持

如果遇到问题，请检查：
1. 依赖包是否正确安装
2. 模型文件是否存在
3. 数据格式是否正确
4. 硬件资源是否充足

## 🎉 总结

偏好优化系统通过结合人类反馈和机器学习，能够生成更符合人类偏好的CVRP解。系统具有良好的可扩展性和实用性，可以应用于实际的物流优化场景。 