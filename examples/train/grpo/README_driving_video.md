# 汽车驾驶视频多分类任务 GRPO 训练

本项目实现了基于 Qwen2.5-VL 模型的汽车驾驶视频多分类任务，使用 GRPO (Group Relative Policy Optimization) 训练方法。

## 任务描述

- **训练方法**: GRPO
- **数据类型**: 视频
- **领域**: 汽车驾驶
- **任务类型**: 多分类（一个视频可以属于多个类别）
- **奖励函数设计**: 增加准确率奖励的方差 + 引入推理长度奖励
- **模型**: Qwen2.5-VL

## 文件结构

```
examples/
├── custom/
│   └── dataset.py                    # 自定义数据集处理
├── train/
│   └── grpo/
│       ├── plugin/
│       │   └── plugin.py            # 奖励函数实现
│       ├── driving_video_classification.sh  # 训练脚本
│       └── README_driving_video.md  # 本文件
└── configs/
    └── ds_config_zero2.json         # DeepSpeed 配置
```

## 实现组件

### 1. 自定义数据集 (`dataset.py`)

**类**: `DrivingVideoMultiClassificationPreprocessor`

**功能**:
- 处理汽车驾驶视频多分类数据
- 支持一个视频属于多个类别的场景
- 生成结构化的问答格式

**支持的分类**:
- 白天驾驶 (Day Driving)
- 夜间驾驶 (Night Driving)
- 城市道路 (Urban Road)
- 高速公路 (Highway)
- 乡村道路 (Rural Road)
- 雨天驾驶 (Rainy Driving)
- 雪天驾驶 (Snowy Driving)
- 拥堵交通 (Traffic Jam)
- 停车场景 (Parking)
- 其他 (Other)

**输出格式**:
```
<think>
[详细推理过程，分析光照、道路类型、天气、交通密度等因素，支持多分类判断]
</think>
<answer>
[逗号分隔的多个分类标签，例如："白天驾驶,城市道路,拥堵交通"]
</answer>
```

### 2. 奖励函数 (`plugin.py`)

#### DrivingVideoClassificationReward
- **准确率权重**: 0.7
- **推理长度权重**: 0.3
- **方差惩罚系数**: 0.1
- **特点**: 使用F1分数评估多分类准确率

#### DrivingVideoClassificationRewardV2 (推荐)
- **准确率权重**: 0.6
- **推理长度权重**: 0.2
- **格式权重**: 0.2
- **特点**: 
  - 使用F1分数评估多分类准确率
  - 格式检查
  - 更细致的奖励计算

## 使用方法

### 1. 准备数据集

确保你的数据集包含以下字段:
- `video_path`: 视频文件路径
- `label`: 分类标签（支持多种格式）

**数据集格式示例**:

```json
// 字符串格式（逗号分隔）
{
  "video_path": "/path/to/video1.mp4",
  "label": "白天驾驶,城市道路,拥堵交通"
}

// 列表格式
{
  "video_path": "/path/to/video2.mp4", 
  "label": ["夜间驾驶", "高速公路"]
}

// 单个标签
{
  "video_path": "/path/to/video3.mp4",
  "label": "雨天驾驶"
}
```

### 2. 注册数据集

在 `dataset.py` 中修改数据集ID:
```python
register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/driving_video_classification',
        hf_dataset_id='your_actual_dataset_id',  # 替换为实际数据集ID
        preprocess_func=DrivingVideoMultiClassificationPreprocessor(),
    ))
```

### 3. 运行训练

```bash
# 给脚本执行权限
chmod +x examples/train/grpo/driving_video_classification.sh

# 运行训练
./examples/train/grpo/driving_video_classification.sh
```

### 4. 自定义参数

可以修改训练脚本中的参数:

```bash
# 模型配置
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# 训练参数
BATCH_SIZE=1
LEARNING_RATE=5e-6
MAX_EPOCHS=3

# GRPO 参数
GRPO_BETA=0.1
GRPO_LAMBDA=0.1
GRPO_ALPHA=0.1

# 奖励函数
REWARD_FUNC="driving_video_classification_reward_v2"
```

## 奖励函数详解

### 多分类准确率评估（F1分数）
- **精确率**: 预测正确的标签数 / 预测的总标签数
- **召回率**: 预测正确的标签数 / 真实的总标签数
- **F1分数**: 2 * 精确率 * 召回率 / (精确率 + 召回率)

**示例**:
- 真实标签: ["白天驾驶", "城市道路", "拥堵交通"]
- 预测标签: ["白天驾驶", "城市道路"]
- 精确率: 2/2 = 1.0
- 召回率: 2/3 = 0.67
- F1分数: 2 * 1.0 * 0.67 / (1.0 + 0.67) = 0.80

### 推理长度奖励
- 理想长度: 100-400字符
- 过短 (<50字符): 0.0分
- 过长 (>600字符): 逐渐减少奖励

### 格式奖励
- 正确格式: 1.0分
- 部分正确: 0.5分
- 格式错误: 0.0分

### 方差惩罚
- 计算最近5次预测的F1分数方差
- 方差越大，惩罚越重
- 鼓励模型预测的稳定性

## 多分类任务特点

### 1. 标签格式支持
- **字符串格式**: "标签1,标签2,标签3"
- **列表格式**: ["标签1", "标签2", "标签3"]
- **单个标签**: "标签1"

### 2. 评估指标
- **F1分数**: 平衡精确率和召回率
- **精确率**: 避免过度预测
- **召回率**: 避免遗漏标签

### 3. 实际应用场景
- 一个视频可能同时包含多种驾驶场景
- 例如：白天在城市道路上遇到交通拥堵
- 标签组合：["白天驾驶", "城市道路", "拥堵交通"]

## 注意事项

1. **数据集格式**: 确保数据集支持多标签格式
2. **视频处理**: 模型需要能够处理视频输入
3. **内存需求**: 根据GPU内存调整batch_size
4. **训练时间**: 视频处理较慢，建议使用多GPU
5. **评估指标**: 关注F1分数和推理质量

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 启用gradient_checkpointing
   - 使用DeepSpeed ZeRO-2

2. **数据集加载失败**
   - 检查数据集ID是否正确
   - 确认数据集格式符合要求
   - 验证标签格式是否正确

3. **奖励函数错误**
   - 检查输出格式是否正确
   - 确认标签在预定义范围内
   - 验证多标签解析是否正常

### 调试建议

1. 先用小数据集测试
2. 检查日志输出
3. 验证奖励函数计算
4. 监控训练损失变化
5. 检查F1分数变化趋势

## 扩展功能

### 添加新的分类标签

1. 在 `DrivingVideoMultiClassificationPreprocessor` 中更新prompt
2. 在 `DrivingVideoClassificationRewardV2` 中更新 `valid_categories`
3. 重新训练模型

### 自定义奖励函数

1. 继承 `ORM` 类
2. 实现 `__call__` 方法
3. 在 `plugin.py` 中注册
4. 在训练脚本中指定

## 联系方式

如有问题，请提交Issue或联系开发团队。 