# Swift Rollout 调试故障排除指南

## 问题分析

您遇到的错误是：
```
ValueError: Please set --model <model_id_or_path>`, model: None
```

这个错误表明在创建 `RolloutArguments` 实例时，没有正确传递 `--model` 参数。

## 解决方案

### 方案1: 使用修复后的调试脚本

我已经修复了 `debug_rollout.py` 脚本，现在它包含：
- 参数验证
- 更好的错误处理
- 完整的参数列表

### 方案2: 使用简化的快速调试脚本

我创建了一个新的 `quick_debug.py` 脚本，它：
- 更简单直接
- 使用与原始命令相同的参数
- 减少复杂性

## 推荐的调试入口

### 1. 使用 Quick Debug Swift Rollout (推荐)

在VSCode中选择 "Quick Debug Swift Rollout" 配置，它会：
- 使用 `quick_debug.py` 作为入口
- 在入口处停止，让您逐步调试
- 包含所有必要的环境变量

### 2. 直接运行快速调试脚本

```bash
python quick_debug.py
```

### 3. 使用修复后的详细调试脚本

```bash
python debug_rollout.py --mode step
```

## 调试步骤

### 第一步：验证模型路径

确保模型路径存在且可访问：
```bash
ls -la /mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct
```

### 第二步：检查环境变量

确保环境变量正确设置：
```bash
echo $CUDA_VISIBLE_DEVICES
echo $VIDEO_MAX_PIXELS
echo $FPS_MAX_FRAMES
echo $MAX_PIXELS
```

### 第三步：开始调试

1. 在VSCode中打开调试面板 (`Ctrl+Shift+D`)
2. 选择 "Quick Debug Swift Rollout" 配置
3. 点击绿色播放按钮开始调试
4. 程序会在 `quick_debug.py` 的 `main()` 函数处停止

### 第四步：逐步调试

使用以下快捷键：
- `F10`: 单步跳过
- `F11`: 单步进入
- `F5`: 继续执行

## 关键断点位置

在调试过程中，建议在以下位置设置断点：

1. **`swift/llm/infer/rollout.py:347`**
   ```python
   def rollout_main(args: Union[List[str], RolloutArguments, None] = None) -> None:
       SwiftRolloutDeploy(args).main()  # 设置断点
   ```

2. **`swift/llm/infer/rollout.py:148`**
   ```python
   def __init__(self, args: Union[List[str], RolloutArguments, None] = None):
       super().__init__(args)  # 设置断点
   ```

3. **`swift/llm/infer/rollout.py:160`**
   ```python
   def _start_data_parallel_workers(self):
       for data_parallel_rank in range(self.num_connections):
           # 设置断点，观察进程创建
   ```

4. **`swift/llm/infer/rollout.py:339`**
   ```python
   def run(self):
       uvicorn.run(self.app, host=args.host, port=args.port, log_level=args.log_level)  # 设置断点
   ```

## 常见问题解决

### 问题1: 模型路径不存在
```bash
# 检查模型路径
ls -la /mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct

# 如果路径不存在，修改quick_debug.py中的路径
```

### 问题2: GPU内存不足
```python
# 在quick_debug.py中添加更多参数
args = [
    "--model", "/mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct",
    "--tensor_parallel_size", "1",
    "--data_parallel_size", "1",
    "--max_model_len", "2048"  # 减少最大模型长度
]
```

### 问题3: 端口被占用
```python
# 修改端口号
args = [
    "--model", "/mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct",
    "--port", "8001"  # 使用不同的端口
]
```

### 问题4: 权限问题
```bash
# 检查GPU权限
nvidia-smi

# 检查文件权限
ls -la /mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct
```

## 调试技巧

### 1. 观察变量值
在VSCode调试面板中观察：
- `args`: 传入的参数列表
- `rollout_args`: 解析后的参数对象
- `deploy`: SwiftRolloutDeploy实例

### 2. 使用条件断点
在关键位置设置条件断点：
```python
# 只在特定条件下触发断点
if "Qwen2.5-VL-7B-Instruct" in str(args):
    # 设置断点
    pass
```

### 3. 日志输出
注意观察控制台输出：
- 参数解析结果
- 模型加载状态
- 进程启动信息
- 错误信息

## 总结

现在您应该使用 **"Quick Debug Swift Rollout"** 配置作为起始调试入口，它会：
1. 使用修复后的参数处理
2. 提供清晰的调试流程
3. 包含所有必要的环境变量
4. 在入口处停止，让您逐步调试

如果仍然遇到问题，请检查：
1. 模型路径是否正确
2. 环境变量是否正确设置
3. GPU资源是否可用
4. 文件权限是否正确 