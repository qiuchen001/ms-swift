# Swift Rollout 调试指南

## 概述

本指南将帮助您使用VSCode调试swift rollout命令。我们提供了多种调试配置，以适应不同的调试需求。

## 调试配置说明

### 1. Debug Swift Rollout (基础配置)
- **用途**: 直接调试rollout.py入口文件
- **特点**: 最接近原始命令的执行方式
- **适用场景**: 快速验证基本功能

### 2. Debug Swift Rollout with Custom Args (自定义参数)
- **用途**: 使用更多自定义参数进行调试
- **特点**: 包含完整的服务器配置参数
- **适用场景**: 需要测试不同配置参数

### 3. Debug Swift CLI Main (CLI主入口)
- **用途**: 调试完整的CLI路由流程
- **特点**: 模拟完整的swift命令执行
- **适用场景**: 调试CLI层面的问题

### 4. Debug SwiftRolloutDeploy (核心类调试)
- **用途**: 直接调试SwiftRolloutDeploy类
- **特点**: 使用自定义调试脚本，提供更精细的控制
- **适用场景**: 深入调试核心逻辑

## 使用方法

### 方法1: 使用VSCode调试面板

1. 打开VSCode
2. 按 `Ctrl+Shift+D` (或 `Cmd+Shift+D` on Mac) 打开调试面板
3. 在调试配置下拉菜单中选择合适的配置
4. 点击绿色的播放按钮开始调试

### 方法2: 使用自定义调试脚本

```bash
# 运行基础调试
python debug_rollout.py --mode step

# 运行main函数调试
python debug_rollout.py --mode main

# 运行断点调试
python debug_rollout.py --mode breakpoints
```

## 关键断点位置

### 1. 参数解析阶段
```python
# 文件: swift/llm/infer/rollout.py
def rollout_main(args: Union[List[str], RolloutArguments, None] = None) -> None:
    SwiftRolloutDeploy(args).main()  # 在这里设置断点
```

### 2. SwiftRolloutDeploy初始化
```python
# 文件: swift/llm/infer/rollout.py
class SwiftRolloutDeploy(SwiftPipeline):
    def __init__(self, args: Union[List[str], RolloutArguments, None] = None):
        super().__init__(args)  # 在这里设置断点
        # ... 初始化代码
```

### 3. FastAPI应用创建
```python
# 文件: swift/llm/infer/rollout.py
def __init__(self, args: Union[List[str], RolloutArguments, None] = None):
    # ...
    self.app = FastAPI(lifespan=self.lifespan)  # 在这里设置断点
    self._register_rl_rollout_app()
```

### 4. 工作进程启动
```python
# 文件: swift/llm/infer/rollout.py
def _start_data_parallel_workers(self):
    for data_parallel_rank in range(self.num_connections):
        # 在这里设置断点，观察进程创建过程
        process = Process(target=worker_func, args=(self.args, data_parallel_rank, self.master_port, child_conn))
```

### 5. uvicorn服务器启动
```python
# 文件: swift/llm/infer/rollout.py
def run(self):
    args = self.args
    uvicorn.run(self.app, host=args.host, port=args.port, log_level=args.log_level)  # 在这里设置断点
```

## 环境变量说明

调试配置中包含了以下环境变量：

```json
{
    "CUDA_VISIBLE_DEVICES": "3",        // 指定GPU设备
    "VIDEO_MAX_PIXELS": "50176",        // 视频最大像素数
    "FPS_MAX_FRAMES": "12",             // 最大帧数
    "MAX_PIXELS": "1003520"             // 最大像素数
}
```

## 调试技巧

### 1. 观察变量值
在调试过程中，您可以在VSCode的调试面板中观察以下关键变量：

- `self.args`: 解析后的参数对象
- `self.app`: FastAPI应用实例
- `self.connections`: 进程间通信连接
- `self.processes`: 工作进程列表
- `self.master_port`: 主端口号

### 2. 单步执行
使用以下快捷键进行单步调试：
- `F10`: 单步跳过 (Step Over)
- `F11`: 单步进入 (Step Into)
- `Shift+F11`: 单步跳出 (Step Out)
- `F5`: 继续执行 (Continue)

### 3. 条件断点
在关键位置设置条件断点，例如：
```python
# 只在特定条件下触发断点
if self.args.model == "/mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct":
    # 设置断点
    pass
```

### 4. 日志输出
在调试过程中，注意观察控制台输出：
- 参数解析结果
- 模型加载状态
- 进程启动信息
- 错误信息

## 常见问题排查

### 1. 模型路径问题
确保模型路径正确且可访问：
```bash
ls -la /mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct
```

### 2. GPU内存问题
如果遇到GPU内存不足，可以调整以下参数：
- `tensor_parallel_size`: 减少张量并行大小
- `data_parallel_size`: 减少数据并行大小
- `max_model_len`: 减少最大模型长度

### 3. 端口冲突问题
如果端口被占用，可以修改端口号：
```bash
--port 8001  # 使用不同的端口
```

### 4. 权限问题
确保有足够的权限访问模型文件和GPU设备：
```bash
# 检查GPU权限
nvidia-smi
# 检查文件权限
ls -la /mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct
```

## 调试脚本使用

`debug_rollout.py` 脚本提供了三种调试模式：

### step模式 (默认)
逐步执行每个阶段，适合理解整个流程：
```bash
python debug_rollout.py --mode step
```

### main模式
直接调用rollout_main函数，适合快速测试：
```bash
python debug_rollout.py --mode main
```

### breakpoints模式
在关键位置设置断点，适合深入调试：
```bash
python debug_rollout.py --mode breakpoints
```

## 总结

通过以上配置和方法，您可以有效地调试swift rollout命令。建议从基础配置开始，逐步深入到更复杂的调试场景。如果遇到特定问题，可以根据错误信息调整调试策略。 