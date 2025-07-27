#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script for Swift Rollout
This script allows you to debug the SwiftRolloutDeploy class step by step
"""

import os
import sys
from typing import List, Union

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swift.llm.infer.rollout import SwiftRolloutDeploy, rollout_main
from swift.llm import RolloutArguments


def debug_rollout_step_by_step():
    """Debug the rollout process step by step"""
    print("=== Starting Swift Rollout Debug ===")
    
    # 1. 创建参数
    print("\n1. Creating RolloutArguments...")
    args = [
        "--model", "/mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor_parallel_size", "1",
        "--data_parallel_size", "1"
    ]
    
    # 2. 解析参数
    print("2. Parsing arguments...")
    rollout_args = RolloutArguments()
    rollout_args, remaining = rollout_args.parse_args_into_dataclasses(return_remaining_strings=True, args=args)
    print(f"Parsed args: {rollout_args}")
    print(f"Remaining args: {remaining}")
    
    # 验证必需参数
    if not rollout_args.model:
        print("✗ Error: --model parameter is required")
        print("Please provide a valid model path or ID")
        return
    
    # 3. 创建SwiftRolloutDeploy实例
    print("\n3. Creating SwiftRolloutDeploy instance...")
    try:
        deploy = SwiftRolloutDeploy(rollout_args)
        print("✓ SwiftRolloutDeploy created successfully")
        print(f"  - App: {type(deploy.app)}")
        print(f"  - Master port: {deploy.master_port}")
        print(f"  - Connections: {len(deploy.connections)}")
        print(f"  - Processes: {len(deploy.processes)}")
    except Exception as e:
        print(f"✗ Error creating SwiftRolloutDeploy: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 运行主程序
    print("\n4. Running main program...")
    try:
        deploy.main()
    except KeyboardInterrupt:
        print("\n✓ Debug session interrupted by user")
    except Exception as e:
        print(f"✗ Error in main: {e}")
        import traceback
        traceback.print_exc()


def debug_rollout_main():
    """Debug the rollout_main function directly"""
    print("=== Starting rollout_main Debug ===")
    
    args = [
        "--model", "/mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor_parallel_size", "1",
        "--data_parallel_size", "1"
    ]
    
    try:
        rollout_main(args)
    except KeyboardInterrupt:
        print("\n✓ Debug session interrupted by user")
    except Exception as e:
        print(f"✗ Error in rollout_main: {e}")
        import traceback
        traceback.print_exc()


def debug_with_breakpoints():
    """Debug with specific breakpoints"""
    print("=== Starting Debug with Breakpoints ===")
    
    # 设置断点位置
    print("Setting up breakpoints...")
    
    # 1. 在参数解析处设置断点
    print("Breakpoint 1: Argument parsing")
    args = [
        "--model", "/mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor_parallel_size", "1",
        "--data_parallel_size", "1"
    ]
    
    # 2. 在SwiftRolloutDeploy初始化处设置断点
    print("Breakpoint 2: SwiftRolloutDeploy initialization")
    rollout_args = RolloutArguments()
    rollout_args, remaining = rollout_args.parse_args_into_dataclasses(return_remaining_strings=True, args=args)
    
    # 验证必需参数
    if not rollout_args.model:
        print("✗ Error: --model parameter is required")
        return
    
    # 3. 在FastAPI应用创建处设置断点
    print("Breakpoint 3: FastAPI app creation")
    deploy = SwiftRolloutDeploy(rollout_args)
    
    # 4. 在uvicorn启动处设置断点
    print("Breakpoint 4: uvicorn server start")
    deploy.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Swift Rollout")
    parser.add_argument("--mode", choices=["step", "main", "breakpoints"], 
                       default="step", help="Debug mode")
    
    args = parser.parse_args()
    
    if args.mode == "step":
        debug_rollout_step_by_step()
    elif args.mode == "main":
        debug_rollout_main()
    elif args.mode == "breakpoints":
        debug_with_breakpoints()
    else:
        print("Invalid mode. Using step-by-step debug.")
        debug_rollout_step_by_step() 