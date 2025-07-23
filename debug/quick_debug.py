#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick debug script for Swift Rollout
Simplified version for quick testing
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swift.llm.infer.rollout import rollout_main


def main():
    """Quick debug main function"""
    print("=== Quick Debug Swift Rollout ===")
    
    # 使用与原始命令相同的参数
    args = [
        "--model", "/mnt/data/ai-ground/models/Qwen/Qwen2.5-VL-7B-Instruct"
    ]
    
    print(f"Arguments: {args}")
    print("Starting rollout_main...")
    
    try:
        rollout_main(args)
    except KeyboardInterrupt:
        print("\n✓ Debug session interrupted by user")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 