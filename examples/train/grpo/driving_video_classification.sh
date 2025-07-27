#!/bin/bash

# 汽车驾驶视频多分类任务 GRPO 训练脚本
# 使用 Qwen2.5-VL 模型

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 基础配置
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_NAME="swift/driving_video_classification"
OUTPUT_DIR="./outputs/driving_video_classification_grpo"

# 训练参数
BATCH_SIZE=1
LEARNING_RATE=5e-6
MAX_LENGTH=2048
MAX_EPOCHS=3
SAVE_STEPS=100
LOGGING_STEPS=10

# GRPO 特定参数
GRPO_BETA=0.1
GRPO_LAMBDA=0.1
GRPO_ALPHA=0.1

# 奖励函数配置
REWARD_FUNC="driving_video_classification_reward_v2"
EXTERNAL_PLUGINS="examples/train/grpo/plugin/plugin.py"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行 GRPO 训练
swift llm train \
    --model_name_or_path $MODEL_NAME \
    --dataset $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $MAX_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 2 \
    --evaluation_strategy "steps" \
    --eval_steps $SAVE_STEPS \
    --load_best_model_at_end \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --bf16 true \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --ddp_timeout 180000000 \
    --gradient_checkpointing true \
    --ddp_find_unused_parameters false \
    --report_to none \
    --deepspeed configs/ds_config_zero2.json \
    --trainer grpo \
    --grpo_beta $GRPO_BETA \
    --grpo_lambda $GRPO_LAMBDA \
    --grpo_alpha $GRPO_ALPHA \
    --external_plugins $EXTERNAL_PLUGINS \
    --reward_funcs $REWARD_FUNC \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed 42

echo "训练完成！模型保存在: $OUTPUT_DIR" 