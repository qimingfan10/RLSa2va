#!/bin/bash

# 快速验证：阈值扫描实验
# 目标：验证是否只需调整阈值就能提升Dice到0.85+

echo "========================================"
echo "快速验证：阈值扫描实验"
echo "========================================"

# 配置
MODEL_PATH="/home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
DATA_ROOT="/home/ubuntu/Sa2VA/data/merged_vessel_data"
OUTPUT_DIR="/home/ubuntu/Sa2VA/threshold_validation_output"
MAX_SAMPLES=50  # 使用50张图像进行快速验证
GPU=1  # 使用GPU1

# Prompt (可以尝试不同的prompt)
PROMPT="Please segment the blood vessel."

# 阈值范围
MIN_THRESHOLD=0.1
MAX_THRESHOLD=0.9
THRESHOLD_STEP=0.05

echo ""
echo "配置信息:"
echo "  - 模型: $MODEL_PATH"
echo "  - 数据集: $DATA_ROOT"
echo "  - 样本数: $MAX_SAMPLES"
echo "  - GPU: GPU$GPU"
echo "  - Prompt: $PROMPT"
echo "  - 阈值范围: [$MIN_THRESHOLD, $MAX_THRESHOLD)"
echo "  - 阈值步长: $THRESHOLD_STEP"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行验证
echo "开始阈值扫描..."
echo ""

CUDA_VISIBLE_DEVICES=$GPU python3 /home/ubuntu/Sa2VA/quick_threshold_validation.py \
    --model_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --prompt "$PROMPT" \
    --output_dir $OUTPUT_DIR \
    --max_samples $MAX_SAMPLES \
    --min_threshold $MIN_THRESHOLD \
    --max_threshold $MAX_THRESHOLD \
    --threshold_step $THRESHOLD_STEP \
    --gpu 0

echo ""
echo "========================================"
echo "验证完成！"
echo "========================================"
echo ""
echo "结果文件:"
echo "  - JSON: $OUTPUT_DIR/threshold_scan_results.json"
echo "  - 曲线图: $OUTPUT_DIR/threshold_scan_curves.png"
echo ""
echo "查看结果:"
echo "  cat $OUTPUT_DIR/threshold_scan_results.json | jq '.best_threshold'"
echo ""
