#!/bin/bash

# Sa2VA血管分割训练脚本 - Merged Dataset (1220张图片)
# 使用4个GPU进行训练

echo "=========================================="
echo "Sa2VA血管分割训练 - Merged Dataset"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  数据集: Segment_DATA_Merged_512 (1220张图片)"
echo "  GPU数量: 4"
echo "  Batch size: 1 per GPU"
echo "  梯度累积: 8步"
echo "  有效batch size: 32"
echo "  训练轮数: 3 epochs"
echo "  学习率: 2e-5"
echo ""

# 激活conda环境
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate topo-sarl

# 设置环境变量
export PYTHONPATH=/home/ubuntu/Sa2VA:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 配置文件路径
CONFIG_FILE="/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_merged_vessel_finetune.py"

# 工作目录
WORK_DIR="/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation"

# 创建工作目录
mkdir -p $WORK_DIR

echo "开始训练..."
echo "配置文件: $CONFIG_FILE"
echo "工作目录: $WORK_DIR"
echo ""

# 使用xtuner训练
NPROC_PER_NODE=4 \
python -m xtuner.tools.train $CONFIG_FILE \
    --work-dir $WORK_DIR \
    --deepspeed deepspeed_zero3 \
    2>&1 | tee $WORK_DIR/training.log

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "检查点保存在: $WORK_DIR"
echo "训练日志: $WORK_DIR/training.log"
