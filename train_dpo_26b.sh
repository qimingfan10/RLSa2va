#!/bin/bash
#######################################################################
# Sa2VA 26B DPO Training Script
# 在已达到Dice 0.82的26B模型上进行DPO训练
#######################################################################

# 配置
CONFIG="projects/sa2va/configs/sa2va_dpo_finetune_26b.py"
GPUS=4
WORK_DIR="work_dirs/dpo_vessel_training_26b"
LOG_FILE="/home/ubuntu/dpo_training_26b.log"

echo "=========================================="
echo "Sa2VA 26B DPO Training"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "GPU数量: $GPUS"
echo "工作目录: $WORK_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 设置环境
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate topo-sarl

# 设置环境变量
export PYTHONPATH="${PWD}:${PWD}/projects:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 创建工作目录
mkdir -p "$WORK_DIR"

# 记录开始时间
echo "开始时间: $(date)" | tee -a "$LOG_FILE"

# 启动训练
cd /home/ubuntu/Sa2VA

echo "启动26B DPO训练..."
NPROC_PER_NODE=$GPUS python -m xtuner.tools.train \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --deepspeed deepspeed_zero3 \
    2>&1 | tee -a "$LOG_FILE"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 26B DPO训练完成!"
    echo "   模型保存在: $WORK_DIR"
else
    echo ""
    echo "❌ 训练失败，请查看日志: $LOG_FILE"
fi

echo "结束时间: $(date)" | tee -a "$LOG_FILE"
