#!/bin/bash

# Sa2VA血管分割训练脚本 - 优化显存使用
# 使用4个GPU，更激进的显存分散策略

echo "=========================================="
echo "Sa2VA血管分割训练 - 显存优化版"
echo "=========================================="
echo ""

# 激活conda环境
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate topo-sarl

# 清理GPU缓存
echo "清理GPU缓存..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# 设置环境变量以优化显存使用
export PYTHONPATH=/home/ubuntu/Sa2VA:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# PyTorch显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 减少NCCL的显存使用
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

# DeepSpeed优化
export DEEPSPEED_ZERO_OPTIMIZATION=1

echo "配置信息:"
echo "  数据集: Merged (1220张图片，坐标已缩放)"
echo "  GPU数量: 4"
echo "  Batch size: 1 per GPU"
echo "  梯度累积: 16步"
echo "  有效batch size: 64"
echo "  训练轮数: 3 epochs"
echo "  显存优化: 激进CPU offload + 碎片整理"
echo ""

# 配置文件路径
CONFIG_FILE="/home/ubuntu/Sa2VA/projects/sa2va/configs/sa2va_merged_vessel_finetune_optimized.py"

# 工作目录
WORK_DIR="/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation_optimized"

# 创建工作目录
mkdir -p $WORK_DIR

echo "开始训练..."
echo "配置文件: $CONFIG_FILE"
echo "工作目录: $WORK_DIR"
echo ""

# 显示GPU状态
echo "训练前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s MB / %s MB (%.1f%%)\n", $1, $3, $4, ($3/$4)*100}'
echo ""

# 使用xtuner训练
NPROC_PER_NODE=4 \
python -m xtuner.tools.train $CONFIG_FILE \
    --work-dir $WORK_DIR \
    --deepspeed deepspeed_zero3 \
    2>&1 | tee $WORK_DIR/training.log

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
else
    echo "❌ 训练失败 (退出码: $EXIT_CODE)"
fi
echo "=========================================="
echo "检查点保存在: $WORK_DIR"
echo "训练日志: $WORK_DIR/training.log"
echo ""
echo "最终GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s MB / %s MB (%.1f%%)\n", $1, $3, $4, ($3/$4)*100}'
