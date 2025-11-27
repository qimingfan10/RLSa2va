#!/bin/bash

echo "========================================================================"
echo "Sa2VA真实推理 - 使用训练权重进行实际预测"
echo "========================================================================"

# 设置micromamba路径
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba
export PATH="/home/ubuntu/micromamba/micromamba/bin:$PATH"

# 初始化micromamba
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"

# 激活环境
echo "激活topo-sarl环境..."
micromamba activate topo-sarl

if [ $? -ne 0 ]; then
    echo "❌ 环境激活失败"
    exit 1
fi

echo "✅ 环境激活成功"
echo ""

# 检查GPU
echo "检查GPU状态:"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | head -4
echo ""

# 设置环境变量
export PYTHONPATH="/home/ubuntu/Sa2VA:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/ubuntu/Sa2VA

echo "========================================================================"
echo "开始真实Sa2VA推理..."
echo "========================================================================"
echo ""

python real_sa2va_inference.py

echo ""
echo "========================================================================"
echo "完成！"
echo "========================================================================"
echo ""
echo "查看结果:"
echo "  ls -lh real_sa2va_inference_results/predictions/"
echo "  cat real_sa2va_inference_results/real_inference_results.json"
