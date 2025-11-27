#!/bin/bash

echo "========================================================================"
echo "Sa2VA多GPU推理 - 使用4张GPU"
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
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# 设置环境变量
export PYTHONPATH="/home/ubuntu/Sa2VA:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/ubuntu/Sa2VA

echo "========================================================================"
echo "开始多GPU推理..."
echo "========================================================================"
echo ""

python multi_gpu_inference.py

echo ""
echo "========================================================================"
echo "完成！"
echo "========================================================================"
echo ""
echo "查看结果:"
echo "  ls -lh multi_gpu_inference_results/predictions/"
echo "  cat multi_gpu_inference_results/evaluation_results.json"
