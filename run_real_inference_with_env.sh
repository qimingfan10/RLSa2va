#!/bin/bash

echo "========================================================================"
echo "Sa2VA真实权重推理 - 使用topo-sarl环境"
echo "========================================================================"

# 设置micromamba路径
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba
export PATH="/home/ubuntu/micromamba/micromamba/bin:$PATH"

# 初始化micromamba
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"

# 激活环境
echo "激活topo-sarl环境..."
micromamba activate topo-sarl

if [ $? -eq 0 ]; then
    echo "✅ 环境激活成功"
    echo ""
    echo "Python版本:"
    python --version
    echo ""
    echo "检查关键包:"
    python -c "import mmengine; print('mmengine:', mmengine.__version__)" 2>/dev/null || echo "mmengine: 未安装"
    python -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || echo "torch: 未安装"
    echo ""
else
    echo "❌ 环境激活失败"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="/home/ubuntu/Sa2VA:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

cd /home/ubuntu/Sa2VA

echo "========================================================================"
echo "开始推理..."
echo "========================================================================"
echo ""

python real_model_inference.py

echo ""
echo "========================================================================"
echo "完成！"
echo "========================================================================"
