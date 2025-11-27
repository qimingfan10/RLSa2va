#!/bin/bash

echo "========================================="
echo "Sa2VA模型推理脚本"
echo "========================================="

# 激活conda/micromamba环境
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# 尝试激活topo-sarl环境
if command -v conda &> /dev/null; then
    echo "使用conda激活环境..."
    conda activate topo-sarl 2>/dev/null || echo "conda环境激活失败"
elif [ -f ~/micromamba/bin/micromamba ]; then
    echo "使用micromamba激活环境..."
    eval "$(~/micromamba/bin/micromamba shell hook --shell bash)"
    micromamba activate topo-sarl 2>/dev/null || echo "micromamba环境激活失败"
fi

echo ""
echo "Python版本:"
python3 --version
echo ""

# 设置环境变量
export PYTHONPATH="/home/ubuntu/Sa2VA:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

cd /home/ubuntu/Sa2VA

echo "开始推理..."
echo ""

python3 inference_with_trained_model.py

echo ""
echo "========================================="
echo "推理完成！"
echo "========================================="
echo "查看结果: ls -lh inference_results/predictions/"
