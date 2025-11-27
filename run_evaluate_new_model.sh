#!/bin/bash

echo "========================================================================"
echo "使用新模型 (iter_3672) 进行推理评估"
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

# 设置环境变量
export PYTHONPATH="/home/ubuntu/Sa2VA:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/ubuntu/Sa2VA

echo "========================================================================"
echo "模型信息"
echo "========================================================================"
echo ""
echo "新模型: /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf"
echo "  - 源checkpoint: iter_3672.pth"
echo "  - 训练时间: Nov 23 21:41"
echo "  - 训练步数: 3672步 (3 epochs)"
echo "  - Loss: 13.76 → 1.08"
echo ""
echo "旧模型: /home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
echo "  - 源checkpoint: iter_12192.pth"
echo "  - 训练时间: Nov 22 09:09"
echo "  - 训练步数: 12192步"
echo ""

echo "========================================================================"
echo "开始评估..."
echo "========================================================================"
echo ""

python evaluate_new_model.py

echo ""
echo "========================================================================"
echo "完成！"
echo "========================================================================"
echo ""
echo "查看结果:"
echo "  ls -lh new_model_evaluation_results/predictions/"
echo "  cat new_model_evaluation_results/new_model_evaluation_results.json"
echo "  cat new_model_evaluation_results/new_model_evaluation_report.md"
