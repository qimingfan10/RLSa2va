#!/bin/bash

echo "========================================================================"
echo "转换最新训练的模型 (iter_3672.pth) 到新目录"
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

cd /home/ubuntu/Sa2VA

echo "========================================================================"
echo "模型信息"
echo "========================================================================"
echo ""
echo "旧模型 (保留): /home/ubuntu/Sa2VA/models/sa2va_vessel_hf"
echo "  - 来源: iter_12192.pth (Nov 22)"
echo "  - 大小: 30GB"
echo "  - 状态: 保留不动"
echo ""
echo "新模型 (待转换): work_dirs/merged_vessel_segmentation/iter_3672.pth"
echo "  - 训练时间: Nov 23 21:41"
echo "  - 训练步数: 3672步 (3 epochs)"
echo "  - 配置: sa2va_merged_vessel_finetune.py"
echo "  - 目标: /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf"
echo ""

echo "当前磁盘空间:"
df -h /home/ubuntu | tail -1
echo ""

echo "========================================================================"
echo "开始转换..."
echo "========================================================================"
echo ""

python tools/convert_to_hf.py \
    projects/sa2va/configs/sa2va_merged_vessel_finetune.py \
    work_dirs/merged_vessel_segmentation/iter_3672.pth \
    --save-path /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf

EXIT_CODE=$?

echo ""
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 转换完成！"
    echo "========================================================================"
    echo ""
    echo "新HuggingFace模型保存在: /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf"
    echo ""
    echo "检查新模型:"
    ls -lh /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf/ | head -20
    echo ""
    echo "磁盘空间:"
    df -h /home/ubuntu | tail -1
    echo ""
    echo "现在有两个模型:"
    echo "  1. sa2va_vessel_hf (旧，iter_12192) - 30GB"
    echo "  2. sa2va_vessel_iter3672_hf (新，iter_3672) - ~35GB"
else
    echo "❌ 转换失败！"
    echo "========================================================================"
    echo "退出代码: $EXIT_CODE"
    echo ""
    echo "磁盘空间:"
    df -h /home/ubuntu | tail -1
fi
