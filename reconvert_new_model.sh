#!/bin/bash

echo "========================================================================"
echo "重新转换新模型 (iter_3672) - 使用修复版脚本"
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
echo "源checkpoint: work_dirs/merged_vessel_segmentation/iter_3672.pth"
echo "  - 训练时间: Nov 23 21:41"
echo "  - 训练步数: 3672步 (3 epochs)"
echo "  - 配置: sa2va_merged_vessel_finetune.py"
echo ""
echo "目标路径: /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf_fixed"
echo ""
echo "⚠️  重要改进："
echo "  - 使用修复版转换脚本"
echo "  - 不加载Sa2VA-26B.pth预训练权重"
echo "  - 只使用训练checkpoint的权重"
echo "  - 确保保留训练结果的所有差异"
echo ""

echo "========================================================================"
echo "步骤1: 删除旧的转换结果（如果存在）"
echo "========================================================================"
echo ""

if [ -d "/home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf_fixed" ]; then
    echo "发现旧的转换结果，删除中..."
    rm -rf /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf_fixed
    echo "✅ 删除完成"
else
    echo "没有旧的转换结果"
fi

echo ""
echo "========================================================================"
echo "步骤2: 使用修复版脚本转换"
echo "========================================================================"
echo ""

python convert_without_pretrained.py \
    projects/sa2va/configs/sa2va_merged_vessel_finetune.py \
    work_dirs/merged_vessel_segmentation/iter_3672.pth \
    --save-path /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf_fixed

EXIT_CODE=$?

echo ""
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 转换完成！"
    echo "========================================================================"
    echo ""
    echo "新的HF模型: /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf_fixed"
    echo ""
    echo "检查文件:"
    ls -lh /home/ubuntu/Sa2VA/models/sa2va_vessel_iter3672_hf_fixed/ | head -20
    echo ""
    echo "磁盘空间:"
    df -h /home/ubuntu | tail -1
    echo ""
    echo "现在有三个模型:"
    echo "  1. sa2va_vessel_hf (旧，iter_12192，有预训练权重问题) - 30GB"
    echo "  2. sa2va_vessel_iter3672_hf (新，iter_3672，有预训练权重问题) - 30GB"
    echo "  3. sa2va_vessel_iter3672_hf_fixed (新，iter_3672，修复版) - ~30GB ✨"
    echo ""
    echo "下一步: 使用修复版模型重新评估"
else
    echo "❌ 转换失败！"
    echo "========================================================================"
    echo "退出代码: $EXIT_CODE"
    echo ""
    echo "磁盘空间:"
    df -h /home/ubuntu | tail -1
fi
