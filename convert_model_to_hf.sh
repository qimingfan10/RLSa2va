#!/bin/bash

echo "========================================================================"
echo "转换Sa2VA模型为HuggingFace格式"
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
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/ubuntu/Sa2VA

# 检查文件
echo "检查文件..."
if [ ! -f "tools/convert_to_hf.py" ]; then
    echo "❌ 转换脚本不存在: tools/convert_to_hf.py"
    exit 1
fi

if [ ! -f "work_dirs/merged_vessel_segmentation/iter_3672.pth" ]; then
    echo "❌ Checkpoint不存在"
    exit 1
fi

if [ ! -f "projects/sa2va/configs/sa2va_merged_vessel_finetune.py" ]; then
    echo "❌ 配置文件不存在"
    exit 1
fi

echo "✅ 文件检查通过"
echo ""

# 创建输出目录
mkdir -p models/sa2va_vessel_hf

echo "========================================================================"
echo "开始转换..."
echo "========================================================================"
echo ""

python tools/convert_to_hf.py \
    projects/sa2va/configs/sa2va_merged_vessel_finetune.py \
    work_dirs/merged_vessel_segmentation/iter_3672.pth \
    --save-path models/sa2va_vessel_hf

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ 转换成功！"
    echo "========================================================================"
    echo "HuggingFace模型保存在: models/sa2va_vessel_hf"
    echo ""
    ls -lh models/sa2va_vessel_hf/
else
    echo ""
    echo "========================================================================"
    echo "❌ 转换失败"
    echo "========================================================================"
    exit 1
fi
