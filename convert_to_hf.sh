#!/bin/bash

echo "========================================================================"
echo "将训练checkpoint转换为HuggingFace格式"
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
echo "开始转换模型..."
echo "========================================================================"
echo ""
echo "源checkpoint: /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth"
echo "目标目录: /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation_hf"
echo ""

python tools/convert_to_hf.py \
    projects/sa2va/configs/sa2va_merged_vessel_finetune.py \
    /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/iter_3672.pth \
    --save-path /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation_hf

echo ""
echo "========================================================================"
echo "转换完成！"
echo "========================================================================"
echo ""
echo "HuggingFace模型保存在: /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation_hf"
