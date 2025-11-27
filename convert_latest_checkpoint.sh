#!/bin/bash

echo "========================================================================"
echo "将最新训练checkpoint (iter_3672.pth) 转换为HuggingFace格式"
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
echo "检查源文件..."
echo "========================================================================"
echo ""
echo "配置文件: projects/sa2va/configs/sa2va_merged_vessel_finetune.py"
ls -lh projects/sa2va/configs/sa2va_merged_vessel_finetune.py
echo ""
echo "源checkpoint: work_dirs/merged_vessel_segmentation/iter_3672.pth"
ls -lh work_dirs/merged_vessel_segmentation/iter_3672.pth
echo ""
echo "训练时间: Nov 23 21:41"
echo "训练步数: 3672步"
echo "训练epoch: 3个"
echo ""

echo "目标目录: work_dirs/merged_vessel_segmentation_iter3672_hf"
echo ""

echo "========================================================================"
echo "开始转换模型..."
echo "========================================================================"
echo ""

python tools/convert_to_hf.py \
    projects/sa2va/configs/sa2va_merged_vessel_finetune.py \
    work_dirs/merged_vessel_segmentation/iter_3672.pth \
    --save-path work_dirs/merged_vessel_segmentation_iter3672_hf

EXIT_CODE=$?

echo ""
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 转换完成！"
    echo "========================================================================"
    echo ""
    echo "HuggingFace模型保存在: work_dirs/merged_vessel_segmentation_iter3672_hf"
    echo ""
    echo "检查转换结果:"
    ls -lh work_dirs/merged_vessel_segmentation_iter3672_hf/
    echo ""
    echo "下一步: 使用新转换的模型进行推理"
else
    echo "❌ 转换失败！"
    echo "========================================================================"
    echo ""
    echo "可能的原因:"
    echo "  1. 磁盘空间不足（需要约40GB）"
    echo "  2. 配置文件不匹配"
    echo "  3. checkpoint文件损坏"
    echo ""
    echo "当前磁盘空间:"
    df -h /home/ubuntu
fi
