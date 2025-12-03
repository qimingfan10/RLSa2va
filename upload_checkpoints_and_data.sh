#!/bin/bash

# 上传训练checkpoint和数据集到HuggingFace

set -e

LOG_FILE="/home/ubuntu/Sa2VA/upload_extras.log"
cd /home/ubuntu/Sa2VA

echo "========================================================================"
echo "上传Sa2VA Checkpoints和数据集"
echo "========================================================================"
echo ""

# 检查登录
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ 未登录HuggingFace"
    exit 1
fi

echo "✅ 已登录: $(huggingface-cli whoami 2>&1 | grep -v Warning)"
echo ""

# ============================================================================
# 上传Checkpoint 1: iter_12192.pth
# ============================================================================
echo "========================================================================"
echo "1. 上传Checkpoint: iter_12192.pth"
echo "========================================================================"
echo "文件: work_dirs/vessel_segmentation/iter_12192.pth"
echo "大小: $(du -sh work_dirs/vessel_segmentation/iter_12192.pth | cut -f1)"
echo "目标: ly17/sa2va-checkpoints"
echo ""

if [ -f "work_dirs/vessel_segmentation/iter_12192.pth" ]; then
    echo "开始上传..."
    
    huggingface-cli upload \
        ly17/sa2va-checkpoints \
        work_dirs/vessel_segmentation/iter_12192.pth \
        iter_12192.pth \
        --repo-type model \
        --commit-message "Training checkpoint at iteration 12192"
    
    echo ""
    echo "✅ Checkpoint 1上传完成"
    echo ""
else
    echo "⚠️  文件不存在，跳过"
    echo ""
fi

# ============================================================================
# 上传Checkpoint 2: iter_3672.pth
# ============================================================================
echo "========================================================================"
echo "2. 上传Checkpoint: iter_3672.pth"
echo "========================================================================"
echo "文件: work_dirs/merged_vessel_segmentation/iter_3672.pth"
echo "大小: $(du -sh work_dirs/merged_vessel_segmentation/iter_3672.pth | cut -f1)"
echo "目标: ly17/sa2va-checkpoints"
echo ""

if [ -f "work_dirs/merged_vessel_segmentation/iter_3672.pth" ]; then
    echo "开始上传..."
    
    huggingface-cli upload \
        ly17/sa2va-checkpoints \
        work_dirs/merged_vessel_segmentation/iter_3672.pth \
        iter_3672.pth \
        --repo-type model \
        --commit-message "Training checkpoint at iteration 3672"
    
    echo ""
    echo "✅ Checkpoint 2上传完成"
    echo ""
else
    echo "⚠️  文件不存在，跳过"
    echo ""
fi

# ============================================================================
# 上传数据集
# ============================================================================
echo "========================================================================"
echo "3. 上传数据集: Segment_DATA_Merged_512"
echo "========================================================================"
echo "目录: Segment_DATA_Merged_512/"
echo "大小: $(du -sh Segment_DATA_Merged_512 | cut -f1)"
echo "目标: ly17/sa2va-vessel-dataset"
echo ""

if [ -d "Segment_DATA_Merged_512" ]; then
    echo "开始上传..."
    
    huggingface-cli upload \
        ly17/sa2va-vessel-dataset \
        Segment_DATA_Merged_512 \
        . \
        --repo-type dataset \
        --commit-message "Sa2VA OCT vessel segmentation dataset with annotations"
    
    echo ""
    echo "✅ 数据集上传完成"
    echo ""
else
    echo "⚠️  目录不存在，跳过"
    echo ""
fi

# ============================================================================
# 完成
# ============================================================================
echo "========================================================================"
echo "✅ 所有上传完成！"
echo "========================================================================"
echo ""
echo "HuggingFace仓库:"
echo "  1. Checkpoint: https://huggingface.co/ly17/sa2va-checkpoints"
echo "  2. Dataset: https://huggingface.co/ly17/sa2va-vessel-dataset"
echo ""
echo "下载命令:"
echo ""
echo "# 下载checkpoint"
echo "huggingface-cli download ly17/sa2va-checkpoints --local-dir checkpoints"
echo ""
echo "# 下载数据集"
echo "huggingface-cli download ly17/sa2va-vessel-dataset --local-dir dataset --repo-type dataset"
echo ""
echo "========================================================================"
