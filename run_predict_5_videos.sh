#!/bin/bash

echo "========================================================================"
echo "Sa2VA 5个视频序列预测"
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
echo "开始预测5个视频序列"
echo "========================================================================"
echo ""
echo "配置:"
echo "  - 模型: models/sa2va_vessel_hf"
echo "  - 数据: data/merged_vessel_data/"
echo "  - 输出: video_prediction_5_videos/"
echo "  - 视频数量: 5"
echo ""
echo "⚠️  预计需要时间: 10-15分钟"
echo ""

python predict_5_videos.py 2>&1 | tee video_prediction_5_videos.log

EXIT_CODE=$?

echo ""
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 5个视频预测完成！"
    echo "========================================================================"
    echo ""
    echo "查看结果:"
    echo "  1. 综合报告:"
    echo "     cat video_prediction_5_videos/SUMMARY_5_VIDEOS.md"
    echo ""
    echo "  2. JSON数据:"
    echo "     cat video_prediction_5_videos/summary_5_videos.json"
    echo ""
    echo "  3. 各视频MP4:"
    echo "     ls video_prediction_5_videos/video_*/*.mp4"
    echo ""
    echo "  4. 完整日志:"
    echo "     cat video_prediction_5_videos.log"
    echo ""
else
    echo "❌ 预测失败！"
    echo "========================================================================"
    echo "退出代码: $EXIT_CODE"
    echo ""
    echo "查看日志: cat video_prediction_5_videos.log"
fi
