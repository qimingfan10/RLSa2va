#!/bin/bash

echo "========================================================================"
echo "Sa2VA视频预测 - 生成对比MP4视频"
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
echo "开始视频预测..."
echo "========================================================================"
echo ""

python predict_video.py

echo ""
echo "========================================================================"
echo "完成！"
echo "========================================================================"
echo ""
echo "查看结果:"
echo "  ls -lh video_prediction_results/"
echo "  cat video_prediction_results/video_evaluation_report.md"
echo ""
echo "播放视频:"
echo "  ffplay video_prediction_results/comparison.mp4"
echo "  # 或下载到本地使用视频播放器查看"
