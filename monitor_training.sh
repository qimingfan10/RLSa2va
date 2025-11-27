#!/bin/bash

# 监控Sa2VA训练进度

LOG_FILE="/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/training_20251123_034229.log"
PID_FILE="/home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation/training.pid"

echo "=========================================="
echo "Sa2VA训练监控"
echo "=========================================="
echo ""

# 检查进程是否在运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 训练进程正在运行 (PID: $PID)"
    else
        echo "❌ 训练进程已停止 (PID: $PID)"
    fi
else
    echo "⚠️  找不到PID文件"
fi

echo ""
echo "最新日志 (最后30行):"
echo "=========================================="
tail -30 "$LOG_FILE"

echo ""
echo "=========================================="
echo "查看完整日志: tail -f $LOG_FILE"
echo "查看训练loss: grep -i 'loss' $LOG_FILE | tail -20"
echo "查看训练进度: grep -E '(epoch|iter)' $LOG_FILE | tail -20"
echo "=========================================="
