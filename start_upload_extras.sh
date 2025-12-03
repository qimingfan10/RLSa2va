#!/bin/bash

# 后台上传checkpoints和数据集

LOG_FILE="/home/ubuntu/Sa2VA/upload_extras.log"
SCRIPT="/home/ubuntu/Sa2VA/upload_checkpoints_and_data.sh"
PID_FILE="/home/ubuntu/Sa2VA/upload_extras.pid"

echo "========================================================================"
echo "启动Checkpoints和数据集后台上传"
echo "========================================================================"
echo ""

cd /home/ubuntu/Sa2VA

# 赋予执行权限
chmod +x "$SCRIPT"

# 检查是否已有进程
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "⚠️  已有上传进程在运行: PID $(cat "$PID_FILE")"
    exit 1
fi

echo "启动后台上传..."
echo "日志: $LOG_FILE"
echo ""

# 后台运行
nohup bash "$SCRIPT" > "$LOG_FILE" 2>&1 &
PID=$!

echo $PID > "$PID_FILE"
disown $PID 2>/dev/null

echo "✅ 上传已启动（后台运行）"
echo ""
echo "进程ID: $PID"
echo "PID文件: $PID_FILE"
echo "日志文件: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  tail -f $LOG_FILE"
echo ""
echo "预计时间: ~35分钟"
echo "  - Checkpoint 1 (2.5GB): ~15分钟"
echo "  - Checkpoint 2 (2.5GB): ~15分钟"
echo "  - Dataset (440MB): ~5分钟"
echo ""
