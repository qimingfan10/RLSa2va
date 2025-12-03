#!/bin/bash

# 后台启动上传新增内容

LOG_FILE="/home/ubuntu/Sa2VA/upload_new_content.log"
SCRIPT="/home/ubuntu/Sa2VA/upload_new_content.sh"
PID_FILE="/home/ubuntu/Sa2VA/upload_new.pid"

echo "========================================================================"
echo "启动Sa2VA新增内容后台上传"
echo "========================================================================"
echo ""

cd /home/ubuntu/Sa2VA

chmod +x "$SCRIPT"

# 检查是否已有进程
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "⚠️  已有上传进程在运行: PID $(cat "$PID_FILE")"
    exit 1
fi

echo "启动后台上传..."
nohup bash "$SCRIPT" > "$LOG_FILE" 2>&1 &
PID=$!

echo $PID > "$PID_FILE"
disown $PID 2>/dev/null

echo "✅ 上传已启动"
echo ""
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  tail -f $LOG_FILE"
echo ""
echo "预计时间: 5-10分钟"
echo ""
