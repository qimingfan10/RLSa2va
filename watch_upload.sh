#!/bin/bash

# 实时监控上传进度的便捷脚本

LOG_FILE="/home/ubuntu/Sa2VA/upload_models.log"
PID_FILE="/home/ubuntu/Sa2VA/upload.pid"

echo "========================================================================"
echo "Sa2VA模型上传实时监控"
echo "========================================================================"
echo ""

# 检查进程状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 上传进程正在运行"
        echo "PID: $PID"
        
        # 显示运行时间
        ELAPSED=$(ps -p $PID -o etime= 2>/dev/null)
        echo "运行时间: $ELAPSED"
    else
        echo "❌ 上传进程未运行"
        echo "可能已完成或出错"
    fi
else
    echo "⚠️  未找到PID文件"
fi

echo ""
echo "日志文件: $LOG_FILE"
echo "大小: $(du -h "$LOG_FILE" 2>/dev/null | cut -f1 || echo '0')"
echo ""
echo "========================================================================"
echo "最新日志（自动刷新）"
echo "========================================================================"
echo ""
echo "按Ctrl+C退出监控"
echo ""
sleep 2

# 实时监控日志
tail -f "$LOG_FILE"
