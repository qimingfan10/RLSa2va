#!/bin/bash

# 完全后台独立运行的上传脚本（不受终端关闭影响）

LOG_FILE="/home/ubuntu/Sa2VA/upload_models.log"
SCRIPT="/home/ubuntu/Sa2VA/upload_models_background.sh"
PID_FILE="/home/ubuntu/Sa2VA/upload.pid"

echo "========================================================================"
echo "启动Sa2VA模型完全后台上传（独立运行）"
echo "========================================================================"
echo ""

cd /home/ubuntu/Sa2VA

# 检查脚本存在
if [ ! -f "$SCRIPT" ]; then
    echo "❌ 错误：找不到上传脚本 $SCRIPT"
    exit 1
fi

# 赋予执行权限
chmod +x "$SCRIPT"

# 检查是否已有上传进程在运行
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "⚠️  警告：检测到已有上传进程在运行"
    echo "PID: $(cat "$PID_FILE")"
    echo ""
    read -p "是否终止旧进程并重新开始? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $(cat "$PID_FILE") 2>/dev/null
        sleep 2
        echo "✅ 已终止旧进程"
    else
        echo "已取消"
        exit 0
    fi
fi

echo "准备启动完全后台上传..."
echo "日志文件: $LOG_FILE"
echo ""

# 使用nohup和disown确保进程完全独立
nohup bash "$SCRIPT" > "$LOG_FILE" 2>&1 &
PID=$!

# 保存PID
echo $PID > "$PID_FILE"

# 让进程完全独立
disown $PID 2>/dev/null

echo "✅ 上传已在后台启动！（完全独立运行）"
echo ""
echo "进程ID: $PID"
echo "PID文件: $PID_FILE"
echo "日志文件: $LOG_FILE"
echo ""
echo "========================================================================"
echo "监控命令"
echo "========================================================================"
echo ""
echo "1. 实时查看日志:"
echo "   tail -f $LOG_FILE"
echo ""
echo "2. 查看进程状态:"
echo "   ps aux | grep $PID"
echo ""
echo "3. 快速检查状态:"
echo "   bash check_upload_status.sh"
echo ""
echo "4. 终止上传:"
echo "   kill $PID"
echo ""
echo "5. 查看最新30行:"
echo "   tail -30 $LOG_FILE"
echo ""
echo "========================================================================"
echo "特点"
echo "========================================================================"
echo ""
echo "✅ 完全独立运行 - 关闭终端也不会中断"
echo "✅ 自动记录日志 - 所有输出保存到日志文件"
echo "✅ 支持断点续传 - 中断后可以继续"
echo "✅ PID文件管理 - 方便后续管理"
echo ""
echo "预计时间: 2-4小时 (60GB)"
echo "========================================================================"
echo ""
echo "您现在可以:"
echo "1. 关闭此终端（上传会继续）"
echo "2. 打开新终端监控: tail -f $LOG_FILE"
echo "3. 稍后检查状态: bash check_upload_status.sh"
echo ""
