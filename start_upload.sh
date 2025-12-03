#!/bin/bash

# 启动后台上传的便捷脚本

LOG_FILE="/home/ubuntu/Sa2VA/upload_models.log"
SCRIPT="/home/ubuntu/Sa2VA/upload_models_background.sh"

echo "========================================================================"
echo "启动Sa2VA模型后台上传"
echo "========================================================================"
echo ""

# 检查脚本存在
if [ ! -f "$SCRIPT" ]; then
    echo "❌ 错误：找不到上传脚本 $SCRIPT"
    exit 1
fi

# 赋予执行权限
chmod +x "$SCRIPT"

# 检查是否已有上传进程在运行
if pgrep -f "upload_models_background.sh" > /dev/null; then
    echo "⚠️  警告：检测到已有上传进程在运行"
    echo ""
    echo "运行中的进程："
    ps aux | grep upload_models_background.sh | grep -v grep
    echo ""
    read -p "是否终止旧进程并重新开始? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f upload_models_background.sh
        sleep 2
        echo "✅ 已终止旧进程"
    else
        echo "已取消"
        exit 0
    fi
fi

echo "准备启动后台上传..."
echo "日志文件: $LOG_FILE"
echo ""

# 使用nohup后台运行
nohup bash "$SCRIPT" > "$LOG_FILE" 2>&1 &

PID=$!

echo "✅ 上传已在后台启动！"
echo ""
echo "进程ID: $PID"
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
echo "   ps aux | grep upload_models_background"
echo ""
echo "3. 检查上传进度:"
echo "   watch -n 5 'tail -20 $LOG_FILE'"
echo ""
echo "4. 终止上传:"
echo "   kill $PID"
echo "   或"
echo "   pkill -f upload_models_background.sh"
echo ""
echo "5. 查看完整日志:"
echo "   cat $LOG_FILE"
echo ""
echo "========================================================================"
echo "预计时间: 2-4小时 (两个模型共60GB)"
echo "========================================================================"
echo ""
echo "开始监控日志..."
sleep 2
tail -f "$LOG_FILE"
