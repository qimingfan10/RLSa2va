#!/bin/bash

# 检查上传状态的脚本

LOG_FILE="/home/ubuntu/Sa2VA/upload_models.log"

echo "========================================================================"
echo "Sa2VA模型上传状态检查"
echo "========================================================================"
echo ""

# 检查进程
echo "【1】进程状态"
echo "----------------------------------------"
if pgrep -f "upload_models_background.sh" > /dev/null; then
    echo "✅ 上传进程正在运行"
    echo ""
    ps aux | grep upload_models_background.sh | grep -v grep | grep -v check_upload
    echo ""
    
    # 显示进程已运行时间
    PID=$(pgrep -f "upload_models_background.sh" | head -1)
    ELAPSED=$(ps -p $PID -o etime= 2>/dev/null)
    if [ -n "$ELAPSED" ]; then
        echo "运行时间: $ELAPSED"
    fi
else
    echo "❌ 没有检测到上传进程"
    echo ""
    echo "可能原因:"
    echo "  - 上传已完成"
    echo "  - 上传出错终止"
    echo "  - 尚未启动上传"
fi

echo ""
echo "【2】日志最新内容 (最后20行)"
echo "----------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
    echo ""
    echo "日志文件大小: $(du -h "$LOG_FILE" | cut -f1)"
else
    echo "⚠️  日志文件不存在: $LOG_FILE"
fi

echo ""
echo "【3】模型目录状态"
echo "----------------------------------------"
if [ -d "models/sa2va_vessel_hf" ]; then
    echo "模型1: $(du -sh models/sa2va_vessel_hf | cut -f1) (sa2va_vessel_hf)"
else
    echo "⚠️  模型1目录不存在"
fi

if [ -d "models/sa2va_vessel_iter3672_hf" ]; then
    echo "模型2: $(du -sh models/sa2va_vessel_iter3672_hf | cut -f1) (sa2va_vessel_iter3672_hf)"
else
    echo "⚠️  模型2目录不存在"
fi

echo ""
echo "【4】网络连接"
echo "----------------------------------------"
if ping -c 2 huggingface.co > /dev/null 2>&1; then
    echo "✅ HuggingFace可访问"
else
    echo "❌ HuggingFace连接失败"
fi

echo ""
echo "【5】快捷命令"
echo "----------------------------------------"
echo "实时监控日志: tail -f $LOG_FILE"
echo "查看完整日志: cat $LOG_FILE"
echo "查看进程详情: ps aux | grep upload"
echo "终止上传: pkill -f upload_models_background.sh"
echo ""
echo "========================================================================"
