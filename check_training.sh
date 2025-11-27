#!/bin/bash

# 检查训练状态的便捷脚本

LOG_DIR="/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation"
PID_FILE="${LOG_DIR}/training.pid"

echo "=========================================="
echo "Sa2VA 训练状态检查"
echo "=========================================="
echo ""

# 检查PID文件是否存在
if [ ! -f "${PID_FILE}" ]; then
    echo "未找到训练进程ID文件"
    echo "训练可能尚未启动或已经完成"
    echo ""
    echo "最近的日志文件："
    ls -lt "${LOG_DIR}"/training_*.log 2>/dev/null | head -5
    exit 0
fi

# 读取PID
TRAIN_PID=$(cat "${PID_FILE}")
echo "训练进程ID: ${TRAIN_PID}"
echo ""

# 检查进程是否还在运行
if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
    echo "✓ 训练进程正在运行"
    echo ""
    
    # 显示进程信息
    echo "进程详情："
    ps aux | grep ${TRAIN_PID} | grep -v grep
    echo ""
    
    # 显示GPU使用情况
    echo "GPU使用情况："
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "无法获取GPU信息"
    echo ""
    
    # 找到最新的日志文件
    LATEST_LOG=$(ls -t "${LOG_DIR}"/training_*.log 2>/dev/null | head -1)
    if [ -n "${LATEST_LOG}" ]; then
        echo "最新日志文件: ${LATEST_LOG}"
        echo ""
        echo "最后20行日志："
        echo "----------------------------------------"
        tail -20 "${LATEST_LOG}"
        echo "----------------------------------------"
        echo ""
        echo "实时查看日志："
        echo "  tail -f ${LATEST_LOG}"
    fi
else
    echo "✗ 训练进程已停止"
    echo ""
    
    # 找到最新的日志文件
    LATEST_LOG=$(ls -t "${LOG_DIR}"/training_*.log 2>/dev/null | head -1)
    if [ -n "${LATEST_LOG}" ]; then
        echo "最新日志文件: ${LATEST_LOG}"
        echo ""
        echo "日志末尾（可能包含错误信息）："
        echo "----------------------------------------"
        tail -50 "${LATEST_LOG}"
        echo "----------------------------------------"
    fi
fi

echo ""
echo "所有日志文件："
ls -lh "${LOG_DIR}"/training_*.log 2>/dev/null
echo ""
echo "检查点文件："
ls -lh "${LOG_DIR}"/*.pth 2>/dev/null || echo "暂无检查点文件"
echo "=========================================="
