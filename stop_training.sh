#!/bin/bash

# 停止训练的脚本

LOG_DIR="/home/ubuntu/Sa2VA/work_dirs/vessel_segmentation"
PID_FILE="${LOG_DIR}/training.pid"

echo "=========================================="
echo "停止 Sa2VA 训练"
echo "=========================================="
echo ""

# 检查PID文件是否存在
if [ ! -f "${PID_FILE}" ]; then
    echo "未找到训练进程ID文件"
    echo "训练可能尚未启动或已经停止"
    exit 0
fi

# 读取PID
TRAIN_PID=$(cat "${PID_FILE}")
echo "训练进程ID: ${TRAIN_PID}"

# 检查进程是否还在运行
if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
    echo "正在停止训练进程..."
    
    # 先尝试优雅地停止（SIGTERM）
    kill ${TRAIN_PID}
    
    # 等待最多10秒
    for i in {1..10}; do
        if ! ps -p ${TRAIN_PID} > /dev/null 2>&1; then
            echo "✓ 训练进程已成功停止"
            rm -f "${PID_FILE}"
            exit 0
        fi
        echo "等待进程停止... ($i/10)"
        sleep 1
    done
    
    # 如果还没停止，强制终止（SIGKILL）
    echo "进程未响应，强制终止..."
    kill -9 ${TRAIN_PID}
    sleep 1
    
    if ! ps -p ${TRAIN_PID} > /dev/null 2>&1; then
        echo "✓ 训练进程已强制停止"
        rm -f "${PID_FILE}"
    else
        echo "✗ 无法停止进程"
        exit 1
    fi
else
    echo "训练进程已经停止"
    rm -f "${PID_FILE}"
fi

echo "=========================================="
