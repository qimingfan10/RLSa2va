#!/bin/bash

echo "========================================"
echo "实验一快速评估监控"
echo "========================================"
echo ""

LOG_FILE="/home/ubuntu/Sa2VA/rl_prompt_optimization/evaluation_exp1_quick.log"

echo "配置信息:"
echo "  样本数: 50张"
echo "  GPU: GPU 2（空闲）"
echo "  预计时间: 30-60分钟"
echo ""

echo "实时监控命令:"
echo "  tail -f $LOG_FILE"
echo ""

echo "查看进度命令:"
echo "  grep '评估样本' $LOG_FILE | tail -5"
echo ""

echo "查看最新Dice结果:"
echo "  tail -50 $LOG_FILE | grep 'Dice='"
echo ""

echo "========================================"
echo "当前状态"
echo "========================================"

# 检查进程
PROCESS=$(ps aux | grep "evaluate_rl_prompt_quick.py" | grep -v grep)
if [ -n "$PROCESS" ]; then
    echo "✅ 评估进程运行中"
else
    echo "❌ 评估进程未运行"
fi

# 检查GPU
echo ""
echo "GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader | grep "^2,"

# 检查进度
echo ""
echo "评估进度:"
COMPLETED=$(grep -c "评估样本" $LOG_FILE 2>/dev/null || echo "0")
echo "  已完成: $COMPLETED / 50"

# 最近的Dice结果
echo ""
echo "最近5次Dice结果:"
tail -100 $LOG_FILE 2>/dev/null | grep "Dice=" | tail -5

echo ""
echo "========================================"
