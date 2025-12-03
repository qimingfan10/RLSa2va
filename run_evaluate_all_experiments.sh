#!/bin/bash

# 评估所有实验的脚本

echo "========================================"
echo "评估所有实验"
echo "========================================"

# 实验一：Prompt优化RL
echo ""
echo "1. 评估实验一（Prompt优化RL）"
echo "========================================"

cd /home/ubuntu/Sa2VA/rl_prompt_optimization

# 找到最新的模型
MODEL_PATH="/home/ubuntu/Sa2VA/rl_prompt_optimization/outputs/rl_prompt_20251129_154906/final_model.zip"

if [ -f "$MODEL_PATH" ]; then
    echo "找到模型: $MODEL_PATH"
    echo "开始评估..."
    
    CUDA_VISIBLE_DEVICES=1 python3 evaluate_rl_prompt.py \
        --rl_model_path "$MODEL_PATH" \
        --model_path /home/ubuntu/Sa2VA/models/sa2va_vessel_hf \
        --data_root /home/ubuntu/Sa2VA/data/merged_vessel_data \
        --output_dir ./evaluations \
        --max_steps 3 \
        --split val \
        2>&1 | tee evaluation_exp1.log
    
    echo "✅ 实验一评估完成"
else
    echo "❌ 未找到模型: $MODEL_PATH"
fi

# 实验二：后处理优化RL
echo ""
echo "2. 实验二（后处理优化RL）"
echo "========================================"
echo "结论: 阈值扫描验证显示此方法不可行"
echo "原因: Sa2VA返回二值mask，不是概率图"
echo "状态: 方案已放弃，无需评估"

echo ""
echo "========================================"
echo "评估完成！"
echo "========================================"
echo ""
echo "查看结果:"
echo "  实验一: /home/ubuntu/Sa2VA/rl_prompt_optimization/evaluations/eval_*/evaluation_results.json"
echo ""
