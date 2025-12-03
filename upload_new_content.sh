#!/bin/bash

# Sa2VA 新增内容上传脚本
# 创建时间: 2025-12-03

set -e

LOG_FILE="/home/ubuntu/Sa2VA/upload_new_content.log"
cd /home/ubuntu/Sa2VA

echo "========================================================================"
echo "Sa2VA 新增内容上传"
echo "========================================================================"
echo "开始时间: $(date)"
echo ""

# ============================================================================
# 第一部分：更新 .gitignore
# ============================================================================
echo "【1】更新 .gitignore..."

# 确保大文件不会被上传到GitHub
cat >> .gitignore << 'EOF'

# LoRA训练输出（权重文件）
lora_ppo_training/output/
lora_ppo_training/output_v2/
lora_ppo_training/*.log
lora_ppo_training/__pycache__/
lora_ppo_training/=*

# RL Prompt优化输出
rl_prompt_optimization/outputs/
rl_prompt_optimization/evaluations/
rl_prompt_optimization/logs/
rl_prompt_optimization/*.log
rl_prompt_optimization/__pycache__/
rl_prompt_optimization/env/

# 视频数据
Segment_DATA_Videos_512/

# 其他大文件
*.safetensors
*.bin
*.pth
*.pt
EOF

echo "✅ .gitignore 已更新"
echo ""

# ============================================================================
# 第二部分：上传代码到GitHub
# ============================================================================
echo "【2】上传代码到GitHub..."

# 添加新文件
git add .gitignore
git add docs/
git add lora_ppo_training/*.py
git add lora_ppo_training/*.sh
git add lora_ppo_training/*.md
git add lora_ppo_training/requirements.txt 2>/dev/null || true
git add rl_prompt_optimization/*.py
git add rl_prompt_optimization/*.sh
git add rl_prompt_optimization/*.md
git add rl_prompt_optimization/requirements.txt

# 添加新增的文档
git add *.md

# 添加修改的代码
git add models/sa2va_vessel_hf/*.py
git add projects/sa2va/models/*.py
git add projects/sa2va/models/**/*.py
git add tools/*.py
git add *.py

# 提交
git commit -m "Add LoRA PPO training, RL prompt optimization, and documentation

New features:
- LoRA PPO training pipeline (lora_ppo_training/)
- RL prompt optimization (rl_prompt_optimization/)
- DPO training documentation
- Training methodology documentation
- Multiple experiment reports and status documents

Modified:
- Model inference code improvements
- HuggingFace model conversion updates
- Various bug fixes and optimizations"

# 推送到GitHub
echo "推送到GitHub..."
git push origin main

echo "✅ GitHub上传完成"
echo ""

# ============================================================================
# 第三部分：上传LoRA权重到HuggingFace
# ============================================================================
echo "【3】上传LoRA权重到HuggingFace..."

# 检查登录状态
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ 未登录HuggingFace，请先运行: huggingface-cli login"
    exit 1
fi

echo "✅ 已登录: $(huggingface-cli whoami 2>&1 | grep -v Warning)"

# 上传best_lora
LORA_PATH="lora_ppo_training/output_v2/sa2va_lora_ppo_20251129_175137/best_lora"
if [ -d "$LORA_PATH" ]; then
    echo ""
    echo "上传LoRA PPO最佳权重..."
    echo "源目录: $LORA_PATH"
    echo "大小: $(du -sh $LORA_PATH | cut -f1)"
    
    huggingface-cli upload \
        ly17/sa2va-vessel-lora-ppo \
        "$LORA_PATH" \
        best_lora \
        --repo-type model \
        --commit-message "Upload LoRA PPO best checkpoint"
    
    echo "✅ best_lora 上传完成"
fi

# 上传final_lora
FINAL_LORA_PATH="lora_ppo_training/output_v2/sa2va_lora_ppo_20251129_175137/final_lora"
if [ -d "$FINAL_LORA_PATH" ]; then
    echo ""
    echo "上传LoRA PPO最终权重..."
    echo "源目录: $FINAL_LORA_PATH"
    echo "大小: $(du -sh $FINAL_LORA_PATH | cut -f1)"
    
    huggingface-cli upload \
        ly17/sa2va-vessel-lora-ppo \
        "$FINAL_LORA_PATH" \
        final_lora \
        --repo-type model \
        --commit-message "Upload LoRA PPO final checkpoint"
    
    echo "✅ final_lora 上传完成"
fi

echo ""
echo "✅ HuggingFace LoRA权重上传完成"
echo "地址: https://huggingface.co/ly17/sa2va-vessel-lora-ppo"
echo ""

# ============================================================================
# 第四部分：上传视频数据到HuggingFace Dataset（可选）
# ============================================================================
echo "【4】上传视频数据到HuggingFace Dataset..."

VIDEO_DATA_PATH="Segment_DATA_Videos_512"
if [ -d "$VIDEO_DATA_PATH" ]; then
    echo "源目录: $VIDEO_DATA_PATH"
    echo "大小: $(du -sh $VIDEO_DATA_PATH | cut -f1)"
    
    huggingface-cli upload \
        ly17/sa2va-vessel-dataset \
        "$VIDEO_DATA_PATH" \
        Segment_DATA_Videos_512 \
        --repo-type dataset \
        --commit-message "Add video segmentation data"
    
    echo "✅ 视频数据上传完成"
else
    echo "⚠️  视频数据目录不存在，跳过"
fi

echo ""

# ============================================================================
# 完成
# ============================================================================
echo "========================================================================"
echo "✅ 所有上传完成！"
echo "========================================================================"
echo "完成时间: $(date)"
echo ""
echo "上传内容："
echo "  1. GitHub: 代码和文档"
echo "     - lora_ppo_training/ (训练代码)"
echo "     - rl_prompt_optimization/ (优化代码)"
echo "     - docs/ (文档)"
echo "     - 40+ MD文档"
echo ""
echo "  2. HuggingFace Model:"
echo "     - ly17/sa2va-vessel-lora-ppo"
echo "       - best_lora/ (159MB)"
echo "       - final_lora/ (159MB)"
echo ""
echo "  3. HuggingFace Dataset:"
echo "     - ly17/sa2va-vessel-dataset"
echo "       - Segment_DATA_Videos_512/ (46MB)"
echo ""
echo "验证链接："
echo "  - GitHub: https://github.com/qimingfan10/RLSa2va"
echo "  - HF Model: https://huggingface.co/ly17/sa2va-vessel-lora-ppo"
echo "  - HF Dataset: https://huggingface.co/datasets/ly17/sa2va-vessel-dataset"
echo ""
echo "========================================================================"
