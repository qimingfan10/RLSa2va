#!/bin/bash

# Sa2VA血管分割训练脚本 - Merged Dataset (1220张图片)
# 使用与第一次训练相同的方式：torch.distributed.run + DeepSpeed Zero-3

# 激活micromamba环境
echo "激活micromamba环境: topo-sarl"
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate topo-sarl

# 检查环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "topo-sarl" ]; then
    echo "错误: 无法激活topo-sarl环境"
    exit 1
fi

echo "环境激活成功: $CONDA_DEFAULT_ENV"
echo "开始Sa2VA血管分割训练 - Merged Dataset"
echo "配置文件: projects/sa2va/configs/sa2va_merged_vessel_finetune.py"
echo "数据路径: /home/ubuntu/Sa2VA/data/merged_vessel_data/"
echo "数据集: 1220张图片 (坐标已缩放)"
echo "模型路径: /home/ubuntu/Sa2VA-26B.pth"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/ubuntu/Sa2VA:$PYTHONPATH

# 设置Hugging Face缓存目录
export HF_HOME=/home/ubuntu/huggingface_cache
export TRANSFORMERS_CACHE=/home/ubuntu/huggingface_cache
export HF_DATASETS_CACHE=/home/ubuntu/huggingface_cache

# 禁用在线模式，强制使用本地缓存
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 设置PyTorch CUDA内存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 创建输出目录
mkdir -p /home/ubuntu/Sa2VA/work_dirs/merged_vessel_segmentation

# 检查数据是否存在
if [ ! -f "/home/ubuntu/Sa2VA/data/merged_vessel_data/annotations.json" ]; then
    echo "错误: merged annotations.json文件不存在"
    exit 1
fi

if [ ! -d "/home/ubuntu/Sa2VA/data/merged_vessel_data/images" ]; then
    echo "错误: merged images目录不存在"
    exit 1
fi

if [ ! -f "/home/ubuntu/Sa2VA-26B.pth" ]; then
    echo "错误: Sa2VA-26B.pth模型文件不存在"
    exit 1
fi

echo "数据和模型检查完成，开始训练..."

# 启动训练 - 使用4个GPU + DeepSpeed Zero-3
cd /home/ubuntu/Sa2VA

# 设置日志文件路径
LOG_DIR="work_dirs/merged_vessel_segmentation"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/training.pid"

echo "训练日志将保存到: ${LOG_FILE}"
echo "进程ID将保存到: ${PID_FILE}"

# 后台运行训练
# 使用torch.distributed.run进行4卡训练，启用DeepSpeed Zero-3
nohup /home/ubuntu/micromamba/envs/topo-sarl/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29501 \
    /home/ubuntu/micromamba/envs/topo-sarl/lib/python3.10/site-packages/xtuner/tools/train.py \
    projects/sa2va/configs/sa2va_merged_vessel_finetune.py \
    --work-dir work_dirs/merged_vessel_segmentation \
    --deepspeed deepspeed_zero3.json \
    > "${LOG_FILE}" 2>&1 &

# 保存进程ID
TRAIN_PID=$!
echo $TRAIN_PID > "${PID_FILE}"

echo ""
echo "=========================================="
echo "训练已在后台启动！"
echo "=========================================="
echo "进程ID: $TRAIN_PID"
echo "日志文件: ${LOG_FILE}"
echo "PID文件: ${PID_FILE}"
echo ""
echo "数据集信息:"
echo "  总样本数: 1220"
echo "  坐标已缩放: 600个样本 (800×800 → 512×512)"
echo "  训练轮数: 3 epochs"
echo ""
echo "查看实时日志："
echo "  tail -f ${LOG_FILE}"
echo ""
echo "查看训练进程状态："
echo "  ps aux | grep $TRAIN_PID"
echo ""
echo "停止训练："
echo "  kill $TRAIN_PID"
echo "  或者: kill \$(cat ${PID_FILE})"
echo ""
echo "检查点保存在: work_dirs/merged_vessel_segmentation/"
echo "=========================================="
