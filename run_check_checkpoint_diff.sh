#!/bin/bash

# 设置micromamba路径
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba
export PATH="/home/ubuntu/micromamba/micromamba/bin:$PATH"

# 初始化micromamba
eval "$(/home/ubuntu/micromamba/micromamba/bin/micromamba shell hook --shell bash)"

# 激活环境
micromamba activate topo-sarl

cd /home/ubuntu/Sa2VA
python check_checkpoint_diff.py
