# HuggingFace上传完整指南

## ✅ **问题2答案：是否支持断点续传？**

### **答案：是的！HuggingFace CLI支持断点续传** ⭐

---

## 🔍 **断点续传机制**

### 1. **自动断点续传**

HuggingFace CLI使用Git LFS，天然支持断点续传：

```bash
huggingface-cli upload qimingfan10/sa2va-vessel-hf models/sa2va_vessel_hf
```

**如果上传中断**：
- ✅ 已上传的文件不会重新上传
- ✅ 部分上传的大文件会从断点继续
- ✅ 自动处理，无需额外参数

### 2. **工作原理**

```
第一次上传:
├── file1.safetensors (4.5GB) → 上传50% → ❌ 网络中断
├── file2.safetensors (4.6GB) → ✅ 完成
└── file3.safetensors (4.6GB) → 未开始

重新运行命令:
├── file1.safetensors (4.5GB) → 从50%继续 ✅
├── file2.safetensors (4.6GB) → 跳过（已完成）
└── file3.safetensors (4.6GB) → 开始上传
```

---

## 📋 **实际使用方法**

### 方法1: 直接重新运行（推荐）

上传中断后，直接重新运行相同的命令：

```bash
# 第一次运行（中断了）
huggingface-cli upload qimingfan10/sa2va-vessel-hf models/sa2va_vessel_hf

# 网络中断或Ctrl+C后...

# 直接重新运行相同命令
huggingface-cli upload qimingfan10/sa2va-vessel-hf models/sa2va_vessel_hf
```

**CLI会自动**：
- 检查远程仓库已有的文件
- 跳过已完成的文件
- 继续未完成的文件

### 方法2: 使用Python API（更多控制）

```python
from huggingface_hub import HfApi, create_commit
from huggingface_hub import CommitOperationAdd
import os

api = HfApi()

# 准备文件列表
operations = []
local_dir = "models/sa2va_vessel_hf"

for root, dirs, files in os.walk(local_dir):
    for file in files:
        local_path = os.path.join(root, file)
        path_in_repo = os.path.relpath(local_path, local_dir)
        
        operations.append(
            CommitOperationAdd(
                path_in_repo=path_in_repo,
                path_or_fileobj=local_path
            )
        )

# 上传（支持断点续传）
api.create_commit(
    repo_id="qimingfan10/sa2va-vessel-hf",
    operations=operations,
    commit_message="Upload Sa2VA model",
    repo_type="model",
)
```

---

## 🛡️ **如何确保断点续传有效？**

### 1. **不要删除Git缓存**

HuggingFace CLI使用本地Git仓库：
```bash
# 不要删除这个目录！
~/.cache/huggingface/
```

### 2. **使用相同的命令**

确保重试时使用**完全相同的命令**：
```bash
# ✅ 正确：相同的命令
huggingface-cli upload qimingfan10/sa2va-vessel-hf models/sa2va_vessel_hf

# ❌ 错误：改变了路径或参数
huggingface-cli upload qimingfan10/sa2va-vessel-hf ./models/sa2va_vessel_hf  # 不同路径
```

### 3. **保持网络环境一致**

如果可能，在重试时：
- 使用相同的网络
- 相同的机器
- 相同的用户

---

## 📊 **上传进度追踪**

### 查看上传进度

```bash
# 上传时会显示进度
huggingface-cli upload qimingfan10/sa2va-vessel-hf models/sa2va_vessel_hf

# 输出示例：
# Uploading files:   0%|          | 0/7 [00:00<?, ?it/s]
# model-00001-of-00007.safetensors: 100%|██| 4.5G/4.5G [15:23<00:00, 4.87MB/s]
# model-00002-of-00007.safetensors:  45%|█  | 2.1G/4.6G [08:12<09:42, 4.29MB/s]
# ^C  ← 中断
```

### 验证已上传的文件

访问HuggingFace查看：
```
https://huggingface.co/qimingfan10/sa2va-vessel-hf/tree/main
```

或使用CLI：
```bash
huggingface-cli scan-cache
```

---

## ⚡ **优化上传速度**

### 1. **使用稳定网络**

```bash
# 检查网络速度
curl -o /dev/null http://speedtest.tele2.net/100MB.zip
```

### 2. **增加超时时间**

```bash
# 设置更长的超时
export HF_HUB_TIMEOUT=3600  # 60分钟

huggingface-cli upload qimingfan10/sa2va-vessel-hf models/sa2va_vessel_hf
```

### 3. **使用镜像（中国用户）**

```bash
# 使用HF镜像站
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli upload qimingfan10/sa2va-vessel-hf models/sa2va_vessel_hf
```

### 4. **分批上传**

如果30GB太大，可以分批上传：

```bash
# 只上传前3个文件
huggingface-cli upload qimingfan10/sa2va-vessel-hf \
    models/sa2va_vessel_hf/model-00001-of-00007.safetensors \
    model-00001-of-00007.safetensors

huggingface-cli upload qimingfan10/sa2va-vessel-hf \
    models/sa2va_vessel_hf/model-00002-of-00007.safetensors \
    model-00002-of-00007.safetensors

# 然后上传剩余文件
huggingface-cli upload qimingfan10/sa2va-vessel-hf models/sa2va_vessel_hf
```

---

## 🔧 **实用脚本**

### 创建上传脚本（带重试）

```bash
#!/bin/bash
# upload_with_retry.sh

MODEL_DIR="models/sa2va_vessel_hf"
REPO_ID="qimingfan10/sa2va-vessel-hf"
MAX_RETRIES=3

for i in $(seq 1 $MAX_RETRIES); do
    echo "尝试 $i/$MAX_RETRIES..."
    
    if huggingface-cli upload "$REPO_ID" "$MODEL_DIR" \
        --repo-type model \
        --commit-message "Upload Sa2VA model"; then
        echo "✅ 上传成功!"
        exit 0
    else
        echo "❌ 上传失败，等待30秒后重试..."
        sleep 30
    fi
done

echo "❌ 达到最大重试次数，上传失败"
exit 1
```

使用：
```bash
chmod +x upload_with_retry.sh
bash upload_with_retry.sh
```

---

## 📝 **常见问题**

### Q1: 如何知道上传是否完整？

**A**: 检查HuggingFace仓库的Files页面，确认所有文件都存在：
```
✅ model-00001-of-00007.safetensors (4.5GB)
✅ model-00002-of-00007.safetensors (4.6GB)
✅ model-00003-of-00007.safetensors (4.6GB)
✅ model-00004-of-00007.safetensors (4.7GB)
✅ model-00005-of-00007.safetensors (4.7GB)
✅ model-00006-of-00007.safetensors (4.0GB)
✅ model-00007-of-00007.safetensors (3.1GB)
✅ config.json
✅ README.md
✅ ... 其他配置文件
```

### Q2: 上传速度太慢怎么办？

**A**: 
1. 检查网络：`speedtest-cli`
2. 使用有线网络
3. 避免高峰时段
4. 考虑使用镜像站（中国）

### Q3: 上传中断了很多次怎么办？

**A**: 
- 每次重新运行相同命令即可
- CLI会记住已上传的部分
- 不需要重新开始

### Q4: 如何取消上传？

**A**: 
- 按 `Ctrl+C` 即可
- 不会损坏远程仓库
- 下次运行会继续

### Q5: 上传后如何验证？

**A**: 
```bash
# 测试下载
cd /tmp
huggingface-cli download qimingfan10/sa2va-vessel-hf \
    --local-dir test_download

# 检查文件大小
du -sh test_download/
```

---

## 🎯 **推荐上传流程**

### 完整步骤

```bash
# 1. 确保已登录
huggingface-cli login

# 2. 测试网络
ping huggingface.co -c 5

# 3. 开始上传（自动断点续传）
huggingface-cli upload \
    qimingfan10/sa2va-vessel-hf \
    models/sa2va_vessel_hf \
    --repo-type model \
    --commit-message "Upload Sa2VA vessel model (iter_12192)"

# 4. 如果中断，直接重新运行步骤3

# 5. 验证上传
# 访问: https://huggingface.co/qimingfan10/sa2va-vessel-hf
```

### 预计时间

| 网络速度 | 单个模型(30GB) | 两个模型(60GB) |
|----------|---------------|---------------|
| 10 MB/s  | 50分钟        | 100分钟 (1.7小时) |
| 5 MB/s   | 100分钟 (1.7小时) | 200分钟 (3.3小时) |
| 2 MB/s   | 250分钟 (4.2小时) | 500分钟 (8.3小时) |

---

## ✅ **总结**

### 断点续传支持

| 特性 | 支持 | 说明 |
|------|------|------|
| **自动续传** | ✅ Yes | 重新运行相同命令即可 |
| **跳过已上传** | ✅ Yes | 自动检测已完成文件 |
| **部分文件续传** | ✅ Yes | Git LFS自动处理 |
| **需要特殊参数** | ❌ No | 完全自动 |
| **需要记住进度** | ❌ No | CLI自动管理 |

### 关键要点

1. ✅ **HuggingFace CLI支持断点续传**
2. ✅ **无需额外配置或参数**
3. ✅ **重新运行相同命令即可**
4. ✅ **基于Git LFS，非常可靠**
5. ⚠️ **不要删除`~/.cache/huggingface/`**

---

**准备好开始上传了吗？** 🚀

```bash
cd /home/ubuntu/Sa2VA
huggingface-cli upload qimingfan10/sa2va-vessel-hf models/sa2va_vessel_hf
```

**创建时间**: 2025-11-27  
**状态**: 随时可以开始上传
