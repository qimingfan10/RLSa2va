# Sa2VA GitHub + HuggingFace 上传检查清单

## 📋 上传前准备

### 步骤1: 文件清理

- [ ] 删除临时文件和日志
  ```bash
  find . -name "*.log" -type f -delete
  find . -name "*.pyc" -type f -delete
  find . -name "__pycache__" -type d -exec rm -rf {} +
  ```

- [ ] 删除大型结果文件
  ```bash
  rm -rf video_prediction_5_videos/
  rm -rf *_results/
  rm -rf *.mp4
  ```

- [ ] 备份重要数据（如果需要）
  ```bash
  # 备份模型
  cp -r models/ /backup/sa2va_models/
  
  # 备份训练checkpoints
  cp -r work_dirs/ /backup/sa2va_checkpoints/
  ```

### 步骤2: 更新配置文件

- [ ] 更新.gitignore
  ```bash
  cp .gitignore_updated .gitignore
  ```

- [ ] 检查.gitignore是否包含所有大文件
  ```bash
  # 测试哪些文件会被git追踪
  git status --short
  ```

- [ ] 确认requirements.txt完整
  ```bash
  cat requirements.txt
  ```

### 步骤3: 文档检查

- [ ] README.md 包含模型下载说明
- [ ] SA2VA_METHODOLOGY.md 完整
- [ ] SA2VA_TECHNICAL_DETAILS.md 完整
- [ ] DOCUMENTATION_INDEX.md 已更新
- [ ] LICENSE 文件存在

### 步骤4: 脚本检查

- [ ] scripts/download_models.sh 可执行
  ```bash
  chmod +x scripts/download_models.sh
  chmod +x scripts/upload_to_github.sh
  chmod +x scripts/upload_to_huggingface.sh
  ```

- [ ] 所有.sh脚本有执行权限
  ```bash
  chmod +x *.sh
  chmod +x run_*.sh
  ```

---

## 🐙 GitHub上传流程

### 准备阶段

- [ ] 创建GitHub仓库
  - 仓库名: RLSa2va
  - 可见性: Public
  - 描述: Sa2VA: Segment Anything to Vessel Analysis
  - 不要勾选"Initialize with README"（已有README）

### Git配置

- [ ] 配置用户信息
  ```bash
  git config --global user.name "Qiming Fan"
  git config --global user.email "your_email@example.com"
  ```

- [ ] 检查SSH密钥（推荐）或使用HTTPS
  ```bash
  # 生成SSH密钥（如果没有）
  ssh-keygen -t ed25519 -C "your_email@example.com"
  
  # 添加到GitHub
  cat ~/.ssh/id_ed25519.pub
  # 复制并添加到 GitHub Settings > SSH Keys
  ```

### 执行上传

- [ ] 运行上传脚本
  ```bash
  bash scripts/upload_to_github.sh
  ```

- [ ] 或手动执行：
  ```bash
  cd /home/ubuntu/Sa2VA
  
  # 初始化（如果需要）
  git init
  git remote add origin git@github.com:qimingfan10/RLSa2va.git
  
  # 添加文件
  git add .
  
  # 提交
  git commit -m "Initial commit: Sa2VA code and documentation"
  
  # 推送
  git branch -M main
  git push -u origin main
  ```

### 验证

- [ ] 访问 https://github.com/qimingfan10/RLSa2va
- [ ] 检查README.md正确显示
- [ ] 检查文件完整性
- [ ] 确认没有大文件（>100MB）
- [ ] 测试克隆仓库
  ```bash
  cd /tmp
  git clone https://github.com/qimingfan10/RLSa2va.git
  cd RLSa2va
  ls -lah
  ```

---

## 🤗 HuggingFace上传流程

### 准备阶段

- [ ] 注册HuggingFace账号
  - 网址: https://huggingface.co/join

- [ ] 创建Access Token
  - Settings > Access Tokens > New Token
  - 权限: Write

### 安装和登录

- [ ] 安装huggingface_hub
  ```bash
  pip install huggingface_hub
  ```

- [ ] 登录HuggingFace
  ```bash
  huggingface-cli login
  # 粘贴Access Token
  ```

- [ ] 验证登录
  ```bash
  huggingface-cli whoami
  ```

### 创建模型仓库

- [ ] 创建仓库1: sa2va-vessel-hf
  - 访问: https://huggingface.co/new
  - Owner: qimingfan10
  - Model name: sa2va-vessel-hf
  - License: apache-2.0
  - Visibility: Public

- [ ] 创建仓库2: sa2va-vessel-iter3672-hf
  - Model name: sa2va-vessel-iter3672-hf

### 准备模型卡片

- [ ] 复制MODEL_CARD_TEMPLATE.md
  ```bash
  # 对于仓库1
  cp scripts/MODEL_CARD_TEMPLATE.md models/sa2va_vessel_hf/README.md
  
  # 对于仓库2
  cp scripts/MODEL_CARD_TEMPLATE.md models/sa2va_vessel_iter3672_hf/README.md
  ```

- [ ] 编辑README.md，更新：
  - 模型名称
  - 训练迭代次数
  - 训练时间
  - 特定说明

### 上传模型

- [ ] 运行上传脚本
  ```bash
  bash scripts/upload_to_huggingface.sh
  ```

- [ ] 或手动上传模型1
  ```bash
  cd /home/ubuntu/Sa2VA
  
  huggingface-cli upload qimingfan10/sa2va-vessel-hf \
      models/sa2va_vessel_hf \
      --repo-type model \
      --commit-message "Upload Sa2VA vessel model (iter_12192)"
  ```

- [ ] 或手动上传模型2
  ```bash
  huggingface-cli upload qimingfan10/sa2va-vessel-iter3672-hf \
      models/sa2va_vessel_iter3672_hf \
      --repo-type model \
      --commit-message "Upload Sa2VA vessel model (iter_3672)"
  ```

### 完善模型页面

- [ ] 编辑README.md（模型卡片）
- [ ] 添加标签(Tags):
  - medical-imaging
  - vessel-segmentation
  - oct
  - multimodal
  - vision-language

- [ ] 设置许可证: Apache 2.0
- [ ] 添加示例代码
- [ ] 上传示例图片（如果有）

### 验证

- [ ] 访问模型页面
  - https://huggingface.co/qimingfan10/sa2va-vessel-hf
  - https://huggingface.co/qimingfan10/sa2va-vessel-iter3672-hf

- [ ] 检查文件完整性
  - 7个safetensors文件
  - config.json
  - README.md

- [ ] 测试下载
  ```bash
  cd /tmp
  huggingface-cli download qimingfan10/sa2va-vessel-hf \
      --local-dir test_download
  ls -lh test_download/
  ```

---

## ✅ 最终验证

### GitHub仓库

- [ ] README完整显示
- [ ] 代码可以克隆
- [ ] 文档链接正常
- [ ] 无敏感信息
- [ ] 无大文件警告

### HuggingFace模型

- [ ] 模型可以下载
- [ ] README正确显示
- [ ] 标签完整
- [ ] 示例代码可运行

### 完整流程测试

- [ ] 在新环境测试完整流程：
  ```bash
  # 1. 克隆代码
  git clone https://github.com/qimingfan10/RLSa2va.git
  cd RLSa2va
  
  # 2. 安装依赖
  pip install -r requirements.txt
  
  # 3. 下载模型
  bash scripts/download_models.sh
  
  # 4. 测试推理
  python test_existing_hf_model.py
  ```

---

## 📊 上传时间估算

| 任务 | 大小 | 预计时间 |
|------|------|----------|
| GitHub代码推送 | ~500MB | 5-10分钟 |
| HF模型1上传 | 30GB | 1-2小时 |
| HF模型2上传 | 30GB | 1-2小时 |
| **总计** | **~60GB** | **2-4小时** |

*时间取决于网络速度*

---

## 💡 提示

### 加速上传

1. **使用更快的网络**
   - 校园网/公司网络通常更快
   - 避免高峰时段

2. **分批上传**
   - 可以先上传一个模型
   - 验证无误后上传第二个

3. **使用镜像**（下载时）
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

### 避免问题

1. **确保.gitignore正确**
   - 在push前检查文件大小
   - 使用`git status`确认

2. **保持网络稳定**
   - 使用有线网络
   - 避免中断上传

3. **备份重要数据**
   - 上传前备份模型
   - 以防万一

---

## 📝 问题记录

如遇到问题，记录在此：

| 时间 | 问题 | 解决方案 |
|------|------|----------|
|      |      |          |

---

**创建时间**: 2025-11-27  
**最后更新**: 2025-11-27  
**状态**: 待执行
