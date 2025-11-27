# 🔧 修复GitHub认证问题

## 问题诊断

您遇到了两个问题：

### ❌ 问题1: 远程仓库URL错误
```
当前: https://github.com/bytedance/Sa2VA.git
应该: https://github.com/qimingfan10/RLSa2va.git
```

### ❌ 问题2: 认证失败
```
fatal: Authentication failed
```

---

## ✅ 快速修复

### 方案A: 使用Personal Access Token (最简单) ⭐

**第1步**: 修复远程仓库
```bash
cd /home/ubuntu/Sa2VA
git remote remove origin
git remote add origin https://github.com/qimingfan10/RLSa2va.git
```

**第2步**: 生成GitHub Token
1. 访问: https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选权限: `repo` (全部)
4. 生成并**复制token**（只显示一次！）

**第3步**: 推送代码
```bash
git push -u origin main
```

提示时输入：
- **Username**: `qimingfan10`
- **Password**: 粘贴你的token（不是GitHub密码！）

---

### 方案B: 使用SSH密钥（推荐长期使用）

**一键执行**:
```bash
cd /home/ubuntu/Sa2VA
chmod +x fix_github_remote.sh
bash fix_github_remote.sh
```

然后选择方式B，脚本会自动设置SSH。

**或手动设置**:
```bash
# 1. 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 查看公钥
cat ~/.ssh/id_ed25519.pub

# 3. 复制公钥内容，添加到GitHub
# 访问: https://github.com/settings/ssh/new

# 4. 更新远程仓库
git remote remove origin
git remote add origin git@github.com:qimingfan10/RLSa2va.git

# 5. 推送
git push -u origin main
```

---

## 🚀 推荐步骤（最快）

### 使用一键脚本：

```bash
cd /home/ubuntu/Sa2VA
chmod +x fix_github_remote.sh
bash fix_github_remote.sh
```

选择**方式A**（Token），然后：

1. 打开浏览器访问: https://github.com/settings/tokens
2. 生成token并复制
3. 回到终端按Enter
4. 输入用户名: `qimingfan10`
5. 输入密码: 粘贴token

完成！

---

## 📝 验证成功

推送成功后会看到：
```
Enumerating objects: xxx, done.
Counting objects: 100% (xxx/xxx), done.
...
To https://github.com/qimingfan10/RLSa2va.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

访问仓库确认: https://github.com/qimingfan10/RLSa2va

---

## ❓ 常见问题

### Q1: Token在哪里生成？
A: https://github.com/settings/tokens → Generate new token (classic) → 勾选 `repo`

### Q2: Token应该有什么权限？
A: 勾选 `repo` 即可（包含所有仓库权限）

### Q3: SSH密钥添加到哪里？
A: https://github.com/settings/ssh/new

### Q4: 如何查看当前远程仓库？
A: `git remote -v`

### Q5: 如何重新设置远程仓库？
A: 
```bash
git remote remove origin
git remote add origin <新地址>
```

---

## 🎯 下一步

推送成功后：

1. ✅ **验证GitHub仓库**: 访问 https://github.com/qimingfan10/RLSa2va
2. ✅ **上传模型到HuggingFace**: `bash scripts/upload_to_huggingface.sh`
3. ✅ **完善README**: 添加模型下载说明

---

**状态**: 等待修复  
**预计时间**: 5-10分钟
