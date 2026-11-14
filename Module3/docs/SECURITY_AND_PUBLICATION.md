# GitHub 隐私与数据发布说明

## 仓库中刻意排除的内容

- 原始 M5 数据和价格文件。
- 完整预测 Parquet 树，约 901 MB。
- Python 缓存、编译字节码和编辑器临时文件。
- 姓名、学生 EID、私人邮箱、本机用户名和绝对路径。
- 模型权重、登录凭据和环境密钥。

## 首次提交前

Git 提交作者会写入提交历史。个人电脑如果配置了私人邮箱，应先改为 GitHub 提供的 noreply 邮箱：

```bash
git config user.name "Your GitHub Name"
git config user.email "YOUR_GITHUB_NOREPLY_EMAIL"
```

此配置必须在第一次提交前完成。即使之后修改配置，旧提交中的作者信息也不会自动消失。

## 推荐发布方式

1. Git 只保存代码、小型契约文件和诊断摘要。
2. 完整预测写入 `.gitignore` 中的本地目录，例如 `prediction_outputs/`。
3. 如需共享完整结果，单独使用受控附件或 GitHub Release，并先核实数据许可。
4. 合并到小组总仓库时，还要检查其他模块文档是否含组员姓名或 EID。

## 发布前扫描

建议使用 `gitleaks` 或 GitHub secret scanning，并人工检查代码、Markdown、JSON 和提交作者信息。再确认仓库中没有 Python 缓存、操作系统元数据、个人目录或私人邮箱。

公开包只保留角色描述，例如 “Module 3 lead”，不保留具体学生身份。

