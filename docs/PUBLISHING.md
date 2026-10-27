# 上传 GitHub

此文件夹是独立整理的源码发布副本。初始化 Git 时必须进入该文件夹；不要在包含训练数据和权重的上一级项目执行 `git add .`。

```bat
cd /d C:\work\Speech_agent_GitHub
git init
git add .
git status --short
git diff --cached --stat
```

检查待提交文件，确认没有 `.env`、录音、数据集、模型、特征、运行结果、临时文件和个人服务器信息。只有 `.env.example`、手写 JSONL 及未录制模板属于源码样例。

检查完成后：

```bat
git commit -m "Initial source release"
git branch -M main
```

然后在自己的 GitHub 账号下创建空仓库，不添加与本地冲突的初始化文件。复制 GitHub 显示的真实仓库地址，再执行：

```bat
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

占位账号及仓库名需要替换。认证在自己的电脑完成，不把访问令牌写进 URL、源码或聊天。

当前尚未选择源码许可证。维护者正式发布前应选择适合的许可证并补充作者信息；第三方模型和数据各自遵循其条款。这里没有替维护者授予额外的代码或数据授权。

根目录 `.gitignore` 排除了下载资源及生成结果；已有 Git 追踪的文件不受新增忽略规则影响。首次提交之前就应核对文件列表。代码发布不要求生成公网网页链接，README 可用架构图、复现说明和之后自行录制的脱敏演示视频展示功能。
