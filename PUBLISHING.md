# 匿名 GitHub 发布操作

以下命令只在 `anonymous_inference_artifact` 目录内操作，不会把上级完整项目加入仓库。

## 发布前注意事项

1. 使用专门的匿名 GitHub 账号和不含真实姓名的仓库名；不要 fork 现有仓库，fork
   关系会暴露来源。
2. 不要用个人邮箱提交。下面的 `git config --local` 只影响此仓库。
3. 三个大文件合计约 1.25 GB，必须使用 Git LFS。推送前确认匿名账号的 LFS
   存储和带宽配额足够。
4. 在 GitHub 网页创建一个空仓库，不要自动添加 README、LICENSE 或 `.gitignore`。

## 初始化与检查

```bash
cd /path/to/anonymous_inference_artifact

git init -b main
git config --local user.name "Anonymous Authors"
git config --local user.email "anonymous@users.noreply.github.com"

git lfs install --local
git lfs track "*.arrow"
git lfs track "*.safetensors"

git add .
git status
git lfs ls-files
```

`git lfs ls-files` 应列出且仅需包含下列三个发布工件：

```text
checkpoints/model.safetensors
data/tmwf_closed_world_test.arrow
data/mtwf_closed_world_test.arrow
```

提交前扫描常见身份信息和绝对路径：

```bash
rg -n -i '/home/|/Users/|/mnt/|C:\\Users\\|affiliation|university|e-mail|email' \
  --glob '!uv.lock' \
  --glob '!*.arrow' \
  --glob '!*.safetensors' .

git diff --cached --stat
git diff --cached -- . ':(exclude)*.arrow' ':(exclude)*.safetensors'
```

人工确认输出中没有真实姓名、学校、邮箱、个人用户名、项目内部路径和论文作者信息。
然后提交：

```bash
git commit -m "Release anonymous inference artifact"
git log -1 --format=fuller
```

确认 `Author` 和 `Commit` 都是匿名身份。

## 推送到 GitHub

先在匿名 GitHub 账号下创建空仓库，然后替换下面两个占位符：

```bash
git remote add origin https://github.com/ANONYMOUS_ACCOUNT/ANONYMOUS_REPOSITORY.git
git remote -v
git push -u origin main
```

HTTPS 推送要求使用该匿名账号的 token，而不是 GitHub 密码。也可以为匿名账号单独配置
SSH key 后改用 SSH remote。

推送后重新克隆一次做最终验收：

```bash
cd /tmp
git clone https://github.com/ANONYMOUS_ACCOUNT/ANONYMOUS_REPOSITORY.git artifact-check
cd artifact-check
git lfs pull
git lfs ls-files
uv sync --frozen
```

最后在 GitHub 网页检查仓库首页、commit 作者、文件历史和 Releases，不要在仓库简介、
Issues、Actions 日志或 commit message 中留下身份信息。若会议使用匿名链接服务，请再按
会议当年的匿名仓库政策处理 GitHub URL。
