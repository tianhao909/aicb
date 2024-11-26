
# github fork过的仓库 ， 怎么保证和原仓库同步， 原仓库的更新怎么同步到fork的仓库里面来？  中文回答

在 GitHub 上，如果你 fork 了一个仓库，并希望将原仓库的更新同步到你的 fork 中，可以按照以下步骤操作：

添加上游仓库 (Upstream Repository) 作为远程仓库： 首先，确保你的本地仓库已经关联了原仓库（即上游仓库）。如果你没有添加过上游仓库，可以通过以下命令添加：

git remote add upstream <原仓库的URL>
例如，如果原仓库是 https://github.com/original_owner/repository_name.git，你可以执行：

git remote add upstream https://github.com/original_owner/repository_name.git

git remote add upstream git@github.com:aliyun/aicb.git
获取原仓库的最新更新： 使用 git fetch 从上游仓库获取最新的更新：

git fetch upstream
合并上游的更新到你的 fork： 获取到上游仓库的更新后，你需要将其合并到你的本地分支。假设你想将更新合并到 main 分支，可以执行以下命令：

git checkout main  # 切换到你的 main 分支
git merge upstream/main  # 将 upstream 仓库的 main 分支合并到你的本地 main 分支
如果你在其他分支上工作，也可以将其合并到对应的分支。

解决冲突（如果有的话）： 如果合并时出现冲突，Git 会提示你手动解决冲突。解决冲突后，记得提交变更：

git add .
git commit -m "Resolved merge conflicts"
推送到你的 GitHub 仓库： 最后，更新合并后的分支推送到你自己的 GitHub 仓库：

git push origin main
小结
通过以上步骤，你可以将原仓库的更新同步到你的 fork 仓库中，确保你的 fork 保持与原仓库的同步。

你也可以定期执行 git fetch upstream 和 git merge upstream/main 来确保你的 fork 不会过时。


https://github.com/tianhao909/aicb/tree/master

点击synb fork

然后 git pull  就更新了
