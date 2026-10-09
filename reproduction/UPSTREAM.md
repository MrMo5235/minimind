# 复现代码起点

记录日期：2026-10-09（Asia/Shanghai）。

| 项目 | 值 |
| --- | --- |
| 原仓库 | https://github.com/jingyaogong/minimind |
| 个人 Fork | https://github.com/MrMo5235/minimind |
| 上游来源分支 | master |
| 复现实验分支 | reproduce |
| 起点 commit | `f659b55761b754d306bd140573493a6543cafd7f` |
| 起点提交说明 | `[fix] residual step order` |
| 本地目录 | `F:\project\work\Minimind` |

此 commit 在首次记录时为本地 HEAD 与 `origin/master` 指向的代码版本。该记录不声称它一直是上游最新版本。

远程仓库用途：

- `origin` 指向个人 Fork，用于推送代码和成果。
- `upstream` 指向原项目，用于查看和获取更新。

基线源码：

- 模型：[`model/model_minimind.py`](../model/model_minimind.py)
- 预训练：[`trainer/train_pretrain.py`](../trainer/train_pretrain.py)
- SFT：[`trainer/train_full_sft.py`](../trainer/train_full_sft.py)
- 依赖：[`requirements.txt`](../requirements.txt)
- Tokenizer：`model/` 中随此版本提供的文件。

预训练脚本基线默认值：768 维、8 层、batch size 32、梯度累积 8、序列长度 340、2 epochs、学习率 5e-4。
SFT 脚本基线默认值：768 维、8 层、batch size 16、梯度累积 1、序列长度 768、2 epochs、学习率 1e-5。
个人显卡的候选配置见 [PLAN.md](PLAN.md)，实际采用值以各次实验记录为准。

复现期间不自动合并 upstream 更新。确实需要更新时，记录新旧 commit、原因和兼容性影响，并保留已有实验对应的代码版本。
保留项目原有许可证和署名；新增工作放在 `reproduction/` 中，必要的训练代码改动逐次说明。
