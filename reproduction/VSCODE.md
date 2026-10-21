# 在 VS Code 中配置和启动训练

参数文件：项目根目录的 `.vscode/launch.json`。这是支持中文注释的 JSON 配置，可在编辑器中逐项修改；训练曲线和指标图表需另外配置。

## 使用

1. 用 VS Code 打开整个 `Minimind` 文件夹。需要 Microsoft 的 Python 和 Python Debugger 扩展，本机已检测到安装。
2. 打开 `.vscode/launch.json`，修改选定配置的 `args`，保存文件。
3. 按 `Ctrl+Shift+D`，在“运行和调试”的下拉列表中选择 `01 预训练试跑（128条，先运行这个）`。
4. 按 `Ctrl+F5` 普通运行；需要断点调试时按 `F5`。日志显示在集成终端中。停止按钮可终止运行。

请从“运行和调试”的命名配置启动；编辑器右上角“运行 Python 文件”可能使用另一套参数。
训练中修改配置不会改变当前进程，需停止并重新启动。

## 配置含义

- `python`：本机 Conda 环境解释器 `C:/Anaconda/envs/minimind/python.exe`。明确指定后不依赖终端是否已激活该环境；其他电脑需修改此路径。
- `program`：预训练或 SFT 脚本。
- `cwd`：固定为 `trainer`，兼容原脚本 `../model`、`../out` 等相对路径。
- `args`：原训练入口支持的命令行参数，文件中已显式列出数值及路径参数。开关 `--use_wandb` 默认省略（关闭）；原代码实际使用 SwanLab。
- `${workspaceFolder}`：当前打开的项目根目录。

例如，`"--batch_size", "2"` 表示每批 2 条，`"--learning_rate", "0.0005"` 表示学习率。值用字符串表示，由原脚本的 argparse 转为整数或小数。
模型中未开放为命令行参数的架构细节（如词表大小、注意力头数）仍由 `model/model_minimind.py` 定义，本次不修改训练或模型源码。

## 三套配置的范围

| 配置 | 数据与目的 | 权重输出 |
| --- | --- | --- |
| 01 预训练试跑 | 原预训练文件前 128 条，batch=2、累积=1、1 epoch，预计 64 步 | `out/smoke/pretrain_smoke_768.pth` |
| 02 完整预训练 | 完整 mini 预训练数据，候选 batch=4、累积=64 | `out/pretrain_768.pth` |
| 03 完整 SFT | 完整 mini SFT 数据，候选 batch=2、累积=8 | `out/full_sft_768.pth` |

本次准备 `dataset/pretrain_smoke.jsonl` 为原预训练数据的前 128 条，仅用于验证流程，并非独立验证集。
两套完整训练参数尚未经过本机训练和性能验证；先试跑，再测显存和耗时，并完成 SFT 小规模验证后再使用。
SFT 的 `from_weight=pretrain` 需要先有 `out/pretrain_768.pth`；试跑权重保存在另一个目录，不能直接替代完整预训练成果。
断点状态由原代码保存到 `checkpoints/`，其文件名前缀随 `save_weight` 变化。`from_resume=0` 表示重新开始；要恢复对应实验需改为 `1` 并保持数据、模型结构与配置一致。

## 本次检查（2026-10-20）

- 已验证 Python 3.10 环境路径、PyTorch `2.6.0+cu124`、CUDA 可用、RTX 4060 Laptop GPU、BF16 支持及微小矩阵运算。
- `pip check` 未发现已安装包的依赖冲突；这不代表逐项核验了完整依赖清单。
- 两份原始数据各前 128 条可解析且具有 `text` / `conversations` 必要字段；尚未完整扫描数据。
- 配置交付只检查配置结构、参数解析与路径；不启动训练，不宣称训练阶段验收完成。

提交到 GitHub 的内容包括运行配置、说明和忽略规则。原始数据、试跑数据、`out/` 权重及 `checkpoints/` 断点不进入 Git。

VS Code 官方说明：https://code.visualstudio.com/docs/python/debugging

## 训练前后交互比较（2026-10-21 新增）

“运行和调试”现在还有 **04 训练前后交互续写对比**，它启动 `scripts/compare_pretrain.py`，只运行推理。按 Ctrl+F5 后在终端输入文本，同一个提示会分别让 seed 重建的初始化基线和本次试跑权重续写。
输入 `/examples` 运行固定示例，`/exit` 退出。用法、基线限制及记录位置见 [COMPARE_PRETRAIN.md](COMPARE_PRETRAIN.md)。

编号 04 是启动配置编号，与 PLAN 中的训练性能测试阶段编号没有对应关系。
