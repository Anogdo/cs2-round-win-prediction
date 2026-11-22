# 本地训练监控界面

从项目根目录运行，然后访问 **http://127.0.0.1:8765**：

```powershell
uv sync --frozen --extra dev --extra ml
.venv\Scripts\python.exe -m cs2_pipeline.monitor
```

macOS/Linux 使用 `.venv/bin/python -m cs2_pipeline.monitor`。没有新增运行依赖，
服务使用 Python 标准库，只监听本机回环地址。它不需要云服务、数据库或前端构建工具。
保持服务进程运行即可随时打开页面；关闭进程或重启电脑后，重新执行上述启动命令。
端口冲突可加 `--port 8766`，然后访问相同端口的地址。

## 页面可以做什么

- **开始训练**：可选时序 Transformer、多赛事 V1、GPU 树模型或历史 Rio V1/V2，启动对应的 Python 子进程。
- **固定模型评估**：可以在已观察的 50 场上重跑同 34 场拟合的模型对照，或在 Bounty 和 EWC 上验证 Porto 选定的时序模型。
- **真实阶段进度**：数据校验 → 模型拟合 → 验证与评估 → 保存并重读。每秒自动刷新。
- **运行耗时和迭代数**：拟合时显示实际耗时，结束后显示 sklearn 返回的真实优化迭代数。
- **实时日志与指标**：流式落盘的进程输出、验证集 Log Loss / Brier / AUC / ECE。
- **停止任务**：只终止此服务启动且仍持有的训练子进程，未完成输出不会标成成功。
- **历史记录**：查看已有模型结果，点击新运行记录可回看其阶段与日志。
- **额外赛事评估**：实验记录列出冻结模型的 ESL 24 留出集评分，明确标记为额外赛事评估。
- **批量数据进度**：50 场测试解析单独显示已处理、通过、失败数量和最近更新时间；进度按比赛计数。
- **数据概览**：所选数据集的集合场次、回合、时点和玩家状态数量。

百分比条表示四个阶段中已经完成的阶段，不代表优化器训练完成比例。
当前 sklearn LBFGS 接口没有提供逐迭代回调，所以不显示虚构的 epoch、实时损失曲线或 ETA。
拟合通常很快；无需为了显示动画而延迟真实训练。

页面的上方指标在选择任务后显示**该次任务的验证集**；历史表单独标明测试指标。
没有新任务时，上方显示此前已完成的 Cologne 校准模型参考结果。
所有这些旧评估均已被观察，页面复跑不构成新的独立测试。

## 数据与模型边界

时序预设使用 Rio 和 Atlanta 拟合，在 Porto 选取检查点，再评估已经看过的 50 场比赛。
该预设依赖 `temporal` extra 和归档数据。页面会显示真实的 epoch、耗时与验证指标。
[时序实验与环境说明](mirage_temporal_experiment.md)

两个固定模型评估预设使用 CUDA 和归档数据。34 场对照复用已保存的检查点，统一评估先前观察过的 50 场比赛。
Bounty 和 EWC 预设复用首轮 Porto 选定的检查点，在 10 场历史开发数据上评分。
[34 场对照](mirage_temporal_experiment.md#follow-up-identical-34-map-training-set-november-21-2026) · [历史验证](mirage_historical_validation_20261118.md)

多赛事配置使用 Rio、Atlanta、Cologne、Porto 训练，Beijing 验证、BLAST Rivals 测试。
ESL 24 的 6 场只用于固定模型的补充回顾性评估，显示在历史记录中，不在训练预设里。
历史 Rio 配置仍沿用 7 场训练、1 场验证、1 场已观察测试的原划分。
Bucharest 保留集仍未读取或加入；Cologne 已从先前测试集转为多赛事配置的训练数据。
预处理、特征白名单、回合权重和模型参数保持原有规则；日志不会成为模型输入。
可视化开关不改变模型系数，已有回归测试验证了这一点。

每次输出一个新目录，原有参考模型保持原样：

```text
runs/<运行ID>/
├── run.json          # 子进程状态、启动时间、PID、输出位置
├── progress.jsonl    # 各阶段真实事件及指标
└── training.log      # stdout / stderr

models/monitor_runs/<运行ID>/
├── model.json
├── report.json
└── predictions.parquet
```

刷新浏览器不会停止训练。已完成记录在服务重启后仍保留。服务如果在训练中重启，
不能安全重连原进程时会显示“状态待确认”，而非误报完成；原进程稍后写入终止事件时可恢复状态。
也可明确归档未知记录后启动新任务，**归档不会终止原进程**。

模型拟合阶段由现有训练入口执行，独立运行时也可以追加阶段日志：

```powershell
.venv\Scripts\python.exe -m cs2_pipeline.train --dataset data/datasets/mirage_rio_v1 --out models/my_new_run --progress-file runs/my_new_run/progress.jsonl
```

页面监控的是从该页面启动并登记的任务，不会自动接管其他终端中的任意进程。
训练失败时会保留具体异常和日志；没有 `report.json` 的退出进程不会显示为成功。
服务不使用 shell 拼接执行训练命令；写操作检查本地 Host、Origin 和页面会话令牌。

## 本次验证

已从浏览器点击按钮实际完成一轮 Rio V1 训练，四个阶段和日志均更新，实际优化 23 次。
验证集 Log Loss 为 0.4292319602，原测试基准 Log Loss 为 0.5336477445，
与原参考模型一致。历史文件没有被覆盖。

也已从页面选择并完成 V2：实际优化 32 次，验证集 Log Loss 为 0.5797374702，
原测试基准为 0.6786672755，与原结果一致。浏览器没有 JavaScript 错误；服务重启和
页面刷新后，已完成的 V1/V2 日志与指标仍能恢复。全量 **103 项测试通过**，Ruff 通过。


新增测试覆盖启动配置限制、单任务约束、停止、失败、服务重启状态、未知任务归档、
进度记录不改变模型、HTTP 接口路由及令牌/来源校验、重复端口占用和 ESL 留出评估展示。
2026 多赛事扩充和 ESL 评估完成后，全部 **112 项测试通过**，Ruff 检查通过。
50 场测试批次完成后，新增了分段回合恢复、档案校验、进度写入和测试数据展示检查；
该批次完成后全量 **118 项测试通过**。加入同 34 场对照、历史验证和后续实验代码后，本次发布检查为 **149 项测试通过**，Ruff 检查通过。

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check code/src scripts tests
```
