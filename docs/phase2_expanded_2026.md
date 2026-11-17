# Phase 2：2026 多赛事扩充训练

2026-11-17 完成 14 张新增 Mirage 地图的获取、解析、质量检查和训练：
8 张来自 [OpenCS2 公开镜像](https://huggingface.co/datasets/blanchon/opencs2_dataset_demo)
中此前未使用的 Rio 比赛，6 张来自用户提供的 HLTV 压缩包。
后者分别是 [BLAST Open Porto](https://www.hltv.org/events/8249/blast-open-porto-2026)、
IEM Beijing 和 [BLAST Rivals S2](https://www.hltv.org/events/8251/blast-rivals-2026-season-2)。
这些压缩包的 Mirage 地图比分与六个 HLTV 比赛页面逐一核对一致。

共 41 个录制文件对应 38 场完整地图；Cologne 三组多分段保留各自局部 tick。
8 张镜像 Demo 校验了镜像声明的大小与 LFS SHA256；6 个用户压缩包和解压文件
记录了本地 SHA256，但不能将其描述成 HLTV 官方提供的校验值。未加入 Bucharest 保留集。

## 统一解析与数据质量

本轮使用 [Awpy 3.3.0](https://pypi.org/project/awpy/3.3.0/) 统一重解析所有 41 个文件，
保留旧版 3.1.1 的原始处理结果和旧模型。由于 3.2 以后修复了死亡玩家身份及手雷数量，
避免在同一个新数据集里混用两种解析口径。

以 Atlanta 那场 Demo 的对照为例：新版直接保留此前需按死亡事件补齐的 499 条死亡身份；
两版的存活状态一致，197 条存活玩家状态的闪光弹数量发生改变。新版数据已作为新实验独立输入，
不能把旧模型与新模型的差异完全归因于训练比赛数量。

全部 41 个录制文件的逐行对照中，146,300 个 `(tick, steamid)` 键完全一致，
阵营、队名和存活状态均未改变；其中双方版本都标记为存活的记录里，有 12,734 条
闪光弹数量不同。详细字段和逐文件统计见 `reports/awpy_3_3_comparison.json`。

所有 41 份结果通过写入与重读验证，回合编号、战斗边界、胜方和有效性均与旧解析一致；
原有 Spirit–9z 的一个缺失采样 tick 保持警告，没有补造。总计保留 108 名选手、
811 回合、14,630 个有效时点和 146,300 个玩家节点。Steam ID 继续为 UInt64。

## 训练与赛事划分

在模型训练和本次测试评分之前固定了数据、特征及比较模型；不同赛事不跨集合。
此前 Cologne 的结果已经观察，因此本次允许将它转作训练，不能再称其为新模型的独立测试。

| 集合 | 赛事 | 比赛 | 回合 | 时点 |
| --- | --- | ---: | ---: | ---: |
| 训练 | Rio、Atlanta、Cologne、Porto | 34 | 730 | 13,208 |
| 验证 | IEM Beijing | 2 | 40 | 729 |
| 测试 | BLAST Rivals S2 | 2 | 41 | 693 |

使用固定 V1 的 47 个特征，训练集拟合缺失值处理、加权标准化与逻辑回归，
每回合总权重为 1；C=1、lbfgs、随机种子 42。训练实际收敛于 39 次优化迭代。
本轮新模型**未做概率校准**，也没有用测试集拟合任何参数。

对照是事先固定的旧 Cologne 校准模型；两个模型在同一批 Awpy 3.3.0 解析的样本上评分。
旧模型训练时使用旧版解析器，所以差异同时包含训练数据量和解析口径的影响。
指标按回合加权，Log Loss、Brier 和 ECE 越低越好：

| 赛事 / 模型 | Log Loss | Brier | AUC | ECE |
| --- | ---: | ---: | ---: | ---: |
| Beijing 验证：扩充 V1 | 0.4411 | 0.1498 | 0.8713 | 0.0879 |
| Beijing 验证：旧校准模型 | 0.5121 | 0.1734 | 0.8465 | 0.1280 |
| BLAST 测试：扩充 V1 | 0.5327 | 0.1840 | 0.7995 | 0.1317 |
| BLAST 测试：旧校准模型 | 0.5464 | 0.1927 | 0.7905 | 0.1192 |
| BLAST 测试：训练集常数先验 | 0.6986 | 0.2527 | 0.5000 | — |

测试中两场都涉及 FURIA。新模型整体 Log Loss 略低，但其中一场高于旧模型，
且整体 ECE 高于旧校准模型。只有两场比赛，不能据此断定泛化或概率校准稳定改善。
BLAST 测试现在也已被观察，后续调参需重新设置未见过的赛事测试。

## ESL Pro League Season 24 补充赛事留出评估

模型和训练集在收到这批文件之前已固定。用户提供了六场含 Mirage 的 HLTV 压缩包：
Falcons–TYLOO、MOUZ–M80、Vitality–Natus Vincere、Falcons–Spirit、
Aurora–Vitality、MOUZ–Spirit。解析后均通过数据质量和公开地图比分核对；
队伍别名只用于核对比分，Parquet 中保留 Demo 中的原始真实队名。
赛事留出集包含 **6 场、120 回合、2,237 个时点**，与现有全部训练、验证和测试比赛及 Demo 哈希均不重叠。

| 评估对象 | Log Loss | Brier | AUC | ECE |
| --- | ---: | ---: | ---: | ---: |
| 冻结的扩充 V1 模型 | 0.4468 | 0.1505 | 0.8715 | 0.0569 |
| 原训练集常数先验 | 0.7107 | 0.2587 | 0.5000 | 0.0991 |

指标按回合加权：每个回合的时点总权重为 1，故不把同回合多个时点当成独立回合。
此表比较的是简单的训练胜率先验，不代表战胜了其他复杂模型。比赛按场次重采样得到的区间
和逐场结果保存在 `models/evaluations/esl24_v1/report.json`，只能反映这一个赛事内的描述性不确定性；
赛事仅选取了六场且存在重复战队，不能推断赛事间泛化能力。

ESL 比赛日期晚于模型训练截止日 2026-09-06，但早于已用作验证的 Beijing；这是一项回顾性
赛事留出评估，不是实时部署的前瞻测试。选择 ESL 前已经看过 Beijing 和 BLAST 的结果；
其后固定模型、在获取 ESL 文件前列出这六场，并且没有利用 ESL 结果训练、校准或调参。
该赛事现已观察，未来再次选择模型应预留新的赛事测试集。Bucharest 继续保持未触碰。

数据集由 `configs/esl24_holdout_plan.json`、`configs/esl24_holdout_sources.json` 和
`configs/esl24_user_hltv_sources.json` 约束，生成 `data/datasets/esl24_holdout_v1/`；
其 `holdout_only` 标记会阻止训练命令读取，包括 smoke test。
评估阶段检查模型、计划、训练集和文件哈希；预测与标签分开保存。

如需重建留出集及用同一冻结模型重新评分，请为输出指定新目录，避免覆盖已有结果：

```powershell
.venv\Scripts\python.exe -m cs2_pipeline.holdout build --out data/datasets/esl24_rebuild
.venv\Scripts\python.exe -m cs2_pipeline.holdout score --dataset data/datasets/esl24_rebuild --out models/evaluations/esl24_rebuild
```

## 复现与检查

```powershell
uv sync --frozen --extra dev --extra ml --extra context

# 已验证的下载与用户压缩包可复用；用户文件必须仍位于固定清单所列位置
.venv\Scripts\python.exe -m cs2_pipeline.acquire --manifest configs/rio_additional_downloads.json --raw data/raw/rio_additional --processed data/processed/rio_additional --dataset-manifest configs/phase2_rio_additional.json
.venv\Scripts\python.exe scripts/import_user_hltv.py
.venv\Scripts\python.exe scripts/reparse_consistent.py

# 已有输出目录不能覆盖；复现时使用新的名称
.venv\Scripts\python.exe -m cs2_pipeline.dataset --manifest configs/phase2_expanded_2026.json --out data/datasets/mirage_expanded_rebuild --group-by event
.venv\Scripts\python.exe -m cs2_pipeline.train --dataset data/datasets/mirage_expanded_rebuild --out models/expanded_rebuild

# 复核正式数据集的源文件、赛事划分与 HLTV 地图比分
.venv\Scripts\python.exe scripts/audit_expanded_data.py
```

训练还可在 [本地训练界面](http://127.0.0.1:8765) 选择“2026 多赛事 · V1 扩充训练”。
该界面按真实阶段更新，输出到新目录。已有模型和报告：

- 数据清单：`configs/phase2_expanded_2026.json`。
- 数据质量审计：`reports/expanded_2026_data_audit.json`。
- 模型与预测：`models/monitor_runs/20261117T052929-9d7d1b12/`。
- 两模型重推理比较：`reports/expanded_2026_model_audit.json`。
- 统一解析计划：`configs/reparse_expanded_2026.json`。
- 新旧解析器字段对照：`reports/awpy_3_3_comparison.json`。

下一步应增加赛事覆盖，并在未参与模型选择的新赛事上评估；另需保留独立比赛做概率校准。
若要比较 V2，必须为全部新增来源提取因果时钟和武器上下文，不将旧版上下文直接混入新解析。
