# Phase 2 进展：独立比赛评估与因果上下文

> 此页保留早期 Rio 试验结果。最新 [Cologne 实验](phase2_cologne.md) 将 Rio 4 月 19 日
> 的原测试比赛用于校准；它已不再是当前实验的独立测试集。原文件和历史分数均保留。

2026-10-28 已实际完成 9 场 IEM Rio 2026 Mirage 的下载、解析、上下文提取、
训练集构建、两组固定逻辑回归训练、独立比赛评估和分数审计。
这是一轮同赛事小规模试验，不代表已经获得可靠的跨赛事胜率模型。

## 真实数据与划分

固定选取 9 场公开原始 Demo，共约 3.92 GB，来源和许可见 [数据署名](DATA_SOURCES.md)。
九个文件的大小和 SHA256 全部匹配，Awpy 解析质量检查全部通过。
每场总回合数也与来源目录独立统计一致。
来源清单只依据可获得性和日期选取，未根据模型结果增删比赛。
原先 Atlanta 单场流程检查的数据没有加入本次 Rio 训练/验证/测试。

| 集合 | 日期 | 比赛数 | 有效回合 | 采样时点 |
| --- | --- | ---: | ---: | ---: |
| train | 2026-04-13 至 2026-04-17 | 7 | 132 | 2,281 |
| validation | 2026-04-18 | 1 | 17 | 315 |
| test | 2026-04-19 | 1 | 24 | 442 |
| 合计 |  | 9 | 173 | 3,038 |

保留 **30,380 个玩家节点**。同比赛、同日期锁定分组，按日期划分；
两组实验的样本 ID、标签、时点、比赛身份和集合划分全部一致。
内置 Bucharest 保留赛事排除规则仍然有效。
本轮是按比赛隔离的同赛事评估；按赛事隔离模式需要更多独立赛事，未假称已经完成。

## 新增数据

`context.py` 逐 tick 保存：

- 当前回合的游戏内经过时间，以及未安放炸弹时的回合剩余时间。
- 炸弹是否安放、A/B 点位，以及安放后经过的游戏时间。
- 当前是否暂停；暂停或暂停状态未知时，时钟保持 null。
- 已在安放前记录引信时长时的炸弹剩余时间；否则为 null。

主解析器仍为 **Awpy 3.1.1**，负责回合、玩家、身份、装备和炸弹事件。
补充依赖 **demoparser2 0.42.1** 只按选定 tick 读取游戏规则时钟，避免 Awpy
`ticks(players_only=False)` 全量导出所有实体的内存成本。
接口及网络属性已核对安装包与官方源代码，并用真实 Demo 验证。

倒计时使用当前 `game_time`、`m_fRoundStartTime` 和 `m_iRoundTime`，
不使用最终回合结束时间。炸弹计时只查看当前 tick 之前/当时的安放事件。
引信时长只接受安放前的 `server_cvar` 记录，不使用整场最后值或后续爆炸时刻推断。

本批 3,038 个时点全部具有游戏内经过时间，2,597 个未安放时点有回合倒计时，
441 个安放后时点有安放经过时间。**本批没有录制可用于此用途的引信时长，
所以炸弹剩余时间 3,038 行均为 null。** 未填充默认 40/45 秒。
完整性统计与来源哈希保存在每份 `context_metadata.json`。

## 特征版本与兼容性

- V1：47 个特征，保留原有行为及旧模型兼容性。
- V2：96 个特征，移除 V1 的 tick 经过时间作为模型输入，加入上述游戏内上下文，
  并加入仅存活玩家的 AWP、AK、M4、其他步枪/狙击枪、冲锋枪、霰弹枪、机枪、
  重型/普通手枪、空武器栏和未知武器类别的阵营统计及差值。

武器栏未知与“已观测到空栏”分开处理，使用观察掩码；死亡玩家不计入武器统计。
未知武器名称会进入 unknown 类别，不猜测它属于哪种枪。
Steam ID 继续保留为 UInt64，姓名、真实队名、回合编号和最终赢家不进入模型。
两种特征都有固定白名单；旧的单场 V1 JSON 模型重新推理结果与原文件完全一致。

缺失值填补、缩放和分类器仍只拟合训练集。每回合总样本权重为 1。
为消除常量列加权方差的浮点负值警告，缩放采用两遍计算的非负加权方差；
未改变数据划分或正则化参数，并验证 JSON 推理与 sklearn 一致。

## 两组固定模型结果

指标按回合加权，Log Loss 和 Brier 越低越好。两组都使用 C=1 的逻辑回归，
没有因测试分数调参或继续筛选特征。

| 模型 | 特征数 | 验证 Log Loss | 测试 Log Loss | 测试 Brier |
| --- | ---: | ---: | ---: | ---: |
| 训练回合胜率常数先验 | — | 0.7380 | 0.7440 | 见完整报告 |
| V1 状态基线 | 47 | **0.4292** | **0.5336** | **0.1811** |
| V2 上下文与武器 | 96 | 0.5797 | 0.6787 | 0.2163 |

按预定的“验证 Log Loss 较小”规则，当前参考模型为 **V1**。
V2 在本轮没有改善结果，因此没有宣称加入特征必然提升性能。
验证与测试各只有一场比赛，来源都属于同一赛事，尚未进行跨赛事泛化检验或概率校准。
新增上下文仍可用于后续更大数据集、消融实验或玩家集合模型。

`reports/rio_pilot.json` 会验证两个数据集的标签和划分完全相同，
核对模型报告绑定的数据集哈希，再从预测文件重新计算指标。
模型选择只使用验证集；测试分数不进入选择规则。

## 已生成的主要文件

| 路径 | 内容 |
| --- | --- |
| configs/corpus_downloads.json | 固定下载地址、日期、事件、SHA256 和署名 |
| configs/phase2_corpus.json | 9 场已验证 Phase 1 结果清单 |
| configs/phase2_corpus_v2.json | 对应上下文及 feature_version=2 清单 |
| data/raw/corpus/ | 真实 .dem、来源 JSON 和 acquisition_report.json |
| data/processed/corpus/ | 每场的 Phase 1 四个文件 |
| data/context/corpus/ | 每场 context.parquet 和 context_metadata.json |
| data/datasets/mirage_rio_v1/ | V1 玩家节点、时点、特征、标签、划分与审计清单 |
| data/datasets/mirage_rio_v2_checked/ | V2 数据，包含上下文输入的哈希 |
| models/logistic_rio_v1_stable/ | 参考模型、独立评估报告和留出预测 |
| models/logistic_rio_v2_stable/ | V2 候选模型、报告和预测 |
| reports/rio_pilot.json | 重新计算并审计过的对比报告 |

原始 Demo、上下文、数据集和模型均已被 `.gitignore` 排除；小型来源清单和比较报告可保留。
初始实验目录 `mirage_rio_v2`、`logistic_rio_v1`、`logistic_rio_v2` 是过程产物；
正式引用上表中的 checked/stable 路径。

## 重现流程

安装全部依赖：

```powershell
uv sync --frozen --extra dev --extra ml --extra context
```

获取并解析固定语料；已存在的文件必须重新通过哈希或质量验证才会复用：

```powershell
uv run --frozen --extra ml --extra context python -m cs2_pipeline.acquire
uv run --frozen --extra ml --extra context python scripts/enrich_corpus.py
```

数据和模型目录已存在。以下命令使用新的 reproduce 目录，以避免覆盖已有结果：

```powershell
uv run --frozen --extra ml python -m cs2_pipeline.dataset --manifest configs/phase2_corpus.json --out data/datasets/rio_v1_reproduce
uv run --frozen --extra ml python -m cs2_pipeline.train --dataset data/datasets/rio_v1_reproduce --out models/rio_v1_reproduce
uv run --frozen --extra ml python -m cs2_pipeline.dataset --manifest configs/phase2_corpus_v2.json --out data/datasets/rio_v2_reproduce
uv run --frozen --extra ml python -m cs2_pipeline.train --dataset data/datasets/rio_v2_reproduce --out models/rio_v2_reproduce
uv run --frozen --extra ml python -m cs2_pipeline.compare --baseline-dataset data/datasets/rio_v1_reproduce --baseline-model models/rio_v1_reproduce --candidate-dataset data/datasets/rio_v2_reproduce --candidate-model models/rio_v2_reproduce --out reports/rio_reproduce.json
```

用选定的参考 JSON 模型预测同地图的新特征：

```powershell
uv run --frozen --extra ml python -m cs2_pipeline.predict --model models/logistic_rio_v1_stable/model.json --features data/datasets/mirage_rio_v1/features.parquet --out models/logistic_rio_v1_stable/predicted_all_reproduce.parquet
```

这条命令输出所有时点供检查；其中训练时点不构成独立评估。
正式留出指标来自 `report.json` 和仅包含验证/测试时点的 `predictions.parquet`。

单独补充其他真实 Demo 的上下文：

```powershell
uv run --frozen --extra context python -m cs2_pipeline.context --demo data/raw/match.dem --processed data/processed/match --out data/context/match_new
```

V2 来源清单在每个 Phase 1 `path` 旁增加 `context_path`，路径均相对清单文件目录。
若来源出现登录、403 或限流，下载程序停止，不进行重试绕过；可按 Phase 1 说明人工提供文件。

## 测试与下一步

```powershell
uv run --frozen --extra dev --extra ml --extra context pytest -q --basetemp .pytest_cache/pilot-tmp
uv run --frozen --extra dev --extra ml --extra context ruff check src tests scripts
```

当前 **81 passed**，包含 5 项真实数据集成检查；其余为明确标注的模拟表/HTTP 测试。
新检查包括下载哈希和访问保护、未来事件不能回填过去、暂停未知值、引信在安放前锁定、
全局规则冲突、武器观察掩码、V1/V2 标签划分一致、JSON 推理和指标重算审计。
受限环境运行 pytest 时，basetemp 只能指向可清空的专用临时目录。

下一步应补充**其他 MVP 赛事**，保留新的时间/赛事盲测集，再研究校准与模型扩展。
本轮没有实现 Player Rating、Team Elo、GNN/Transformer、地图/整场胜率或市场交易功能。
