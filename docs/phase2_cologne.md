# Phase 2：Cologne 新赛事测试与胜率校准

2026-10-28 已实际解析用户提供的 17 个 `.dem`，全部来自 **IEM Cologne Major 2026**。
17 个录制文件对应 14 场比赛的 14 张完整 Mirage 地图，包含三组 p1/p2，共 333 回合。
本轮使用此前在 Rio 验证集选出的 V1 模型，冻结其 47 个输入、预处理和全部系数，
只用两场旧比赛拟合额外的 sigmoid 校准参数，随后评估全部 Cologne 地图。

## 原始来源与分段

原文件仍在 `data/datasets/mirage demo/`，未改名或覆盖。各文件 SHA256、大小、
原始回合编号、比赛日期和 HLTV 链接固定在 `configs/cologne_local.json`。
赛事身份由用户确认；日期与地图根据各比赛公开页面及 Demo 回合核对。
例如 [G2–Spirit 四分之一决赛](https://www.hltv.org/matches/2394998/g2-vs-spirit-iem-cologne-major-2026)
和 [FURIA–Falcons 决赛](https://www.hltv.org/matches/2395002/furia-vs-falcons-iem-cologne-major-2026)。
本轮使用用户已有的本地文件，没有再下载或绕过站点访问保护；Bucharest 保留集没有被读取或加入。

| 比赛 | 日期 | 地图序号 | 回合 | 录制文件数 |
| --- | --- | ---: | ---: | ---: |
| Vitality–9z | 2026-06-12 | 2 | 22 | 1 |
| FUT–G2 | 2026-06-12 | 3 | 22 | 1 |
| B8–FUT | 2026-06-13 | 2 | 22 | 1 |
| Falcons–Monte | 2026-06-13 | 3 | 23 | 1 |
| Vitality–MOUZ | 2026-06-13 | 1 | 22 | 1 |
| NAVI–The MongolZ | 2026-06-13 | 3 | 19 | 1 |
| Spirit–9z | 2026-06-13 | 2 | 29 | 1 |
| NAVI–Falcons | 2026-06-14 | 2 | 20 | 1 |
| BETBOOM–FUT | 2026-06-15 | 1 | 20 | 1 |
| NAVI–G2 | 2026-06-15 | 3 | 23 | 2 |
| 9z–FURIA | 2026-06-18 | 2 | 22 | 2 |
| G2–Spirit | 2026-06-19 | 3 | 47 | 2 |
| Spirit–Falcons | 2026-06-20 | 2 | 21 | 1 |
| FURIA–Falcons | 2026-06-21 | 1 | 21 | 1 |

多分段先由 Awpy `Demo([p1, p2])` 原生校验地图、build、边界和回合连续性；
然后分别写出 Phase 1 四个文件，保留每段的 SHA256、局部 tick 和原始 `round_num`。
**不拼接二进制文件，也不平移 tick。** `map_id` 关联完整地图，`part_index` 标明录制顺序。
同一 `match_id` 所有分段必须处于同一集合；同图原始回合不允许在多个分段重复。
跨文件关联应使用 `demo_sha256 + round_id`；全局采样键为 `sample_id`，不能只用 tick。

## 数据质量

17 份 Phase 1 输出都通过 Polars/PyArrow 重读、Steam ID、血量、坐标、阵营、
每时点人数、重复键和回合映射检查。新语料包含：

- 65 名选手，Steam ID 保持 UInt64；真实战队身份与 T/CT 阵营分开。
- 6,087 个有效时点、60,870 条玩家状态，全部时点双方各 5 名选手。
- 49,869 条直接观测状态，以及 11,001 条由此前死亡事件保留的死亡身份记录。
- 死亡记录的当前坐标、血量、经济、装备保持 null；不会当成存活者参与聚合。
- Spirit–9z 请求的 525 个 tick 中有 1 个没有返回状态，实际保留 524 个，并记录警告。
- 质量异常数为 0；缺失 tick 和死亡实体缺失的警告没有被隐藏或补造。

原始武器栏 null 还可能表示空栏，不视为某种武器。完整逐列缺失统计与逐文件质量摘要
保存在 `reports/cologne_experiment.json`，原始报告位于各文件的 `metadata.json`。
本轮模型仍使用 V1 状态特征；没有为 Cologne 拟合 V2、Rating、Elo 或玩家集合模型。

## 固定实验与防泄漏

实验计划在拟合校准器和首次测试评分前保存于 `configs/phase2_cologne_experiment.json`。
计划包含每场分组、原模型 SHA256、参考数据集哈希、固定校准参数和 bootstrap 随机种子。

| 集合 | 来源/日期 | 比赛数 | 回合 | 时点 |
| --- | --- | ---: | ---: | ---: |
| train | Rio，4 月 13–17 日 | 7 | 132 | 2,281 |
| validation | Rio，4 月 18 日 | 1 | 17 | 315 |
| calibration | Rio，4 月 19 日；Atlanta，5 月 13 日 | 2 | 42 | 737 |
| test | Cologne，6 月 12–21 日 | 14 | 333 | 6,087 |
| 总计 | 三个赛事 | 24 | 524 | 9,420 |

每个回合的时点权重之和为 1，防止较长回合获得更多权重。共保留 **94,200 个玩家节点**，
没有把十人节点结构丢弃。`features.parquet` 使用固定白名单，`labels.parquet` 单独存放最终胜方。
日期必须严格按四集合顺序排列；同场、同日期不跨集合，Cologne 只能出现在 test。

基线文件是 `models/logistic_rio_v1_stable/model.json`，没有重新拟合它的缺失填补、
标准化或系数。运行时逐项比较当前与参考数据集的训练特征、标签和样本键，必须完全一致。
验证集只负责此前的 V1/V2 选择；本轮没有再比较候选模型。

校准形式为 `sigmoid(a × 原模型logit + b)`。参考
[scikit-learn 官方概率校准文档](https://scikit-learn.org/stable/modules/calibration.html)，
分类器与校准器的拟合数据必须独立。这里使用单变量正则逻辑回归，固定 C=1、lbfgs、
max_iter=2000、seed=42，并继续按回合加权；不是额外搜索的超参数。
得到 a=0.6324044127，b=-0.0898762468。正斜率保持排序，因此 AUC 与原模型一致。
JSON 模型先保存冻结，再读取并评分；便携推理接口支持旧 JSON 和新校准 JSON。

**Rio 4 月 19 日的原测试比赛已在旧试验中看过结果，本轮将其明确改作 calibration，
不能继续把它作为新的独立测试。** Cologne 在本轮首次评分后也已被观察，后续调参需要
另一个尚未使用的赛事测试集；本次不能作为无限次调参后的独立评估。

## 真实测试结果

以下指标按回合加权。Log Loss、Brier、ECE 越低越好；ECE 使用固定十个等宽概率箱。

| 模型 | Log Loss | Brier | AUC | ECE |
| --- | ---: | ---: | ---: | ---: |
| 训练集 CT 胜率常数先验 | 0.684063 | 0.245400 | 0.500000 | 0.041633 |
| 冻结 V1 | 0.473132 | 0.156716 | 0.854564 | 0.058587 |
| 冻结 V1 + 旧比赛校准 | 0.459119 | 0.152833 | 0.854564 | 0.031416 |

按 14 场比赛整组进行 1,000 次配对 bootstrap，seed=42，保留同场时点与回合相关性。
校准相对原模型的 Log Loss 差值 95% 区间为 **[-0.0303, 0.0030]**，
Brier 差值为 **[-0.00864, 0.00141]**。均跨过 0，所以此次观察到改善，
但不足以确认稳定收益。区间只反映这一个测试赛事内部的比赛抽样不确定性，
不估计赛事之间的变化。两场校准比赛的覆盖面仍然很小。

`reports/cologne_experiment.json` 经独立脚本核对模型/数据 SHA256、重新推理、
重新计算指标以及逐个 Phase 1 文件重读，审计通过。

## 生成文件与复现命令

| 文件或目录 | 用途 |
| --- | --- |
| code/src/cs2_pipeline/local.py | 本地清单导入、哈希检查、原生多分段校验、可验证缓存复用 |
| code/src/cs2_pipeline/calibrate.py | 冻结模型核验、仅校准集拟合、跨赛事评分、比赛聚类区间 |
| code/src/cs2_pipeline/split.py | 保留原三集合，新增固定四集合及独立测试赛事规则 |
| code/src/cs2_pipeline/dataset.py | 分段身份、重复地图回合检查、冻结策略重读校验 |
| code/src/cs2_pipeline/predict.py | 提取未校准 logit，兼容可选 JSON 校准参数 |
| tests/test_local_and_calibration.py | 分段、四集合、校准隔离、UInt8 运算、真实语料回归检查 |
| scripts/audit_cologne.py | 独立重算和质量摘要 |
| configs/cologne_local.json | 17 文件来源与 SHA256 固定清单 |
| configs/phase2_cologne.json | 17 份实际输出的 Phase 2 来源清单 |
| configs/phase2_cologne_experiment.json | 27 份录制、24 场比赛的固定四集合计划 |
| data/processed/cologne/ | 17 份 Phase 1 输出和导入报告 |
| data/datasets/mirage_cologne_experiment_v1/ | 五张数据表与 dataset.json |
| models/logistic_cologne_calibrated_v1/ | 校准模型、报告、测试预测及全量推理 |
| reports/cologne_experiment.json | 小型审计报告、缺失统计、指标和区间 |

依赖与旧版相同，已有 `.venv` 可以直接运行。严格复现安装：

```powershell
uv sync --frozen --extra dev --extra ml --extra context
```

使用 Python 模块入口，不需要重新安装额外 CLI。命令均从项目根目录执行；
macOS/Linux 将 `.venv\Scripts\python.exe` 替换成 `.venv/bin/python`。

```powershell
# 本地导入可重复执行：核对原始 SHA256、来源、5 秒采样与缓存质量后复用
.venv\Scripts\python.exe -m cs2_pipeline.local

# 以下目录已有结果，重建时请使用新的 --out，避免覆盖
.venv\Scripts\python.exe -m cs2_pipeline.dataset --manifest configs/phase2_cologne_experiment.json --out data/datasets/mirage_cologne_rebuild
.venv\Scripts\python.exe -m cs2_pipeline.calibrate --dataset data/datasets/mirage_cologne_rebuild --reference-dataset data/datasets/mirage_rio_v1 --base-model models/logistic_rio_v1_stable/model.json --out models/logistic_cologne_rebuild

# 对已有正式产物进行审计
.venv\Scripts\python.exe scripts/audit_cologne.py

# 固定模型推理，不拟合任何参数；输出文件也应是新路径
.venv\Scripts\python.exe -m cs2_pipeline.predict --model models/logistic_cologne_calibrated_v1/model.json --features data/datasets/mirage_cologne_experiment_v1/features.parquet --out data/datasets/cologne_inference.parquet

.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests scripts
```

没有真实文件的其他机器可运行合成表测试；真实 Cologne 集成测试会明确跳过，
不会把合成测试计为真实 Demo 解析证据。原始文件、Parquet、模型目录继续排除 Git。

本次实际执行结果为 **94 passed，0 skipped**，其中 6 项是带 `integration` 标记的
真实数据检查，包括真实 Demo 端到端重解析、新 Cologne 语料完整性和上下文检查。
Ruff 通过。另已重新执行本地导入命令，确认 17 份原始 SHA256 与 5 秒缓存均可验证复用；
已对校准模型执行全量 9,420 条推理并重读 Parquet。

下一步应扩大训练与校准赛事，做按日期推进的赛事验证，再比较 V2 的因果时钟、
炸弹和武器特征。当前 Cologne 可作为已观察的回归基准；新的最终测试赛事需在
后续选择之前固定。Bucharest 继续保留。本阶段没有实现市场接口或交易系统。
