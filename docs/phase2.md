# Phase 2：训练集与回合胜率基线

本页记录单场 V1 首版。**最新进展见 [Rio 独立比赛试验](phase2_pilot.md)**：
新增 9 场真实 MVP 赛事 Mirage、96 维 V2 特征和按日期隔离的验证/测试。
原有 V1 数据及 JSON 模型仍可使用，已检查预测结果完全一致。

已在真实 Mirage 数据上完成训练集构建、基线拟合、模型导出和重新加载推理。
当前只有 1 场比赛、18 回合，正式验证集和测试集均为空。
此次训练明确标记为 **smoke_test（流程检查）**，没有报告准确率、AUC 或泛化性能。

## 安装和完整重跑

在项目根目录执行：

```powershell
uv sync --frozen --extra dev --extra ml
```

ML 依赖为 scikit-learn 1.9.1 和 NumPy。`uv.lock` 固定全部解析、训练及测试依赖。
为保持 Python 3.11+ 兼容性，锁文件按 Python 版本选择兼容的 NumPy/SciPy；
本机 Python 3.12.14 实际使用 NumPy 2.5.4。尚未在其他操作系统运行。

已经生成 `data/datasets/mirage_v1/` 和 `models/logistic_smoke_v1/`。
以下重跑命令使用新的 v2 目录，避免覆盖既有结果：

```powershell
uv run --frozen --extra ml python -m cs2_pipeline.dataset --manifest configs/phase2.json --out data/datasets/mirage_v2
uv run --frozen --extra ml python -m cs2_pipeline.train --dataset data/datasets/mirage_v2 --out models/logistic_smoke_v2 --smoke-test
uv run --frozen --extra ml python -m cs2_pipeline.predict --model models/logistic_smoke_v2/model.json --features data/datasets/mirage_v2/features.parquet --out models/logistic_smoke_v2/predicted_all.parquet
```

本机也可以直接用 `.venv\Scripts\python.exe -m ...`，macOS/Linux 使用 `.venv/bin/python`。
另提供 `cs2-build-dataset`、`cs2-train`、`cs2-predict` 三个命令入口。
命令失败时返回 1，并输出具体原因；已有输出一律拒绝覆盖。

## 来源清单

`configs/phase2.json` 是已可运行的真实来源清单。每个来源声明：

| 字段 | 用途 |
| --- | --- |
| path | Phase 1 输出目录，相对当前清单文件所在目录解析 |
| match_id | 同一整场比赛共享标识，例如 `hltv:2394170`；不包含地图后缀 |
| match_date | 真实比赛日期，严格使用 YYYY-MM-DD 格式 |
| event | 赛事标识，例如 `iem-atlanta-2026` |

比赛日期及身份必须显式提供。Phase 1 有外部来源时，会核对 SHA256、日期、赛事及 HLTV 比赛 ID。
Demo 本身没有的元数据不会凭文件名或比分猜测。
不同 SHA256 的同场录像会共享 match_id；相同 SHA256 的重复来源只保留一次。
把相同 Demo 声明成不同比赛会被拒绝。

`pgl-masters-bucharest-2026` 为内置保留赛事，清单或 Phase 1 来源出现该赛事时，
会在载入玩家表前排除，不能用于当前训练、调参或评估。
所有排除原因写入 `dataset.json`。

当前版本每次只处理清单指定的一张地图，默认 de_mirage。
不会把不同地图的坐标直接混用。以后扩展多地图共训时，需要统一的比赛分组清单与地图编码。

## 生成的数据

```text
data/datasets/mirage_v1/
├── dataset.json
├── samples.parquet
├── nodes.parquet
├── features.parquet
├── labels.parquet
└── splits.parquet
```

| 文件 | 当前真实结果 | 含义 |
| --- | --- | --- |
| samples.parquet | 295 行 | 时点索引，含比赛/地图、原始 tick、回合 ID 和经过时间 |
| nodes.parquet | 2,950 行 | 每时点 10 名选手，保留 UInt64 Steam ID、实时状态及观察掩码 |
| features.parquet | 295 行、47 个数值特征 | 逻辑回归用的白名单特征，另有 sample_id 关联键 |
| labels.parquet | 295 行 | 仅 sample_id 与 y_ct_win；CT 胜为 1，T 胜为 0 |
| splits.parquet | 295 行 | 每时点的 train/validation/test 归属 |
| dataset.json | 1 份 | 来源文件哈希、排除信息、特征版本、划分策略和行数 |

`sample_id` 由 Demo SHA256、Demo 内 round_id、原始 tick 组成，跨文件稳定且唯一。
原始 round_num 仍保存在索引表中。Phase 1 文件没有被改写。
GNN / Transformer 后续可按 sample_id 读取 nodes，保留选手之间的空间关系。
当前基线特征表不取代玩家粒度数据。

只接受 Phase 1 重新验证通过的输出、正常有效回合，以及双方各 5 名选手身份明确的完整时点。
缺少选手的时点整体排除，不补造未观测到的存活或死亡状态。
Demo 缺少实测 tick_rate 时拒绝训练集构建，避免使用未经核实的默认速率。

死亡派生记录继续保留 null，并提供 `position_observed`、`economy_observed`、
`equipment_observed` 和 `primary_slot_observed` 掩码。
当前样本有 499 条事件派生死亡记录，其位置、经济和装备不参与存活队伍统计。

所有 Parquet 写入后会重读核对，保存文件 SHA256。训练时会再次检查哈希和表间键，
防止修改文件后继续沿用旧划分或标签。

## 47 个特征及标签隔离

`features.py` 使用固定白名单：

- 从冻结结束到当前 tick 的经过秒数。
- T 和 CT 各自的存活人数。
- 仅存活玩家的血量、护甲、现金、装备价值、烟/闪/雷/燃烧弹、头盔、拆弹器和持弹状态总量。
- 仅存活玩家的 XYZ 坐标均值及双方 XY 中心距离。
- 上述阵营统计的 CT 减 T 差值。

没有存活玩家的一方，其总量为 0、位置均值为 null。
某个存活玩家的数值未知时，对应总量保持 null，不把缺失当作 0。
经过秒数是 Demo tick 时间，可能包含暂停，不是权威的剩余回合倒计时。

赢家、结束 tick、结束原因、回合编号、Steam ID、姓名、真实队名、划分标签和解析质量标记
均不进入模型特征。经过时间不使用 `end_tick - tick` 计算。
标签仅在训练/评估阶段按 sample_id 关联。
数据质量筛选可能依赖完整回合结果，这些筛选标记也不能直接作为实时特征。

## 时间划分和样本权重

默认按 match_id 锁定分组，随后按比赛日期划分。相同日期不能跨集合，
同一比赛在清单中的全部录像也不能跨集合。
指定 `--group-by event` 会将整场赛事锁在同一集合，重叠赛事日期区间会先合并。
默认目标比例为约 70%/15%/15%；小样本和日期区间合并会导致实际比例偏离。

至少要有 **3 个不重叠的时间区间** 才能形成三个非空集合。
不足时全部保留在 train，`evaluation_ready=false`。
程序不会退回到随机拆分 Snapshot 或拆分同场比赛。

当前真实数据只有一个比赛组，因此：

```text
train: 1 场 / 18 回合 / 295 个时点
validation: 0
test: 0
```

每个时点的样本权重为 `1 / 同回合采样时点数`，使每回合总权重为 1。
这避免持续时间长的回合因为时点较多而占据更多训练与评估权重。
Scaler、分类器和评估指标使用这些权重；缺失值填补只使用训练集的中位数。
当前按回合平衡的 CT 胜先验为 13/18，与按时点直接统计的胜方比例不同。

## 基线模型与推理

模型流程：训练集中位数填补与缺失指示器 → 标准化 → 固定参数的正则化逻辑回归。
三个阶段都只拟合 train，验证和测试数据不会参与填补、缩放或训练。
训练集某特征全部缺失时保留该列，以 0 填补并添加缺失指示器；
它是训练预处理策略，不会改写原始数据为“已观测的 0”。

```text
models/logistic_smoke_v1/
├── model.json
├── report.json
├── predictions.parquet
└── predicted_all.parquet
```

模型导出为普通 JSON，含特征列表、填补值、指示器索引、缩放参数及线性系数。
保存前会验证 JSON 推理与 sklearn 原始 Pipeline 的概率在 1e-12 容差内一致。
推理只应用已保存的参数，不重新拟合。
输出 `p_ct_win` 表示 CT 在当前回合获胜的概率；T 胜率为 `1-p_ct_win`。

模型固定适用于一张地图。对数据集目录中的 features.parquet，会重新检查数据集和地图；
独立特征文件需要额外传入 `--map de_mirage`，地图不匹配会拒绝推理。

**当前 model.json 的 run_mode 是 smoke_test。**
预测值来自这场比赛自身拟合的模型，仅用于检查管线，尚未证明能预测其他比赛，
也没有概率校准或可交易定价结论。

## 增加独立比赛后的正式评估

先解析更多真实 Demo，并把对应的输出目录、比赛 ID、日期和赛事加入来源清单。
重新构建到新的数据集目录；满足独立划分后，运行训练命令时移除 `--smoke-test`。

正式模式会在 validation 和 test 分别输出：

- 按回合加权的 Log Loss、Brier Score 和 0.5 阈值准确率。
- 双类别都存在时的 ROC AUC；单类别时为 null。
- 加权校准分桶统计，及仅由训练回合胜率构成的常数先验基线。

三个时间区间只是程序的最低要求，不能代表样本量已经充分。
本版没有调参、校准拟合、GNN/Transformer、Player Rating、Team Elo、地图/整场胜率模型或市场接口。
炸弹安放状态、游戏内有效倒计时、暂停状态和具体武器类别仍需下一步补充，
之后再扩展特征和比较模型。

## 自动测试

```powershell
uv run --frozen --extra dev --extra ml pytest -q
uv run --frozen --extra dev --extra ml ruff check src tests scripts
```

受限 Windows 临时目录不可写时，可用项目内专用临时目录：

```powershell
uv run --frozen --extra dev --extra ml pytest -q --basetemp .pytest_cache/phase2-tmp
```

pytest 会管理并清空专用 basetemp，勿指向自己的文件目录。

首版交付时为 **65 passed**，包含 3 项真实数据集成检查：Phase 1 实际 Demo 解析、
Phase 2 实际训练集构建，以及实际数据上的 smoke 训练和重新加载推理。
其余测试使用明确的模拟 DataFrame，包含独立划分评估路径；它们不构成真实比赛泛化证据。

新增检查覆盖：目标列隔离、死者数值掩码、缺失活人值、完整时点筛选、来源日期与比赛 ID、
SHA256 去重、保留赛事排除、同比赛/同日锁组、重叠赛事合并、篡改文件拒绝、按回合权重、
训练集专用预处理、单类别 AUC、无独立集合时拒绝正式训练，以及 JSON 模型推理一致性。

API 依据：[scikit-learn 数据泄漏与 Pipeline 指南](https://scikit-learn.org/stable/common_pitfalls.html)。
实现同时核对了安装版本的 SimpleImputer 和 LogisticRegression 实际签名。
