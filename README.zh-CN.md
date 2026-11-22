# CS2 Mirage 回合胜率预测

本项目用职业 CS2 Mirage 录像预测 CT 回合胜率，重点是在同一训练集上比较传统模型和 Transformer。
用相同的 34 场训练数据，在已观察的 50 场比赛上，逻辑回归、单快照 Transformer、最多 8 快照的
Transformer 和归档 XGBoost 的 log loss 分别为 **0.4322、0.4348、0.4386、0.4435**。
单快照相对逻辑回归的 95% 配对区间跨 0，加入历史快照的效果随随机种子变化。
[34 场同数据对照](docs/mirage_temporal_experiment.md#follow-up-identical-34-map-training-set-november-21-2026)

后续实验已完成 200 轮训练、16／24／32 场训练规模对比、47 场扩充和 dropout 消融。
长训练出现过拟合。面向新加坡 Major 的独立评估已冻结，两个模型都使用 **49 场（1,079 个回合、19,460 个时点）**训练。
主比较采用 seed 42、第 1 轮、dropout 0.10 的单快照 Transformer 和配对逻辑回归。新加坡 Major Mirage 待评分。
[结果与曲线](results/README.md) · [冻结规则](docs/mirage_temporal_experiment.md#frozen-singapore-evaluation)

此前的分段回合审计、人数/时间/下包分组、比赛聚类区间、V1/V2 与 RTX CUDA 对照见
[Mirage 审计及模型对照](docs/phase2_mirage_audit_20261117.md)。50 场已经观察过，只能作为探索集；
`pgl-masters-bucharest-2026` 仍然完全保留。

**Phase 2 安装、构建、训练和推理命令见 [Phase 2 使用说明](docs/phase2.md)。**
**此前 9 场语料、V1/V2 对比和复现命令见 [Rio 试验记录](docs/phase2_pilot.md)。**
**此前 Cologne 四集合实验见 [Cologne 实验记录](docs/phase2_cologne.md)；当前扩充训练见 [多赛事实验记录](docs/phase2_expanded_2026.md)。**
需要完整依赖时执行 `uv sync --frozen --extra dev --extra ml`；只使用解析功能仍可按下文安装。

## 训练监控 UI

```powershell
.venv\Scripts\python.exe -m cs2_pipeline.monitor
```

本地训练监控位于 `http://127.0.0.1:8765`，支持时序 Transformer、GPU 树模型、多赛事 V1 和历史
Rio V1/V2。训练阶段、耗时、日志和指标可在页面查看。时序预设使用可选的 `temporal` 依赖和归档数据。
[时序训练环境](docs/mirage_temporal_experiment.md#cuda-verification-and-reproduction)
实验记录也显示 ESL 额外赛事留出评估；该赛事没有加入训练预设。
50 场批量解析及上下文补充有单独的进度卡片。GPU 预设需要安装以下可选依赖，
并会核查 XGBoost 的 CUDA 配置和训练时的 GPU 进程；无法确认 GPU 时直接失败，不静默回退：

```powershell
uv sync --frozen --extra dev --extra ml --extra context --extra gpu
```

需要 NVIDIA GPU、兼容驱动与 XGBoost 支持的平台；其他预设仍可在没有 GPU 时运行。
服务仅监听本机，保持进程运行即可使用。详细说明见 [训练 UI 使用说明](docs/training_ui.md)。

## 安装与版本

需要 Python 3.11+，推荐使用 uv。在项目根目录执行：

```powershell
uv sync --frozen --extra dev
```

`uv.lock` 固定全部运行及测试依赖和下载哈希，支持 Windows、macOS、Linux。
`pyproject.toml` 固定 Awpy 3.3.0、Polars 2.1.0、PyArrow 26.0.0、pytest 9.2.0。
Awpy 3.3.0 是 2026-11-17 查询到的稳定版本；已核对安装包接口并在真实 Demo 验证。
官方 `latest` 文档可能比 PyPI 版本更新，不应直接套用后续版本变更。

可选 pip 安装方式（严格复现优先使用上述 uv 锁文件）：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

macOS/Linux 中将 `.venv\Scripts\python.exe` 换成 `.venv/bin/python`。
本机系统 `python` 是不可用的 Windows 占位程序；本次已使用已有 Python 3.12.14 创建 `.venv` 并安装完成。
因此本机也可直接使用 `.venv\Scripts\python.exe`，无需激活环境。

## 获取真实 Demo

已下载的默认输入是 `data/raw/match.dem`，约 347 MB，内容是：

- **2026-05-13，IEM Atlanta 2026，B8 vs Team Vitality，第 1 图 Mirage**。
- 以 2026-10-27 为基准，比赛在最近六个月内。该赛事有 HLTV MVP 奖项，符合 MVP event 类别。
- Mirage 独立比分为 B8 5–13 Vitality，与解析后逐回合胜方核对一致。
- 下载来源为 [Awpy 官方真实测试样例仓库](https://github.com/pnxenopoulos/awpy-fixtures)，不是生成或模拟的 Demo。
- [比赛来源](https://www.hltv.org/matches/2394170/b8-vs-vitality-iem-atlanta-2026)、[赛事与 MVP 信息](https://www.hltv.org/events/8243/iem-atlanta-2026)。
- 未获取或纳入 `pgl-masters-bucharest-2026`；该赛事继续保留为后续测试集。

来源、官方 SHA256、文件大小及独立比分保存在 `docs/demo_source.json`。
可以重新获取同一个公开样本：

```powershell
uv run --frozen python scripts/download_sample.py
```

脚本通过公开 GitHub Release 下载已经解压的 `.dem`，校验大小及 SHA256，
采用临时文件后重命名；已有相同文件会验证后复用，已有不同文件会拒绝覆盖。
SHA256：`321cfe8bd30501f9040a9ae0cfaca900b6db1cc6cc6ab026f56b86a5b3e20ddc`。

如果公开下载失效，人工获取步骤：

1. 用正常浏览器打开 [HLTV Results](https://www.hltv.org/results) 或上述比赛页。
2. 选择已经结束的职业赛事，确认比赛日期、MVP event 类别和地图；排除保留测试赛事。
3. 使用比赛页的 **Demo download** 下载赛事公开提供的文件，并遵守来源使用条件。
4. 用系统解压工具或 7-Zip 解开 `.rar`、`.zip` 等压缩包，找到 Mirage 对应的 `.dem`。
5. 将文件放到 `data/raw/match.dem`，或在命令中传入其完整本地路径。

本次直接访问 HLTV 遇到 Cloudflare 验证后已停止；没有绕过登录、验证码或访问保护。
原始 Demo、压缩包及全部 Parquet 已加入 `.gitignore`，不提交到 Git。

## 解析与重新验证

通用命令（Windows/macOS/Linux 相同）：

```powershell
uv run --frozen python -m cs2_pipeline.parse --demo data/raw/match.dem --out data/processed/match --interval 5
```

当前已经有解析结果。重复执行时请换用尚不存在的输出目录，例如：

```powershell
uv run --frozen python -m cs2_pipeline.parse --demo data/raw/match.dem --out data/processed/match_again --interval 5 --source docs/demo_source.json
```

可选 `--source` 将外部来源写入 `metadata.external_source`，必须与 Demo SHA256 和地图吻合。
Demo 本身未提供的比赛日期和赛事信息仍为 null，与外部来源严格区分。
不要把这个来源文件附到其他 Demo 上。

本机不通过 uv 也可以执行：

```powershell
.venv\Scripts\python.exe -m cs2_pipeline.parse --demo data/raw/match.dem --out data/processed/match_again --interval 5
.venv\Scripts\python.exe -m cs2_pipeline.validate --input data/processed/match
```

重新验证或输出完整 JSON：

```powershell
uv run --frozen python -m cs2_pipeline.validate --input data/processed/match
uv run --frozen python -m cs2_pipeline.validate --input data/processed/match --json
```

也提供 `uv run --frozen cs2-parse` 和 `uv run --frozen cs2-validate` 两个入口。
路径含空格时加引号。PowerShell 无需使用反斜杠换行。
如旧 Windows 终端显示中文乱码，可先执行 `$env:PYTHONUTF8="1"`。

退出码：`0` 表示通过；`1` 表示输入/解析/写入失败；`2` 表示质量验证失败。
失败时会提供具体原因和下一步，缺少文件不会打印难以理解的堆栈。
质量失败的已解码数据仍保留供排查，并在 `metadata.validation.passed` 标记 false。
输出目录必须不存在，避免覆盖用户文件；写入后会分别用 Polars 和 PyArrow 重读检查。

## 数据格式与边界约定

```text
data/processed/match/
├── metadata.json
├── rounds.parquet
├── players.parquet
└── snapshots.parquet
```

| 文件 | 内容 |
| --- | --- |
| metadata.json | Demo 名称、SHA256、地图、版本、游戏 build、tick rate 来源、解析耗时、原始 header、convars、来源及质量报告 |
| rounds.parquet | 原始 round_num、round_id、起止 tick、冻结结束 tick、最终 winner_side、结束原因、刀局/有效回合标记 |
| players.parquet | UInt64 steamid、选手名和别名、真实队名及冲突信息 |
| snapshots.parquet | 每个 tick 每名选手一行，含位置、血量、护甲、经济、武器、手雷、装备、阵营、真实队伍及存活状态 |

具体字段和类型以 `code/src/cs2_pipeline/schema.py` 为准。

- 保留原始 `round_num`，不根据 MR12、换边或加时重新编号。
  `round_id` 是当前 Demo 内唯一的关联键，用于区分重启前后重复的原始编号；不是游戏原始编号。
- 每回合从 `freeze_end_tick` 开始，每隔 `interval × tick_rate` 采样，
  小数间隔对绝对偏移取整，避免累计漂移。边界采用 **`freeze_end_tick <= tick < end_tick`**。
  冻结期、胜负已定的结束 tick 和结束后的状态均排除。
- 使用 tick 关联回合；标签列不参与状态表关联。重叠区间不会任意分配给某个回合。
- 刀局、重复编号、缺失或异常边界、无正常胜方等保留在 rounds 中并标记无效。
  无法确定战斗边界的回合不输出 Snapshot。未被 Awpy 输出的热身/未完成回合不猜测补造。
- `side` 统一为 `T` / `CT`；`team` 是 Awpy `team_clan_name` 提供的真实组织名称。
  team 按 Steam ID 关联，换边不会改变。缺少队名则 null，多队名冲突保留候选并告警。
- Steam ID 全程使用 **UInt64**，拒绝浮点数输入，Parquet 往返保持完整。
  若之后导出 JSON/JavaScript，应转为十进制字符串，避免超过 JavaScript 安全整数范围。
- 仅移除完全一致的重复状态；同一 `(tick, steamid)` 若内容冲突则直接失败，避免任意选取。
- `is_alive` 来自当前血量及**当前 tick 之前或当时**的死亡事件；死亡证据不会跨回合继承。

### 死亡实体缺失的处理

真实样本暴露了 Awpy 3.1.1 的情况：部分死亡后的 pawn 有血量 0 和位置，但已失去 Steam ID。
不能凭位置猜测其身份，也不能把缺行当作存活或断线。

- `state_source="snapshot"`：Awpy 直接返回的有身份状态。
- `state_source="death_event"`：同回合该选手已有过去的死亡事件，且已有过去的身份观察，
  但当前 tick 缺少有身份状态；仅保留其 Steam ID、此前阵营/姓名及已死亡事实。
  **当前 x/y/z、health、经济、装备等全部保持 null**，不复制最后位置或装备。
- `death_observed` 仅表示截至当前 tick 已有死亡证据。缺少死亡证据的玩家不会被补成死亡。
- 每条事件派生状态都计入质量警告，并在缺失值统计中体现；训练时用 `is_alive` 和观察掩码处理。

武器栏 null 也可能表示没有携带该类武器；不要一概当成解析错误。
缺少可选字段会记录在 `metadata.cleaning.missing_snapshot_columns`，不会填充虚构的 0/false。
若 Demo 缺少播放计时，`tick_rate` 留 null，另存 Awpy 默认采样速率 64 及来源，并给出警告。

### 标签隔离

`winner_side`、回合结束原因等未来结果只放在 `rounds.parquet`。
`snapshots.parquet` 不含当前回合赢家。
`round_id`、`is_valid_round`、`state_source`、清洗及验证信息用于关联/筛选，
**不要直接把整张表作为模型输入**。特别是 `is_valid_round` 的判断可能依赖最终回合状态。

## 自动测试与真实运行记录

```powershell
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev ruff check src tests scripts
```

如果受限 Windows 环境的系统临时目录不可写，可以指定项目内专用测试临时目录：

```powershell
uv run --frozen --extra dev pytest -q --basetemp .pytest_cache/test-tmp
```

`--basetemp` 目录只用于临时测试数据，pytest 会管理并清空该目录，勿指向自己的文件目录。

Phase 1 交付时验证记录：**39 passed**，其中 38 项测试使用模拟 DataFrame/异常输入，
1 项使用真实职业比赛 Demo 进行端到端解析和独立比分核对。
没有本地 Demo 时，真实解析测试明确显示 skipped；不会把模拟测试声称为真实解析成功。
可通过 `CS2_TEST_DEMO` 环境变量指定另一个真实 Demo，或用 `pytest -m "not integration"` 仅跑单元测试。

覆盖回合边界、冻结/结束过滤、换边、加时、重启重复编号、重叠区间、去重、队伍身份、
Steam ID 精度、死亡不可复活、死亡证据不可回填过去、可选字段、异常输入、损坏文件和 Parquet 往返读取。

默认真实样本结果：

| 指标 | 结果 |
| --- | --- |
| 地图 | de_mirage |
| 回合 / 唯一选手 | 18 / 10 |
| 采样 tick / 有效采样 tick | 295 / 295 |
| Snapshot 总行数 | 2,950 |
| 直接观测状态 / 事件派生死亡状态 | 2,451 / 499 |
| 异常 | 0 |
| 警告 | 499 条事件派生死亡状态，当前字段未知已留 null |
| 独立比分核对 | B8 5–13 Team Vitality |
| 四个文件重读 / 验证 | 通过 / 通过（含上述警告） |

已在本机 Windows + Python 3.12.14 实跑。macOS/Linux 使用 pathlib、跨平台依赖及相同 uv 锁文件，
尚未在对应系统执行，未声称已经测试全部操作系统。

## 批量扩展与 Phase 2

`parse_demo(Path(...), Path(...), interval=5)` 可从 Python 逐文件调用；每个 Demo 使用独立输出目录。
批处理清单应保存 Demo SHA256、比赛日期、赛事来源、输出路径和验证结果，重复下载用 SHA256 识别。
目前不加入数据库、容器或分布式框架。

以下是 Phase 2 的数据处理原则，前四项已实现；完整运行记录见 [Phase 2 说明](docs/phase2.md)：

1. 先选择验证通过的 Demo 和正常有效回合；检查死亡派生状态、缺失字段及中途断线。
2. 以 `(demo_sha256, round_id, tick)` 为时点，保留每名选手一行及存活/观测掩码，
   由位置、血量、经济、装备及经过的回合时间构建状态输入。
3. 单独从 rounds 关联监督标签；例如 CT 胜记为 1，T 胜记为 0。
   winner、结束原因、结束 tick、解析后质量标记等不得进入实时特征。
4. 按比赛/赛事分组并按时间划分训练、验证和测试集，不能随机拆同一回合的 Snapshot。
   同一比赛多图应放同一个集合，继续完全隔离 `pgl-masters-bucharest-2026`。
5. 以后按 Steam ID 关联**比赛发生前已知**的 Player Rating，按真实队伍关联比赛前 Elo，
   再研究图结构/序列输入和概率校准；本阶段尚未实现这些步骤。

当前字段尚未包含完整炸弹安放时间、剩余回合计时或暂停状态。Phase 2 前应增加相应事件/规则状态，
不要仅用 `end_tick - tick` 计算“剩余时间”，那会泄漏未来。

## API 依据

- [Awpy Getting started](https://awpy.readthedocs.io/en/latest/getting-started.html)
- [Awpy API](https://awpy.readthedocs.io/en/latest/api.html)
- [Awpy datasets](https://awpy.readthedocs.io/en/latest/datasets.html)
- [Awpy reference](https://awpy.readthedocs.io/en/latest/reference.html)
- [PyPI 稳定包版本](https://pypi.org/project/awpy/3.3.0/)

实现使用 `Demo(path)`、`demo.load("rounds", "players", "kills")`、
`demo.snapshots(ticks=[...])`、`demo.header`、`demo.tick_rate` 和 `demo.convars`。
没有使用旧 CS:GO `DemoParser`，也没有套用旧版 `demo.parse()` / `tickrate=` 接口。
