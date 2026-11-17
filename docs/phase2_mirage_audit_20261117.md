# Mirage 回合胜率：数据审计与探索性模型对照

日期：2026-11-17。已有玩家快照和回合数据均复用，没有重新解析；补充读取了同一批 Demo 的炸弹事件和游戏规则，并按录制文件 SHA256、回合 ID 和原始 tick 与快照对齐。50 场的结果此前已被观察，以下所有 50 场上的评分均为**探索分析**。不能将它们称为全新盲测，也不能将已看过的结果用来声称未来赛事表现已被验证。

## 数据、标签和因果性

- 审计已有的 38 场历史数据（811 回合、14,630 个时点）及 50 场探索数据（1,099 回合、20,180 个时点）。逐时点核对标签是否等于源回合 CT 胜方、原始回合编号、经过秒数以及 `[freeze_end_tick, end_tick)` 边界；均通过。
- 两个首段恢复的比赛 `hltv:2398745`、`hltv:2398721` 分别保留独立录制 tick。本轮复核源回合和标签关联；之前的独立审计还用同一恢复算法逐项对照了各自后段的 9 和 8 回合，最终比分与公开比赛页一致。详见 [50 场原审计](holdout_50_results.md) 和 `reports/holdout_50_audit_v2.json`。
- 两批数据的选手 ID 均为 UInt64。每个时点均保留 10 个不同 ID、T/CT 各 5 人；同比赛内无已知 Steam ID 的职业队伍变更、无同一队伍同时处于双方阵营、无同一阵营混入多个队伍或同回合换边，也无已记录死亡后仍存活或同回合复活的记录。此前 `hltv:2398743` 异常换边的 4 个完整时点仍被排除，未补造状态。
- V1/V2 都使用固定特征白名单。最终胜方、回合结束原因与结束 tick、选手 ID、队名和赛事均未作为模型特征；死亡状态只使用当前及之前的死亡事件。战斗窗口和有效回合是事后筛选，因此结果是**有完整可用录制的回合条件下**的表现，未来实时系统仍需单独处理进行中的回合、缺失选手和暂停。
- 下包、A/B 点位来自当前游戏规则与截止当前 tick 的已记录事件；游戏时钟使用当前游戏规则，不使用后续爆炸时间推断。赛后结果只用于数据质量核对和监督标签，不生成实时特征。
- 93 个录制文件中，4 个最新 BLAST 录制的规则实体无法读取，但可以读取炸弹事件；其中 1 个属于旧历史测试、3 个属于 50 场。回退前检查了炸弹爆炸/拆除回合具有此前下包事件；下包状态使用事件推断并明确标记，游戏时钟、暂停及炸弹计时保持 null。50 场有 993 个时点属于这 3 场，另有 27 个时点的游戏经过时间因暂停或规则状态未知不可用。全部 50 场均无可靠的炸弹引信配置，因此炸弹剩余秒数为 null，未假设固定秒数。炸弹分组只用其余 47 场可读取游戏规则的时点；与全部 50 场的敏感性比较见 JSON。
- 限制：对录制质量、事件完整性、比赛发布信息和当前 Awpy 属性语义的检查不能证明未来所有 Demo 均无隐性泄漏；实盘数据源还需要与离线字段逐项对齐。

完整机器可读审计：`reports/phase2_data_audit_v2.json`。上下文内容和元数据哈希与源回合、快照哈希绑定；原始文件不作覆盖。

## 严格划分和模型设置

固定 V1 与 `simple_full34` 均使用原来 34 张地图、730 回合训练；固定 V1 文件未改变。其余候选采用严格的**赛事及时间分离**：

| 用途 | 赛事 | 比赛 / 回合 | 日期 |
| --- | --- | ---: | --- |
| 拟合 | IEM Rio、IEM Atlanta | 18 / 353 | 4 月 13 日—5 月 13 日 |
| 校准 | IEM Cologne Major | 14 / 333 | 6 月 12 日—21 日 |
| 验证 | BLAST Open Porto | 2 / 44 | 9 月 5 日—6 日 |
| 已观察的探索集 | FISSURE、StarLadder、Thunderpick、ESL、BLAST Rivals | 50 / 1,099 | 9 月 8 日—11 月 14 日 |

同比赛、赛事和录制不会跨这些集合；`pgl-masters-bucharest-2026` 不参与训练、校准、调参或本轮评估。18 场候选仅拟合 353 回合；Cologne 校准虽然有 14 场，但仅覆盖一个赛事，Porto 验证只有 2 场，均不足以支持稳定的跨赛事模型选择。固定 V1 本身也曾用到后于其中部分探索比赛的 Beijing 验证结果，故更不能把整个 50 场报告解释为严格前瞻评估。

V1 候选为 47 特征逻辑回归，V2 为加入因果时钟、下包、包点与活人武器类别的 96 特征逻辑回归。简单基线只用存活人数差和已过去时间。CPU 非线性候选为浅层 HistGradientBoosting；GPU 候选为 47 特征、120 棵浅树的 XGBoost，关闭随机时点早停。候选、参数、种子与校准方法记录在 `configs/phase2_comparison_plan.json` 和 `configs/phase2_gpu_plan.json`。训练和评分每回合总权重均为 1；指标均为 CT 回合胜率。校准仅在独立 Cologne 数据上拟合 sigmoid，不能用它校准已包含 Cologne 的固定 V1 或 34 场 GPU 模型。

## 同一批 50 场上的结果

| 模型 | 拟合地图 | 校准 | Log Loss ↓ | Brier ↓ | AUC ↑ | ECE ↓ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 固定 V1 | 34 | 无 | 0.4321 | 0.1448 | 0.8714 | 0.0213 |
| 人数差 + 时间 | 34 | 无 | 0.5125 | 0.1758 | 0.7916 | 0.0247 |
| V1 逻辑回归 | 18 | 无 | 0.4420 | 0.1484 | 0.8660 | 0.0293 |
| V1 逻辑回归 | 18 | Cologne | 0.4395 | 0.1473 | 0.8660 | 0.0191 |
| V2 逻辑回归 | 18 | 无 | 0.4791 | 0.1606 | 0.8503 | 0.0621 |
| V2 逻辑回归 | 18 | Cologne | 0.4615 | 0.1557 | 0.8503 | 0.0263 |
| CPU 浅树 | 18 | 无 | 0.4487 | 0.1486 | 0.8636 | 0.0165 |
| CPU 浅树 | 18 | Cologne | 0.4482 | 0.1484 | 0.8636 | 0.0139 |
| CUDA XGBoost 浅树 | 18 | 无 | 0.4538 | 0.1495 | 0.8613 | 0.0174 |
| CUDA XGBoost 浅树 | 18 | Cologne | 0.4532 | 0.1494 | 0.8613 | 0.0138 |
| CUDA XGBoost 浅树 | 34 | 无 | 0.4435 | 0.1465 | 0.8682 | 0.0179 |

18 场候选使用相同拟合集，均在完全相同的 50 场、20,180 个时点评分；固定 V1 与 34 场候选训练量更大，应单独理解。V2 和两种浅树在这轮没有超过对应条件下的 V1 逻辑回归；这不是新赛事上的最终结论。V2 同时改变了时钟、炸弹和武器相关输入，不能把差异归因于其中某一个特征。排除 3 场规则无法读取的 BLAST 后，在同样的 47 场上，V1 18 场为 0.4429，V2 18 场为 0.4781，V2 校准后为 0.4606，结论未因这 3 场而反转。

按比赛进行 2,000 次成对 bootstrap：固定 V1 的 Log Loss 95% 区间为 **[0.4134, 0.4524]**；人数差基线减 V1 的 Log Loss 差为正，区间 **[0.0646, 0.0962]**；34 场 GPU 减固定 V1 的差值区间为 **[0.0049, 0.0182]**，对应 Brier 差区间 **[-0.00065, 0.00440]**。这些是所抽比赛内的不确定性，不能代表未来赛事之间的差异或多次探索后的确认性检验。

概率校准效果不一致：18 场 V1 的 Log Loss 从 0.4420 降至 0.4395，差值区间为 **[-0.0059, 0.0006]**；V2 从 0.4791 降至 0.4615，差值区间 **[-0.0261, -0.0103]**；18 场 GPU 从 0.4538 降至 0.4532，差值区间 **[-0.0018, 0.0008]**。10 箱 ECE 易受分箱和样本组成影响，低 ECE 不等于各状态下都可靠。

Porto 两场的未校准 Log Loss：V1 18 场 0.3759、V2 18 场 0.3714、CPU 浅树 0.4216、GPU 浅树 0.4292；样本太少，不作强结论。因为固定 V1 与 34 场模型已使用 Porto 训练，不在此报告它们的 Porto 验证分数。

## 固定 V1 的误差和可靠度

各组内部重新按回合赋权；同一回合可以在不同人数、时间或炸弹状态组中出现。不能把各组时点数当作独立样本。

| 分组 | 比赛 / 覆盖回合 | Log Loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| 双方存活人数相同 | 50 / 1,099 | 0.5970 | 0.2081 | 0.0419 |
| CT 多 1 人 | 50 / 640 | 0.4938 | 0.1603 | 0.0508 |
| CT 少 1 人 | 50 / 534 | 0.4695 | 0.1540 | 0.0559 |
| CT 多 2 人 | 50 / 450 | 0.1910 | 0.0503 | 0.0192 |
| CT 少 2 人 | 50 / 292 | 0.1364 | 0.0326 | 0.0090 |
| 未下包，排除仅事件回退录制 | 47 / 1,043 | 0.4910 | 0.1669 | 0.0406 |
| 已下包，排除仅事件回退录制 | 47 / 507 | 0.2162 | 0.0647 | 0.0443 |

时间窗口全量分组为 0–15 秒 0.6114、15–30 秒 0.5737、30–45 秒 0.4810、45–60 秒 0.4214、60 秒以后 0.2769，但覆盖回合不同。另取**同一批同时有 0 秒和 60 秒状态的 906 回合**，逐回合配对：0 秒 Log Loss 0.6392，60 秒 0.4021，60 秒减 0 秒的比赛聚类区间 **[-0.2660, -0.2045]**。这是以回合持续到 60 秒为条件的描述，不能推断时间本身导致改善或推广到较早结束的回合。

各赛事 Log Loss：FISSURE 0.4274（11 场）、StarLadder 0.3854（9 场）、Thunderpick 0.4579（8 场）、ESL 0.4525（19 场）、BLAST Rivals 0.4057（3 场）。各组比分布、队伍和比赛数不同，不能直接解释为赛事难度。整体可靠度箱中，预测均值约 0.747 的时点 CT 实际胜率约 0.702，预测均值约 0.148 的时点实际为 0.201；其他箱及逐赛事区间见 `reports/phase2_exploration_final.json`。

## GPU 执行证据

本机为 NVIDIA GeForce RTX 4070 Ti、驱动 621.01，XGBoost 3.4.1 编译信息为 CUDA 13.3。两次训练期间，`nvidia-smi` 均记录了训练进程及对应 GPU UUID；XGBoost 模型配置为 `cuda:0`，没有 CPU 回退警告。Windows 进程显存字段返回 `[N/A]`，因此不声称实测显存占用。模型重读审计用 CPU 独立推理，最大概率差小于 2.4e-7，校准与保存的评分一致。含设备标识的本地审计文件可按下方命令重新生成。GPU 运行配置见 [XGBoost 官方 CUDA 文档](https://xgboost.readthedocs.io/en/stable/gpu/)。

## 下一批数据

最紧迫的是补充**多个不同赛事**的 Mirage，而非把 50 场重新命名成盲测。下列比赛均实际打了 Mirage，页面可寻找 “Demo download”。请把 `.rar` 或解压后的 `.dem` 放到 `data/raw/phase2_validation_more/`，保留原文件名。它们可用于扩充 **18 场拟合 + Cologne 校准**方案的历史验证和模型开发；它们早于包含 9 月 Porto 的固定 V1 训练截止日期，不能冒充固定 V1 的未来测试。用这些结果选模型后，必须另留未看过的新赛事。

BLAST Bounty 2026 Season 2 Finals：

1. [FaZe vs The MongolZ](https://www.hltv.org/matches/2396016/faze-vs-the-mongolz-blast-bounty-2026-season-2-finals)
2. [Astralis vs MOUZ](https://www.hltv.org/matches/2396019/astralis-vs-mouz-blast-bounty-2026-season-2-finals)
3. [Spirit vs MOUZ](https://www.hltv.org/matches/2396021/spirit-vs-mouz-blast-bounty-2026-season-2-finals)

Esports World Cup 2026：

4. [magic vs The MongolZ](https://www.hltv.org/matches/2396572/magic-vs-the-mongolz-esports-world-cup-2026)
5. [FUT vs TYLOO](https://www.hltv.org/matches/2396569/fut-vs-tyloo-esports-world-cup-2026)
6. [B8 vs Vitality](https://www.hltv.org/matches/2396583/b8-vs-vitality-esports-world-cup-2026)
7. [FUT vs MOUZ](https://www.hltv.org/matches/2396584/fut-vs-mouz-esports-world-cup-2026)
8. [The MongolZ vs Natus Vincere](https://www.hltv.org/matches/2396587/the-mongolz-vs-natus-vincere-esports-world-cup-2026)
9. [FURIA vs 9z](https://www.hltv.org/matches/2396588/furia-vs-9z-esports-world-cup-2026)
10. [JiJieHao vs Aurora](https://www.hltv.org/matches/2396592/jijiehao-vs-aurora-esports-world-cup-2026)
11. [FURIA vs Aurora](https://www.hltv.org/matches/2396600/furia-vs-aurora-esports-world-cup-2026)
12. [FUT vs magic](https://www.hltv.org/matches/2396601/fut-vs-magic-esports-world-cup-2026)
13. [GamerLegion vs MOUZ](https://www.hltv.org/matches/2396602/gamerlegion-vs-mouz-esports-world-cup-2026)
14. [Legacy vs Falcons](https://www.hltv.org/matches/2396609/legacy-vs-falcons-esports-world-cup-2026)

之后可在固定候选与评估计划后，将 [PGL Major Singapore 2026](https://www.hltv.org/major/matches) 的未进行比赛作为新赛事候选；截至本报告，该 Major 定于 11 月 24 日开始，Demo 尚未产生，实际地图须赛后确认。它与 `pgl-masters-bucharest-2026` 是不同赛事，后者继续保留。

## 复现

所有输出必须使用新目录，避免覆盖已保存结果；下面假定在项目根目录运行。

```powershell
uv sync --frozen --extra dev --extra ml --extra context --extra gpu
.venv\Scripts\python.exe -m scripts.context_audit_batch
.venv\Scripts\python.exe -m scripts.audit_phase2_data --out reports/audit_recheck.json
.venv\Scripts\python.exe -m cs2_pipeline.phase2_compare --out models/phase2_compare_recheck --progress-file runs/phase2_compare_recheck/progress.jsonl
.venv\Scripts\python.exe -m scripts.explore_phase2 --model-out models/phase2_compare_recheck --out reports/exploration_recheck.json
.venv\Scripts\python.exe -m cs2_pipeline.phase2_gpu --out models/phase2_gpu_recheck --progress-file runs/phase2_gpu_recheck/progress.jsonl
.venv\Scripts\python.exe -m scripts.audit_phase2_gpu --model-out models/phase2_gpu_recheck --out reports/gpu_audit_recheck.json
.venv\Scripts\python.exe -m pytest -q
```

GPU 脚本通过固定计划引用已保存的 `models/phase2_compare_final` 基线，并在无法确认 GPU 时拒绝执行。若新环境没有该基线，先将上面 CPU 对照的输出目录设为 `models/phase2_compare_final`。当前机器可从训练界面启动 `Mirage · RTX CUDA 模型对照`；使用 `python -m cs2_pipeline.monitor --port 8769` 后打开 `http://127.0.0.1:8769`。报告、逐时点预测、GPU 进程证据和校准参数均保存在独立输出目录。
