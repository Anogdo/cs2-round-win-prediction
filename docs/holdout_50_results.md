# 50 场 Mirage 留出比赛：固定模型评估

## 范围与结果

用户提供的 50 个 HLTV 压缩包均已处理，包含 **50 张 Mirage、1,099 回合、20,180 个采样时点和 201,800 条玩家节点**；两张地图分别有两个录制分段，因此共处理 52 个 `.dem`。每场的地图、完整回合数及双方比分均与 `configs/holdout_50_reference.json` 中的 HLTV 比赛页面核对一致。比赛 ID 和 Demo 哈希与原有训练、验证、测试及此前 ESL 六场留出集不重叠。

固定模型为 `models/monitor_runs/20261117T052929-9d7d1b12/model.json`，SHA256 为
`166cdac8ae4225c48245ca845f3ac262a5e306ef79588aa5eb7c70f7ab3e622e`。
本批比赛仅用于评估：没有用这些数据拟合、校准或调整模型。
结果使用每回合总权重为 1 的指标；Log Loss、Brier 和 ECE 越低越好。

| 范围 | 比赛 | 回合 | 时点 | Log Loss | Brier | AUC | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 50 场 | 50 | 1,099 | 20,180 | 0.4321 | 0.1448 | 0.8714 | 0.0213 |
| 前 3 个此前未用赛事 | 28 | 651 | 12,098 | 0.4221 | 0.1415 | 0.8782 | 0.0236 |
| 曾评估赛事中的新比赛 | 22 | 448 | 8,082 | 0.4466 | 0.1495 | 0.8610 | 0.0306 |
| 全部 50 场：训练集常数先验 | 50 | 1,099 | 20,180 | 0.6878 | 见报告 | 0.5000 | 见报告 |

按赛事分别为：FISSURE 11 场 Log Loss 0.4274、StarLadder 9 场 0.3854、
Thunderpick 8 场 0.4579、ESL Pro League 19 场 0.4525、BLAST Rivals 3 场 0.4057。
它们的队伍和赛事样本数量不同，不应将差异直接解释为某个赛事的难度。

在回合刚开始的 0–15 秒，模型 Log Loss 为 0.6114；60 秒及以后的采样时点为 0.2769。
每个时间窗口分别对其所包含的回合归一化权重；前者覆盖 1,099 回合，后者只覆盖 906 个持续到该窗口的回合，不能把这两个结果当作同一批回合的因果比较。

选择这批比赛前已经观察过以前的验证和测试结果；虽然比赛均晚于 2026-09-06 的模型训练截止日，部分比赛早于已使用过的 Beijing 验证赛事，故本轮是**回顾性留出评估**，不是完全盲法或前瞻性测试。模型本身未做概率校准，五个赛事的结果不能保证未来赛事、地图或市场中的表现。按比赛重采样的区间只反映这批赛事内部的不确定性，不代表赛事之间的变化。

## 异常和数据质量

- 50 场、52 个录制文件重新读取验证全部通过；唯一警告是下面隔离的 4 个采样 tick。原始 `.rar` 和 `.dem` 均保留，本地 SHA256 用于完整性校验，不声称是 HLTV 官方校验值。
- [FURIA vs Aurora](https://www.hltv.org/matches/2398743/furia-vs-aurora-esl-pro-league-season-24) 的最后一回合出现选手异常切换阵营，导致四个时点变成 6 T / 4 CT。质量修正副本排除了这四个完整时点（40 条玩家记录），没有推测或补造阵营；根据 HLTV 比赛阵容将 yuurih 的固定职业战队身份修正为 FURIA。原始解析结果单独保留，修正细节记录在元数据 `quality_adjustment`。
- [Falcons vs Natus Vincere](https://www.hltv.org/matches/2398745/falcons-vs-natus-vincere-esl-pro-league-season-24) 和 [Aurora vs 9z](https://www.hltv.org/matches/2398721/aurora-vs-9z-esl-pro-league-season-24) 的首段，Awpy 3.3.0 直接读取到零个回合。严格恢复流程使用录制中 `demoparser2` 游戏规则状态与 Awpy 的冻结结束事件恢复回合边界和胜方；玩家状态和死亡事件仍由 Awpy 读取。相同恢复方法在各自后段与 Awpy 的所有 **9 / 8 回合**边界、胜方及结束原因逐项一致，完整地图的连续编号和最终比分也均一致。每个恢复分段的元数据包含方法、版本、后段哈希和对照回合数。原始局部 tick 未跨分段拼接。
- 模型输入与胜方标签仍分开保存。重推理得到的 20,180 个概率与已存储结果逐条一致；重新计算整体及各赛事指标并复核来源文件见 `reports/holdout_50_audit_v2.json`。

## 文件和复现

- 事先固定的比赛和模型计划：`configs/holdout_50_plan.json`。
- 独立核对的日期、地图顺序、比分：`configs/holdout_50_reference.json`。
- 解析及比分校验：`scripts/import_holdout_50.py`；实时状态：`runs/holdout_50/progress.json`。
- 训练界面：`http://127.0.0.1:8765`；50 场处理进度显示在单独卡片中。
- 测试数据集：`data/datasets/mirage_holdout_50_v1/`。
- 评估报告和预测：`models/evaluations/holdout_50_v1/`。
- 独立审计：`reports/holdout_50_audit_v2.json`；两份恢复元数据描述修订：`reports/holdout_50_metadata_revision.json`。

需安装锁定的上下文依赖以恢复特殊分段。已有输出目录不会被覆盖；如重新构建或评分，请使用新路径：

```powershell
uv sync --frozen --extra dev --extra ml --extra context
.venv\Scripts\python.exe scripts/import_holdout_50.py resume
.venv\Scripts\python.exe -m scripts.evaluate_holdout_50 build --out data/datasets/mirage_holdout_50_rebuild
.venv\Scripts\python.exe -m scripts.evaluate_holdout_50 score --dataset data/datasets/mirage_holdout_50_rebuild --out models/evaluations/holdout_50_rebuild
.venv\Scripts\python.exe -m scripts.audit_holdout_50 --out reports/holdout_50_audit_recheck.json
```
