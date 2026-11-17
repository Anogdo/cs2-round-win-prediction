# 数据来源与署名

## 原始单场样本

- Awpy maintainers, [awpy-fixtures](https://github.com/pnxenopoulos/awpy-fixtures)。
- B8 vs Team Vitality，IEM Atlanta 2026，2026-05-13，Mirage。
- 固定来源和 SHA256：`docs/demo_source.json`。
- 原始 `.dem` 未提交 Git。

## IEM Rio 2026 扩展语料

- **OpenCS2 / blanchon**，[HLTV CS2 Demos Dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset_demo)。
- 数据集卡声明许可：[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)。
- 来源数据集卡：[README](https://huggingface.co/datasets/blanchon/opencs2_dataset_demo/blob/912a363098feec3f220e41d68634d2f71676c28b/README.md)。
- 固定仓库版本：`912a363098feec3f220e41d68634d2f71676c28b`。
- 比赛原始来源为 HLTV；[IEM Rio 2026 赛事/MVP 信息](https://www.hltv.org/events/8242/iem-rio-2026)。
- 本项目根据原始 Demo 重新使用 Awpy 解析，并补充因果游戏时钟、武器特征和按比赛划分。
  没有把来源数据集的 Rating、ADR、最终比分或未来回合信息当作实时输入。

| 日期 | HLTV 比赛 ID | 比赛 | 地图 | 字节数 |
| --- | --- | --- | --- | ---: |
| 2026-04-13 | 2393087 | Aurora vs HOTU | Mirage | 364,296,972 |
| 2026-04-13 | 2393088 | Legacy vs MOUZ | Mirage | 380,515,711 |
| 2026-04-14 | 2393226 | Spirit vs Falcons | Mirage | 396,499,473 |
| 2026-04-14 | 2393228 | Natus Vincere vs FURIA | Mirage | 364,333,863 |
| 2026-04-15 | 2393236 | Vitality vs Falcons | Mirage | 504,107,083 |
| 2026-04-15 | 2393237 | G2 vs Spirit | Mirage | 458,987,520 |
| 2026-04-17 | 2393240 | Spirit vs MOUZ | Mirage | 485,468,181 |
| 2026-04-18 | 2393242 | Falcons vs Spirit | Mirage | 403,194,394 |
| 2026-04-19 | 2393244 | Falcons vs FURIA | Mirage | 562,248,182 |

每个 Demo 的不可变下载地址、文件大小、LFS SHA256、比赛链接及署名均固定在
`configs/corpus_downloads.json`。总下载量约 3.92 GB；仅获取这 9 个原始 Demo。
允许用户在项目私有数据目录重现下载；分享衍生分析时请保留本页来源、署名及许可链接。

用户已明确允许纳入 2026 年较早的 MVP 赛事，因此本批使用 4 月比赛；
没有将它们标记为最近六个月。`pgl-masters-bucharest-2026` 继续完全排除。

本项目使用原始数据重解析的 9 场总回合数，与来源目录中的独立 `rounds_played` 字段逐场一致。
目录内其他赛事和预先生成的统计数据没有被纳入本次训练。

## 2026-11 多赛事扩充

- 同一 OpenCS2 镜像和固定版本中，其余 8 场 IEM Rio Mirage 记录已加入训练；下载 URL、
  上游 LFS SHA256、大小及署名记录在 `configs/rio_additional_downloads.json`。
- 用户从 HLTV 提供 6 个压缩包，分别包含 BLAST Open Porto、IEM Beijing 和 BLAST Rivals 的
  Mirage。压缩包与解压 Demo 的本地 SHA256、比赛页面和日期见 `configs/user_hltv_sources.json`
  及 `configs/user_hltv_local.json`。这些本地计算的哈希用于完整性追踪，不冒充 HLTV 官方校验值。
- 数据集卡的 CC BY 4.0 标注只归属于 OpenCS2 镜像；没有假定其他 HLTV 资料具有相同许可。
- 所有原始 Demo 均以 Awpy 3.3.0 重新解析，旧版解析结果与报告单独保留；
  分组、训练与测试详情见 [多赛事实验记录](phase2_expanded_2026.md)。

## ESL Pro League Season 24 额外留出评估

- 用户提供六个 [ESL Pro League Season 24](https://www.hltv.org/events/8244/esl-pro-league-season-24)
  比赛的 HLTV 压缩包。比赛地址、日期、地图号、比分及压缩包本地 SHA256 见
  `configs/esl24_user_hltv_sources.json`，解压后的 Demo 本地哈希见 `configs/esl24_local.json`。
- 六张 Mirage 地图使用 Awpy 3.3.0 解析，并核对公开比分；赛事留出集只对固定模型评分，
  没有用于模型训练或校准。说明、评估计划及使用限制见 [多赛事实验记录](phase2_expanded_2026.md)。
- 这些 HLTV 文件的本地计算哈希不是官方校验值，OpenCS2 镜像的 CC BY 标注也不自动适用于这些文件。

## 50 场 Mirage 额外测试

- 用户依照 `configs/holdout_50_plan.json` 提供 50 个 HLTV 比赛压缩包；各比赛原始页面、
  日期、地图顺序和独立比分见 `configs/holdout_50_reference.json`。
- 50 场仅供固定模型测试。52 个录制分段及其哈希、一个阵营异常的处理、两场分段回合恢复
  和复核结果见 [50 场测试记录](holdout_50_results.md)。
- 这些文件的本地 SHA256 用于追踪完整性，并非 HLTV 官方发布的校验值；OpenCS2 镜像
  的 CC BY 标注不自动适用于用户提供的 HLTV 压缩包。
