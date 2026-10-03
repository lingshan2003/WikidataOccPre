# DBpedia 关系细分、R-GCN / GraphMask 重跑与可视化交接

更新：2026-10-02。以下以本仓库可见的代码和数据为准；用户反馈服务器上的第一轮 R-GCN、GraphMask 已初步跑通，但**服务器训练结果尚未同步到本地工作区**。本文件是下一轮工作的交接，不表示细分实验或新图已经完成。

## 1. 当前数据和实验口径

- 原始输入是 `external_data/dbpedia/2022.12.01/` 下的三份 `ttl.bz2`。`DBpedia/run.py` 依次筛人物、社会关系、生卒年、职业证据，最终输出 `external_data/dbpedia/processed/04_graph/`。
- 完整筛选图有 **55,690 人物端点、67,090 条原方向人物关系三元组、33 种谓词**。`DBpedia/report_final_relation_distribution.py` 导出的 `final_relation_distribution.tsv` 和 `final_relation_group_distribution.tsv` 给出逐谓词/现有九组的计数与占比。模型预处理额外加入反向边，因此全图模型输入有 **134,180 条有向边、66 个关系 ID**；不要把反向边当作新观察到的社会事实。
- `DBpedia/export_q_r_q_extended.py` 为每人选唯一职业：`dbo:occupation` → `dbo:profession` → 职业型 `dbo:` → 审查通过的 Wikidata Q 类。同来源多值时以源文件最早出现的事实为准，全部备选保留在 `processed/model_input/occupation_assignments.tsv.gz`。这里的 `occupation_level1` 只是兼容现有预处理的单标签字段，**不是原论文的三级职业本体**。
- `run.py prepare` 在全图保留至少 20 人的职业类别：82 个可监督类别、50,486 个可监督人物。少量低频类别的人仍留在图中，但其标签为 `y=-1`。时期子图会重新筛除该期局部低频职业的监督标签，不删除节点。
- `config/dbpedia_life_periods_20y_v1.json` 定义 8 张时期图：≤1500、1501–1900、1901–1920、1921–1940、1941–1960、1961–1980、1981–2000、≥2001。完整生卒区间与一个时期相交即可入期；只有单侧年份时仅进入该端点所在时期。**这表示人物生命时期可能与窗口重合，不表示社会关系发生于该窗口。**边只在两端都入期时保留。
- `DBpedia/run_rgcn_graphmask_20y.sh` 是现有一条龙脚本：全图 + 8 个时期各训练一个使用原始 33 种谓词的 R-GCN，再为各检查点训练 GraphMask 并生成测试报告。训练默认 seed 42、两层、采样扇出 `15,10`、50 epoch、职业特征层级 `1`、辅助特征 `temporal`；GraphMask 默认沿用检查点采样扇出 `auto`。
- 现有 `DBpedia/rules/relation_policy.tsv` 的 `social_group` 是**抽取和描述性统计标签**，并未把模型的 33 种谓词重编码成 9 组。`config/dbpedia_tie_taxonomy_v1.json` 仅提供训练器需要的 inherited/acquired 审计口径；默认训练没有执行分组消融。若要测试细分关系组作为模型关系类型的效果，必须另建重编码图并重新训练。

### 本地与服务器产物边界

本地已有原始抽取、兼容边表、全图及 8 张时期图；可复查 `artifacts/dbpedia_2022_priority_v1/`、`artifacts/dbpedia_2022_priority_periods_20y_v1/`。本地当前**没有** `runs_graphmask/dbpedia_2022_priority_20y_v1/`，`runs/dbpedia_2022_priority_20y_v1/` 也只有数据准备日志。因此不能在这台机器上宣称已有九张图的正式准确率、Macro-F1 或 GraphMask 解释结果。下一轮先从服务器核对或同步：

```text
runs/dbpedia_2022_priority_20y_v1/{full,时期ID}/seed_42/{best_model.pt,metrics.json,test_predictions.csv}
runs_graphmask/dbpedia_2022_priority_20y_v1/{full,时期ID}/seed_42/{graphmask_probe.pt,validation.json,training_history.json}
runs_graphmask/dbpedia_2022_priority_20y_v1/{full,时期ID}/seed_42/test_report/{test_metrics.json,relations_base.csv,relations_directed.csv,root_top_edges.csv.gz,manifest.json}
```

仅制图时可以先同步 JSON/CSV；继续训练或重跑解释时需要检查点和 probe。`DBpedia/summarize_graphmask_matrix.py` 要求九份测试报告齐全，生成汇总表。检查本体映射、seed、扇出、训练和 GraphMask 保真度设置是否一致，不以文件夹存在代替结果完整性检查。

## 2. 当前关系分布与细分动机

完整最终图的九个现有描述组（分母均为 67,090 条原方向关系）如下：

| 现有组 | 边数 | 占比 |
| --- | ---: | ---: |
| kinship | 23,552 | 35.105% |
| education_mentorship | 14,658 | 21.848% |
| intellectual_influence | 13,371 | 19.930% |
| office_succession | 10,624 | 15.835% |
| political_working_context | 2,154 | 3.211% |
| appointment_religious_act | 1,705 | 2.541% |
| partnership | 530 | 0.790% |
| sports_performance | 318 | 0.474% |
| professional | 178 | 0.265% |

`spouse` 单独有 8,868 条，混在 kinship 中会掩盖与血缘/其他亲属的差别；`trainer`、`training` 与博士生导师关系也不完全同义；`political_working_context` 和 `appointment_religious_act` 内部同样混有不同语义。这些是细分理由，不代表下表已获最终认可。

### 待审的细分草案：13 组，覆盖 33 个原始谓词

| 候选模型组 | 原始谓词（不写 `__rev`） | 全图原方向边数 |
| --- | --- | ---: |
| inherited | child, parent, father, mother, relative | 14,684 |
| marital | spouse | 8,868 |
| academic_mentorship | doctoralAdvisor, doctoralStudent, academicAdvisor, notableStudent | 11,043 |
| sports_mentorship | trainer, training, coach | 3,916 |
| intellectual_influence | influenced, influencedBy | 13,371 |
| office_succession | successor, predecessor | 10,624 |
| political_leadership | president, primeMinister, governor, governorGeneral, monarch, chancellor | 1,930 |
| political_assistance | lieutenant, vicePresident, deputy | 224 |
| religious_recognition | beatifiedBy, canonizedBy | 1,687 |
| appointment | appointer | 18 |
| partnership | partner | 530 |
| workplace | associate, employer | 178 |
| sports_opposition | opponent | 17 |

草案每个谓词恰好出现一次，合计 67,090 条。`inherited` 名称须保留：现有 `run.py collapse-relations` 要求它存在，并据此生成训练兼容的 `collapsed_tie_taxonomy.json`；`spouse` 沿用当前二分本体，单列于 inherited 外。`appointment`、`sports_opposition`、`workplace`、`political_assistance` 支持量很低，正式训练前要逐期查看边数，决定保留作低支持组、与语义相近的组审慎合并，或仅做描述性分析。`president`、`chancellor` 等也可能是机构职务上下文，不能直接解释为亲密或自愿社会关系。

实现时新增**版本化**配置，例如 `config/dbpedia_relation_taxonomy_fine_v1.json`。用 `training.tie_taxonomy.load_relation_taxonomy` 对全图和每个时期的 `relation_to_id` 验证：所有 33 个基础谓词恰好覆盖一次；反向关系随基础谓词进入同组。审核完成后固定映射，后续改动另起版本，不覆盖第一轮规则或产物。

## 3. 重跑方案

1. **冻结第一轮基线。** 从服务器收齐并核对九张图的 `metrics.json`、GraphMask `validation.json` 与 `test_metrics.json`。记录原模型和掩码后 Macro-F1、保真度、根节点数、扇出及运行 seed。若第一轮尚有缺失，先补齐第一轮结果，再比较新组。
2. **生成细分图。** `run.py collapse-relations --data <原图/graph_data.pt> --relation-taxonomy <新配置> --output-dir <新版本目录>` 已能按组重编码，同时保留 `__rev` 方向。对全图及 8 张时期图各生成一个独立产物；逐一检查 `relation_collapse_manifest.json` 中的 `edges_before`、`edges_after`、`duplicates_removed`。不同原谓词在同一人物对上合组后可能合并，故细分图的边数不一定与原图相等。人物、标签、划分和特征应保持一致。
3. **重新训练 R-GCN。** 复用第一轮的模型、seed、采样、职业特征、早停及评价口径；`--data` 指向细分图，`--tie-taxonomy` 指向该图生成的 `collapsed_tie_taxonomy.json`。使用新的 `artifacts/`、`runs/`、`runs_graphmask/` 版本目录；不能在已有结果上只换分组文件，因为模型关系嵌入维度和关系 ID 已变化。
4. **重新训练 GraphMask。** 每个新检查点配一个 probe，使用与检查点一致的图和 `--num-neighbors auto`。测试报告仍须生成 `relations_base.csv`、`relations_directed.csv`、`test_metrics.json` 等文件。默认验证保真度阈值为相对 Macro-F1 差 5%；若某期未满足，检查 `training_history.json`，不要悄悄修改阈值后与原结果混报。
5. **做对照汇总。** 每个时期列原始精确谓词模型与细分组模型的原模型测试 Macro-F1、GraphMask 掩码后 Macro-F1、prediction agreement、hard retention rate、实际边数和被合并边数。注明两套模型独立训练，GraphMask 的 gate 值不能被当作同一固定模型内的直接因果差异。可参考 `scripts/run_period_relation_graphmask.sh` 的重编码/训练矩阵组织方式，但该脚本针对旧 Wikidata R-GAT 路径，不能直接当作 DBpedia 命令运行。

## 4. 可视化代码及应交付结果

新建可重复执行的 `DBpedia/plot_relation_group_results.py`，输入旧、新实验根目录和细分本体配置，输出到被 Git 忽略的 `visualization/dbpedia_relation_groups_fine_v1/`。代码应验证九个上下文、报告文件、时期顺序与组覆盖后再绘图；另导出支撑每张图的整洁 TSV。旧的 `notebooks/graphmask_period_relation_group_heatmaps.ipynb` 可参考指标定义与布局，但其 R-GAT 路径和四时期配置不适用于 DBpedia，不能直接复用输出。

至少交付以下三类图，各自保存 PNG 和可编辑矢量格式（SVG 或 PDF）：

1. **数据构成图**：8 个时期（可加完整图）的原方向边数与细分组占比。原始关系表或时期 `edges.csv` 仅计非 `__rev` 边；同一人物可以进入多个时期，时期总量不可相加成全图总量。另附非孤立节点数，来源是 `DBpedia/report_period_graph_sizes.py` 的 `graph_sizes.tsv`，不要将包含孤点的全部入期节点数当作实际关系图规模。
2. **模型结果图**：各时期精确谓词与细分组的测试 Macro-F1，以及 GraphMask 掩码前后 Macro-F1 或差值。每个面板标注 test roots / 局部可监督类别数；不能把低支持时期的波动解释为组划分的确定收益。
3. **GraphMask 解释图**：按层分别画“时期 × 细分组”热图，并区分 `hard_retention_rate = 该组保留消息数 / 该组采样消息数` 与 `retained_edge_share = 该组保留消息数 / 本层全部保留消息数`。前者是组内留存率，后者是保留预算份额。读取 `relations_base.csv` 时按 `message_observations` 或保留消息数汇总，**不要对逐谓词百分比做简单平均**；若画精确谓词基线的组级图，先依新本体聚合；某期无该组消息记为缺失或明确标 0 支持。可另用 `relations_directed.csv` 展示正向与生成反向的差别。

可选第四张图比较“图中原方向边占比”与“测试时 GraphMask 保留消息占比”，但两者分母不同：后者是重复采样后的消息观测，不能直接称为原图边的重要性。每张图附输入路径、seed、时期版本、关系本体版本、层号和分母说明。`root_top_edges.csv.gz` 用于个案审查，别把其 top-k 截断结果当成全图关系频率。

## 5. 完成条件与下一位执行者的首项检查

- 新本体覆盖 33 个谓词且各出现一次；每个分期产物能用当前模型代码加载，并记录关系重编码后合并了多少边。
- 旧、新版本的九个 R-GCN 与九个 GraphMask 报告各自完整；模型参数、划分、职业特征、seed 与验证阈值可核对。必要时记录无法通过保真度的时期，而不把缺失结果填 0。
- 可视化脚本在不依赖 Notebook 手工改路径的情况下重现全部图和底层 TSV；图中的关系比例与 `final_relation_distribution.tsv` / 时期 `edges.csv` 可互相核对。
- **首先核实服务器第一轮结果是否真的九个上下文全部完成**，再定稿细分本体。当前本地没有正式 GPU 结果；这项状态不得从用户“初步跑通”的描述推断为全部完整。
