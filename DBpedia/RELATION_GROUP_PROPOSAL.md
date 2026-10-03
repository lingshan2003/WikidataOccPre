# DBpedia 关系分组建议：对齐主实验的审阅草案

日期：2026-10-03。状态：供审阅，尚未作为新实验口径执行。

依据：`DBpedia/HANDOFF_relation_groups_and_visualization.md`、`config/tie_taxonomy_acquired_subgroups_v1.json`、`config/dbpedia_tie_taxonomy_v1.json`，以及本地最终图 `external_data/dbpedia/processed/04_graph/graph_edges.tsv.gz`。已逐条重新计数并与 `final_relation_distribution.tsv` 核对。本方案针对现有 33 种原始谓词，保留原 inherited/acquired 边界，优先与主实验的 acquired 主题组对齐。

## 一级划分

| 一级组 | 原始谓词数 | 原方向三元组数 | 全图占比 |
| --- | ---: | ---: | ---: |
| inherited | 5 | 14,684 | 21.887% |
| acquired | 28 | 52,406 | 78.113% |
| 合计 | 33 | 67,090 | 100.000% |

`inherited` 沿用本项目的家庭归属口径：`child, parent, father, mother, relative`。`relative` 是宽泛亲属标签，不能断言这些边全部为生物血缘。`spouse` 和 `partner` 进入 acquired。acquired 是 inherited 之外的社会关系集合，不等于自愿选择的关系，也不保证双方生前有直接互动。

## 二级划分：一个 inherited 组，五个 acquired 主题组，一个 acquired 兜底组

| 二级组 | 中文解释 | 全部原始谓词 | 谓词数 | 原方向三元组数 | 全图占比 | acquired 内占比 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| inherited | 家庭归属／亲属 | child, parent, father, mother, relative | 5 | 14,684 | 21.887% | — |
| intimate_partnership | 婚姻与亲密伴侣 | spouse, partner | 2 | 9,398 | 14.008% | 17.933% |
| education_mentorship | 教育、师承与训练指导 | doctoralAdvisor, doctoralStudent, academicAdvisor, notableStudent, trainer, training, coach | 7 | 14,959 | 22.297% | 28.544% |
| professional_collaboration | 职业合作与雇佣 | associate, employer | 2 | 178 | 0.265% | 0.340% |
| influence_succession | 思想影响与职务接替 | influenced, influencedBy, successor, predecessor | 4 | 23,995 | 35.765% | 45.787% |
| religious_authority_recognition | 宗教权威／认可 | beatifiedBy, canonizedBy | 2 | 1,687 | 2.515% | 3.219% |
| other_acquired | 其他后天关系：政治任职、任命、对立 | president, primeMinister, governor, governorGeneral, monarch, chancellor, lieutenant, vicePresident, deputy, appointer, opponent | 11 | 2,189 | 3.263% | 4.177% |

分母为全图 67,090 条原方向三元组；acquired 内占比的分母为 52,406。四舍五入后百分比可能略有加总误差。正反向、互为逆谓词等观察仍按原始三元组分别计数，不代表独立人物对数；不计预处理生成的 `__rev` 边。

## 与主实验的对应和必要调整

1. **婚姻／伴侣**：对齐主实验的 `spouse, unmarried_partner, significant_person` 主题。DBpedia 的 `partner` 暂归亲密伴侣，不能看到英文 partner 就照搬主实验的 `partner_in_business_or_sport`。本地图中样例有 `Alberto_Fernández → Fabiola_Yáñez`、`Alberto_Moravia → Dacia_Maraini`、`Albert_Rivera → Malú`。在线 [DBpedia 人物页](https://dbpedia.org/page/Alberto_Fern%C3%A1ndez)也显示前者的 `dbo:partner`。这是结合样例的谓词级归类建议，并非逐条人工验证全部 530 条边。
2. **教育／师承**：主实验已经把 `trained_by, head_coach` 纳入教育指导，所以 DBpedia 的 `trainer, coach` 留在这一大组。`training` 的本地样例是 `Abraham_de_Haen → Cornelis_Pronk`、`Achille_Etna_Michallon → Jacques-Louis_David`，含艺术师承，不宜全部叫体育关系。[DBpedia 官方映射](https://mappings.dbpedia.org/server/ontology/classes/MusicalArtist)也将 training 列为 Artist 的属性，且当前声明的 range 为 EducationalInstitution；本图只保留人物端点，其实际用法与 schema range 并不完全一致，仍需以保留下来的数据为准。
3. **职业合作／雇佣**：`employer` 与主实验同名主题对应；`associate` 按现有抽取口径和本地同僚样例暂纳入职业合作。associate 本身较宽，样例不少是政治同僚，不能解释为全部商业合作。该组只有 178 条；现有 ≤1500 时期为零条，≥2001 时期只有 6 条，正式分期比较需要报告支持量，零支持不能解释为模型不需要该组。
4. **影响／接替**：对齐主实验的 `influenced_by, inspired_by, replaces, replaced_by` 主题。主表保持合组，便于对照；解释时可再分思想影响（13,371 条）和职务接替（10,624 条），避免把接任直接解释为思想传承。
5. **宗教组**：主实验的 `religious_ordination` 只有 `consecrator`，属于祝圣关系。DBpedia 的 `beatifiedBy, canonizedBy` 是列福／封圣，故用 `religious_authority_recognition` 明确语义差别。它是宗教制度关系层面的主题对应，不能当作与主实验同一种机制。[DBpedia 的 canonizedBy 定义](https://dbpedia.org/ontology/canonizedBy)与这一用语一致。认可可能发生在当事人死后，例如 [Hildegard of Bingen](https://dbpedia.org/page/Hildegard_of_Bingen)；更不能将该组直接解释为生前人际接触。
6. **兜底组**：政治领导、协同任职等谓词主要记录任职上下文；任命也不等于合作。为了保持主实验职业合作组的含义，本草案将它们保留在 `other_acquired`，不为增加职业合作组支持量而并入。`opponent` 的本地样例有 `Brett_Carter_(politician) → Diane_Black`、`Jon_Grunseth → Arne_Carlson` 和历史冲突人物，不能统一叫体育对手。

## DBpedia 内部的辅助细分

下面可以作为组内解释标签，不改变主表的七组结构：

| 主表组 | 辅助标签 | 谓词 | 原方向三元组数 |
| --- | --- | --- | ---: |
| education_mentorship | academic_mentorship | doctoralAdvisor, doctoralStudent, academicAdvisor, notableStudent | 11,043 |
| education_mentorship | artistic_or_other_training | training | 401 |
| education_mentorship | training_coaching | trainer, coach | 3,515 |
| influence_succession | intellectual_influence | influenced, influencedBy | 13,371 |
| influence_succession | office_succession | successor, predecessor | 10,624 |
| other_acquired | political_institutional_context | president, primeMinister, governor, governorGeneral, monarch, chancellor, lieutenant, vicePresident, deputy, appointer | 2,172 |
| other_acquired | opposition | opponent | 17 |

如果后续重点是 DBpedia 自身的制度关系，可以把 political_institutional_context 提升为第八组；当前建议主表先对齐主实验，并在附表单独展示这 2,172 条政治制度边。

## 文件与使用边界

- 草案配置：`config/dbpedia_tie_taxonomy_acquired_subgroups_draft_v1.json`。显式列出全部 33 个谓词，新谓词必须重新审阅；不使用自动吸收新谓词的 all_remaining。
- 逐谓词核对表：`DBpedia/relation_group_proposal_mapping.tsv`。
- 支持量核对表：`DBpedia/relation_group_proposal_support.tsv`，包含完整图及现有八个时期，均排除 `__rev`。
- 验证：33 个原始谓词互斥且完整覆盖，二级 inherited 成员与现有二分配置一致，原方向总数为 67,090；本地原始图与逐谓词分布表一致。没有改动原图或运行新的模型实验。

现有 R-GCN / GraphMask 以 33 种谓词训练。**只做关系组分析，可直接按本草案聚合既有 GraphMask 报告，不需要重新训练。**组内 hard retention rate 应以保留消息数除以消息观测数，不能简单平均各谓词百分比；反向消息跟随基础谓词归组。只有把组名替换为模型的关系类型、或运行新的分组消融，才需要相应的图处理或训练。

现有 taxonomy loader 会拒绝配置里出现而词表中缺席的谓词。后续实际核查确认：本项目现有八张时期图均保留全图的 66 个有向关系 ID，即使某类边在该时期没有出现，也不会从 metadata 词表删除。因此这些时期图可以直接加载全图分组配置，保留统一的模型关系 ID；无边组须记录零支持，GraphMask 留存率留空。若未来其他构图流程缩减词表，则需先处理缺席谓词。按生命时期构造子图不代表关系发生于该时期，跨期数量不可相加为全图总量。

该口径已用于正式运行配置 `config/dbpedia_tie_taxonomy_acquired_subgroups_v1.json`；分时期二分组／多组 RGCN + GraphMask 运行说明见 `DBpedia/GROUPED_PIPELINE.md`。
