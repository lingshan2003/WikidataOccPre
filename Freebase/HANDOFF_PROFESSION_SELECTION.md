# Freebase Easy 人物职业处理交接

更新：2026-10-03。当前已完成数据抽取、描述性统计及职业大类映射的智能体初审草案，尚未冻结训练图或单职业标签。下一位接手者应先阅读本文件、[`README.md`](README.md)、[`relation_rules.json`](relation_rules.json)、[`descriptive_report.md`](../docs/freebase_easy_audit_2026-10-01/descriptive_report.md) 和 [`职业 L1 初审报告`](../docs/freebase_profession_review_2026-10-03/review_report.md)。

## 当前数据与结论

服务器已经从 `facts.txt` 完整重扫并按六阶段导出；本地下载包已解到 `external_data/freebase/descriptive_v2_local/`（该目录被 Git 忽略）。可复现脚本是 `Freebase/extract.py`，本地描述统计脚本是 `Freebase/describe_local.py`。关键结果：

| 阶段 | 数量 | 含义 |
| --- | ---: | --- |
| `is-a Person` | 3,970,878 | 不同名称，尚非全部核实的真实人物 |
| `Profession` 至少一个，且出生或死亡有可解析年份 | 598,078 | 当前资料完整候选人物队列 |
| 队列中有多个不同 `Profession` 值 | 420,421（70.30%） | **全部保留**，不选第一个、不删人物 |
| 队列两端的原始名称匹配三元组 | 148,445 | 60 种原始谓词，自环和反向记录仍在 |
| 20 个初审关系谓词 | 147,454 | 探索性关系集合 |
| 11 个主分析关系谓词 | 145,876 | 原始事实，非去重边数 |
| 11 谓词涉及的人物名称 | 100,756 | 其中 67,502 人有多个原始职业 |

此前误把“至少一个职业”实现为“恰好一个职业”，曾因多职业排除 121,890 条候选记录。`scripts/freebase_easy_build_extended_v1.py` 及其 14,451 条边的结果是**历史错误口径**，不要作为当前最终数据。新版 `Freebase/extract.py` 输出的节点表用 JSON 数组保存全部职业值：`05_final/nodes.csv` 的 `Professions`、`05_final/main_relation_facts.csv` 的 `Node1_Professions`/`Node2_Professions`。`02_cohort/cohort_people.tsv` 保存 598,078 人的全部职业与生卒年；不必为职业问题再扫 362,243,831 行 `facts.txt`。

## 困境具体是什么

`Profession` 是多值属性，不提供“主职业”或重要性排序。旧职业审计记录 3,084,516 条 `Profession` 事实，对应同样数量的不同“人物—原始职业值”组合，**没有重复事实**；因此也不能靠某职业在同一个人身上出现次数较多来选主职业。全库有 3,899 种原始职业字符串；稀少的 `Occupation` 谓词另有 49 条候选人物事实，尚未并入 `Profession`。

多值至少有两种可能来源，必须分开处理：

1. **宽泛类与细分类共存**：在 598,078 人的当前队列中，`Artist`＋`Musician` 同人出现 73,417 次，`Athlete`＋`Football player` 51,120 次，`Writer`＋`Screenwriter` 39,456 次，`Lawyer`＋`Criminal defense lawyer` 17,695 次。这些计数由本地 `cohort_people.tsv` 精确求交得到；它们说明职业值并非相互独立，但不能单凭英文词面自动证明全部上下位关系。
2. **可能的跨领域经历或多个职位**：如同一个人同时标注演员、作家、导演、音乐人。即使删掉宽泛标签，仍可能保留多个合理职业。不能把这种情况一概当数据错误，或按字母顺序、全局流行度、罕见程度任意取一个。

2026-09-28 的 `profession_crosswalk_draft_v1` 只是与旧数据职业字符串比对的**未审核草案**：3,899 个值中有 3,391 个标为 `unmatched`，382 个只是 `exact_text_candidate`。它不是已批准的语义映射，不能直接当成 L1/L2/L3 或单职业真值。本轮也不需要先建立职业层次。

## 当前建议的决策边界

- **描述性统计阶段**：保持多职业原值；职业频率应统计“拥有该职业的人数”，一个人可贡献给多个职业。明确写出分母和多标签口径。现有 [`descriptive_report.md`](../docs/freebase_easy_audit_2026-10-01/descriptive_report.md)按此处理。
- **若之后必须做单标签预测**：从原始职业数组派生一个**独立的新标签列**，不要覆盖 `Professions`。先手工审定有限的同义词与宽泛/细分规则；细分值取代宽泛值只在规则明确时发生。若规则处理后仍有多个不同领域职业，应标记 `ambiguous` 并暂不赋唯一标签，或改用多标签任务；不要因为标签未定就从原始人物与关系表中删节点。
- **样本单位**：优先用 Freebase 链接 ID 辅助身份核对，但名称—MID 映射仍是暂定证据。当前 11 谓词人物中 100,282 个名称有唯一 ID，474 个缺 ID；缺 ID 不应自动删去。职业选择规则与身份审计是两件事。
- **关系单位**：当前 145,876 是原始事实；其中有反向重复、自环、可能的同名碰撞。训练时若要去重或规范关系方向，另开版本，不要改写原始抽取表。

建议新增派生审计表，至少包括 `person_name`、`freebase_id`、`raw_professions_json`、`mapped_professions_json`、`chosen_profession`、`resolution_status`、`rule_id`、`review_version`。`resolution_status` 可用 `single_raw`、`single_after_mapping`、`multiple_valid`、`needs_review`；空的 `chosen_profession` 必须与无职业区分。若需要回溯每条职业事实的原始行号，服务器上的 `02_cohort/person_attribute_facts.tsv` 保留了 `source_line`，无需再读全量库。

## 下一步按顺序做

1. **只读审计高频职业组合**：分别统计 598,078 人和 100,756 个关系参与者中的职业数量分布、最常见职业对及相关关系类型。优先检查宽泛/细分共存的高频组合和确实跨领域的组合；保存人工复核样例。
2. **制定小规模规则草案**：只覆盖经人工确认的同义词、拼写变体和明确的宽泛/细分关系；每条规则给出理由、例子和例外。暂时不要求覆盖全部 3,899 个值。
3. **测量代价**：报告规则前后有唯一候选职业的人数、仍多职业的人数、失去标签但仍留在关系数据中的人数，以及各关系组/出生时期的覆盖变化。不要只报告总体准确率或总体保留率。
4. **再定任务**：如果大量人物仍有多个合理职业，优先讨论多标签评价；如果研究必须单标签，明确弃权范围与由此产生的选择偏差，然后才生成实验输入。

## 2026-10-03：当前关系图的职业大类映射初审

用户指定优先处理当前关系图实际涉及的 **1,817** 个原始职业值。三名 GPT-6 Luna（high）子智能体按共同四类口径分批提出候选，主智能体二次复核前 150 个高频词及其他边界，共 288 个值。方法包含明确词表规则、旧标签文本参照和智能体词义判断，**不等同逐条人工批准，也未核实每个人的职业事实**。

审核目录：`docs/freebase_profession_review_2026-10-03/`，其中 `REVIEW_POLICY.md` 写明本体边界；`profession_l1_crosswalk_draft.tsv` 保存全部 1,817 行、理由、证据、规则和待人工审核状态；三批原始决定及主智能体修订另存。旧本体的 `Leadership` 包括政治、法律、宗教、军事、贵族与企业经营，不等于狭义组织领导。

当前初审草案为：1,095 个值有有效大类候选、416 个待审、306 个在当前四类之外。对全部 100,756 个人物，只有全部原始职业都有有效候选且大类去重后唯一，才派生唯一候选标签；任何未解决或四类之外的值都阻止这一判定。得到 **81,621** 人唯一候选大类（81.01%）、**6,475** 人多个候选大类（6.43%）、**12,660** 人映射未完整（12.57%）。这些是草案覆盖，不是已批准监督样本数。

人物派生审计保存于 `external_data/freebase/profession_l1_review_v1/person_l1_audit.tsv`。原始人物、职业和 145,876 条关系事实全部保留。关系组和出生时期的覆盖见初审报告，不因未获得唯一标签删人物或边。

复现盘点：`python Freebase/review_professions.py inventory`；修改批次及 `primary_review.tsv` 后重算：`python Freebase/review_professions.py merge`。主智能体复核会覆盖对应批次候选，必须同步维护其决定，不能仅改单个批次就假定最终表已更新。规则/数据源变化须另开审核版本，不能把草案结果直接喂入训练。

下一步优先审定高频泛化词、本体边界和中置信候选；`unresolved_person_coverage.tsv` 同时报告某值涉及人数与“唯一阻塞值且已有一个有效大类”的人数，二者不是同一回事。例如 Manager 涉及 4,155 人，但只对 44 人构成这种单一阻塞；Operator 对 1,300 人构成单一阻塞。复核前不能据此保证能获得多少新唯一标签，也不能用同人的其他职业或邻居关系填补待定职业的词义。

## 关于 `Parent` 的补充核对

原始 `facts.txt` **有** `Parent` 谓词：全库 8,808 条。旧全量人物名称对清单里，只有 **2 条** 的主客体都匹配 `is-a Person` 候选名称；本次严格队列的 148,445 条人物对中，`Parent` 为 **0 条**。因此当前 11 谓词表没有 `Parent`，不是我们把它转换成了 `Children`，也不是漏掉一个本应很大的亲子关系。`Children` 在严格队列中有 31,541 条原始事实；按已审样例的父母→子女方向，将来可由其原向派生 `parent_of`、反向派生 `child_of`，并保留来源与方向规则。

## 运行与核查

本地已下载的六阶段文件在 `external_data/freebase/descriptive_v2_local/`。重新生成描述统计：

```bash
python Freebase/describe_local.py
```

报告中 598,078、148,445、147,454、145,876 等数量已逐项与阶段 `summary.json` 核对。首次全量抽取的命令和 tmux 用法见 `Freebase/README.md`。后续职业选择只需读本地 `02_cohort/cohort_people.tsv`、`05_final/nodes.csv` 和必要时的服务器职业属性子集；请勿重新下载或全量扫描原始 23 GB `facts.txt`，除非发现新的抽取规则确实需要原始事实。
