# CBDB 关系分组 v1（2026-10-06）

状态：细分审计层已执行、已导出、已验证。**当前实验模型粒度改用 [12 个大关系组](cbdb_experiment_relation_groups_v1.md)；本页的 31 组/75 类型保留为语义和方向追溯层。**以 `flat_triples_v1` 的同一批人物、年份和试验职业为输入，仅处理关系粒度。未分配时期，未生成训练/验证/测试划分，未训练模型。职业标签仍为 v1 试验标签，关系分组不等于职业质量已经通过。

## 分组口径

原始 SQLite 有 981 个正关系码（KIN 485、ASSOC 496）。最终显式映射覆盖全部正码，不把新增码自动吸收进旧组。当前 v1 图实际出现 630 个原始码，全部成功映射到 31 个语义组；结合角色方向得到 75 种实际出现的模型关系。

分组依据是原始中文/英文名称、亲属路径、官方交往子类及反向码。官方 A/P/M 仅作辅助：它不必然对应关系名称的施受语义；中文与英文角色冲突时不选择一方作事实真值，标为 `unspecified` 并记录 qualifier。此版本是码表层面的语义映射，不宣称逐条核验了全部历史事件。

亲属分为一代亲子、祖裔、兄弟姐妹、旁系、姻亲、养继/嗣承亲子及祖裔；配偶与伴侣单列。亲子称谓不自动证明生物学血缘，嫡母、庶母、继承身份等保留说明。“养继”表示亲属角色来源，不断言两人绝无其他血缘。

交往保留师承、科举考校、学术影响、学术与文艺联系、同僚、上下级、政治支持、荐举、对抗、朋友、社会交往，以及医疗、宗教、经济、家内服务和家庭交往。师承不与考试官关系混为一类；研读著作、私淑、风格效法不当作直接授课；“其后代出于Y之门”不把本人写成Y的学生。

文字关系分为书札、墓葬纪念、序跋题词、传记和其他文字。委托撰文与实际撰文用不同角色；为第三方求得文字也注明实际作者不在这两个端点中。墓志、传记、纪念碑等可以跨越生死年代，不能把这些组解释为双方生前直接接触。

`RelationFamily` 只分 `kinship`、`partnership`、`association`，不将前者宣称为全部生物血缘，亦未在本次强行冻结跨数据源 inherited/acquired 的同义对应。

## 方向与去重

所有输出保留原始 Node1 → Node2 顺序，不交换人物或伪造反向事实。例如：

| 原始码 | 码表含义 | 模型关系 |
| --- | --- | --- |
| KIN:75 | 目标是源人物的父亲 | kin_parent_child__to_parent |
| KIN:180 | 目标是源人物的儿子 | kin_parent_child__to_child |
| ASSOC:22 | 源人物为目标的学生 | education_mentorship__to_teacher |
| ASSOC:23 | 目标为源人物的学生 | education_mentorship__to_student |
| ASSOC:558 | 源人物为目标的考官 | education_assessment__to_examinee |
| ASSOC:44 | 源人物为目标作墓志 | text_epitaph__actor_to_recipient |
| ASSOC:43 | 源人物的墓志由目标所作 | text_epitaph__recipient_to_actor |
| ASSOC:154 | 源人物请目标为第三方作墓志 | text_epitaph__to_writer |

`peer` 表示码表给出的对等角色，不保证观察到两条镜像记录；`unspecified` 表示没有冻结精确角色，也不把原始端点顺序抹去。各组内 `actor_to_recipient` 按具体行动识别施为与受事，不能将所有施为者概括为作者或所有受事者概括为学生。

同一 Node1、同一组与角色、同一 Node2 才合并。不同角色、反向人物对及不同语义组不合并。合并后 `SupportRows` 是原始数据库支持行之和，`RawTripleCount` 是 v1 原始类型三元组数；它们都不是独立事件数，也不自动作为训练边权。原始码集合完整存入 `OriginalRelations`。

未增加模型预处理的 `__rev` 消息边。原库已经存在的反向记录仍保留；后续构图要显式说明消息反向边的处理，不把这里的边数与加反向消息后的边数混报。

## 实际支持量

下表的原始三元组分母为 92,168，分组后分母为 87,126；组内人数可能重叠，不能相加得到总人物数。所有 31 组均保留，没有按支持量删掉稀少组。

| 组名 | 中文含义 | v1 原始三元组 | 分组后边 | 涉及人物 | 方向关系数 |
| --- | --- | ---: | ---: | ---: | ---: |
| academic_exchange | 学术与文艺联系 | 832 | 815 | 609 | 4 |
| education_assessment | 科舉考試關係 | 144 | 144 | 113 | 2 |
| education_mentorship | 師承教學 | 2,631 | 2,422 | 1,388 | 2 |
| family_social | 家庭與姻親交往 | 56 | 56 | 42 | 5 |
| financial_support | 財務與物質往來 | 50 | 50 | 63 | 3 |
| friendship | 友誼 | 2,452 | 2,407 | 1,176 | 1 |
| household_service | 家内与侍从服务 | 8 | 8 | 8 | 2 |
| influence_succession | 学术影响与志业传承 | 256 | 248 | 183 | 2 |
| intimate_partnership | 配偶与伴侣 | 709 | 709 | 669 | 1 |
| kin_affinal | 姻亲 | 2,883 | 2,874 | 2,341 | 1 |
| kin_ancestor_descendant | 祖先与后裔 | 1,484 | 1,484 | 1,177 | 2 |
| kin_collateral | 旁系亲属 | 799 | 799 | 690 | 1 |
| kin_parent_child | 父母与子女 | 19,483 | 19,420 | 12,780 | 2 |
| kin_sibling | 兄弟姐妹 | 1,378 | 1,378 | 1,108 | 1 |
| kin_social_ancestor_descendant | 养继与嗣承祖裔 | 6 | 6 | 6 | 2 |
| kin_social_parent_child | 养继及非亲生亲子 | 60 | 58 | 58 | 2 |
| medical_care | 醫療照護 | 2 | 2 | 2 | 2 |
| military_support | 軍事支持 | 149 | 147 | 133 | 3 |
| official_colleagues | 同僚 | 237 | 237 | 224 | 1 |
| official_hierarchy | 官場上下級 | 646 | 616 | 506 | 2 |
| opposition | 對抗與衝突 | 1,790 | 1,671 | 877 | 4 |
| other_association | 其他或未明交往 | 173 | 171 | 146 | 2 |
| political_support | 政治支持 | 1,340 | 1,311 | 874 | 4 |
| recommendation | 薦舉保任 | 1,145 | 1,135 | 781 | 2 |
| religious_relation | 宗教關係 | 15 | 15 | 17 | 2 |
| social_association | 社會交往 | 291 | 289 | 243 | 4 |
| text_biography | 傳記文字 | 3,438 | 3,402 | 1,764 | 3 |
| text_correspondence | 書札往來 | 10,915 | 9,857 | 2,194 | 2 |
| text_epitaph | 墓葬紀念文字 | 12,896 | 12,301 | 6,061 | 4 |
| text_other | 其他文字作品 | 17,378 | 15,661 | 3,730 | 3 |
| text_preface | 序跋題詞 | 8,522 | 7,433 | 2,025 | 4 |

总人物数仍为 **22,392**；分组前后来源支持行总和均为 **98,557**。合并掉的是同组同方向同人物对的 5,042 条平行原始类型三元组，没有删除人物或来源证据。职业分布保持 v1 的分布，“做官”仍为 21,504 人。

## 歧义与码表异常

本图中，14 个出现的原始码、220 条原始类型三元组存在中文/英文角色描述冲突。它们仍在标注表及分组表中，以 qualifier 记录冲突，并用未明确角色类型；具体名单可以筛选标注表的 `RelationQualifier`。这不影响关系码完整覆盖，不代表这些历史事件已得到逐条纠错。

审计另记录 3 个不具备同组反向匹配的正码：KIN:572/573 的妾之子女名称与“非亲生父母”反向码有定义差异；KIN:578 外曾祖母的反向码指向外曾祖父，双方都在祖先端。这三个码当前 v1 的出现次数均为 0，映射保留说明，没有改写原库。KIN:368 的一个替代反向码同样指向后裔，但另一个反向码有效，因此不作为无可用反向匹配记录；也不强制采用无效替代码。

交往正码中，34 个完全没有官方类型关联，另有 656/657 仅关联根级 `01` 而非叶级子类。它们依名称显式分组。官方三级/四位类别与此处语义分组不是同一概念，完整映射表保留官方类别以便追溯。

## 文件与复现

- 最终映射：[config/cbdb_relation_groups_v1.json](/Users/wangyue/RGCN/config/cbdb_relation_groups_v1.json)。逐码包含组、角色、qualifier 和理由；两份 `*_groups_review_v1.json` 是初审留档，含主控修订前的判断，执行以最终映射为准。
- 保留原始三元组的分组标注：[person_relation_triples_annotated.tsv.gz](/Users/wangyue/RGCN/external_data/cbdb/2026.09.14/grouped_relations_v1/person_relation_triples_annotated.tsv.gz)。此表 `Relation` 仍为原码，`RelationGroup` 是新语义组，旧分组在 `OriginalRelationGroup`，`ModelRelation` 是组加角色。
- 合并后的模型关系表：[person_relation_triples_grouped.tsv.gz](/Users/wangyue/RGCN/external_data/cbdb/2026.09.14/grouped_relations_v1/person_relation_triples_grouped.tsv.gz)。此表 `Relation` 已是组加角色，原始码在 JSON 列 `OriginalRelations`。
- 独立人物表：[nodes.tsv.gz](/Users/wangyue/RGCN/external_data/cbdb/2026.09.14/grouped_relations_v1/nodes.tsv.gz)。单层试验职业、姓名和原始生卒年，不制造职业层级或国家值以迎合现有其他数据源 loader。
- 可读逐码映射：[relation_mapping.tsv](/Users/wangyue/RGCN/external_data/cbdb/2026.09.14/grouped_relations_v1/relation_mapping.tsv)；组支持量：[group_support.tsv](/Users/wangyue/RGCN/external_data/cbdb/2026.09.14/grouped_relations_v1/group_support.tsv)；方向类型支持量：同目录 `directional_relation_support.tsv`。
- 统计与审计：同目录 `summary.json`、`inverse_code_audit.tsv`。完整图词表 `relation_vocabulary.json` 提供 75 个稳定 ID；`relation_group_taxonomy.json` 将这些方向类型归回 31 个语义组，便于后续分组报告。未来分期可保留全图词表，即使某组在一个时期为零边，也不要自动改 ID。

复现命令：

```bash
python3 scripts/cbdb_group_relations.py
python3 -m unittest tests.test_cbdb_group_relations
```

脚本只读 SQLite、读取现有 v1 三元组，写入独立的 `grouped_relations_v1`。未知码、重复输入三元组、人物属性冲突、未审核的反向映射异常均会报错，不静默丢弃。全量导出另已核对：标注表除分组字段外与原始输入逐行一致；合并表可展开恢复全部原始三元组键；来源支持数及人物集合守恒，输出没有重复的有向分组类型三元组。7 项语义及导出测试通过。

## 下一阶段

时期划分待另行确定。人物生卒年不是关系发生年；当前节点允许只有生年或只有卒年，后续应明确时期采用出生队列、生命跨度还是其他研究口径，并检查每期人物、边、职业和关系组的支持量。稀少组和职业失衡限制仍然存在。本次不预设时期边界或按死亡年反推缺失生年。
