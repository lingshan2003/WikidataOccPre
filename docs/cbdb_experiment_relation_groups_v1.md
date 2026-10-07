# CBDB 主实验对齐的关系大组 v1（2026-10-06）

这是当前 CBDB **实验模型使用的关系大类**。此前的 31 个语义组与 75 个角色方向类型保留为审计层，不再作为本轮实验粒度。用户明确要求把亲属表归入 inherited、另行注意伴侣，并参考主实验的多关系组设计。

## 一级与多关系组口径

二分组为 inherited / acquired。inherited 是家庭归属口径：KIN 表除本人直接伴侣外全部纳入，含姻亲、养继和嗣承；ASSOC 中明确义女、假子(养子)角色也归入。它不等于全部经过核验的生物血缘。婚姻与伴侣进入 acquired；其余所有社会联系也进入 acquired，acquired 不表示自愿或双方生前直接交往。

多关系组共有 12 个基础模型类型。inherited、intimate_partnership、education_mentorship、professional_collaboration、influence_succession 与主实验使用相同主题名；CBDB 宗教组的具体语义比主实验 religious_ordination 更宽，因此用 religious_relation。other_acquired 为显式剩余组。CBDB 独特的著述、考校、政军联系、一般社会联系和冲突主题单独保留。

| 基础关系组（模型 Relation） | 含义与边界 | 合并后原方向边 |
| --- | --- | ---: |
| inherited | 亲属与家庭归属 | 25,989 |
| intimate_partnership | 婚姻与伴侣 | 709 |
| education_mentorship | 教育与师承 | 2,408 |
| professional_collaboration | 职业合作与服务 | 345 |
| influence_succession | 影响与志业传承 | 248 |
| religious_relation | 宗教制度与活动 | 15 |
| textual_relations | 著述与文字关系 | 41,674 |
| education_assessment | 科举与考校 | 144 |
| political_institutional | 政军制度与支持 | 3,059 |
| social_association | 朋友与学术社会联系 | 3,331 |
| opposition_conflict | 对抗与冲突 | 1,511 |
| other_acquired | 其他后天社会关系 | 270 |

重要的归类边界：

- 配偶、妾、明确订婚伴侣单列 intimate_partnership；岳父、媳妇、亲家等姻亲路径仍进入 inherited，不能把配偶的所有亲属当成本人的伴侣。
- 教育与师承不含单纯私淑、研读作品或风格效法；这些进入 influence_succession。考官、科举次第评定独立进入 education_assessment。
- 同僚、家内与侍从服务属于职业服务主题；共撰、编辑、校订、刊刻作品及明确聘教也进入 professional_collaboration。一般赠诗、墓志和评论不自动叫职业合作；聘用者也不自动叫学生。
- textual_relations 合并墓志、序跋、书札、传记和其他文字活动。它是文字行为关系，不能把全组叫朋友或生前互动；精确文字类型、作者/请托角色仍可查细分映射及原始码。
- political_institutional 包括官场上下级、荐举、政军支持和庇护等；opposition_conflict 将政治、军事、学术、司法冲突独立为冲突主题。
- social_association 包括朋友、同乡同会、同门同年、拜访及一般学术文艺联系，不能把全组解释为亲密友谊。
- 经济、医疗、其余家庭事务及未明交往保留在 other_acquired；没有将稀少组的零支持解释为关系机制不重要。

## 模型粒度与边数

多关系组表的 `Relation` 直接等于上述大组名称，**不附加 to_parent/to_teacher/actor_to_recipient 等细角色后缀**。原始 Node1 → Node2 端点顺序保持；模型预处理依主实验约定生成 `__rev` 反向消息类型。

| 表示 | 基础类型数 | 加模型反向后的类型数 | 合并后原方向边 | 加反向并去重后消息边 |
| --- | ---: | ---: | ---: | ---: |
| multi_group | 12 | 24 | 79,703 | 159,406 |
| binary | 2 | 4 | 76,881 | 153,762 |

二分组边为 inherited 25,989、acquired 50,892。多关系组中的 inherited 也是 25,989；acquired 拆为其余 11 组后，同一个有向人物对可能出现在多个主题，所以二分组总边数比多关系组少。这里各计数是原方向三元组或预处理消息边，不是独立人物对或历史事件数。

每个表示都保留同一批 **22,392 名人物、98,557 条原库支持行**。多关系组合并 12,465 条组内平行三元组，二分组合并 15,287 条。合并保留 `SupportRows`、`RawTripleCount`、`OriginalRelations`、`ExperimentGroups` 和细分组/角色集合，原始标注表仍有全部 92,168 条记录。姓名、生卒年和职业保持输入原值。

## 输出文件

根目录：[experiment_groups_v1](/Users/wangyue/RGCN/external_data/cbdb/2026.09.14/experiment_groups_v1)。

- 执行口径：[config/cbdb_experiment_relation_groups_v1.json](/Users/wangyue/RGCN/config/cbdb_experiment_relation_groups_v1.json)，显式覆盖 31 个细分组，并给出 19 个职业协作/明确收养角色的逐码覆盖规则；最终覆盖原码表全部 981 个正码。
- 多关系组数据：[multi_group/person_relation_triples.tsv.gz](/Users/wangyue/RGCN/external_data/cbdb/2026.09.14/experiment_groups_v1/multi_group/person_relation_triples.tsv.gz)。
- 二分组数据：[binary/person_relation_triples.tsv.gz](/Users/wangyue/RGCN/external_data/cbdb/2026.09.14/experiment_groups_v1/binary/person_relation_triples.tsv.gz)。
- 两个表示目录均有 `Q_R_Q_extended.csv.gz`、`relation_taxonomy.json`、`tie_taxonomy.json` 和全图关系词表 `relation_vocabulary.json`。taxonomy 按当前基础关系组显式且互斥覆盖，供既有训练和 GraphMask 分组报告使用。模型 ID 与主实验的 sorted 词表顺序一致。
- 根目录 `raw_relation_taxonomy.json` / `raw_tie_taxonomy.json` 覆盖本次图中的 630 个原始码，可用于先准备原码图、再调用既有 collapse-relations / collapse-ties 的路线；它们与已经合并的两个表示目录中的 taxonomy 不能混用。
- 根目录 `relation_mapping.tsv` 记录原码、细组、角色、大组及覆盖理由；`group_support.tsv`、`summary.json` 记录数量；`nodes.tsv.gz` 是同一批人物的原始单层属性。

## 复现与接入现有主实验

```bash
python3 scripts/cbdb_build_experiment_groups.py
python3 -m unittest tests.test_cbdb_experiment_groups tests.test_cbdb_group_relations
```

默认从原始类型的 `flat_triples_v1/person_relation_triples.tsv.gz` 重新合并，而不是从已合并的 75 类型表倒推，避免丢掉同组内原始来源支持分配。

输入 CSV 适配既有 `run.py prepare` 接口：真实单层职业只放在 `Node1/2_occ_level1`，L2/L3 和 Country 为空。这是字段桥接，不构造职业层级、不补造国家。后续应使用 `--target-level 1` 和训练的 `--occupation-feature-levels 1`。例如多关系组全图接入：

```bash
python run.py prepare   --input external_data/cbdb/2026.09.14/experiment_groups_v1/multi_group/Q_R_Q_extended.csv.gz   --output-dir artifacts/cbdb_experiment_groups_v1/multi_group   --target-level 1 --min-class-count 20 --seed 42

python run.py train   --data artifacts/cbdb_experiment_groups_v1/multi_group/graph_data.pt   --output-dir runs/cbdb_experiment_groups_v1/multi_group/seed_42   --model rgcn --num-bases 24   --occupation-feature-levels 1 --auxiliary-features temporal   --tie-taxonomy external_data/cbdb/2026.09.14/experiment_groups_v1/multi_group/tie_taxonomy.json   --relation-taxonomy external_data/cbdb/2026.09.14/experiment_groups_v1/multi_group/relation_taxonomy.json
```

这些是现有 CLI 的接入示例，尚未执行图准备、训练划分或模型训练。示例阈值 20 和 seed 42 为现有默认，不是已冻结的 CBDB 正式实验设计。二分组接入时使用对应 binary CSV、taxonomy 和独立输出目录，num-bases 可设为 4。两种表示应采用相同人物划分；当前人物与排序一致，正式分期仍应显式核对划分一致性。

未知码、遗漏或重叠分组、重复输入三元组及人物属性冲突会报错，不静默删记录。全量核对已确认：两个表示可展开覆盖全部原始三元组键；来源支持数与人物集合守恒；CSV 的职业、年份及端点与原表一致，未制造 L2/L3/国家；词表及单双 taxonomy 完整互斥覆盖。13 项测试通过。实际 PyG 运行尚未验证。

## 保留的未定事项

时期没有划分。职业仍为试建标签，类别失衡问题没有因为关系合并而解决。用户下一步再考虑时期口径；后续时期图应保留完整的 12 组 / 24 类型词表（binary 为 2 / 4），无支持的组报告零边，不重排模型 ID。
