# CBDB 数据重建交接（更新至 2026-10-07）

## 当前目标与已确定的口径

从 **CBDB 2026-09-14 原始 SQLite** 重新构建人物关系数据，不沿用以前导出的关系图或其中的职业标签。目标是人物—关系—人物扁平表：关系两端各有一个明确的**单层职业**，且各自至少有明确生年或卒年。关系覆盖亲属与交往两张原始表，建模时不使用数百个过细的原始关系码。

- 官位无法可靠排序，因此所有可接受的任官记录暂统一记为 **“做官”**；不建立官位等级，也不要求 L1/L2/L3 职业层级。
- 非官职业需从 `STATUS_DATA` 审核后提取。每人最终只保留一个职业标签；妻、孝子、贞妇等身份/品行不能当职业。
- 生卒年只读 `BIOG_MAIN.c_birthyear`、`c_deathyear`；`0`/`NULL` 视为缺失，不用索引年或活动年代补值。公元前负年份保留。
- 亲属与交往原始细码保留供追溯。**当前实验关系口径为 12 个大组（全图加模型反向为 24 类型）；二分组为 inherited/acquired（加反向为 4 类型）。原有 31 组/75 角色类型保留为审计层**。时期现已固定为7组：700年前、700–899、900–1099、1100–1299、1300–1499、1500–1699、1700年及以后；按已有生卒年任一落入归组。职业映射仍为试验口径。

## 原始数据库是什么

SQLite 文件：`external_data/cbdb/2026.09.14/cbdb_20260914.sqlite3`。`BIOG_MAIN` 一人一行，共 **661,712 人**；`KIN_DATA` 一条亲属记录一行，共 **562,367 行**；`ASSOC_DATA` 一条交往记录一行，共 **190,044 行**。职业相关的 `POSTED_TO_OFFICE_DATA` 是任官记录，`STATUS_DATA` 是身份/特征记录，同一人均可出现多次。职务和状态的中文名称要分别连接 `OFFICE_CODES`、`STATUS_CODES`。

全库有明确出生年的 **59,855 人**、明确死亡年的 **71,298 人**、两者都有的 **37,457 人**、至少一个有的 **93,696 人**（年份非空且非 0）。其中 5 人出生晚于死亡，另有 3 个五位数出生年；试建表排除了这 8 人。

详细原始关系盘点见 `docs/cbdb_relation_audit_2026-09-29/README.md`；职业和年份盘点见 `docs/cbdb_occupation_audit_2026-09-29.md`。

## 职业提取的真正难点

| 来源 | 全库规模 | 能说明什么 | 不能直接说明什么 |
| --- | ---: | --- | --- |
| `POSTED_TO_OFFICE_DATA` | 591,369 行、299,130 人 | 曾有具体官职/官衔记录，如知縣、知府、教諭 | 官位高低；并非每条都是实际任事，含虚衔、爵位等 |
| `STATUS_DATA` | 73,469 行、56,511 人、270 种实际出现的码 | 身份、特征与部分职业，如诗人、画家、医师、商人 | 不能把整表或每人的第一条状态直接视为职业 |
| 官方 `STATUS_TYPES=01`“事業” | 23,481 行、22,916 人、26 种已出现的码 | 提供一批职业候选 | 漏掉被放在其他类别的诗人、画家、僧人；自身也有学生、有官衔等需审查的值 |
| `ENTRY_DATA` | 264,925 行、220,762 人 | 科举/入仕途径与资格 | 进士、举人、生员不是职业 |

`POSTED_TO_OFFICE_DATA.c_office_category_id` 不能用作职业大类：只有 **2.96%** 的任官行有非零值，类别是“寄祿官”“本官”“差遣”“爵”等官职性质。`OFFICE_TYPE_TREE` 顶层主要按朝代组织，也不是现成职业本体。

`STATUS_DATA` 的官方类别不能代替人工复核。例如“诗人”在 `[未詳]`、“画家”在“藝術”、“僧人”在“宗教”；“孝子/孝女”“贞妇/节妇”“妻”则不是职业。一个人还能有多个状态码。当前 `config/cbdb_occupation_whitelist_v1.json` 的 **92 个状态码、11 个宽泛标签只是试建白名单**，尚未逐码确认，不应称为最终职业真值。尤其“学术”“教育”“写作”边界，以及官衔和实际任职的界限，仍需检查。

## 已做的试建版本及限制

`scripts/cbdb_build_flat_triples.py` 从原始 SQLite 构建试验性的 `external_data/cbdb/2026.09.14/flat_triples_v1/person_relation_triples.tsv.gz`；规则和字段见 `docs/cbdb_flat_triples_v1.md`，统计见同目录 `summary.json`。运行：

```bash
python3 scripts/cbdb_build_flat_triples.py
```

试建版本要求两端各有至少一个可用生卒年和单层职业；带亲属中介或第三人物的交往行不强行写成直接二人关系；完全相同的有向、原始类型三元组合并，并用 `SupportRows` 保留来源行数。原始细码保留供核查，不意味着模型必须使用 630 种关系类型。

| 试建漏斗 | 数量 |
| --- | ---: |
| 有可用生年或卒年的人 | 93,688 |
| 同时被试建职业规则标记的人 | 38,789 |
| 去重后的有向三元组 | 92,168 |
| 三元组涉及的人物 | 22,392 |

**最大风险是类别失衡：**输出人物中 **21,504 / 22,392（96.0%）** 被标为“做官”，三元组中 **87,071 / 92,168（94.5%）** 是“做官—做官”。这张表验证了管线可以跑通，**尚不能直接作为多职业预测和泛化结论的数据集**。先前输出的数字都依赖上述试建白名单与严格的双端筛选，白名单变动后必须重新统计。

## 2026-10-06：当前主实验对齐的大组口径

用户明确要求按主实验设置少量多关系大类：亲属表整体归 inherited，直接伴侣单列，其余社会联系参考主实验并保留 CBDB 独特主题。此前 31 组/75 角色类型为细分审计，不再作为当前模型粒度。

执行 `python3 scripts/cbdb_build_experiment_groups.py`；配置 `config/cbdb_experiment_relation_groups_v1.json`。完整定义及接入命令见 [实验大组说明](docs/cbdb_experiment_relation_groups_v1.md)。

- multi_group 有 12 个基础组：inherited、intimate_partnership、education_mentorship、professional_collaboration、influence_succession、religious_relation、textual_relations、education_assessment、political_institutional、social_association、opposition_conflict、other_acquired。
- KIN 除本人直接伴侣全部进入 inherited，含姻亲、养继与嗣承；ASSOC 明确义亲/收养亲子也纳入。不把 inherited 等同全部生物血缘。伴侣进入 acquired。
- 多组表示 79,703 条原方向边；二分组 76,881 条（inherited 25,989、acquired 50,892）。模型关系名直接是大组，不再追加细角色后缀；遵循主实验预处理的 __rev 约定后分别为 24 / 4 个类型。
- 两种表示都保留 22,392 人和 98,557 条来源支持行。职业、年份不变，原码及细分组可追溯。
- 输出根目录 `external_data/cbdb/2026.09.14/experiment_groups_v1/{multi_group,binary}/`。各目录有数据表、主实验接口 CSV 及完整的 relation/tie taxonomy。真实单层职业只放 Level1 字段，Level2/3 和国家为空。
- 两种表示已通过全量覆盖、守恒、属性及 taxonomy 核对。后续分期与服务器流水线见下一节。

## 2026-10-06：七时期与服务器一条龙

运行入口 `bash CBDB/run_pipeline.sh run all --device cuda:0`，说明见 [CBDB/README.md](CBDB/README.md)，配置 `config/cbdb_pipeline_v1.json`。默认从清洗好的原始关系三元组开始，依次执行大组关系导出、分期、PyG准备、RGCN、GraphMask train/test report及汇总；默认7期×2种表示×seed42。`--from-sqlite` 可加入原库清洗步骤，SQLite保持只读。

分期脚本 `scripts/cbdb_split_periods.py`，配置 `config/cbdb_periods_v1.json`，默认独立输出 `external_data/cbdb/2026.09.14/periods_v1`。归组口径为已知 Birth/Death 任一落入闭区间；跨期人物/关系可重复。两端都属于该期才保留三元组；不插补年份。

| 时期ID | 归期人数 | 参与期内边人数 | multi_group三元组 | binary三元组 |
|---|---:|---:|---:|---:|
| to_699 | 2,345 | 2,031 | 3,762 | 3,736 |
| 700_899 | 8,263 | 7,807 | 16,038 | 15,867 |
| 900_1099 | 3,163 | 2,771 | 12,311 | 11,619 |
| 1100_1299 | 4,178 | 3,985 | 20,702 | 19,287 |
| 1300_1499 | 3,113 | 2,875 | 12,942 | 12,535 |
| 1500_1699 | 3,038 | 2,928 | 15,952 | 15,523 |
| 1700_plus | 1,489 | 1,423 | 3,704 | 3,576 |

各期保留 eligible/active 两份人物档。构图用 active 人物，职业为Level1单层标签，其它层级与国家为空；每期职业人数不足3的类别留图但不监督，70/10/20分层划分。两种表示的节点顺序、标签、特征和划分均逐项比较一致。全局有5,210条多组边两端无共同归期，被所有期内子图排除；10,918条多组边跨两期重复，累计85,411条期内三元组。每期taxonomy只包含实际出现的组，模型词表随该期支持变化。

新runner带只读plan、环境/采样预检、阶段完成记录、输入/输出mtime与大小检查、同设置续跑、作业失败汇总与非零退出码、输出目录锁。正式与CPU smoke默认目录分开，失败不会自动放宽GraphMask保真约束。

30项CBDB自动测试通过；7期×2表示已实际完成14/14条CPU小模型端到端链路，严格保真阈值0.05通过，同命令重跑14/14复用检查点。另验证过独立目录的1700+两表示链路；新runner从只读SQLite重建结果也已核对一致。正式50轮CUDA实验仍待服务器运行。

700年前active人物职业为做官2,027、写作2、宗教1、艺术1，按每类至少3人的门槛仅一个监督类，汇总标为 `single_class`；该期F1=1不能解释为多职业预测性能。其余6期监督类别数为4、7、9、8、6、7。

## 2026-10-07：Git 同步和 GPU 选择

运行方式改为Git同步后在仓库根目录执行。已移除压缩包方案；默认清洗输入复制到可由Git提交的 `CBDB/data/person_relation_triples.tsv.gz`（约1.8MB），不再依赖被忽略的 `external_data/` 默认路径。原库SQLite仍单独提供，可使用 `--from-sqlite --database PATH`。

开跑前先 `nvidia-smi` 查看卡占用；例如确认物理2号卡空闲后执行 `export CUDA_DEVICE_ORDER=PCI_BUS_ID`、`export CUDA_VISIBLE_DEVICES=2`，再执行 `bash CBDB/run_pipeline.sh --check-env --device cuda:0` 和 `bash CBDB/run_pipeline.sh run all --device cuda:0`。CUDA在进程内重编号，因此始终以逻辑 `cuda:0` 指向所选卡。runner继承这些环境变量，并在预检显示可见卡、逻辑device和名称。

## 2026-10-06：细分关系审计层（前一版）

运行 `python3 scripts/cbdb_group_relations.py`，执行配置为 `config/cbdb_relation_groups_v1.json`。详细定义、支持量、方向和审计见 [关系分组说明](docs/cbdb_relation_groups_v1.md)。

- 对原始码表全部 981 个正码显式映射；v1 图实际出现的 630 个码全部覆盖。
- 31 个语义组，结合角色方向后为 75 个实际模型关系类型。亲属区分亲子、祖裔、旁系、姻亲及养继等；配偶单列；交往区分师承、考校、学术影响、政治及不同文字关系。
- 输出 `external_data/cbdb/2026.09.14/grouped_relations_v1/person_relation_triples_annotated.tsv.gz`（92,168 条原始三元组，含分组标注）及 `person_relation_triples_grouped.tsv.gz`（87,126 条按同组同角色同人物对合并后的边）。
- 人物仍为 22,392 人，支持的原始数据库行仍为 98,557；原始码集合和合并数量均可追溯。职业和年份不变。
- 220 条原始三元组的中文/英文角色描述冲突，保留为角色未决并记录 qualifier；没有强选一方或交换端点。
- 原码表另有 3 个反向码分组/方向异常，当前 v1 均未出现，已记审计表。没有修改原库。
- 已通过全量人物、属性、原始三元组覆盖及支持数守恒检查，以及 7 项测试。

2026-10-03 职业探查见 `docs/cbdb_occupation_probe_2026-10-03/recommendation.md`。当前已完成关系分组、七期划分及可执行模型流水线；职业标签仍未逐码审核完成，CPU smoke只验证代码执行，正式GPU实验另行运行。

## 下一步交接任务

1. **逐码审核 `STATUS_DATA`（未完成）。** 270 个状态码及样例已在 `docs/cbdb_occupation_probe_2026-10-03/status_inventory.tsv` 导出，仍需逐个标记 `职业可用 / 非职业 / 有歧义`；职业可用者再映射到少量单层标签。重点复核现有 92 码白名单，并记录每项纳入/排除理由。不要按英文关键词或官方大类自动决定。
2. **复核“做官”的边界。** 抽查 `POSTED_TO_OFFICE_DATA` 中虚衔、爵位和其他非实际任事的记录，决定是否全部仍并入“做官”。不需要构建官位重要性排序。
3. **冻结职业映射后重新跑漏斗。** 分别报告全库、日期合格人物、关系端点合格人物、最终三元组中的各职业人数和边数；尤其检查“做官”比例是否仍使多职业任务不可行。多职业状态冲突须有明确规则，不能简单取第一条。
4. **实验大组 v1 已执行。** 当前模型使用 `experiment_groups_v1` 的 12 大组或 2 大组关系名，细角色只作审计。以对应表示目录的 taxonomy 进行训练/报告；职业规则变动后可从原始类型表复用映射重新导出。
5. **七期划分已执行。** 用 `CBDB/run_pipeline.sh` 在服务器运行完整实验；保留每期职业及关系支持、稀有类别处理和GraphMask保真失败记录。不要把跨期重复人物当作互斥样本。

所有统计与判断应从原始 SQLite 及明确的映射文件重算；不要用以前的关系图或其标签作分母。
