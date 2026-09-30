# Freebase Easy 严格人物子图：扩展三元组导出 v1

> 历史记录：这版把职业条件误设为“恰好一个”，排除了大量多职业人物。现已由[Freebase Easy 人物抽取 v2](/Users/wangyue/RGCN/Freebase/README.md)取代；请勿再用本页的 v1 命令生成最终数据。

目标是从服务器已存在的 `person_cohort_profession_dated_v1/cohort_name_pairs.tsv` 构建一份与主数据 `Q_R_Q_extended.txt` **同样以关系边为行**的 CSV。它是独立的 Freebase 数据版本，不覆盖主数据；每人使用一个原始职业值，不构建职业 L1/L2/L3。规则写在 `config/freebase_easy_relation_export_v1.json`，执行脚本为 `scripts/freebase_easy_build_extended_v1.py`。

## v1 关系范围

- 亲属/伴侣：`Children`、`Sibling`、`Spouse (or domestic partner)`。
- 教育/师承：`Academic advisor`、`Martial Art Instructor(s)`。保留 Freebase Easy 的原始方向，规范名带 `_raw`，不宣称已彻底核实学生与教师方向。
- 社交：`Peer`、`Celebrity friend`、`Romantic relationship (with celebrities)`。各自保留关系类型，后两项有来源选择偏差。
- 思想/艺术影响：`Influenced By` 及两条哲学影响谓词。统一为有向 `influenced_by`，其中 `kp_lw/philosophy_influencer/influencee` 反向；同一规范边合并来源。这一组单列 `influence`，不解释为直接相识。

共 11 个原始谓词，过滤后的 148,445 条原始人物对中约有 145,876 条命中；**命中数不是最终边数**。其余 49 个谓词不写入此版本，包括明显属性/作品/地点碰撞，以及目前定义不稳的合作、任职、比赛等关系。所有原始文件保持不变，以便以后加入新关系或对照过滤造成的覆盖损失。

## 清洗与保留痕迹

1. 两端必须各有**恰好一个不同的原始 `Profession` 值**及至少一个可解析出生或死亡年，并且各自最多只有一个不同出生年、一个不同死亡年。多职业人物暂不指定“主职业”；其相关记录写入 `row_audit.tsv`，汇总数量写入 `summary.json`。这会缩小图，不能把剩下的职业分布解释为全体人物的职业分布。
2. 两端须从 `name_mid_status.tsv` 找到各自唯一的链接 ID；缺 ID、两端是同一 ID、或一个 ID 对应多个已选名称的记录不进 v1 图，原因保存在 `row_audit.tsv`。唯一链接 ID 仍只是身份线索，不能完全消除 Easy 名称扁平化的风险。
3. `Sibling`、伴侣、`Peer`、名人朋友和名人恋爱视为无向类型，正反记录合为一条规范边；原始 `source_line` 列表和支持条数保留。`Children`、师承和影响保留方向。
4. 同一人物对若同时有互逆 `Children`，或与 `Sibling`/伴侣类型冲突，相关亲属边都暂不写入 v1 图。互逆 `Academic advisor` 也暂缓。不同性质的非冲突关系可并存，例如导师与思想影响、恋爱与伴侣；各自保留关系类型。
5. 职业列就是唯一的原始 `Profession` 字符串，不做职业映射或 L1/L2/L3。生卒矛盾及超过 125 岁在节点表中标记 `DateWarnings`，不自动删掉全部记录。

## 输出

- `Q_R_Q_extended_freebase_v1.csv`：每行一条去重后的带类型边。`Node1`、`Node2` 是链接文件中的 Freebase ID；`Relation` 是规范关系名；`RelationGroup` 分为 `kinship`、`education`、`social`、`influence`。两端的姓名、生年、卒年、单个原始职业值随边重复写入，类似主数据的边表。
- `nodes_v1.csv`：每个实际入图节点一行，`Occupation` 为该人物唯一的原始 `Profession` 值。
- `row_audit.tsv`：每条命中 11 个原始谓词的记录及其纳入/排除原因；不含未选的 49 个谓词，未选行数在 `summary.json` 报告。
- `summary.json`：输入路径、原始/选中记录数、各排除原因、最终边数/节点数和逐关系、逐关系组计数。

此文件格式**不是现有 `data/extended.py` 的即插即用输入**：旧加载器要求三层职业和国家字段，而本版只有单个原始职业值。后续构图/训练应写专门的 Freebase 适配器，并制定防泄漏协议。

## 服务器运行

在仓库根目录、`wywikidata` Conda 环境中，将新配置和脚本同步到服务器后执行：

```bash
python scripts/freebase_easy_build_extended_v1.py \
  --pairs external_data/freebase_easy/person_cohort_profession_dated_v1/cohort_name_pairs.tsv \
  --attributes external_data/freebase_easy/person_attributes_v1/person_attributes.tsv \
  --name-mids external_data/freebase_easy/name_mid_audit_v3/name_mid_status.tsv \
  --config config/freebase_easy_relation_export_v1.json \
  --output-dir external_data/freebase_easy/extended_graph_v1
```

脚本只读已导出的子集和属性/ID 表，使用 CPU。新目录必须为空，避免覆盖先前结果。完成后先看 `summary.json` 和 CSV 前几行，再决定是否作为训练数据使用。

## 首次服务器运行结果（2026-09-30）

服务器回传 `status=complete_export`，读取 148,445 条严格子集姓名对，其中 145,876 条命中 v1 的 11 个原始谓词。`row_decisions` 中 121,890 条因至少一端有多个不同 `Profession` 值而未入图，占命中记录的 **83.56%**；另有 1,498 条 ID 未解析、47 条自连、16 条亲属冲突。22,425 条原始记录纳入去重，得到 **14,451 条规范边、20,059 个节点**。这不是原始库人物或关系的总体规模。

| 关系组 | 去重边数 | 占全部边 |
| --- | ---: | ---: |
| 亲属/伴侣 | 12,712 | 87.97% |
| 教育/师承 | 820 | 5.67% |
| 影响 | 857 | 5.93% |
| 社交 | 62 | 0.43% |

具体关系：`child` 4,815、`partner` 4,447、`sibling` 3,450、`influenced_by` 857、`academic_advisor_raw` 820、`peer` 29、`celebrity_friend` 20、`celebrity_romantic_relationship` 13。`martial_arts_instructor_raw` 本次没有入图边。样例显示 `Francesco Severi → Corrado Segre` 等导师事实及两端单职业 `Mathematician`；这只验证表结构与表面合理性，尚未完成系统的语义抽样审查。

结论：导出流程可用，但“恰好一个原始职业”的严格口径造成很大的选择损失，且输出图仍高度偏向亲属关系。下一步宜先审查每类边的样例、单职业人物的职业分布以及图的连通性，再决定是否用明确的职业映射规则纳入多职业人物。不要把这 14,451 条边直接当作对 Freebase Easy 人际关系的完整覆盖。
