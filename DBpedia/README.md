# DBpedia 人物社会关系与职业候选图抽取

`DBpedia/` 保存**可提交到仓库、可在服务器运行的 Python 脚本和规则**。三个原始 `.ttl.bz2` 文件放在 `external_data/dbpedia/2022.12.01/`；所有生成数据写到 `external_data/dbpedia/processed/`。`processed` 是常用的“已处理数据”目录名，这里只代表完成了本流程的规则处理，**不代表人物身份或职业标签已人工验证**。整个 `external_data/` 已由仓库 `.gitignore` 忽略。

## 运行

要求 Python 3.10+，只使用标准库。从仓库根目录执行：

```bash
python3 DBpedia/run.py \
  --raw-dir external_data/dbpedia/2022.12.01 \
  --output-dir external_data/dbpedia/processed
```

脚本检查以下三个文件存在：

```text
instance-types_lang=en_transitive.ttl.bz2
mappingbased-objects_lang=en.ttl.bz2
mappingbased-literals_lang=en.ttl.bz2
```

若前两步已成功，可以从日期步继续：

```bash
python3 DBpedia/run.py --from-stage dates
```

也可用 `--through-stage relations` 停在指定步骤。后续步骤需要前序步骤的完整输出，脚本会检查这些文件。重新执行某步会覆盖该步输出；不要并发执行两次。运行完成后，`processed/pipeline_summary.json` 汇总每一步人数、边数、排除原因和原始文件大小；不做 SHA256 校验。

## 四个阶段

| 步骤 | 输入 | 主要规则 | 输出目录 |
| --- | --- | --- | --- |
| `people` | transitive instance types | `dbo:Person`、`schema:Person`、`foaf:Person`、`wikidata:Q5` 类型并集；只标记可疑 URI | `processed/01_people/` |
| `relations` | mapping-based objects + 人物候选 | 保留两端为候选的三元组；用 `rules/relation_policy.tsv` 排除 `currentMember`、音乐关联等，选择社会关系；去自环与完全重复 s-p-o | `processed/02_relations/` |
| `dates` | mapping-based literals + 社会关系 | 从 `birthDate`、`birthYear`、`deathDate`、`deathYear` 抽年；一侧冲突时该侧不合格，另一侧仍可用；边两端都至少有一个唯一的生卒年 | `processed/03_dates/` |
| `occupations` | types、objects + 日期图 | `dbo:occupation`、`dbo:profession` 资源值经过硬排除；20 个职业型 `dbo:` 类；`rules/q_class_review.tsv` 中 `accept` 的 36 个 Wikidata Q 类；边两端都要有职业候选 | `processed/04_graph/` |

当前已审查的 2022.12.01 文件应产生的关键规模：人物候选 **1,860,208**，原始候选间对象三元组 **706,833**，清理后的社会边 **238,908**，双方有年边 **136,503**，最终职业候选图 **55,690 个端点、67,090 条边**。这些是对照基准；同名但内容不同的原始文件可能给出不同数字。关系谓词出现规则表未审查的新项时默认停止并输出 `unreviewed_predicates.tsv`，避免悄悄改变关系口径。若只是想跳过新谓词并在报告中保留计数，可显式传 `--allow-unreviewed-predicates`。

## 最终数据与证据

- `processed/04_graph/graph_nodes.tsv.gz`：图端点，出生年、死亡年、日期来源/冲突、首选职业证据来源、全部来源、具体 QID、复合 URI 标志。
- `processed/04_graph/graph_edges.tsv.gz`：保留原始方向、谓词、源行号、关系分组和直接/上下文口径的有向边。
- `processed/04_graph/candidate_evidence.tsv.gz`：职业证据、原始值、来源文件和行号；包含未进入最终有边图的日期合格候选。
- `processed/03_dates/date_evidence.tsv.gz`：原始生卒日期字面量、解析出的年份、质量标志和源行号。
- `processed/03_dates/all_participant_years.tsv.gz`：所有社会关系端点的年字段与冲突标志，包括因无可用年份而没有进入日期图的端点。
- `processed/04_graph/direct_value_policy.tsv`、`rejected_direct_evidence.tsv.gz`：职业资源值的硬排除判定与被拒绝的原始事实。
- 各阶段 `summary.json`、`predicate_flow.tsv`、`relation_flow.tsv`、`group_flow.tsv`：保留每一步影响了多少数据。

## 与分时期实验的边界

已有实验的当前时期配置是 [`config/historical_life_periods_v2.json`](../config/historical_life_periods_v2.json)：≤500、501–1500、1501–1800、≥1801，已知完整生卒区间可与多个时期相交。**本流程不预先给人物分时期，也不构造时期子图。**它保留有符号的 `birth_year`、`death_year`、缺失、来源和冲突信息，供后续实验按该配置计算成员资格。只有一个年份的人不会在抽取时被补出未知寿命。

## 规则文件

- `rules/relation_policy.tsv`：冻结本轮审查的 97 个关系谓词决定（含 `currentMember`）。
- `rules/q_class_review.tsv`：54 个具体 Q 类的逐类判定，其中 36 个 `accept` 进入图。
- `occupations.py` 中的 20 个职业型 `dbo:` 类和直接职业资源值硬排除规则，与本轮审查保持一致；学科/活动类型只做待审标记，不靠关键词硬删。

这是**抽取流程**，不包括三级职业本体映射、按时期构图、R-GCN 特征处理或 GraphMask 实验接入。
