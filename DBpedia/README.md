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

已有 Wikidata 实验的时期配置是 [`config/historical_life_periods_v2.json`](../config/historical_life_periods_v2.json)：≤500、501–1500、1501–1800、≥1801。DBpedia 的新实验采用 [`config/dbpedia_life_periods_20y_v1.json`](../config/dbpedia_life_periods_20y_v1.json)：≤1500、1501–1900、1901 年起每 20 年一段，末段为 ≥2001。两套配置都沿用已知完整生卒区间与时期相交、只有单侧年份则仅归入该端点所在时期的规则。**原始抽取流程不预先给人物分时期，也不构造时期子图。**它保留有符号的 `birth_year`、`death_year`、缺失、来源和冲突信息，供后续实验计算成员资格。只有一个年份的人不会在抽取时被补出未知寿命。

若只想预览“职业证据候选人物”按现有生命时期规则构图的规模，可在完成抽取后运行：

```bash
python3 DBpedia/report_period_occupation_coverage.py \
  --processed-dir external_data/dbpedia/processed \
  --period-config config/historical_life_periods_v2.json
```

报告写入 `processed/period_occupation_preview/`，分别给出入期候选人数、同一时期内有边的端点人数、孤点、边数、谓词数、职业证据首选来源及一人多值情况。首选来源按 `occupation` → `profession` → `dbo_role_type` → `wikidata_role_type`，这只是**职业线索来源的优先级**；多个职业值或不同来源间的语义冲突仍需在模型标签规则中处理。报告不会生成训练用时期图。

## 规则文件

- `rules/relation_policy.tsv`：冻结本轮审查的 97 个关系谓词决定（含 `currentMember`）。
- `rules/q_class_review.tsv`：54 个具体 Q 类的逐类判定，其中 36 个 `accept` 进入图。
- `occupations.py` 中的 20 个职业型 `dbo:` 类和直接职业资源值硬排除规则，与本轮审查保持一致；学科/活动类型只做待审标记，不靠关键词硬删。

## 导出与现有模型兼容的边表

本轮先按 `occupation` → `profession` → `dbo_role_type` → `wikidata_role_type` 为最终有边图的每个人选一个职业。同一渠道多个不同值时，取原始来源文件中最先出现的事实；审计表保留全部候选值。这是确定性的试跑规则，不宣称所选值比同渠道的其他值更准确。

```bash
python3 DBpedia/export_q_r_q_extended.py \
  --processed-dir external_data/dbpedia/processed

python run.py prepare \
  --input external_data/dbpedia/processed/model_input/DBpedia_R_R_extended.csv \
  --output-dir artifacts/dbpedia_2022_priority_v1 \
  --target-level 1 --min-class-count 20 --seed 42

python scripts/prepare_life_period_induced_artifacts.py \
  --source-data artifacts/dbpedia_2022_priority_v1/graph_data.pt \
  --output-root artifacts/dbpedia_2022_priority_periods_20y_v1 \
  --life-periods config/dbpedia_life_periods_20y_v1.json \
  --split-seed 20260814
```

导出的 CSV 使用原 `Q_R_Q_extended.txt` 的 15 列格式。**`occupation_level1` 在这里仅是兼容现有单标签预处理的字段，内容为规范化的 DBpedia 职业名，不是 Wikidata 的一级本体。**`occupation_level2`、`occupation_level3` 和 `country` 均为空；训练时使用 `--occupation-feature-levels 1 --auxiliary-features temporal`。关系保留原 33 种 `dbo:` 谓词的局部名称，现有预处理会加入反向边。

`processed/model_input/occupation_assignments.tsv.gz` 记录每人的来源、所选原始值、源行号与全部同渠道备选值；`summary.json` 和 `occupation_label_counts.tsv` 记录规模与类别频数。`prepare` 默认会把少于 `min-class-count` 的职业保留在图中但设为 `y=-1`，不计入监督训练；时期脚本会在每期对支持量再次过滤。模型产物保存在仓库原有的、被 Git 忽略的 `artifacts/` 下。

## 一条命令构图、训练和解释

在装有 CUDA 版 PyTorch、PyTorch Geometric 及本仓库其他依赖的服务器环境中，从仓库根目录运行：

```bash
export DBPEDIA_PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
bash DBpedia/run_rgcn_graphmask_20y.sh plan all
bash DBpedia/run_rgcn_graphmask_20y.sh run all
```

`plan` 仅打印即将执行的命令和现有产物状态；`run all` 从三份原始压缩文件开始，已有完整的抽取、导出或构图产物会复用，然后依次训练全图和 8 个时期的 R-GCN，训练 GraphMask，生成测试集关系解释及汇总。运行前会检查所选 Python 的 CUDA 可用性、实际 GPU 和 PyG 的 `NeighborLoader`。默认一次只执行一个 GPU 任务。可用 `run data`、`run train`、`run graphmask` 分段运行；同一命令重复执行会跳过完整产物。若原始文件不在默认位置，可设置 `DBPEDIA_RAW_DIR`，已有抽取结果不需要再次提供原始文件。

上传到服务器时需要同步本目录的新脚本、两个 `config/dbpedia_*.json` 配置，以及 `training/train.py` 的时期缺席类别处理；只复制 Bash 文件不足以运行整套流程。

主要结果在 `runs_graphmask/dbpedia_2022_priority_20y_v1/`：每个 `full` 或时期目录的 `seed_42/test_report/` 包含 `test_metrics.json`、`relations_base.csv`、`relations_directed.csv` 和 `root_top_edges.csv.gz`；根目录的 `graphmask_matrix_summary.tsv` 汇集九张图的规模、测试指标和解释保留率。R-GCN 检查点和训练日志保存在 `runs/dbpedia_2022_priority_20y_v1/`。`artifacts/`、`runs/`、`runs_graphmask/` 和 `external_data/` 均被 Git 忽略。

显存不足时可以在**新实验输出目录**下设置 `DBPEDIA_TRAIN_BATCH_SIZE`、`DBPEDIA_GRAPHMASK_BATCH_SIZE` 或 `DBPEDIA_TRAIN_FANOUTS`。GraphMask 默认使用 `auto`，复用相应 R-GCN 检查点的邻居采样配置。已有检查点只按产物完整性跳过，不会因为环境变量改变而自动重训；如果改变实验设置，须同时设置新的 `DBPEDIA_MODEL_ROOT` 和 `DBPEDIA_GRAPHMASK_ROOT`。GraphMask 若找不到满足默认验证保真度阈值的 probe，会保留 `training_history.json` 并停止；检查后可以用 `DBPEDIA_GRAPHMASK_MAX_RELATIVE_F1_DIFF` 显式调整阈值。
