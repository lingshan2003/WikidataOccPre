# Freebase Easy 人物抽取 v2

当前职业多值问题、已确定口径与后续决策步骤见 [`HANDOFF_PROFESSION_SELECTION.md`](HANDOFF_PROFESSION_SELECTION.md)。

2026-10-03 已完成当前关系图 1,817 个职业值的并行智能体初审，候选映射和覆盖测量见 [`职业 L1 初审报告`](../docs/freebase_profession_review_2026-10-03/review_report.md)。这些是未人工批准的草案，不是冻结训练标签。`review_professions.py` 可复现词表盘点和人物层面的映射审计；原始抽取文件未修改。

这一版按用户确认的口径重做：`is-a Person` 作为暂定人物名单；一个人物只要有 **至少一个不同的原始 `Profession` 值**，且出生年或死亡年有至少一个结构上可解析的年份，就进入资料完整队列。**所有职业值均保留为 JSON 数组**，不选主职业、不做 L1/L2/L3，也不因多职业删除人物或边。结构上可解析不代表日期或人物身份已经核实。

## 输入和输出

默认输入：

- `external_data/freebase_easy/freebase-easy-latest/facts.txt`
- `external_data/freebase_easy/freebase-easy-latest/freebase-links.txt`
- `Freebase/relation_rules.json`：先前初审的 20 个关系谓词，其中 11 个同时标为主图候选。规则配置仍是探索版，不是训练图的最终白名单。

默认输出根目录：`external_data/freebase/processed/`。脚本依次完成：

| 阶段 | 关键输出 | 含义 |
| --- | --- | --- |
| `01_people` | `person_names.txt` | 所有 `is-a Person` 的不同名称 |
| `02_cohort` | `cohort_people.tsv`, `person_attribute_facts.tsv` | `Profession` 至少一个且有出生/死亡可解析年份的人物；原始属性与全部职业都保留 |
| `03_pairs` | `person_name_pairs.tsv`, `cohort_name_pairs.tsv`, `predicate_coverage.tsv` | 全体候选人物对的所有原始谓词，以及两端都在资料完整队列的子集 |
| `04_relations` | `review_candidate_facts.tsv`, `main_relation_facts.tsv` | 20 个初审谓词和 11 个主图候选谓词的原始三元组 |
| `05_final` | `nodes.csv`, `main_relation_facts.csv`, `name_id_status.tsv` | 可下载的节点/关系表，职业为 JSON 数组；附 Freebase 链接 ID 状态 |
| `06_stats` | `summary.json`, `relation_counts.tsv`, `occupation_people_counts.tsv` | 漏斗、职业数、逐关系数量、互逆/自环、按名称构成的连通分量等描述统计 |

所有关系事实保留原始主客体方向与 `source_line`；`Sibling`、伴侣等双向事实在原始表中仍是两行。思想影响谓词也尚未规范方向。`freebase-links.txt` 的 ID 是辅助身份线索；缺失或歧义不导致人物被删除。名称碰撞、虚构人物和伪人物对仍需后续审计。

本次完整重跑得到 3,970,878 个 `Person` 名称、1,728,197 条全部人物名称对、148,445 条两端满足职业及日期条件的名称对、147,454 条 20 谓词初审记录、145,876 条 11 谓词主图候选原始记录。旧版第一阶段曾报告 3,970,877 个名称，差 1 个；以本次同批次输出为准。脚本不把这些数字写死；若结果不同，先对照各阶段 `summary.json` 和输入文件。

## 服务器运行

先同步 `Freebase/` 目录。在独立的外部终端登录服务器、进入项目根目录，然后建立 tmux 会话：

```bash
cd /mnt/network_data/personal_workspace/siruilai/wy/WikidataOccPre
tmux new -s freebase-extract
```

**进入 tmux 后**，在 tmux 的提示符运行：

```bash
conda activate wywikidata
python -u Freebase/extract.py --stage all 2>&1 | tee external_data/freebase_extract.log
```

上述脚本只用 CPU，不调用 PyTorch/GPU。默认完整扫描 `facts.txt` 三次，外加一次 `freebase-links.txt`，日志每 500 万行打印进度。tmux 暂离：按 `Ctrl-b`，再按 `d`；重连：`tmux attach -t freebase-extract`。运行完成时可看到 `06_stats` 的 `complete_descriptive_scan`。

如已完成部分阶段而需要继续：

```bash
python -u Freebase/extract.py --stage all --resume 2>&1 | tee -a external_data/freebase_extract.log
```

`--resume` 只跳过输入元信息相符且文件齐全的完成阶段；失败阶段的残留目录不会被自动覆盖。改动输入或规则后请使用新的 `--output-dir`，保留旧结果供对照。

完成后请先回传下面命令的输出，随后再决定下载哪些大文件：

```bash
p=external_data/freebase/processed
cat "$p/02_cohort/summary.json"
cat "$p/03_pairs/summary.json"
cat "$p/04_relations/summary.json"
cat "$p/05_final/summary.json"
cat "$p/06_stats/summary.json"
head -n 6 "$p/05_final/nodes.csv"
head -n 6 "$p/05_final/main_relation_facts.csv"
```

若只想分析当前 11 谓词描述子集，`05_final/nodes.csv` 和 `05_final/main_relation_facts.csv` 是最小组合；其余各阶段保留完整审计路径。

## 本地描述性统计

下载包含 `02_cohort/cohort_people.tsv`、`03_pairs/cohort_name_pairs.tsv`、`04_relations/review_candidate_facts.tsv`、`05_final/nodes.csv`、`05_final/main_relation_facts.csv` 及阶段摘要的压缩包后，可解到 `external_data/freebase/descriptive_v2_local/`，运行：

```bash
python Freebase/describe_local.py
```

脚本逐项核对阶段数量，生成 `docs/freebase_easy_audit_2026-10-01/descriptive_report.md`。它统计关系频次、去自环与规范去重后的关系组，以及按出生年分组的人物数和同出生时期原始事实数。时期统计不推断关系发生年份。
