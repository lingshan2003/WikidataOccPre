# DBpedia BHHT L1 第一版数据

用户已授权将上一轮仍需语境的445种职业暂时放入Other；逐职业审核状态和理由保持原样。原图55,690个节点、67,090条原方向事实、33种关系保持不变。

| L1 | 节点数 |
|---|---:|
| Leadership | 27,312 |
| Discovery/Science | 14,230 |
| Culture | 9,055 |
| Sports/Games | 3,816 |
| Other | 1,277 |

Other包含原有查表/LLM建议680个节点，以及临时兜底597个节点（全图1.07%）。`temporary_other_fallback=1`标记后者，原语义L1仍空、原审核状态仍为`needs_context`，不把临时填充当作确定的职业事实。

```bash
python DBpedia/prepare_bhht_l1_v1.py
```

默认生成 `external_data/dbpedia/bhht_l1_provisional_v1/` 和同级 `bhht_l1_provisional_v1.tar.gz`，脚本只依赖Python标准库。已有目录或同名压缩包会被保护；重建应指定新的`--output-dir`。

- `nodes.csv`：逐人L1数据，节点顺序、生卒年、国家与原准备节点表一致。
- `relations.csv`：原方向事实，67,090行；不重复生成反向边。
- `DBpedia_R_R_extended.csv`：既有15列CSV接口，替换两端L1为五大类；其余非职业字段逐行核对一致。
- `occupation_crosswalk.tsv` / `occupation_crosswalk_compact.csv`：2,417种原职业到第一版L1的映射，保留原审查信息并新增训练标签/临时Other标记。
- `node_label_audit.tsv.gz`：将映射、源URI、原唯一职业选择和临时标记落实到55,690个节点；保留同来源多职业审计信息。
- `level1_counts.tsv`：区分已有候选/LLM建议与临时Other。
- `manifest.json`：输入路径、大小、mtime、版本、图规模和保护范围；`generator_snapshot.py`保存生成器代码。

L2/L3在数据输入中暂空；不将上轮尚未裁决的关键词当作已确定细职业。原始死亡日期不填2026：出生>1920且缺死亡日期延至2026，仍属于下一步滑窗构图的有效生命区间规则。

这一步只生成独立数据，没有构造新的split、PyG `graph_data.pt`或启动训练。旧DBpedia时期图和旧职业模型保持原版本；换成五类职业标签后需另建图/训练目录，不能直接套用旧模型权重。
