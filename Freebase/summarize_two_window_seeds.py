#!/usr/bin/env python3
"""Combine completed repeat bundles; one Run 1 pair per independent seed."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, nargs="+", required=True, help="Completed results.json files")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    merged = {}
    for path in args.results:
        for row in json.loads(path.read_text()):
            if row["status"] != "complete":
                raise ValueError(f"Incomplete report: {path}/{row}")
            key = row["seed"], row["run"], row["year"]
            if key in merged and merged[key]["share_pct"] != row["share_pct"]:
                raise ValueError(f"Conflicting baseline: {key}")
            merged[key] = row
    selected = []
    for seed in sorted({r["seed"] for r in merged.values()}):
        a, b = [merged[(seed, 1, year)] for year in (1940, 1942)]
        selected.append(dict(seed=seed, share_1940_pct=a["share_pct"], share_1942_pct=b["share_pct"],
                             gap_percentage_points=b["share_pct"]-a["share_pct"],
                             layer0_enabled_1940=a["layer0_enabled"], layer0_enabled_1942=b["layer0_enabled"],
                             original_f1_1940=a["original_macro_f1"], original_f1_1942=b["original_macro_f1"],
                             masked_f1_1940=a["masked_macro_f1"], masked_f1_1942=b["masked_macro_f1"]))
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("all_runs", list(merged.values())), ("seed_comparison", selected)):
        (out/(name+".json")).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (out/(name+".tsv")).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=columns,delimiter="\t"); w.writeheader(); w.writerows(rows)
    stats = dict(n_seeds=len(selected), positive_gap_seeds=sum(r["gap_percentage_points"]>0 for r in selected),
                 negative_gap_seeds=sum(r["gap_percentage_points"]<0 for r in selected))
    for year in (1940,1942):
        values=[r[f"share_{year}_pct"] for r in selected]
        stats[str(year)]=dict(min_pct=min(values),max_pct=max(values),mean_pct=statistics.mean(values),
                              sample_sd_percentage_points=statistics.stdev(values) if len(values)>1 else None)
    (out/"descriptive_statistics.json").write_text(json.dumps(stats,indent=2)+"\n")
    lines=["每个seed取Run 1；seed=42额外两次重跑保留在all_runs.tsv，不重复计入跨seed统计。单位为Layer 0 Intimate partnership占全部保留消息的百分比。", "",
           "| seed | 1940 | 1942 | 1942−1940（百分点） |", "|---|---:|---:|---:|"]
    for row in selected:
        lines.append(f"| {row['seed']} | {row['share_1940_pct']:.2f}% | {row['share_1942_pct']:.2f}% | {row['gap_percentage_points']:+.2f} |")
    lines += ["",f"1942高于1940：{stats['positive_gap_seeds']}/{stats['n_seeds']}个seed；低于：{stats['negative_gap_seeds']}/{stats['n_seeds']}个seed。", "",
              "数据划分、窗口图和训练预算固定；seed同时控制RGCN、GraphMask和测试邻居采样。统计为描述性结果，不使用bootstrap或平滑。"]
    disabled=[(r["seed"],year) for r in selected for year in (1940,1942) if not r[f"layer0_enabled_{year}"]]
    lines += ["", "所选checkpoint的Layer 0均已启用。" if not disabled else f"Layer 0未启用的点（仍按any-stage规则完整保留）：{disabled}。"]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    (out/"provenance.json").write_text(json.dumps(dict(inputs=[str(x.resolve()) for x in args.results],
        seed_level_policy="One Run1 per seed; no duplicate weighting of seed42 repeats", script=str(Path(__file__).resolve())),indent=2)+"\n")
    print((out/"results.md").read_text())


if __name__=="__main__":
    main()
