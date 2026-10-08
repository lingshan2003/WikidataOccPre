#!/usr/bin/env python3
"""Validate, plot and summarize the five-period Layer-0-enabled supplement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.compare_layer0_checkpoint_selection import read_json, write_table


def markdown(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received-root", type=Path, default=ROOT / "artifacts/dbpedia_layer0_enabled_received_2026_10_07")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "visualization/dbpedia_grouped_layer0_enabled_2026_10_07")
    args = parser.parse_args()
    root, output = args.received_root.resolve(), args.output_dir.resolve()
    config_path = root / "config/dbpedia_multi_group_layer0_enabled_5periods_v1.json"
    config = read_json(config_path)
    original_root = root / "runs_graphmask/dbpedia_grouped_20y_v1"
    supplement_root = root / config["graphmask_root"]
    subprocess.run([
        sys.executable, str(ROOT / "DBpedia/plot_layer0_relation_groups.py"),
        "--received-root", str(root), "--layer0-supplement-root", str(supplement_root),
        "--output-dir", str(output), "--seed", str(config["seed"]),
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(ROOT / "DBpedia/compare_layer0_checkpoint_selection.py"),
        "--config", str(config_path), "--original-root", str(original_root),
        "--supplement-root", str(supplement_root), "--output-dir", str(output),
    ], cwd=ROOT, check=True)
    comparison = read_json(output / "checkpoint_selection_comparison.json")
    with (output / "layer0_period_audit.tsv").open(encoding="utf-8", newline="") as handle:
        audit = {r["period_id"]: r for r in csv.DictReader(handle, delimiter="\t") if r["representation"] == "multi_group"}
    periods = read_json(root / config["period_config"])["periods"]
    rows, report_sources = [], []
    for p in periods:
        a = audit[p["id"]]
        report_dir = Path(a["report_dir"])
        metrics = read_json(report_dir / "test_metrics.json")
        layers = {r["layer"]: r for r in metrics["layers"]}
        rows.append({
            "context": p["id"], "period_label": a["period_label"], "result_source": a["result_source"],
            "selected_epoch": int(a["selected_epoch"]), "layer0_gate_enabled": a["layer0_gate_enabled"] == "True",
            "layer0_hard_retention_rate": layers[0]["hard_retention_rate"],
            "layer1_hard_retention_rate": layers[1]["hard_retention_rate"],
            "hard_retention_rate": metrics["hard_retention_rate"],
            "prediction_agreement": metrics["prediction_agreement"],
            "masked_accuracy": metrics["masked"]["accuracy"], "masked_macro_f1": metrics["masked"]["macro_f1"],
            "original_accuracy": metrics["original"]["accuracy"], "original_macro_f1": metrics["original"]["macro_f1"],
        })
        report_sources.append(str(report_dir / "test_metrics.json"))
    selected = [r for r in rows if r["result_source"] == "layer0_enabled_supplement"]
    if len(selected) != 5 or not all(r["layer0_gate_enabled"] for r in rows):
        raise ValueError("Expected five replacements and Layer 0 gates enabled in all eight plotted periods")
    write_table(output / "new_5period_graphmask_metrics.tsv", selected)
    write_table(output / "updated_8period_multi_group_graphmask_metrics.tsv", rows)
    deltas = [abs(r["supplement_retained_edge_share"] - r["original_retained_edge_share"])
              for r in comparison["layer0_groups"]
              if r["supplement_retained_edge_share"] is not None and r["original_retained_edge_share"] is not None]
    max_share_delta = max(deltas)
    (output / "metrics_provenance.json").write_text(json.dumps({
        "received_root": str(root), "config": str(config_path), "seed": config["seed"],
        "stored_units": "proportions_0_to_1", "display_units": "percent_2_decimals",
        "f1_metric": "macro_f1", "all_eight_layer0_gates_enabled": True,
        "maximum_new_old_layer0_share_absolute_difference": max_share_delta,
        "report_sources": report_sources, "five_new_results": selected, "eight_period_results": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    percent = lambda v: f"{100 * v:.2f}"
    metrics_table = markdown(
        ["时期", "Layer 0 保留率", "Layer 1 保留率", "整体保留率", "预测一致性", "删边后准确率", "删边后 Macro-F1"],
        [[r["period_label"], *(percent(r[k]) for k in (
            "layer0_hard_retention_rate", "layer1_hard_retention_rate", "hard_retention_rate",
            "prediction_agreement", "masked_accuracy", "masked_macro_f1"))] for r in selected],
    )
    labels = {r["context"]: r["period_label"] for r in rows}
    comparison_table = markdown(
        ["时期", "整体保留率：原→新", "预测一致性：原→新", "准确率：原→新", "Macro-F1：原→新"],
        [[labels[r["context"]], *(f"{percent(r['original_' + k])} → {percent(r['supplement_' + k])}"
                                  for k in ("test_hard_retention_rate", "prediction_agreement", "test_accuracy", "test_macro_f1"))]
         for r in comparison["periods"]],
    )
    all_keep = all(r["layer0_hard_retention_rate"] == 1.0 for r in selected)
    conclusion = "五个新 checkpoint 均启用两层门控并满足原验证保真度约束。"
    conclusion += " 五个时期在测试采样图上的 Layer 0 硬门控仍全部保留消息。" if all_keep else " 部分时期的 Layer 0 硬门控产生删减。"
    conclusion += f" 原、新 Layer 0 关系组保留消息占比的最大绝对差为 {100 * max_share_delta:.8f} 个百分点。"
    command = (
        ".venv/bin/python DBpedia/visualize_layer0_supplement.py \\\n"
        f"  --received-root {root.relative_to(ROOT) if root.is_relative_to(ROOT) else root} \\\n"
        f"  --output-dir {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}"
    )
    notes = [
        "# DBpedia Layer 0 门控启用后补充结果",
        conclusion,
        "## 五个时期的新结果\n\n单位：%。整体保留率按两层消息观测合计，F1 为 Macro-F1。\n\n" + metrics_table,
        "## 原结果与新结果\n\n单位：%。\n\n" + comparison_table,
        "## 八时期 Layer 0 热图\n\n五个时期替换为本次新报告，1901–1920、1921–1940、1961–1980 沿用原报告。当前八个时期的 Layer 0 门控均已启用。",
        "![Layer 0](dbpedia_multi_group_layer0_retained_message_share.png)",
        "单元格表示该组保留消息占 Layer 0 所有保留消息的比例，每行合计 100%。它与组内消息保留率含义不同。星号表示该时期 Layer 0 所有采样消息均保留。灰色横线表示无采样消息；1981–2000 的宗教组有极少量消息，显示的 0.0% 是四舍五入。",
        "## 解释边界\n\n这些结果表明，在本轮训练预算与选择规则下，Layer 0 没有形成硬门控删减，不能据此证明该层所有消息都必要。除 ≤1500 外，新结果保留更多 Layer 1 消息，预测更接近冻结的原 RGCN。",
        "模型与采样参数沿用原设置，仅限制 checkpoint 候选阶段。由于需要重跑 GraphMask，1941–1960 和 1981–2000 的训练历史存在数值漂移，因此新旧差异还包含重跑差异，不宜声称完全来自 checkpoint 选择条件。保真度约束在验证集检查。",
        "## 文件与复现\n\n主图包含 PNG、SVG、PDF；两份 metrics TSV 分别覆盖新五期和更新后的八期。其他 TSV／JSON 保留逐组数据、原新对照、门控启用状态、选中 epoch 和来源路径。Binary 图沿用全部原报告，没有替换。",
        "```bash\n" + command + "\n```",
        "ZIP 内 scripts/ 保存本轮三个生成脚本，source_reports/ 保存收到的原始统计包展开内容。复现时将统计包内容作为 --received-root，脚本在仓库环境运行。未包含模型权重，也不重新训练模型。",
    ]
    (output / "README.md").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    bundle = output / "dbpedia_layer0_enabled_figures_and_data.zip"
    files = [p for p in output.iterdir() if p.is_file() and p.suffix != ".zip"]
    with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.name)
        for name in ("visualize_layer0_supplement.py", "plot_layer0_relation_groups.py", "compare_layer0_checkpoint_selection.py"):
            archive.write(ROOT / "DBpedia" / name, "scripts/" + name)
        for p in root.rglob("*"):
            if p.is_file():
                archive.write(p, "source_reports/" + str(p.relative_to(root)))
    print(conclusion)
    print(metrics_table)
    print(f"Files: {output}\nBundle: {bundle}")


if __name__ == "__main__":
    main()
