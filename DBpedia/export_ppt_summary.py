#!/usr/bin/env python3
"""Export DBpedia relation groups and period metrics as concise Markdown tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = {
    "inherited": "Inherited ties",
    "intimate_partnership": "Intimate partnership",
    "education_mentorship": "Education / mentorship",
    "professional_collaboration": "Professional collaboration",
    "influence_succession": "Influence / succession",
    "religious_authority_recognition": "Religious authority / recognition",
    "other_acquired": "Other acquired ties",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def percent(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Expected a proportion between zero and one: {value}")
    return f"{100 * value:.2f}"


def period_label(period):
    if "start" not in period:
        return f"≤{period['end']}"
    if "end" not in period:
        return f"≥{period['start']}"
    return f"{period['start']}–{period['end']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received-root", type=Path, default=ROOT / "artifacts/dbpedia_visualization_received_2026_10_06")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/dbpedia_ppt_summary_2026_10_07")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.received_root.resolve()
    config = read_json(root / "config/dbpedia_grouped_rgcn_graphmask_20y_v1.json")
    periods = read_json(root / config["period_config"])["periods"]
    taxonomy = read_json(root / config["multi_group_taxonomy"])
    members = [r for relations in taxonomy["groups"].values() for r in relations]
    if len(members) != 33 or len(set(members)) != 33 or len(taxonomy["groups"]) != 7:
        raise ValueError("Expected 33 non-overlapping predicates in seven groups")
    records = {}
    for period in periods:
        for rep in ("binary", "multi_group"):
            relative = Path(period["id"]) / rep / f"seed_{args.seed}"
            rgcn_path = root / config["model_root"] / relative / "metrics.json"
            mask_dir = root / config["graphmask_root"] / relative
            mask_path = mask_dir / "test_report/test_metrics.json"
            validation_path = mask_dir / "validation.json"
            split_path = root / config["relation_root"] / period["id"] / rep / "split_summary.json"
            rgcn, mask = read_json(rgcn_path), read_json(mask_path)
            validation, split = read_json(validation_path), read_json(split_path)
            if rgcn["run_config"]["seed"] != args.seed or rgcn["run_config"]["model"] != "rgcn":
                raise ValueError(f"Wrong model/seed: {rgcn_path}")
            if mask["split"] != "test" or mask["roots"] != split["test_nodes"] or mask["labeled_roots"] != split["test_nodes"]:
                raise ValueError(f"Wrong evaluation split/root count: {mask_path}")
            if validation["relative_macro_f1_difference"] > config["graphmask_train"]["max_relative_macro_f1_diff"]:
                raise ValueError(f"Validation fidelity failed: {validation_path}")
            if len(mask["layers"]) != 2:
                raise ValueError(f"Expected two GraphMask layers: {mask_path}")
            observations = sum(layer["message_observations"] for layer in mask["layers"])
            retained = sum(layer["message_observations"] * layer["hard_retention_rate"] for layer in mask["layers"])
            if not observations or not math.isclose(mask["hard_retention_rate"], retained / observations):
                raise ValueError(f"Overall retention disagrees with layer counts: {mask_path}")
            records[(period["id"], rep)] = {
                "period": period["id"], "period_label": period_label(period), "representation": rep,
                "rgcn_test": rgcn["test"], "graphmask_test": mask,
                "validation_relative_macro_f1_difference": validation["relative_macro_f1_difference"],
                "sources": {"rgcn": str(rgcn_path), "graphmask": str(mask_path), "validation": str(validation_path)},
            }
    sections = [
        "# DBpedia PPT 关系分组与原实验结果",
        "数据来自 2026-10-06 下载的原实验，seed=42。正在运行的五时期 Layer 0 门控启用后 checkpoint 补充实验尚未计入。",
        "## Multi-group 关系分组",
        "共 33 种原始谓词，分为 7 组：1 个 inherited 组与 6 个 acquired 子组。下表列原始关系名，不重复列自动生成的反向关系。",
        table(["类别", "关系组", "包含关系"], [
            ["Inherited" if group == "inherited" else "Acquired", GROUP_NAMES[group], ", ".join(relations)]
            for group, relations in taxonomy["groups"].items()
        ]),
        "## RGCN 测试集表现",
        "单位：%。F1 均为 Macro-F1。Binary 表示 inherited/acquired 二分组。",
        table(["时期", "Binary 准确率", "Binary Macro-F1", "Multi-group 准确率", "Multi-group Macro-F1"], [
            [period_label(p), *(percent(records[(p["id"], rep)]["rgcn_test"][metric])
                                for rep in ("binary", "multi_group") for metric in ("accuracy", "macro_f1"))]
            for p in periods
        ]),
    ]
    for rep, label in (("multi_group", "Multi-group"), ("binary", "Binary")):
        sections += [
            f"## GraphMask 测试集表现（{label}）",
            "单位：%。准确率与 Macro-F1 均为消息屏蔽后的预测指标。",
            table(["时期", "整体保留率", "预测一致性", "删边后准确率", "删边后 Macro-F1"], [
                [period_label(p), percent(records[(p["id"], rep)]["graphmask_test"]["hard_retention_rate"]),
                 percent(records[(p["id"], rep)]["graphmask_test"]["prediction_agreement"]),
                 percent(records[(p["id"], rep)]["graphmask_test"]["masked"]["accuracy"]),
                 percent(records[(p["id"], rep)]["graphmask_test"]["masked"]["macro_f1"])]
                for p in periods
            ]),
        ]
    sections += [
        "## 指标口径",
        "整体保留率是两层 hard-retained 采样消息数之和除以两层采样消息观测总数，包含正反方向，不是 Layer 0 单层保留率，也不是去重后的全图边比例。",
        "预测一致性是同一批测试采样图上，屏蔽前后预测类别相同的测试人物比例。",
        "RGCN 表采用训练输出 metrics.json 的 test 指标。GraphMask 表采用 test_report/test_metrics.json 的 masked 指标。两者评估采样设置不同，直接计算前后差值时，应使用 JSON 中 GraphMask 报告配对的 original 与 masked。",
        "保真度阈值在验证集检查，不是对测试集指标差的保证。所有数值来自单次 seed=42 实验，未汇总多次运行的不确定性。",
        "时期按人物生命区间重叠划分，不表示关系发生的时期。宗教权威／认可组是 beatifiedBy/canonizedBy，与主实验的 religious ordination 并非相同关系。",
    ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    (output / "summary_data.json").write_text(json.dumps({
        "received_root": str(root), "experiment": config["name"], "seed": args.seed,
        "result_version": "original_any_stage_checkpoint_selection",
        "f1_metric": "macro_f1", "stored_scale": "proportion_0_to_1", "display_scale": "percent_2_decimals",
        "groups": taxonomy["groups"], "periods": list(records.values()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output / "summary.md")
    print(f"Validated {len(records)} model/probe results")


if __name__ == "__main__":
    main()
