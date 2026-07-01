# -*- coding: utf-8 -*-
"""
85_make_optionalB_summary.py

Create the preliminary report for Optional Experiment B.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/experiment8_optionalB"
TABLE_DIR = REPORT_DIR / "tables"

PROBE_SUMMARY = ROOT / "outputs/qwen15b/experiment8_optionalB/hidden_probe/process_error_layerwise_probe/layerwise_probe_summary.json"
PROBE_METRICS = ROOT / "outputs/qwen15b/experiment8_optionalB/hidden_probe/process_error_layerwise_probe/layerwise_probe_metrics.csv"
CONTROLS = ROOT / "outputs/qwen15b/experiment8_optionalB/controls/process_error_controls.csv"

REPORT_MD = REPORT_DIR / "optionalB_process_error_preliminary_report.md"
PROBE_TABLE = TABLE_DIR / "process_error_probe_summary.csv"
CONTROLS_TABLE = TABLE_DIR / "process_error_controls.csv"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    with PROBE_SUMMARY.open("r", encoding="utf-8") as f:
        probe = json.load(f)
    controls = pd.read_csv(CONTROLS, encoding="utf-8-sig")

    probe_df = pd.DataFrame(
        [
            {
                "model": "Qwen2.5-1.5B-Instruct",
                "target": "before_error_vs_at_or_after_error",
                "num_samples": probe["num_samples"],
                "num_groups": probe["num_groups"],
                "before_error_count": probe["before_error_count"],
                "at_or_after_error_count": probe["at_or_after_error_count"],
                "best_layer": probe["best_layer_by_auroc"],
                "best_auroc_mean": probe["best_auroc_mean"],
                "best_auroc_std": probe["best_auroc_std"],
                "best_accuracy_mean": probe["best_accuracy_mean"],
                "best_f1_mean": probe["best_f1_mean"],
                "split": probe["group_split"],
            }
        ]
    )
    probe_df.to_csv(PROBE_TABLE, index=False, encoding="utf-8-sig")
    controls.to_csv(CONTROLS_TABLE, index=False, encoding="utf-8-sig")

    def metric_row(method: str) -> pd.Series:
        return controls[controls["method"] == method].iloc[0]

    lines = [
        "# 可选实验B：Process Error 小切片初步报告",
        "",
        "## 1. 实验定位",
        "",
        "本实验是主线之外的扩展探索，目标是检查合成多步推理轨迹中，错误发生前后是否存在可解码的 step-level hidden-state signal（步骤级隐藏状态信号）。",
        "",
        "注意：本实验使用受控合成算术/方程轨迹，不是真实开放域 CoT reasoning error detection（链式推理错误检测）。",
        "",
        "## 2. 数据",
        "",
        "- 数据：`data/experiment8_optionalB/process_error_traces_v1.jsonl`。",
        "- problem 数：160。",
        "- corrupted step-prefix 样本：480。",
        "- 二分类目标：`before_error` 160 条 vs `at_or_after_error` 320 条。",
        "- correct trace 的 480 条 `no_error_control` 已保留，但未混入第一版二分类 probe。",
        "",
        "## 3. Hidden-state probe 结果",
        "",
        "评估使用 `GroupKFold(trace_id)`，避免同一题的不同步骤同时出现在训练和测试中。",
        "",
        "| 模型 | 样本数 | groups | 最佳层 | AUROC | Accuracy | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Qwen2.5-1.5B-Instruct | {probe['num_samples']} | {probe['num_groups']} | {probe['best_layer_by_auroc']} | "
        f"{probe['best_auroc_mean']:.4f} ± {probe['best_auroc_std']:.4f} | "
        f"{probe['best_accuracy_mean']:.4f} | {probe['best_f1_mean']:.4f} |",
        "",
        "## 4. 混杂控制",
        "",
        "| 方法 | AUROC | Accuracy | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]

    for _, row in controls.iterrows():
        auroc = f"{row['auroc_mean']:.4f}"
        if pd.notna(row.get("auroc_std")):
            auroc += f" ± {row['auroc_std']:.4f}"
        acc = "" if pd.isna(row["accuracy_mean"]) else f"{row['accuracy_mean']:.4f}"
        f1 = "" if pd.isna(row["f1_mean"]) else f"{row['f1_mean']:.4f}"
        lines.append(f"| {row['method']} | {auroc} | {acc} | {f1} |")

    lines.extend(
        [
            "",
            "## 5. 关键解释",
            "",
            "虽然 hidden-state probe 达到满分，但 step_index、step-prefix length 和 TF-IDF 文本基线也达到 AUROC=1.0。这说明当前 B-v1 的标签与步骤位置和文本长度高度绑定：`before_error` 总是第 1 步，`at_or_after_error` 总是第 2 步或答案步。",
            "",
            "因此，B-v1 不能作为“模型内部已经检测到推理过程错误”的证据。它更适合作为一个 sanity check：当前数据设计存在强混杂，需要设计 B-v2。",
            "",
            "## 6. B-v2 改进方向",
            "",
            "下一版应把错误标签与 step_index 解耦：",
            "",
            "1. 让错误可能出现在第 1、2、3 步，而不是固定第 2 步。",
            "2. 构造同一 step_index 下既有 correct/before-error，也有 at-error 样本。",
            "3. 比较同一题、同一步位置的 correct prefix 与 corrupted prefix。",
            "4. 把主目标改为 `step_correct` vs `step_incorrect`，而不是 `before_error` vs `after_error`。",
            "5. 必须继续报告 step_index、length、TF-IDF 和 label shuffle control。",
            "",
            "## 7. 可写结论",
            "",
            "保守写法：",
            "",
            "```text",
            "In a first synthetic process-error slice, hidden-state probes can perfectly separate before-error and at/after-error step prefixes. However, simple step-index and text baselines achieve the same performance, indicating strong positional and textual confounds. We therefore treat this result as a diagnostic failure case and propose a stricter follow-up design that decouples error labels from step position.",
            "```",
        ]
    )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", PROBE_TABLE)
    print("saved:", CONTROLS_TABLE)
    print("saved:", REPORT_MD)


if __name__ == "__main__":
    main()
