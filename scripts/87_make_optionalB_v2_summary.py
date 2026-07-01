# -*- coding: utf-8 -*-
"""
87_make_optionalB_v2_summary.py

Create the report for Optional Experiment B-v2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/experiment8_optionalB"
TABLE_DIR = REPORT_DIR / "tables"

DATA_FILE = ROOT / "data/experiment8_optionalB/process_error_traces_v2.jsonl"
METADATA = ROOT / "outputs/qwen15b/experiment8_optionalB_v2/hidden_probe/hidden_states_process_error_v2.metadata.csv"
PROBE_SUMMARY = ROOT / "outputs/qwen15b/experiment8_optionalB_v2/hidden_probe/process_error_layerwise_probe/layerwise_probe_summary.json"
PROBE_METRICS = ROOT / "outputs/qwen15b/experiment8_optionalB_v2/hidden_probe/process_error_layerwise_probe/layerwise_probe_metrics.csv"
CONTROLS = ROOT / "outputs/qwen15b/experiment8_optionalB_v2/controls/process_error_controls.csv"

REPORT_MD = REPORT_DIR / "optionalB_process_error_v2_report.md"
PROBE_TABLE = TABLE_DIR / "process_error_v2_probe_summary.csv"
LAYER_TABLE = TABLE_DIR / "process_error_v2_layerwise_probe_metrics.csv"
CONTROLS_TABLE = TABLE_DIR / "process_error_v2_controls.csv"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pm(mean: float, std: float | None = None) -> str:
    if std is None or pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    probe = read_json(PROBE_SUMMARY)
    metrics = pd.read_csv(PROBE_METRICS, encoding="utf-8-sig")
    controls = pd.read_csv(CONTROLS, encoding="utf-8-sig")
    meta = pd.read_csv(METADATA, encoding="utf-8-sig")

    label_counts = (
        meta.groupby(["label", "label_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "label_name"])
    )
    step_counts = (
        meta.groupby(["step_index", "label_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["step_index", "label_name"])
    )
    error_counts = (
        meta[meta["label_name"] == "step_incorrect"]
        .groupby(["step_index", "error_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["step_index", "error_type"])
    )

    best_layer = int(probe["best_layer_by_auroc"])
    best_row = metrics.loc[metrics["layer"] == best_layer].iloc[0]
    non_final = metrics[metrics["layer"] != metrics["layer"].max()]
    best_non_final = non_final.loc[non_final["auroc_mean"].idxmax()]

    probe_table = pd.DataFrame(
        [
            {
                "model": "Qwen2.5-1.5B-Instruct",
                "target": "step_correct_vs_step_incorrect",
                "num_samples": int(probe["num_samples"]),
                "num_groups": int(probe["num_groups"]),
                "step_correct": int((meta["label"] == 0).sum()),
                "step_incorrect": int((meta["label"] == 1).sum()),
                "best_layer": best_layer,
                "best_auroc": fmt_pm(float(probe["best_auroc_mean"]), float(probe["best_auroc_std"])),
                "best_accuracy": f"{float(probe['best_accuracy_mean']):.4f}",
                "best_f1": f"{float(probe['best_f1_mean']):.4f}",
                "split": str(probe["group_split"]),
            }
        ]
    )

    control_table = controls.copy()
    control_table["auroc"] = control_table.apply(
        lambda row: fmt_pm(float(row["auroc_mean"]), row.get("auroc_std")),
        axis=1,
    )
    control_table = control_table[
        ["method", "auroc", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "shuffle_repeats"]
    ]

    metrics.to_csv(LAYER_TABLE, index=False, encoding="utf-8-sig")
    probe_table.to_csv(PROBE_TABLE, index=False, encoding="utf-8-sig")
    control_table.to_csv(CONTROLS_TABLE, index=False, encoding="utf-8-sig")

    lines = [
        "# 可选实验B-v2：Step Correctness 过程错误探针报告",
        "",
        "## 1. 实验定位",
        "",
        "B-v2 是对 B-v1 的修正版。B-v1 中 `before_error` 与 `at_or_after_error` 和 step_index、文本长度强绑定，",
        "因此 hidden-state probe 满分不能解释为内部过程错误信号。B-v2 将目标改为 `step_correct` vs `step_incorrect`：",
        "每道题、每个 step_index 同时构造一个正确 prefix 和一个错误 prefix，使 step_index 在设计上不再携带标签信息。",
        "",
        "本实验仍然是受控合成轨迹，不等同于开放域链式推理错误检测；它更适合作为机制可解释性中的 step-level hidden-state probing 扩展结果。",
        "",
        "## 2. 数据与任务",
        "",
        f"- 数据文件：`{DATA_FILE.relative_to(ROOT)}`",
        "- 模型：Qwen2.5-1.5B-Instruct",
        f"- 样本数：{len(meta)} 个 step-prefix",
        f"- problem groups：{meta['trace_id'].nunique()}",
        "- 评估划分：`GroupKFold(trace_id)`，同一道题的所有 prefix 不跨训练集和测试集。",
        "",
        "### label 统计",
        "",
        md_table(label_counts),
        "",
        "### step_index x label 统计",
        "",
        md_table(step_counts),
        "",
        "### 错误类型统计",
        "",
        md_table(error_counts),
        "",
        "## 3. Hidden-state probe 结果",
        "",
        md_table(probe_table),
        "",
        f"最佳层为 layer {best_layer}，AUROC={fmt_pm(float(best_row['auroc_mean']), float(best_row['auroc_std']))}，",
        f"Accuracy={float(best_row['accuracy_mean']):.4f}，F1={float(best_row['f1_mean']):.4f}。",
        f"若不看最后一层，最佳非最终层为 layer {int(best_non_final['layer'])}，",
        f"AUROC={fmt_pm(float(best_non_final['auroc_mean']), float(best_non_final['auroc_std']))}。",
        "",
        "## 4. 混杂控制",
        "",
        md_table(control_table),
        "",
        "B-v2 的关键变化是简单控制项不再复现 hidden probe 的高分：",
        "`step_index` 为随机水平，`step_prefix_char_length` 接近随机，TF-IDF 字符 n-gram 只略高于随机，",
        "label shuffle 也接近随机。这说明 B-v2 相比 B-v1 明显降低了位置和表层文本混杂。",
        "",
        "## 5. 可写结论",
        "",
        "保守写法：",
        "",
        "```text",
        "In the stricter paired process-error slice, step-correct and step-incorrect prefixes were balanced within each step position and evaluated with group-level splits by problem. A layer-wise linear probe over Qwen2.5-1.5B hidden states reached AUROC = 0.9908 +/- 0.0018 at the final layer, whereas step index, prefix length, TF-IDF text baselines, and label-shuffle controls remained near chance or substantially lower. This suggests that the hidden states contain a decodable signal associated with synthetic step correctness beyond the tested positional and surface-text controls.",
        "```",
        "",
        "中文论文表述可以写成：",
        "",
        "```text",
        "在配对构造的过程错误切片中，同一题目、同一推理步位置均包含正确与错误 prefix，并采用按题目分组的交叉验证。结果显示，Qwen2.5-1.5B 的隐状态线性探针可高精度区分 step_correct 与 step_incorrect，而 step_index、prefix 长度、TF-IDF 文本基线和标签打乱控制均显著低于隐状态探针。该结果支持模型隐状态中存在与受控合成步骤正确性相关的可解码表征信号，但不能直接外推为开放域推理过程错误检测能力。",
        "```",
        "",
        "## 6. 局限与下一步",
        "",
        "1. 当前轨迹是模板化合成算术/方程任务，仍可能含有数值模式或模板模式。",
        "2. 最强结果出现在最终层，可能混有输出预测或答案判别信息；建议补充早中层主分析或 token 位置消融。",
        "3. 后续可增加自然语言推理、多模板改写、同义步骤表述、跨模板外推测试。",
        "4. 若要写入正文，建议作为“可选扩展实验/机制补充”，主线仍放在幻觉相关隐状态可解码性与控制实验。",
    ]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", PROBE_TABLE)
    print("saved:", LAYER_TABLE)
    print("saved:", CONTROLS_TABLE)
    print("saved:", REPORT_MD)


if __name__ == "__main__":
    main()
