# -*- coding: utf-8 -*-
"""
44_make_experiment4_summary.py

生成第四次实验的中文汇总报告。
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


MODELS = [
    {
        "model": "Qwen2.5-0.5B-Instruct",
        "gen_dir": ROOT / "outputs/qwen05b/experiment4/experiment4_generation_behavior",
        "analysis_dir": ROOT / "outputs/qwen05b/experiment4/experiment4_generation_behavior_analysis_answer_key",
    },
    {
        "model": "Qwen2.5-1.5B-Instruct",
        "gen_dir": ROOT / "outputs/qwen15b/experiment4/experiment4_generation_behavior",
        "analysis_dir": ROOT / "outputs/qwen15b/experiment4/experiment4_generation_behavior_analysis_answer_key",
    },
    {
        "model": "Qwen2.5-7B-Instruct",
        "gen_dir": ROOT / "outputs/qwen7b/experiment4/experiment4_generation_behavior",
        "analysis_dir": ROOT / "outputs/qwen7b/experiment4/experiment4_generation_behavior_analysis_answer_key",
    },
]


def read_behavior_by_query(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def get_count(row: pd.Series, col: str) -> int:
    return int(row[col]) if col in row.index and pd.notna(row[col]) else 0


def main() -> None:
    rows = []
    metric_rows = []

    for item in MODELS:
        model = item["model"]
        analysis_dir = item["analysis_dir"]
        by_query = read_behavior_by_query(analysis_dir / "behavior_by_query_label.csv")
        metrics = pd.read_csv(analysis_dir / "behavior_prediction_metrics.csv", encoding="utf-8-sig")

        known = by_query[by_query["label_name"] == "known_or_low_risk"].iloc[0]
        unknown = by_query[by_query["label_name"] == "unknown_or_high_risk"].iloc[0]

        known_total = sum(get_count(known, col) for col in by_query.columns if col != "label_name")
        unknown_total = sum(get_count(unknown, col) for col in by_query.columns if col != "label_name")
        known_correct = get_count(known, "correct")
        known_hallucination = get_count(known, "hallucination")
        known_irrelevant = get_count(known, "irrelevant")
        unknown_hallucination = get_count(unknown, "hallucination")
        unknown_refusal = get_count(unknown, "refusal")
        unknown_irrelevant = get_count(unknown, "irrelevant")

        rows.append(
            {
                "model": model,
                "known_total": known_total,
                "known_correct": known_correct,
                "known_correct_rate": known_correct / known_total,
                "known_hallucination": known_hallucination,
                "known_hallucination_rate": known_hallucination / known_total,
                "known_irrelevant": known_irrelevant,
                "unknown_total": unknown_total,
                "unknown_hallucination": unknown_hallucination,
                "unknown_hallucination_rate": unknown_hallucination / unknown_total,
                "unknown_refusal": unknown_refusal,
                "unknown_refusal_rate": unknown_refusal / unknown_total,
                "unknown_irrelevant": unknown_irrelevant,
            }
        )

        for _, metric in metrics.iterrows():
            metric_rows.append(
                {
                    "model": model,
                    "target": metric["target"],
                    "score": metric["score"],
                    "auroc": metric["auroc"],
                    "accuracy_at_median": metric["accuracy_at_median"],
                    "f1_at_median": metric["f1_at_median"],
                }
            )

    summary_df = pd.DataFrame(rows)
    metrics_df = pd.DataFrame(metric_rows)

    out_dir = ROOT / "reports/experiment4/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "experiment4_behavior_summary.csv"
    metrics_csv = out_dir / "experiment4_behavior_prediction_metrics.csv"
    report_path = ROOT / "reports/experiment4/experiment4_generation_behavior_report.md"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    hidden_metrics = metrics_df[metrics_df["score"] == "hidden_risk_score_oof"]
    entropy_metrics = metrics_df[metrics_df["score"] == "entropy_score"]

    lines = [
        "# 第四次实验：真实生成行为标注初步报告",
        "",
        "## 1. 实验目的",
        "",
        "前三次实验证明，在 controlled_v4 和 entity-group split 设置下，query-level risk label 可以从中后层 hidden states（隐藏状态）中被高度线性解码。",
        "第四次实验进一步检查：这个生成前风险信号是否与模型真实生成行为存在关联。",
        "",
        "注意：这里的 query-level risk label 仍然不是 hallucination label（幻觉标签）。本实验通过模型真实生成答案，再用答案表辅助标注 correct/refusal/hallucination/irrelevant。",
        "",
        "## 2. 设置",
        "",
        "- 数据：`data/experiment3/prompts_controlled_v4.jsonl`，600 条，known/low-risk 300 条，unknown/high-risk 300 条。",
        "- 模型：Qwen2.5-0.5B-Instruct 与 Qwen2.5-1.5B-Instruct。",
        "- 生成方式：greedy decoding，`max_new_tokens=32`，取第一行作为主要答案，同时保留 raw answer。",
        "- 行为标注：使用 `scripts/43_label_generation_behavior_with_answer_key.py` 的答案表辅助标注；仍建议抽样人工复核。",
        "",
        "## 3. 生成行为分布",
        "",
        "| 模型 | known 正确 | known 错误/类幻觉 | high-risk 类幻觉 | high-risk 拒答 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model']} | "
            f"{row['known_correct']}/{row['known_total']} ({row['known_correct_rate']:.1%}) | "
            f"{row['known_hallucination']}/{row['known_total']} ({row['known_hallucination_rate']:.1%}) | "
            f"{row['unknown_hallucination']}/{row['unknown_total']} ({row['unknown_hallucination_rate']:.1%}) | "
            f"{row['unknown_refusal']}/{row['unknown_total']} ({row['unknown_refusal_rate']:.1%}) |"
        )

    lines.extend(
        [
            "",
            "## 4. 风险分数预测真实行为",
            "",
            "这里的目标是 `high_risk_behavior`，即标注为 hallucination/refusal/irrelevant 的生成行为。",
            "",
            "| 模型 | hidden risk score AUROC | entropy AUROC | negative top1 AUROC | negative margin AUROC |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for model in summary_df["model"]:
        model_metrics = metrics_df[metrics_df["model"] == model].set_index("score")
        lines.append(
            f"| {model} | "
            f"{model_metrics.loc['hidden_risk_score_oof', 'auroc']:.4f} | "
            f"{model_metrics.loc['entropy_score', 'auroc']:.4f} | "
            f"{model_metrics.loc['neg_top1_prob_score', 'auroc']:.4f} | "
            f"{model_metrics.loc['neg_margin_score', 'auroc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 初步结论",
            "",
            "1. 两个模型在 high-risk entity-relation query 上都很少拒答，更多是给出具体但不可靠的答案；这支持继续研究生成前风险信号与实际生成行为的关系。",
            "2. Qwen2.5-1.5B 的 known 正确率高于 0.5B，说明更大模型在同一数据上生成质量更好，但 high-risk 查询仍主要表现为类幻觉回答。",
            "3. 对真实行为标签而言，0.5B 上 entropy baseline 的 AUROC 略高于 hidden risk score；1.5B 上 hidden risk score 略高于 entropy baseline。这个结果要诚实写，不应只强调 hidden probe。",
            "4. 本实验的标注仍是答案表辅助标注，不是严格双人独立人工标注；最终论文中应把它作为初步证据，而不是最终幻觉检测结论。",
            "",
            "## 6. 后续建议",
            "",
            "- 抽样复核 `answer_key_behavior_labels.csv` 中 known 错误样本，重点看别名、翻译名、争议事实。",
            "- 下一步可以进入 Experiment 5: OOD 泛化实验，检查 probe 换一批新实体/新关系后是否还能工作。",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", summary_csv)
    print("saved:", metrics_csv)
    print("saved:", report_path)
    print(summary_df.to_string(index=False))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
