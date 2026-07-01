# -*- coding: utf-8 -*-
"""
47_make_experiment4_manual_review_batch.py

Create manual-review sheets for Experiment 4.

The script does not change model outputs or automatic labels. It reads the
answer-key-assisted analysis files and exports high-priority rows for human
review, especially known/low-risk samples that were labeled as hallucination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_FILES = [
    {
        "model_short": "qwen05b",
        "model": "Qwen2.5-0.5B-Instruct",
        "path": ROOT
        / "outputs/qwen05b/experiment4/experiment4_generation_behavior_analysis_answer_key/generation_behavior_merged_scores.csv",
    },
    {
        "model_short": "qwen15b",
        "model": "Qwen2.5-1.5B-Instruct",
        "path": ROOT
        / "outputs/qwen15b/experiment4/experiment4_generation_behavior_analysis_answer_key/generation_behavior_merged_scores.csv",
    },
]


OUT_DIR = ROOT / "reports/experiment4/manual_review"
ALL_OUT = OUT_DIR / "experiment4_manual_review_priority_all.csv"
BATCH_OUT = OUT_DIR / "experiment4_manual_review_batch200.csv"
README_OUT = OUT_DIR / "README_manual_review.md"


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def first_nonempty(row: pd.Series, cols: Iterable[str]) -> str:
    for col in cols:
        if col in row.index:
            value = clean_text(row[col])
            if value:
                return value
    return ""


def load_model_rows(item: dict[str, object]) -> pd.DataFrame:
    path = Path(item["path"])
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.insert(0, "model_short", item["model_short"])
    df.insert(1, "model", item["model"])

    df["label_int"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df["review_auto_label"] = df.apply(
        lambda row: first_nonempty(
            row,
            ["final_behavior_label", "behavior_label", "auto_behavior_label"],
        ),
        axis=1,
    )
    df["hidden_risk_score_oof"] = pd.to_numeric(
        df.get("hidden_risk_score_oof"), errors="coerce"
    )
    df["entropy_score"] = pd.to_numeric(df.get("entropy_score"), errors="coerce")
    return df


def add_review_reasons(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    label_sets = (
        df.groupby("id")["review_auto_label"]
        .apply(lambda values: {clean_text(v) for v in values if clean_text(v)})
        .to_dict()
    )
    disagreement_ids = {
        sample_id for sample_id, labels in label_sets.items() if len(labels) > 1
    }

    df["hidden_risk_percentile"] = df.groupby("model_short")[
        "hidden_risk_score_oof"
    ].rank(pct=True)

    reasons: list[str] = []
    scores: list[int] = []
    priority_names: list[str] = []

    for _, row in df.iterrows():
        label = row["label_int"]
        auto_label = clean_text(row["review_auto_label"])
        sample_id = clean_text(row["id"])
        risk_pct = row["hidden_risk_percentile"]

        row_reasons: list[str] = []
        score = 0

        if label == 0 and auto_label != "correct":
            row_reasons.append("known_non_correct_needs_review")
            score += 100

        if sample_id in disagreement_ids:
            row_reasons.append("model_label_disagreement")
            score += 40

        if label == 1 and auto_label in {"correct", "refusal", "irrelevant"}:
            row_reasons.append("high_risk_unusual_behavior")
            score += 30

        if pd.notna(risk_pct):
            if label == 0 and auto_label == "correct" and risk_pct >= 0.90:
                row_reasons.append("score_label_conflict_high_risk_known_correct")
                score += 20
            if label == 1 and auto_label in {"hallucination", "refusal", "irrelevant"} and risk_pct <= 0.10:
                row_reasons.append("score_label_conflict_low_risk_high_risk_behavior")
                score += 20

        reasons.append(";".join(row_reasons))
        scores.append(score)
        priority_names.append(row_reasons[0] if row_reasons else "")

    df["review_reasons"] = reasons
    df["review_priority_score"] = scores
    df["primary_review_reason"] = priority_names
    return df


def make_outputs(batch_size: int = 200) -> None:
    frames = [load_model_rows(item) for item in MODEL_FILES]
    df = pd.concat(frames, ignore_index=True)
    df = add_review_reasons(df)

    candidates = df[df["review_priority_score"] > 0].copy()
    candidates = candidates.sort_values(
        by=[
            "review_priority_score",
            "primary_review_reason",
            "model_short",
            "id",
        ],
        ascending=[False, True, True, True],
    )

    keep_cols = [
        "model_short",
        "model",
        "id",
        "label",
        "label_name",
        "category",
        "entity",
        "relation",
        "prompt",
        "model_answer",
        "model_answer_raw",
        "review_auto_label",
        "behavior_label",
        "auto_behavior_label",
        "auto_behavior_notes",
        "notes",
        "hidden_risk_score_oof",
        "hidden_risk_percentile",
        "entropy_score",
        "neg_top1_prob_score",
        "neg_margin_score",
        "review_priority_score",
        "primary_review_reason",
        "review_reasons",
    ]
    keep_cols = [col for col in keep_cols if col in candidates.columns]
    review_df = candidates[keep_cols].copy()
    review_df.insert(0, "review_id", [f"exp4_review_{i:04d}" for i in range(len(review_df))])
    review_df["manual_behavior_label"] = ""
    review_df["manual_notes"] = ""
    review_df["manual_reviewer"] = ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(ALL_OUT, index=False, encoding="utf-8-sig")
    review_df.head(batch_size).to_csv(BATCH_OUT, index=False, encoding="utf-8-sig")

    reason_summary = (
        review_df["primary_review_reason"]
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="count")
    )
    model_summary = (
        review_df.groupby(["model_short", "primary_review_reason"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["model_short", "primary_review_reason"])
    )

    def markdown_table(table_df: pd.DataFrame) -> str:
        cols = list(table_df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in table_df.iterrows():
            lines.append("| " + " | ".join(clean_text(row[col]) for col in cols) + " |")
        return "\n".join(lines)

    readme_lines = [
        "# 第四次实验人工复核说明",
        "",
        "本目录用于把 Experiment 4 从“答案表辅助标注”推进到“人工复核标注”。",
        "",
        "## 文件",
        "",
        f"- `{ALL_OUT.name}`：所有高优先级候选样本。",
        f"- `{BATCH_OUT.name}`：优先复核前 200 条，建议先填这个。",
        "",
        "## 建议填写字段",
        "",
        "- `manual_behavior_label`：填写 `correct`、`hallucination`、`refusal`、`irrelevant` 或 `uncertain`。",
        "- `manual_notes`：写一句简短原因，例如“答案为别名，可判 correct”或“答案事实错误”。",
        "- `manual_reviewer`：可填姓名或缩写。",
        "",
        "## 优先级含义",
        "",
        "- `known_non_correct_needs_review`：known/low-risk 样本被答案表标成非 correct，最值得复核。",
        "- `model_label_disagreement`：两个模型在同一 query 上的自动行为标签不同。",
        "- `high_risk_unusual_behavior`：high-risk 样本出现 refusal、irrelevant 或 correct 等少见行为。",
        "- `score_label_conflict_*`：hidden risk score 与行为标签方向不一致，适合检查打分边界。",
        "",
        "## 当前候选统计",
        "",
        "按首要原因：",
        "",
        markdown_table(reason_summary),
        "",
        "按模型和首要原因：",
        "",
        markdown_table(model_summary),
        "",
        "## 论文表述提醒",
        "",
        "在人工复核完成前，第 4 次实验应表述为“答案表辅助标注/初步行为标注”。复核完成后，可以表述为“抽样人工复核后的行为标注”。",
    ]
    README_OUT.write_text("\n".join(readme_lines), encoding="utf-8")

    print("saved:", ALL_OUT)
    print("saved:", BATCH_OUT)
    print("saved:", README_OUT)
    print()
    print("reason summary:")
    print(reason_summary.to_string(index=False))
    print()
    print("model summary:")
    print(model_summary.to_string(index=False))


if __name__ == "__main__":
    make_outputs()
