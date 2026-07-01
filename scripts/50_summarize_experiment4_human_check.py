# -*- coding: utf-8 -*-
"""
50_summarize_experiment4_human_check.py

Summarize agreement after the user fills experiment4_human_check_sample60.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = ROOT / "reports/experiment4/manual_review"
SAMPLE = MANUAL_DIR / "experiment4_human_check_sample60.csv"
SUMMARY = MANUAL_DIR / "experiment4_human_check_sample60_summary.md"
DISAGREE = MANUAL_DIR / "experiment4_human_check_sample60_disagreements.csv"


VALID_LABELS = {"correct", "hallucination", "refusal", "irrelevant", "uncertain"}


def md_table(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype=str).fillna("")
    df["human_behavior_label"] = df["human_behavior_label"].str.strip().str.lower()
    filled = df[df["human_behavior_label"] != ""].copy()

    invalid = sorted(set(filled["human_behavior_label"]) - VALID_LABELS)
    if invalid:
        print("warning: invalid labels:", invalid)

    if filled.empty:
        lines = [
            "# 第四次实验人工确认样本摘要",
            "",
            "还没有填写 `human_behavior_label`，因此暂时无法统计一致率。",
        ]
        SUMMARY.write_text("\n".join(lines), encoding="utf-8")
        print("no filled human labels yet")
        print("saved:", SUMMARY)
        return

    filled["agrees_with_codex"] = filled["human_behavior_label"] == filled["codex_behavior_label"]
    strict = filled[filled["human_behavior_label"] != "uncertain"].copy()
    agreement = strict["agrees_with_codex"].mean() if len(strict) else float("nan")

    label_dist = (
        filled["human_behavior_label"]
        .value_counts()
        .rename_axis("human_behavior_label")
        .reset_index(name="count")
    )
    confusion = (
        filled.groupby(["codex_behavior_label", "human_behavior_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["codex_behavior_label", "human_behavior_label"])
    )
    by_model = (
        filled.groupby(["model_short", "codex_behavior_label", "human_behavior_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["model_short", "codex_behavior_label", "human_behavior_label"])
    )

    disagreements = filled[
        (filled["human_behavior_label"] != "uncertain")
        & (~filled["agrees_with_codex"])
    ].copy()
    disagreements.to_csv(DISAGREE, index=False, encoding="utf-8-sig")

    lines = [
        "# 第四次实验人工确认样本摘要",
        "",
        f"- 样本总数：{len(df)}",
        f"- 已填写：{len(filled)}",
        f"- 不计 uncertain 的已填写样本：{len(strict)}",
        f"- Codex-assisted 标签一致率：{agreement:.4f}" if len(strict) else "- Codex-assisted 标签一致率：NA",
        f"- 不一致样本数：{len(disagreements)}",
        "",
        "## 人工标签分布",
        "",
        md_table(label_dist),
        "",
        "## Codex x Human 混淆表",
        "",
        md_table(confusion),
        "",
        "## 按模型统计",
        "",
        md_table(by_model),
        "",
        "## 不一致样本文件",
        "",
        f"- `{DISAGREE.relative_to(ROOT)}`",
    ]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", SUMMARY)
    print("saved:", DISAGREE)
    print(f"filled={len(filled)} strict={len(strict)} agreement={agreement:.4f}" if len(strict) else "agreement=NA")


if __name__ == "__main__":
    main()
