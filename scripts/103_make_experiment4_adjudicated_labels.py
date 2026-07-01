# -*- coding: utf-8 -*-
"""Create an adjudicated Experiment 4 manual-review table from human checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = ROOT / "reports/experiment4/manual_review"
SOURCE = MANUAL_DIR / "experiment4_manual_review_priority_all.csv"
CHECKS = MANUAL_DIR / "experiment4_human_check_combined_filled.csv"
OUT = MANUAL_DIR / "experiment4_manual_review_priority_all_adjudicated.csv"
SUMMARY = MANUAL_DIR / "experiment4_manual_review_adjudicated_summary.md"
VALID_LABELS = {"correct", "hallucination", "refusal", "irrelevant"}


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
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype=str).fillna("")
    checks = pd.read_csv(CHECKS, encoding="utf-8-sig", dtype=str).fillna("")
    checks["human_behavior_label"] = checks["human_behavior_label"].str.strip().str.lower()
    strict = checks[checks["human_behavior_label"].isin(VALID_LABELS)].copy()

    correction_rows = strict[strict["human_behavior_label"] != strict["codex_behavior_label"]].copy()
    correction_map = correction_rows.set_index("review_id")["human_behavior_label"].to_dict()
    note_map = correction_rows.set_index("review_id")["human_notes"].to_dict()

    adjudicated = source.copy()
    adjudicated["manual_behavior_label_original"] = adjudicated["manual_behavior_label"]
    adjudicated["manual_notes_original"] = adjudicated["manual_notes"]
    adjudicated["adjudication_source"] = ""
    adjudicated["adjudication_notes"] = ""

    for idx, row in adjudicated.iterrows():
        review_id = row["review_id"]
        if review_id in correction_map:
            adjudicated.at[idx, "manual_behavior_label"] = correction_map[review_id]
            adjudicated.at[idx, "manual_notes"] = note_map.get(review_id, "")
            adjudicated.at[idx, "adjudication_source"] = "human_check_disagreement"
            adjudicated.at[idx, "adjudication_notes"] = note_map.get(review_id, "")

    adjudicated.to_csv(OUT, index=False, encoding="utf-8-sig")

    before = (
        source["manual_behavior_label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="before_count")
    )
    after = (
        adjudicated["manual_behavior_label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="after_count")
    )
    counts = before.merge(after, on="label", how="outer").fillna(0)
    counts["before_count"] = counts["before_count"].astype(int)
    counts["after_count"] = counts["after_count"].astype(int)
    counts["delta"] = counts["after_count"] - counts["before_count"]

    corrections_view = correction_rows[
        [
            "review_id",
            "model_short",
            "label_name",
            "category",
            "entity",
            "codex_behavior_label",
            "human_behavior_label",
            "human_notes",
        ]
    ].copy()

    lines = [
        "# 实验 4 人工裁决后标签汇总",
        "",
        "## 范围",
        "",
        f"- 原始表：`{SOURCE.relative_to(ROOT)}`",
        f"- 人工复核合并表：`{CHECKS.relative_to(ROOT)}`",
        f"- 裁决后表：`{OUT.relative_to(ROOT)}`",
        "",
        "仅将人工复核中非 uncertain 且与 Codex-assisted 标签不一致的样本写入裁决修正；uncertain 样本不改动原标签。",
        "",
        "## 标签计数变化",
        "",
        md_table(counts.sort_values("label")),
        "",
        "## 修正样本",
        "",
        md_table(corrections_view) if len(corrections_view) else "无。",
    ]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", OUT)
    print("saved:", SUMMARY)
    print("corrections:", len(correction_rows))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
