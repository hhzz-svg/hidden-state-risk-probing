# -*- coding: utf-8 -*-
"""Summarize Optional Experiment A v2 PK/CK manual review sample."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/experiment7_optionalA"
SAMPLE = REPORT_DIR / "pk_ck_v2_manual_review_sample44.csv"
SUMMARY = REPORT_DIR / "pk_ck_v2_manual_review_summary.md"
DISAGREE = REPORT_DIR / "pk_ck_v2_manual_review_disagreements.csv"
VALID_LABELS = {"pk_follow", "ck_follow", "mixed_or_conflict_ack", "refusal", "other", "uncertain"}


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
    df["human_pkck_label"] = df["human_pkck_label"].str.strip().str.lower()
    filled = df[df["human_pkck_label"] != ""].copy()
    invalid = sorted(set(filled["human_pkck_label"]) - VALID_LABELS)

    if filled.empty:
        SUMMARY.write_text(
            "# 可选实验 A v2 PK/CK 人工复核汇总\n\n尚未填写 `human_pkck_label`，暂时无法统计一致率。\n",
            encoding="utf-8",
        )
        print("no filled rows")
        print("saved:", SUMMARY)
        return

    filled["agrees_with_auto"] = filled["human_pkck_label"] == filled["auto_pkck_label"]
    strict = filled[filled["human_pkck_label"] != "uncertain"].copy()
    agreement = strict["agrees_with_auto"].mean() if len(strict) else float("nan")

    label_dist = (
        filled["human_pkck_label"]
        .value_counts()
        .rename_axis("human_pkck_label")
        .reset_index(name="count")
    )
    confusion = (
        filled.groupby(["auto_pkck_label", "human_pkck_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["auto_pkck_label", "human_pkck_label"])
    )
    disagreements = strict[~strict["agrees_with_auto"]].copy()
    disagreements.to_csv(DISAGREE, index=False, encoding="utf-8-sig")

    lines = [
        "# 可选实验 A v2 PK/CK 人工复核汇总",
        "",
        f"- 抽样总数：{len(df)}",
        f"- 已填写：{len(filled)}",
        f"- 不计 uncertain 的已填写样本：{len(strict)}",
        f"- 自动标签一致率：{agreement:.4f}" if len(strict) else "- 自动标签一致率：NA",
        f"- 不一致样本数：{len(disagreements)}",
        f"- 非法标签：{', '.join(invalid) if invalid else '无'}",
        "",
        "## 人工标签分布",
        "",
        md_table(label_dist),
        "",
        "## Auto x Human 混淆表",
        "",
        md_table(confusion),
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
