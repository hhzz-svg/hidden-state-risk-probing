# -*- coding: utf-8 -*-
"""Summarize all filled Experiment 4 human-check rounds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = ROOT / "reports/experiment4/manual_review"
FILES = [
    MANUAL_DIR / "experiment4_human_check_sample60.csv",
    MANUAL_DIR / "experiment4_human_check_round2_80.csv",
]
SUMMARY_MD = MANUAL_DIR / "experiment4_human_check_combined_summary.md"
DISAGREE_CSV = MANUAL_DIR / "experiment4_human_check_combined_disagreements.csv"
COMBINED_CSV = MANUAL_DIR / "experiment4_human_check_combined_filled.csv"
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
    frames = []
    for path in FILES:
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No human-check files found.")

    df = pd.concat(frames, ignore_index=True)
    df["human_behavior_label"] = df["human_behavior_label"].astype(str).str.strip().str.lower()
    filled = df[df["human_behavior_label"] != ""].copy()
    invalid = sorted(set(filled["human_behavior_label"]) - VALID_LABELS)

    if filled.empty:
        SUMMARY_MD.write_text(
            "# 实验 4 人工确认汇总\n\n尚未填写 `human_behavior_label`，暂时无法统计一致率。\n",
            encoding="utf-8",
        )
        print("no filled rows")
        print("saved:", SUMMARY_MD)
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
    by_round = (
        filled.groupby(["source_file", "codex_behavior_label", "human_behavior_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["source_file", "codex_behavior_label", "human_behavior_label"])
    )
    disagreements = strict[~strict["agrees_with_codex"]].copy()

    filled.to_csv(COMBINED_CSV, index=False, encoding="utf-8-sig")
    disagreements.to_csv(DISAGREE_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# 实验 4 人工确认合并汇总",
        "",
        f"- 抽样总数：{len(df)}",
        f"- 已填写：{len(filled)}",
        f"- 不计 uncertain 的已填写样本：{len(strict)}",
        f"- Codex-assisted 标签一致率：{agreement:.4f}" if len(strict) else "- Codex-assisted 标签一致率：NA",
        f"- 不一致样本数：{len(disagreements)}",
        f"- 非法标签：{', '.join(invalid) if invalid else '无'}",
        "",
        "## 人工标签分布",
        "",
        md_table(label_dist),
        "",
        "## Codex x Human 混淆表",
        "",
        md_table(confusion),
        "",
        "## 按轮次统计",
        "",
        md_table(by_round),
        "",
        "## 输出文件",
        "",
        f"- `{COMBINED_CSV.relative_to(ROOT)}`",
        f"- `{DISAGREE_CSV.relative_to(ROOT)}`",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", SUMMARY_MD)
    print("saved:", COMBINED_CSV)
    print("saved:", DISAGREE_CSV)
    print(f"filled={len(filled)} strict={len(strict)} agreement={agreement:.4f}" if len(strict) else "agreement=NA")


if __name__ == "__main__":
    main()
