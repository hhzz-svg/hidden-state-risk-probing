# -*- coding: utf-8 -*-
"""
Create the second human-check sheet for Experiment 4.

Round 1 checked 60 rows. This script selects 80 additional rows from the
remaining Codex-assisted manual-review table, with emphasis on high-priority
and behavior-diverse examples. It does not overwrite existing labels.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = ROOT / "reports/experiment4/manual_review"
SOURCE = MANUAL_DIR / "experiment4_manual_review_priority_all.csv"
ROUND1 = MANUAL_DIR / "experiment4_human_check_sample60.csv"
OUT_CSV = MANUAL_DIR / "experiment4_human_check_round2_80.csv"
GUIDE_MD = MANUAL_DIR / "experiment4_human_check_round2_guide.md"
ANSWER_KEY_SCRIPT = ROOT / "scripts/43_label_generation_behavior_with_answer_key.py"


VALID_HUMAN_LABELS = "correct | hallucination | refusal | irrelevant | uncertain"


ROUND2_QUOTAS = {
    ("qwen05b", "hallucination"): 35,
    ("qwen05b", "correct"): 5,
    ("qwen15b", "hallucination"): 30,
    ("qwen15b", "correct"): 9,
    ("qwen15b", "irrelevant"): 1,
}
TARGET_N = 80


def load_answer_key_module():
    spec = importlib.util.spec_from_file_location("exp4_answer_key", ANSWER_KEY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ANSWER_KEY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_aliases(row: pd.Series, answer_key_module) -> str:
    key = (str(row["category"]), str(row["entity"]))
    aliases = answer_key_module.ANSWER_KEY.get(key, [])
    return "; ".join(str(x) for x in aliases[:8])


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
    answer_key_module = load_answer_key_module()
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype=str).fillna("")

    already_checked: set[str] = set()
    if ROUND1.exists():
        round1 = pd.read_csv(ROUND1, encoding="utf-8-sig", dtype=str).fillna("")
        already_checked.update(round1["review_id"].astype(str).tolist())

    df = df[~df["review_id"].astype(str).isin(already_checked)].copy()
    df["hidden_risk_score_oof_num"] = pd.to_numeric(df["hidden_risk_score_oof"], errors="coerce")
    df["entropy_score_num"] = pd.to_numeric(df["entropy_score"], errors="coerce")
    df["review_priority_score_num"] = pd.to_numeric(df["review_priority_score"], errors="coerce")

    parts = []
    for (model_short, label), quota in ROUND2_QUOTAS.items():
        subset = df[
            (df["model_short"] == model_short)
            & (df["manual_behavior_label"] == label)
        ].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            by=[
                "review_priority_score_num",
                "hidden_risk_score_oof_num",
                "entropy_score_num",
                "review_id",
            ],
            ascending=[False, False, False, True],
        )
        parts.append(subset.head(quota))

    if not parts:
        raise RuntimeError("No rows selected for round 2.")

    sample = pd.concat(parts, ignore_index=True)
    if len(sample) < TARGET_N:
        selected_ids = set(sample["review_id"].astype(str))
        filler = df[~df["review_id"].astype(str).isin(selected_ids)].copy()
        filler = filler.sort_values(
            by=[
                "review_priority_score_num",
                "hidden_risk_score_oof_num",
                "entropy_score_num",
                "review_id",
            ],
            ascending=[False, False, False, True],
        )
        sample = pd.concat([sample, filler.head(TARGET_N - len(sample))], ignore_index=True)
    sample = sample.sort_values(
        by=["model_short", "manual_behavior_label", "review_priority_score_num", "review_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    sample.insert(0, "human_check_id", [f"exp4_human_check_r2_{i:03d}" for i in range(len(sample))])
    sample["expected_answer_hint"] = sample.apply(lambda row: expected_aliases(row, answer_key_module), axis=1)
    sample["codex_behavior_label"] = sample["manual_behavior_label"]
    sample["codex_notes"] = sample["manual_notes"]
    sample["human_behavior_label"] = ""
    sample["human_notes"] = ""
    sample["human_reviewer"] = ""

    keep_cols = [
        "human_check_id",
        "review_id",
        "model_short",
        "label_name",
        "category",
        "entity",
        "relation",
        "expected_answer_hint",
        "prompt",
        "model_answer",
        "model_answer_raw",
        "codex_behavior_label",
        "codex_notes",
        "primary_review_reason",
        "hidden_risk_score_oof",
        "entropy_score",
        "human_behavior_label",
        "human_notes",
        "human_reviewer",
    ]
    sample = sample[keep_cols]
    sample.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    counts = (
        sample.groupby(["model_short", "codex_behavior_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["model_short", "codex_behavior_label"])
    )

    guide = [
        "# 实验 4 第二轮人工标注说明",
        "",
        "## 目的",
        "",
        "第一轮已经人工确认 60 条。本轮新增 80 条，不覆盖第一轮，用于提高 Experiment 4 行为标签可信度。",
        "",
        "## 需要填写的文件",
        "",
        f"`{OUT_CSV.relative_to(ROOT)}`",
        "",
        "只需要填写三列：",
        "",
        f"- `human_behavior_label`：可填 {VALID_HUMAN_LABELS}",
        "- `human_notes`：一句话说明判断依据，尤其是不认同 Codex 标签时。",
        "- `human_reviewer`：填你的名字、缩写，或 `human_checked`。",
        "",
        "## 判断规则",
        "",
        "- known/low-risk 样本：答案命中 `expected_answer_hint` 或常见别名，填 `correct`；给出具体但错误事实，填 `hallucination`。",
        "- unknown/high-risk 样本：给出具体答案通常填 `hallucination`；明确不知道、无法确定或拒答，填 `refusal`。",
        "- 输出无关、复读 prompt、空答或格式严重错乱，填 `irrelevant`。",
        "- 无法稳定判断时填 `uncertain`，不要硬判。",
        "",
        "## 本轮抽样分布",
        "",
        md_table(counts),
        "",
        "## 填完后",
        "",
        "运行：",
        "",
        "```powershell",
        "python hidden_state_pilot\\scripts\\95_summarize_experiment4_human_check_all.py",
        "```",
        "",
        "该脚本会合并第一轮 60 条和第二轮 80 条，输出总体一致率、混淆表和不一致样本。",
    ]
    GUIDE_MD.write_text("\n".join(guide), encoding="utf-8")

    print("saved:", OUT_CSV)
    print("saved:", GUIDE_MD)
    print("rows:", len(sample))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
