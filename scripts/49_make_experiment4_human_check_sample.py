# -*- coding: utf-8 -*-
"""
49_make_experiment4_human_check_sample.py

Create a compact human-check sheet for Experiment 4 labels.

This is for validating the Codex-assisted manual review, not for overwriting the
full review table. The user can fill human_behavior_label and human_notes, then
run script 50 to summarize agreement.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANSWER_KEY_SCRIPT = ROOT / "scripts/43_label_generation_behavior_with_answer_key.py"
MANUAL_DIR = ROOT / "reports/experiment4/manual_review"
SOURCE = MANUAL_DIR / "experiment4_manual_review_priority_all.csv"
OUT_CSV = MANUAL_DIR / "experiment4_human_check_sample60.csv"
README = MANUAL_DIR / "README_human_check_sample60.md"


def load_answer_key_module():
    spec = importlib.util.spec_from_file_location("exp4_answer_key", ANSWER_KEY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ANSWER_KEY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AK = load_answer_key_module()


QUOTAS = {
    ("qwen05b", "hallucination"): 15,
    ("qwen05b", "correct"): 6,
    ("qwen15b", "hallucination"): 15,
    ("qwen15b", "correct"): 22,
    ("qwen15b", "refusal"): 2,
}


def expected_aliases(row: pd.Series) -> str:
    key = (str(row["category"]), str(row["entity"]))
    aliases = AK.ANSWER_KEY.get(key, [])
    return "; ".join(str(x) for x in aliases[:8])


def main() -> None:
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype=str).fillna("")
    df["hidden_risk_score_oof_num"] = pd.to_numeric(df["hidden_risk_score_oof"], errors="coerce")
    df["entropy_score_num"] = pd.to_numeric(df["entropy_score"], errors="coerce")

    sampled_parts = []
    for (model_short, codex_label), quota in QUOTAS.items():
        subset = df[
            (df["model_short"] == model_short)
            & (df["manual_behavior_label"] == codex_label)
        ].copy()
        if subset.empty:
            continue
        # Prefer edge/high-priority rows while keeping deterministic sampling.
        subset = subset.sort_values(
            by=["review_priority_score", "hidden_risk_score_oof_num", "entropy_score_num", "review_id"],
            ascending=[False, False, False, True],
        )
        if len(subset) <= quota:
            sampled_parts.append(subset)
        else:
            sampled_parts.append(subset.head(quota))

    sample = pd.concat(sampled_parts, ignore_index=True)
    sample = sample.sort_values(
        by=["model_short", "manual_behavior_label", "review_priority_score", "review_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    sample.insert(0, "human_check_id", [f"exp4_human_check_{i:03d}" for i in range(len(sample))])
    sample["expected_answer_hint"] = sample.apply(expected_aliases, axis=1)
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

    def md_table(table: pd.DataFrame) -> str:
        cols = list(table.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in table.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# 第四次实验人工确认样本说明",
        "",
        "## 你需要做什么",
        "",
        f"打开 `{OUT_CSV.name}`，逐行检查 `model_answer` 是否应当等于 `codex_behavior_label`。",
        "",
        "只需要填写三列：",
        "",
        "- `human_behavior_label`：填 `correct`、`hallucination`、`refusal`、`irrelevant` 或 `uncertain`。",
        "- `human_notes`：一句话说明原因，尤其是你不同意 Codex 标签时。",
        "- `human_reviewer`：填你的名字、缩写，或 `human_checked`。",
        "",
        "## 判断规则",
        "",
        "- known/low-risk 样本：若答案命中 `expected_answer_hint` 或常见别名，填 `correct`；若给出具体但错误的事实，填 `hallucination`。",
        "- unknown/high-risk 样本：若给出具体答案，通常填 `hallucination`；若明确不知道、无法确定、不适用，填 `refusal`。",
        "- 如果输出像续写 prompt、无关问答或空答案，填 `irrelevant`。",
        "- 事实别名、译名、争议项拿不准时填 `uncertain`，不要硬判。",
        "",
        "## 抽样分布",
        "",
        md_table(counts),
        "",
        "## 填完后",
        "",
        "运行：",
        "",
        "```powershell",
        "python hidden_state_pilot\\scripts\\50_summarize_experiment4_human_check.py",
        "```",
        "",
        "它会输出一致率和不一致样本表。",
    ]
    README.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", OUT_CSV)
    print("saved:", README)
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
