# -*- coding: utf-8 -*-
"""
48_fill_experiment4_manual_review.py

Fill Experiment 4 manual-review CSVs with Codex-assisted labels.

This is not a double-blind human annotation. It uses the Experiment 4 answer key,
simple refusal/irrelevance rules, and conservative notes. The reviewer field is
set to "codex_assisted" so the output can be described honestly in reports.
"""

from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANSWER_KEY_SCRIPT = ROOT / "scripts/43_label_generation_behavior_with_answer_key.py"
MANUAL_DIR = ROOT / "reports/experiment4/manual_review"
ALL_FILE = MANUAL_DIR / "experiment4_manual_review_priority_all.csv"
BATCH_FILE = MANUAL_DIR / "experiment4_manual_review_batch200.csv"
SUMMARY_FILE = MANUAL_DIR / "experiment4_manual_review_filled_summary.md"


def load_answer_key_module():
    spec = importlib.util.spec_from_file_location("exp4_answer_key", ANSWER_KEY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ANSWER_KEY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AK = load_answer_key_module()


def norm(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = text.replace("²", "2").replace("³", "3").replace("₂", "2").replace("₃", "3")
    text = text.replace("₄", "4").replace("₅", "5").replace("₆", "6").replace("₇", "7")
    text = text.replace("₈", "8").replace("₉", "9").replace("₀", "0")
    text = text.replace("·", "")
    text = re.sub(r"[\s\-\_，。、“”‘’：:；;,.()（）\[\]【】/\\|]", "", text)
    return text


def answer_matches(answer: str, raw: str, aliases: Iterable[str]) -> tuple[bool, str]:
    # The main answer is the first postprocessed answer line. Raw generations
    # often contain later explanations that mention the correct answer after an
    # incorrect first line, so using raw text would over-credit wrong answers.
    text = norm(answer)
    for alias in aliases:
        alias_n = norm(alias)
        if alias_n and alias_n in text:
            return True, alias
    return False, ""


def is_refusal(answer: str, raw: str) -> bool:
    first = str(answer).strip()
    text = f"{first}\n{raw}"[:240]
    normalized = norm(text)
    refusal_terms = [
        "不知道",
        "不清楚",
        "无法确定",
        "无法回答",
        "不能确定",
        "不能回答",
        "没有直接",
        "没有相关",
        "没有明确",
        "不适用",
        "未知",
        "无",
        "none",
        "notapplicable",
    ]
    if norm(first) in {"无", "没有", "未知", "不适用", "none", "na", "n/a"}:
        return True
    return any(norm(term) in normalized for term in refusal_terms)


def is_irrelevant(answer: str) -> bool:
    first = str(answer).strip()
    if not first:
        return True
    irrelevant_terms = [
        "human:",
        "assistant:",
        "以下选项",
        "问题描述",
        "请问",
        "这个查询有什么问题",
        "小天才",
    ]
    first_n = norm(first[:100])
    return any(norm(term) in first_n for term in irrelevant_terms)


def label_review_row(row: pd.Series) -> tuple[str, str]:
    label = int(row["label"])
    category = str(row["category"])
    entity = str(row["entity"])
    relation = str(row["relation"])
    answer = str(row.get("model_answer", "") or "")
    raw = str(row.get("model_answer_raw", "") or "")

    if is_irrelevant(answer):
        return "irrelevant", "codex-assisted: empty or prompt-like unrelated answer"
    if is_refusal(answer, raw):
        return "refusal", "codex-assisted: refusal/uncertainty/no-answer expression"

    if label == 1:
        return "hallucination", "codex-assisted: high-risk mismatched entity-relation query received a specific answer"

    expected = AK.ANSWER_KEY.get((category, entity))
    if not expected:
        return "uncertain", f"codex-assisted: missing answer key for {category}/{entity}/{relation}"

    matched, alias = answer_matches(answer, raw, expected)
    if matched:
        return "correct", f"codex-assisted: matched expected alias '{alias}'"

    expected_text = "; ".join(str(x) for x in expected[:5])
    return "hallucination", f"codex-assisted: expected {expected_text}; model answer appears factual but mismatched"


def fill_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    labels = df.apply(label_review_row, axis=1)
    df["manual_behavior_label"] = [label for label, _ in labels]
    df["manual_notes"] = [note for _, note in labels]
    df["manual_reviewer"] = "codex_assisted"
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return df


def main() -> None:
    all_df = fill_csv(ALL_FILE)
    batch_df = fill_csv(BATCH_FILE)

    all_summary = all_df["manual_behavior_label"].value_counts().rename_axis("label").reset_index(name="count")
    batch_summary = batch_df["manual_behavior_label"].value_counts().rename_axis("label").reset_index(name="count")
    by_model = (
        all_df.groupby(["model_short", "manual_behavior_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["model_short", "manual_behavior_label"])
    )

    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# 第四次实验 Codex-assisted 人工复核填表摘要",
        "",
        "## 边界说明",
        "",
        "本次填写是 Codex-assisted manual review（AI 辅助人工复核），不是双人独立人工标注。论文中建议表述为“AI 辅助复核后的行为标签”，不要写成严格人工金标准。",
        "",
        "## 已填写文件",
        "",
        f"- `{ALL_FILE.relative_to(ROOT)}`",
        f"- `{BATCH_FILE.relative_to(ROOT)}`",
        "",
        "## 全部候选标签分布",
        "",
        md_table(all_summary),
        "",
        "## Batch200 标签分布",
        "",
        md_table(batch_summary),
        "",
        "## 按模型统计",
        "",
        md_table(by_model),
    ]
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("filled:", ALL_FILE)
    print("filled:", BATCH_FILE)
    print("saved:", SUMMARY_FILE)
    print("all summary:")
    print(all_summary.to_string(index=False))
    print("batch summary:")
    print(batch_summary.to_string(index=False))


if __name__ == "__main__":
    main()
