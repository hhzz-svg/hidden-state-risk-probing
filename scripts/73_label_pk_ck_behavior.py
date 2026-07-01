# -*- coding: utf-8 -*-
"""
73_label_pk_ck_behavior.py

Optional Experiment A: label generated answers as pk_follow / ck_follow.

The labels are heuristic, answer-string-assisted labels. They should be treated
as preliminary unless manually reviewed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIRS = [
    {
        "model_short": "qwen05b",
        "model": "Qwen2.5-0.5B-Instruct",
        "out_dir": ROOT / "outputs/qwen05b/experiment7_optionalA/pk_ck_generation",
    },
    {
        "model_short": "qwen15b",
        "model": "Qwen2.5-1.5B-Instruct",
        "out_dir": ROOT / "outputs/qwen15b/experiment7_optionalA/pk_ck_generation",
    },
]

REPORT_DIR = ROOT / "reports/experiment7_optionalA"
TABLE_DIR = REPORT_DIR / "tables"
SUMMARY_CSV = TABLE_DIR / "pk_ck_behavior_summary.csv"
REPORT_MD = REPORT_DIR / "optionalA_pk_ck_behavior_report.md"


REFUSAL_PATTERNS = [
    "不知道",
    "无法确定",
    "不能确定",
    "无法回答",
    "不能回答",
    "不确定",
    "没有足够",
    "无法提供",
    "不清楚",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = text.replace("₂", "2").replace("₃", "3").replace("₄", "4")
    text = text.replace("₅", "5").replace("₆", "6").replace("₇", "7")
    text = text.replace("₈", "8").replace("₉", "9").replace("₀", "0")
    text = re.sub(r"[\s\u3000，。！？、：；,.!?;:()\[\]{}《》“”\"'`~\-_/\\|]+", "", text)
    return text


def contains_answer(answer_text: str, target: str) -> bool:
    ans = normalize_text(answer_text)
    tgt = normalize_text(target)
    return bool(tgt) and tgt in ans


def label_one(row: pd.Series) -> tuple[str, str]:
    answer = str(row.get("model_answer", "") or "")
    raw = str(row.get("model_answer_raw", "") or "")
    text = answer + "\n" + raw
    pk_answer = str(row.get("pk_answer", "") or "")
    ck_answer = str(row.get("ck_answer", "") or "")

    has_pk = contains_answer(text, pk_answer)
    has_ck = contains_answer(text, ck_answer)
    norm_text = normalize_text(text)
    refusal = any(normalize_text(pattern) in norm_text for pattern in REFUSAL_PATTERNS)

    if has_pk and not has_ck:
        return "pk_follow", "answer matched pk_answer only"
    if has_ck and not has_pk:
        return "ck_follow", "answer matched ck_answer only"
    if has_pk and has_ck:
        return "mixed_or_conflict_ack", "answer mentioned both pk_answer and ck_answer"
    if refusal:
        return "refusal", "answer matched refusal/uncertainty pattern"
    if not normalize_text(answer):
        return "other", "empty or unparsable answer"
    return "other", "answer matched neither pk_answer nor ck_answer"


def label_model(item: dict[str, object]) -> pd.DataFrame:
    out_dir = Path(item["out_dir"])
    generated_path = out_dir / "generated_answers.jsonl"
    if not generated_path.exists():
        raise FileNotFoundError(generated_path)
    rows = load_jsonl(generated_path)
    df = pd.DataFrame(rows)
    labels = df.apply(label_one, axis=1)
    df["behavior_label"] = [label for label, _ in labels]
    df["behavior_label_reason"] = [reason for _, reason in labels]
    df["model_short"] = item["model_short"]
    df["model_display_name"] = item["model"]

    out_path = out_dir / "pk_ck_behavior_labels.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)
    return df


def make_report(all_df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary = (
        all_df.groupby(["model_display_name", "behavior_label"])
        .size()
        .rename("count")
        .reset_index()
    )
    total = all_df.groupby("model_display_name").size().rename("total").reset_index()
    summary = summary.merge(total, on="model_display_name")
    summary["rate"] = summary["count"] / summary["total"]
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    pivot = summary.pivot_table(
        index="model_display_name",
        columns="behavior_label",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    lines = [
        "# 可选实验A：PK/CK 生成行为初步报告",
        "",
        "## 1. 实验目的",
        "",
        "本实验探索当 prompt 中的 contextual knowledge（上下文知识，CK）与常识性 parametric knowledge（参数知识，PK）冲突时，模型最终回答更倾向于哪一类知识来源。",
        "",
        "注意：本实验是扩展探索，不替代主线实验 1-6；这里的标签是答案字符串辅助标注，不是严格人工金标准。",
        "",
        "## 2. 数据与标注",
        "",
        "- 数据：`data/experiment7_optionalA/pk_ck_conflict_v1.jsonl`，100 条。",
        "- 类别：国家-大洲、国家-首都、元素符号、化学式、河流-大洲各 20 条。",
        "- 标签：`pk_follow`、`ck_follow`、`mixed_or_conflict_ack`、`refusal`、`other`。",
        "",
        "## 3. 行为分布",
        "",
        "| 模型 | pk_follow | ck_follow | mixed/conflict | refusal | other |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in pivot.iterrows():
        model = row["model_display_name"]
        lines.append(
            f"| {model} | "
            f"{int(row.get('pk_follow', 0))} | "
            f"{int(row.get('ck_follow', 0))} | "
            f"{int(row.get('mixed_or_conflict_ack', 0))} | "
            f"{int(row.get('refusal', 0))} | "
            f"{int(row.get('other', 0))} |"
        )

    lines.extend(
        [
            "",
            "## 4. 下一步",
            "",
            "如果两个模型中至少有一个模型同时出现足够数量的 `pk_follow` 与 `ck_follow` 样本，可以继续抽取 hidden states 并训练 probe 预测最终 PK/CK 行为。",
            "如果某个模型几乎全是同一类行为，则该模型不适合直接做二分类 probe，应先调整 prompt 强度或扩充数据。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", SUMMARY_CSV)
    print("saved:", REPORT_MD)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_short", choices=["all", "qwen05b", "qwen15b"], default="all")
    args = parser.parse_args()

    frames = []
    for item in MODEL_DIRS:
        if args.model_short != "all" and item["model_short"] != args.model_short:
            continue
        frames.append(label_model(item))

    all_df = pd.concat(frames, ignore_index=True)
    make_report(all_df)


if __name__ == "__main__":
    main()
