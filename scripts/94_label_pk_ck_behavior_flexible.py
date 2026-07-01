# -*- coding: utf-8 -*-
"""Flexible PK/CK behavior labeler for Experiment A v2 outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


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

ANSWER_ALIASES = {
    "CH3COOH": ["CH3COOH", "C2H4O2", "醋酸"],
    "首尔": ["首尔", "汉城", "Seoul"],
}

CONFLICT_ACK_PATTERNS = [
    "与常识不符",
    "和常识不符",
    "不符合常识",
    "常识不符",
    "该选项与常识",
    "材料与常识",
    "题目中的信息与常识",
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
    text = re.sub(r"[\s\u3000，。！？、：；,.!?;:()\[\]{}《》“”\"'`~\-_/\\|]+", "", text)
    return text


def contains_answer(answer_text: str, target: str) -> bool:
    targets = [target, *ANSWER_ALIASES.get(str(target), [])]
    for tgt in targets:
        tgt = str(tgt)
        norm_tgt = normalize_text(tgt)
        if not norm_tgt:
            continue
        if re.fullmatch(r"[A-Za-z0-9]+", tgt):
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(tgt)}(?![A-Za-z0-9])", str(answer_text), flags=re.IGNORECASE):
                return True
        elif norm_tgt in normalize_text(answer_text):
            return True
    return False


def has_conflict_ack(text: str) -> bool:
    norm_text = normalize_text(text)
    return any(normalize_text(pattern) in norm_text for pattern in CONFLICT_ACK_PATTERNS)


def label_one(row: pd.Series) -> tuple[str, str]:
    answer = str(row.get("model_answer", "") or "")
    raw = str(row.get("model_answer_raw", "") or "")
    raw_without_prompt_echo = raw.split("Human:", 1)[0]
    primary_text = answer.strip() or raw_without_prompt_echo.strip()
    fallback_text = raw_without_prompt_echo.strip()
    pk_answer = str(row.get("pk_answer", "") or "")
    ck_answer = str(row.get("ck_answer", "") or "")

    has_pk = contains_answer(primary_text, pk_answer)
    has_ck = contains_answer(primary_text, ck_answer)
    conflict_ack = has_conflict_ack(primary_text)
    fallback_has_pk = contains_answer(fallback_text, pk_answer) if fallback_text else False
    fallback_has_ck = contains_answer(fallback_text, ck_answer) if fallback_text else False
    fallback_conflict_ack = has_conflict_ack(fallback_text) if fallback_text else False
    if not has_pk and not has_ck and fallback_text:
        has_pk = fallback_has_pk
        has_ck = fallback_has_ck
        conflict_ack = fallback_conflict_ack
    norm_text = normalize_text(primary_text + "\n" + fallback_text)
    refusal = any(normalize_text(pattern) in norm_text for pattern in REFUSAL_PATTERNS)

    if (has_pk or has_ck) and fallback_has_pk and fallback_has_ck:
        return "mixed_or_conflict_ack", "answer text mentioned both pk_answer and ck_answer"
    if (has_pk or has_ck) and (conflict_ack or fallback_conflict_ack):
        return "mixed_or_conflict_ack", "answer followed or mentioned ck_answer while explicitly acknowledging conflict"
    if has_pk and not has_ck:
        return "pk_follow", "answer matched pk_answer only"
    if has_ck and not has_pk:
        return "ck_follow", "answer matched ck_answer only"
    if has_pk and has_ck:
        return "mixed_or_conflict_ack", "answer mentioned both pk_answer and ck_answer"
    if refusal:
        return "refusal", "answer matched refusal or uncertainty pattern"
    if not normalize_text(answer):
        return "other", "empty or unparsable answer"
    return "other", "answer matched neither pk_answer nor ck_answer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_file", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--model_short", required=True)
    parser.add_argument("--model_display_name", required=True)
    args = parser.parse_args()

    generated_file = Path(args.generated_file)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(load_jsonl(generated_file))
    labels = df.apply(label_one, axis=1)
    df["behavior_label"] = [label for label, _ in labels]
    df["behavior_label_reason"] = [reason for _, reason in labels]
    df["model_short"] = args.model_short
    df["model_display_name"] = args.model_display_name
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby(["model_short", "behavior_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["model_short", "behavior_label"])
    )
    print("saved:", out_csv)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
