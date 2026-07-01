# -*- coding: utf-8 -*-
"""Create a manual-review sample for Optional Experiment A v2 PK/CK labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "outputs/qwen15b/experiment7_optionalA_v2/pk_ck_generation/pk_ck_behavior_labels.csv"
REPORT_DIR = ROOT / "reports/experiment7_optionalA"
OUT_CSV = REPORT_DIR / "pk_ck_v2_manual_review_sample44.csv"
GUIDE_MD = REPORT_DIR / "pk_ck_v2_manual_review_guide.md"


QUOTAS = {
    "mixed_or_conflict_ack": 16,
    "ck_follow": 12,
    "pk_follow": 12,
    "other": 6,
    "refusal": 4,
}


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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LABELS, encoding="utf-8-sig", dtype=str).fillna("")
    parts = []
    for label, quota in QUOTAS.items():
        sub = df[df["behavior_label"] == label].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["category", "id"]).head(quota)
        parts.append(sub)
    if not parts:
        raise RuntimeError("No rows available for manual review.")

    sample = pd.concat(parts, ignore_index=True).reset_index(drop=True)
    sample.insert(0, "pkck_review_id", [f"pkck_v2_review_{i:03d}" for i in range(len(sample))])
    sample["auto_pkck_label"] = sample["behavior_label"]
    sample["human_pkck_label"] = ""
    sample["human_notes"] = ""
    sample["human_reviewer"] = ""

    keep_cols = [
        "pkck_review_id",
        "id",
        "source_id",
        "prompt_style",
        "instruction_strength",
        "category",
        "entity",
        "relation",
        "pk_answer",
        "ck_answer",
        "prompt",
        "model_answer",
        "model_answer_raw",
        "auto_pkck_label",
        "behavior_label_reason",
        "human_pkck_label",
        "human_notes",
        "human_reviewer",
    ]
    keep_cols = [col for col in keep_cols if col in sample.columns]
    sample = sample[keep_cols]
    sample.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    counts = sample["auto_pkck_label"].value_counts().rename_axis("auto_pkck_label").reset_index(name="count")
    guide = [
        "# 可选实验 A v2 PK/CK 人工复核说明",
        "",
        "## 文件",
        "",
        f"`{OUT_CSV.relative_to(ROOT)}`",
        "",
        "## 需要填写",
        "",
        "- `human_pkck_label`：可填 `pk_follow`、`ck_follow`、`mixed_or_conflict_ack`、`refusal`、`other`、`uncertain`。",
        "- `human_notes`：一句话说明理由。",
        "- `human_reviewer`：填你的名字、缩写，或 `human_checked`。",
        "",
        "## 判断规则",
        "",
        "- 只跟随常识/参数知识答案 `pk_answer`，填 `pk_follow`。",
        "- 只跟随 prompt 材料里的冲突答案 `ck_answer`，填 `ck_follow`。",
        "- 同时提到 PK 和 CK，或承认材料冲突后仍给双重解释，填 `mixed_or_conflict_ack`。",
        "- 明确说无法判断、信息不足、不回答，填 `refusal`。",
        "- 不相关、空答、复读 prompt 或完全无法解析，填 `other`。",
        "- 模棱两可就填 `uncertain`。",
        "",
        "## 抽样分布",
        "",
        md_table(counts),
    ]
    GUIDE_MD.write_text("\n".join(guide), encoding="utf-8")
    print("saved:", OUT_CSV)
    print("saved:", GUIDE_MD)
    print("rows:", len(sample))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
