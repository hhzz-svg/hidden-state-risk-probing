# -*- coding: utf-8 -*-
"""Summarize Optional Experiment A v2 generation and behavior labels."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "outputs/qwen15b/experiment7_optionalA_v2/pk_ck_generation/pk_ck_behavior_labels.csv"
DATA = ROOT / "data/experiment7_optionalA/pk_ck_conflict_v2_expanded.jsonl"
REPORT_DIR = ROOT / "reports/experiment7_optionalA"
TABLE_DIR = REPORT_DIR / "tables"
BEHAVIOR_TABLE = TABLE_DIR / "pk_ck_v2_qwen15b_behavior_summary.csv"
STYLE_TABLE = TABLE_DIR / "pk_ck_v2_qwen15b_by_prompt_style.csv"
CATEGORY_TABLE = TABLE_DIR / "pk_ck_v2_qwen15b_by_category.csv"
REPORT_MD = REPORT_DIR / "experiment7_optionalA_v2_generation_summary.md"


def md_table(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def add_rate(table: pd.DataFrame, group_cols: list[str], count_col: str = "count") -> pd.DataFrame:
    table = table.copy()
    total = table.groupby(group_cols)[count_col].transform("sum") if group_cols else table[count_col].sum()
    table["rate"] = table[count_col] / total
    return table


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LABELS, encoding="utf-8-sig", dtype=str).fillna("")
    meta_rows = []
    with DATA.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                meta_rows.append(
                    {
                        "id": row.get("id", ""),
                        "source_id_meta": row.get("source_id", ""),
                        "prompt_style_meta": row.get("prompt_style", ""),
                        "instruction_strength": row.get("instruction_strength", ""),
                        "split_group": row.get("split_group", ""),
                    }
                )
    meta = pd.DataFrame(meta_rows)
    df = df.merge(meta, on="id", how="left")
    if "prompt_style" not in df.columns:
        df["prompt_style"] = df["prompt_style_meta"]
    else:
        df["prompt_style"] = df["prompt_style"].where(df["prompt_style"].astype(str) != "", df["prompt_style_meta"])
    if "source_id" not in df.columns:
        df["source_id"] = df["source_id_meta"]
    else:
        df["source_id"] = df["source_id"].where(df["source_id"].astype(str) != "", df["source_id_meta"])

    behavior = df["behavior_label"].value_counts().rename_axis("behavior_label").reset_index(name="count")
    behavior["rate"] = behavior["count"] / len(df)

    by_style = (
        df.groupby(["prompt_style", "instruction_strength", "behavior_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["prompt_style", "behavior_label"])
    )
    by_style = add_rate(by_style, ["prompt_style", "instruction_strength"])

    by_category = (
        df.groupby(["category", "behavior_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["category", "behavior_label"])
    )
    by_category = add_rate(by_category, ["category"])

    behavior.to_csv(BEHAVIOR_TABLE, index=False, encoding="utf-8-sig")
    by_style.to_csv(STYLE_TABLE, index=False, encoding="utf-8-sig")
    by_category.to_csv(CATEGORY_TABLE, index=False, encoding="utf-8-sig")

    compact_behavior = behavior.copy()
    compact_behavior["rate"] = compact_behavior["rate"].map(lambda x: f"{x:.3f}")

    target = df[df["behavior_label"].isin(["pk_follow", "ck_follow"])].copy()
    target_counts = target["behavior_label"].value_counts()
    lines = [
        "# 可选实验 A v2 生成行为摘要",
        "",
        "## 范围",
        "",
        "- 数据：`data/experiment7_optionalA/pk_ck_conflict_v2_expanded.jsonl`。",
        "- 模型：Qwen2.5-1.5B-Instruct。",
        f"- 已生成并标注：{len(df)} 条。",
        "- v2 由 100 个 source_id 和 4 种 prompt style 组成，后续可做 source_id group split 或 prompt_style held-out split。",
        "",
        "## 全量行为分布",
        "",
        md_table(compact_behavior),
        "",
        "## 可用于二分类 probe 的子集",
        "",
        f"- pk_follow：{int(target_counts.get('pk_follow', 0))}",
        f"- ck_follow：{int(target_counts.get('ck_follow', 0))}",
        f"- 合计：{len(target)}",
        "",
        "这个数量已经明显强于 v1 的 60 条二分类子集，可作为后续 hidden-state probe 的候选输入。但在进入正式 probe 前，建议先完成 PK/CK 人工复核小样本。",
        "",
        "## 保守解释",
        "",
        "当前标签仍是答案字符串辅助标签，不是人工金标准。`mixed_or_conflict_ack` 数量较多，说明模型经常同时提到材料说法和常识说法；这类样本不应直接并入 pk_follow/ck_follow 二分类目标。",
        "",
        "## 输出表",
        "",
        f"- `{BEHAVIOR_TABLE.relative_to(ROOT)}`",
        f"- `{STYLE_TABLE.relative_to(ROOT)}`",
        f"- `{CATEGORY_TABLE.relative_to(ROOT)}`",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", BEHAVIOR_TABLE)
    print("saved:", STYLE_TABLE)
    print("saved:", CATEGORY_TABLE)
    print("saved:", REPORT_MD)
    print(behavior.to_string(index=False))


if __name__ == "__main__":
    main()
