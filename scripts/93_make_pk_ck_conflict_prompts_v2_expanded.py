# -*- coding: utf-8 -*-
"""Expand Optional Experiment A PK/CK prompts from v1 to a paraphrased v2 set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data/experiment7_optionalA/pk_ck_conflict_v1.jsonl"
OUT = ROOT / "data/experiment7_optionalA/pk_ck_conflict_v2_expanded.jsonl"
REPORT_DIR = ROOT / "reports/experiment7_optionalA"
TABLE_DIR = REPORT_DIR / "tables"
PREVIEW = TABLE_DIR / "pk_ck_conflict_v2_expanded_preview.csv"
PLAN_MD = REPORT_DIR / "experiment7_optionalA_expansion_plan_v2.md"


STYLES = [
    {
        "name": "quoted_context_then_question",
        "prompt": "背景：{context}\n问题：{question}\n答案：",
        "instruction_strength": "neutral",
    },
    {
        "name": "use_material_first",
        "prompt": "请只根据下面材料回答。\n材料：{context}\n问：{question}\n答：",
        "instruction_strength": "context_priority",
    },
    {
        "name": "brief_qa",
        "prompt": "材料：{context}\n{question}\n请给出简短答案：",
        "instruction_strength": "neutral_short",
    },
    {
        "name": "conflict_explicit",
        "prompt": "如果材料与常识不一致，请优先复述材料中的说法。\n材料：{context}\n问题：{question}\n答案：",
        "instruction_strength": "explicit_context_priority",
    },
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    base_rows = load_jsonl(V1)
    out_rows = []
    for base in base_rows:
        source_id = str(base["id"])
        for style_idx, style in enumerate(STYLES):
            row = dict(base)
            row["id"] = f"{source_id}_s{style_idx}"
            row["source_id"] = source_id
            row["prompt"] = style["prompt"].format(context=base["context"], question=base["question"])
            row["prompt_style"] = style["name"]
            row["instruction_strength"] = style["instruction_strength"]
            row["split_group"] = source_id
            row["expected_later_splits"] = "source_id_group_split|prompt_style_heldout_split|category_heldout_split"
            out_rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(out_rows)
    df.head(40).to_csv(PREVIEW, index=False, encoding="utf-8-sig")
    summary = (
        df.groupby(["category", "prompt_style", "instruction_strength"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["category", "prompt_style"])
    )

    lines = [
        "# 可选实验 A 扩样计划 v2",
        "",
        "## 已完成",
        "",
        f"- v1 基础样本：{len(base_rows)} 条。",
        f"- v2 扩展样本：{len(out_rows)} 条，来自 100 个 source_id × 4 种 prompt style。",
        f"- 数据：`{OUT.relative_to(ROOT)}`",
        f"- 预览：`{PREVIEW.relative_to(ROOT)}`",
        "",
        "## 为什么这样扩",
        "",
        "v1 的主要弱点是样本少、prompt 模板单一，而且 TF-IDF baseline 不低。v2 不把同一个事实当作独立样本来夸大证据，而是显式加入 `source_id` 和 `prompt_style`，后续可以做 group split 和 held-out prompt style split。",
        "",
        "## 后续运行顺序",
        "",
        "1. 用 v2 数据重新生成 Qwen2.5-1.5B 和 0.5B 答案。",
        "2. 用 flexible label 脚本标注 `pk_follow / ck_follow / mixed_or_conflict_ack / refusal / other`。",
        "3. 抽样人工复核 PK/CK 标签，优先复核 mixed、other、以及 pk/ck 分界不清的样本。",
        "4. 只在 pk_follow 与 ck_follow 数量都足够时抽取 hidden states 并训练 probe。",
        "5. 至少报告 source_id group split；若样本数允许，再报告 prompt_style held-out split。",
        "",
        "## 建议命令",
        "",
        "```powershell",
        "E:\\anaconda3\\anaconda\\Scripts\\conda.exe run -n my_torch python hidden_state_pilot\\scripts\\72_generate_pk_ck_answers.py --model_name Qwen/Qwen2.5-1.5B-Instruct --data_file hidden_state_pilot\\data\\experiment7_optionalA\\pk_ck_conflict_v2_expanded.jsonl --out_dir hidden_state_pilot\\outputs\\qwen15b\\experiment7_optionalA_v2\\pk_ck_generation --local_files_only --overwrite",
        "E:\\anaconda3\\anaconda\\Scripts\\conda.exe run -n my_torch python hidden_state_pilot\\scripts\\94_label_pk_ck_behavior_flexible.py --generated_file hidden_state_pilot\\outputs\\qwen15b\\experiment7_optionalA_v2\\pk_ck_generation\\generated_answers.jsonl --out_csv hidden_state_pilot\\outputs\\qwen15b\\experiment7_optionalA_v2\\pk_ck_generation\\pk_ck_behavior_labels.csv --model_short qwen15b --model_display_name Qwen2.5-1.5B-Instruct",
        "```",
        "",
        "## 分布",
        "",
        "| category | prompt_style | instruction_strength | count |",
        "| --- | --- | --- | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['category']} | {row['prompt_style']} | {row['instruction_strength']} | {int(row['count'])} |"
        )
    PLAN_MD.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", OUT)
    print("saved:", PREVIEW)
    print("saved:", PLAN_MD)
    print("rows:", len(out_rows))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
