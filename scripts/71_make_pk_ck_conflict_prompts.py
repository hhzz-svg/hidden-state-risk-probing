# -*- coding: utf-8 -*-
"""
71_make_pk_ck_conflict_prompts.py

Optional Experiment A: PK/CK knowledge-conflict prompts.

PK means parametric knowledge: the ordinary factual answer expected from the
model's learned parameters. CK means contextual knowledge: a conflicting answer
provided inside the prompt context.

This script only creates the prompt dataset. It does not run generation and it
does not assign pk_follow/ck_follow labels. Those labels should come from the
model's actual generated answers in later steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT / "data/experiment7_optionalA/pk_ck_conflict_v1.jsonl"
REPORT_DIR = ROOT / "reports/experiment7_optionalA"
PREVIEW_OUT = REPORT_DIR / "tables/pk_ck_conflict_v1_preview.csv"
SUMMARY_OUT = REPORT_DIR / "optionalA_pk_ck_conflict_data_summary.md"


EXAMPLES = {
    "country_continent": {
        "relation": "所在大洲",
        "items": [
            ("巴西", "南美洲"),
            ("加拿大", "北美洲"),
            ("埃及", "非洲"),
            ("日本", "亚洲"),
            ("德国", "欧洲"),
            ("澳大利亚", "大洋洲"),
            ("肯尼亚", "非洲"),
            ("阿根廷", "南美洲"),
            ("墨西哥", "北美洲"),
            ("印度", "亚洲"),
            ("法国", "欧洲"),
            ("韩国", "亚洲"),
            ("尼日利亚", "非洲"),
            ("秘鲁", "南美洲"),
            ("西班牙", "欧洲"),
            ("泰国", "亚洲"),
            ("摩洛哥", "非洲"),
            ("智利", "南美洲"),
            ("越南", "亚洲"),
            ("瑞典", "欧洲"),
        ],
    },
    "country_capital": {
        "relation": "首都",
        "items": [
            ("法国", "巴黎"),
            ("日本", "东京"),
            ("加拿大", "渥太华"),
            ("澳大利亚", "堪培拉"),
            ("德国", "柏林"),
            ("意大利", "罗马"),
            ("西班牙", "马德里"),
            ("巴西", "巴西利亚"),
            ("阿根廷", "布宜诺斯艾利斯"),
            ("埃及", "开罗"),
            ("肯尼亚", "内罗毕"),
            ("印度", "新德里"),
            ("墨西哥", "墨西哥城"),
            ("泰国", "曼谷"),
            ("智利", "圣地亚哥"),
            ("瑞典", "斯德哥尔摩"),
            ("挪威", "奥斯陆"),
            ("希腊", "雅典"),
            ("葡萄牙", "里斯本"),
            ("韩国", "首尔"),
        ],
    },
    "element_symbol": {
        "relation": "元素符号",
        "items": [
            ("氧", "O"),
            ("氢", "H"),
            ("钠", "Na"),
            ("铁", "Fe"),
            ("金", "Au"),
            ("银", "Ag"),
            ("碳", "C"),
            ("氮", "N"),
            ("氦", "He"),
            ("铜", "Cu"),
            ("钙", "Ca"),
            ("钾", "K"),
            ("氯", "Cl"),
            ("镁", "Mg"),
            ("锌", "Zn"),
            ("铅", "Pb"),
            ("锡", "Sn"),
            ("汞", "Hg"),
            ("硅", "Si"),
            ("硫", "S"),
        ],
    },
    "chemical_formula": {
        "relation": "化学式",
        "items": [
            ("水", "H2O"),
            ("二氧化碳", "CO2"),
            ("甲烷", "CH4"),
            ("氨", "NH3"),
            ("氯化钠", "NaCl"),
            ("硫酸", "H2SO4"),
            ("盐酸", "HCl"),
            ("硝酸", "HNO3"),
            ("葡萄糖", "C6H12O6"),
            ("乙醇", "C2H5OH"),
            ("乙酸", "CH3COOH"),
            ("碳酸钙", "CaCO3"),
            ("氢氧化钠", "NaOH"),
            ("硝酸钾", "KNO3"),
            ("臭氧", "O3"),
            ("过氧化氢", "H2O2"),
            ("氧气", "O2"),
            ("氮气", "N2"),
            ("一氧化碳", "CO"),
            ("碳酸氢钠", "NaHCO3"),
        ],
    },
    "river_continent": {
        "relation": "主要流经大洲",
        "items": [
            ("尼罗河", "非洲"),
            ("亚马孙河", "南美洲"),
            ("长江", "亚洲"),
            ("密西西比河", "北美洲"),
            ("多瑙河", "欧洲"),
            ("恒河", "亚洲"),
            ("刚果河", "非洲"),
            ("伏尔加河", "欧洲"),
            ("湄公河", "亚洲"),
            ("莱茵河", "欧洲"),
            ("印度河", "亚洲"),
            ("尼日尔河", "非洲"),
            ("墨累河", "大洋洲"),
            ("奥里诺科河", "南美洲"),
            ("赞比西河", "非洲"),
            ("塞纳河", "欧洲"),
            ("泰晤士河", "欧洲"),
            ("底格里斯河", "亚洲"),
            ("幼发拉底河", "亚洲"),
            ("科罗拉多河", "北美洲"),
        ],
    },
}


def rotated_false_answer(items: list[tuple[str, str]], index: int, offset: int = 7) -> str:
    answers = [answer for _, answer in items]
    candidate = answers[(index + offset) % len(answers)]
    if candidate == answers[index]:
        candidate = answers[(index + offset + 1) % len(answers)]
    return candidate


def make_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    idx = 0

    for category, spec in EXAMPLES.items():
        relation = spec["relation"]
        items = spec["items"]
        for item_idx, (entity, pk_answer) in enumerate(items):
            ck_answer = rotated_false_answer(items, item_idx)
            context = f"材料中写道：{entity}的{relation}是{ck_answer}。"
            question = f"{entity}的{relation}是什么？"
            prompt = f"背景：{context}\n问题：{question}\n答案："

            rows.append(
                {
                    "id": f"pkck_{idx:04d}",
                    "entity": entity,
                    "relation": relation,
                    "category": category,
                    "pk_answer": pk_answer,
                    "ck_answer": ck_answer,
                    "context": context,
                    "question": question,
                    "prompt": prompt,
                    "conflict_type": "false_context_conflict",
                    "prompt_style": "quoted_context_then_question",
                    "prompt_language": "zh",
                    "expected_later_labels": "pk_follow|ck_follow|other|refusal|irrelevant",
                }
            )
            idx += 1

    return rows


def main() -> None:
    rows = make_records()
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)

    with DATA_OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(PREVIEW_OUT, index=False, encoding="utf-8-sig")

    summary = df.groupby("category").size().rename("count").reset_index()
    summary_lines = [
        "# 可选实验A：PK/CK 冲突数据 v1",
        "",
        "## 定位",
        "",
        "该数据集用于探索：当 prompt 中给出的 contextual knowledge（上下文知识，CK）与常识性 parametric knowledge（参数知识，PK）冲突时，模型最终回答更倾向于 PK 还是 CK。",
        "",
        "注意：这不是主线 known/unknown query-level risk label，也不是 hallucination label。后续标签必须来自模型真实生成答案。",
        "",
        "## 输出",
        "",
        f"- 数据：`{DATA_OUT.relative_to(ROOT)}`",
        f"- 预览表：`{PREVIEW_OUT.relative_to(ROOT)}`",
        "",
        "## 字段",
        "",
        "- `pk_answer`：通常事实答案，可视为参数知识方向。",
        "- `ck_answer`：prompt 背景中人为提供的冲突答案，可视为上下文知识方向。",
        "- `prompt`：用于模型生成的完整输入。",
        "- `expected_later_labels`：后续生成后再标注的行为标签集合。",
        "",
        "## 类别统计",
        "",
        "| category | count |",
        "| --- | ---: |",
    ]
    for _, row in summary.iterrows():
        summary_lines.append(f"| {row['category']} | {row['count']} |")

    summary_lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 用两个 Qwen2.5 模型生成答案。",
            "2. 根据答案是否匹配 `pk_answer` 或 `ck_answer` 标注 `pk_follow` / `ck_follow` / `other` / `refusal` / `irrelevant`。",
            "3. 抽取生成前 hidden states，训练 probe 预测最终 PK/CK 行为，并与 entropy、TF-IDF、长度基线比较。",
        ]
    )
    SUMMARY_OUT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("saved:", DATA_OUT)
    print("saved:", PREVIEW_OUT)
    print("saved:", SUMMARY_OUT)
    print("num_samples:", len(rows))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
