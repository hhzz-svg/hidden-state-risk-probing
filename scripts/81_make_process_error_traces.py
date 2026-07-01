# -*- coding: utf-8 -*-
"""
81_make_process_error_traces.py

Optional Experiment B: create synthetic process-error traces.

Each record is one step-prefix sample. Corrupted traces have binary labels:
- 0: before_error
- 1: at_or_after_error

Correct traces are kept as label=-1 / no_error_control and should not be mixed
into the first binary probe target.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT / "data/experiment8_optionalB/process_error_traces_v1.jsonl"
REPORT_DIR = ROOT / "reports/experiment8_optionalB"
TABLE_DIR = REPORT_DIR / "tables"
PREVIEW_OUT = TABLE_DIR / "process_error_trace_preview.csv"
SUMMARY_OUT = REPORT_DIR / "optionalB_process_error_data_summary.md"


@dataclass
class TraceTemplate:
    problem: str
    correct_steps: list[str]
    corrupted_steps: list[str]
    error_step_index: int
    error_type: str
    correct_answer: str
    corrupted_answer: str


def make_sum3(rng: random.Random) -> TraceTemplate:
    a = rng.randint(12, 79)
    b = rng.randint(11, 68)
    c = rng.randint(6, 45)
    s1 = a + b
    ans = s1 + c
    delta = rng.choice([-3, -2, -1, 1, 2, 3])
    wrong_ans = ans + delta
    return TraceTemplate(
        problem=f"计算 {a} + {b} + {c}。",
        correct_steps=[
            f"步骤1：先计算 {a} + {b} = {s1}。",
            f"步骤2：再计算 {s1} + {c} = {ans}。",
            f"答案：{ans}。",
        ],
        corrupted_steps=[
            f"步骤1：先计算 {a} + {b} = {s1}。",
            f"步骤2：再计算 {s1} + {c} = {wrong_ans}。",
            f"答案：{wrong_ans}。",
        ],
        error_step_index=2,
        error_type="final_addition_error",
        correct_answer=str(ans),
        corrupted_answer=str(wrong_ans),
    )


def make_mul_add(rng: random.Random) -> TraceTemplate:
    a = rng.randint(3, 18)
    b = rng.randint(4, 16)
    c = rng.randint(5, 60)
    prod = a * b
    ans = prod + c
    delta = rng.choice([-4, -3, -2, 2, 3, 4])
    wrong_ans = ans + delta
    return TraceTemplate(
        problem=f"计算 {a} × {b} + {c}。",
        correct_steps=[
            f"步骤1：先计算 {a} × {b} = {prod}。",
            f"步骤2：再计算 {prod} + {c} = {ans}。",
            f"答案：{ans}。",
        ],
        corrupted_steps=[
            f"步骤1：先计算 {a} × {b} = {prod}。",
            f"步骤2：再计算 {prod} + {c} = {wrong_ans}。",
            f"答案：{wrong_ans}。",
        ],
        error_step_index=2,
        error_type="final_addition_error",
        correct_answer=str(ans),
        corrupted_answer=str(wrong_ans),
    )


def make_parentheses(rng: random.Random) -> TraceTemplate:
    a = rng.randint(5, 30)
    b = rng.randint(4, 25)
    c = rng.randint(2, 9)
    inner = a + b
    ans = inner * c
    error_step = 2
    delta = rng.choice([-5, -4, -3, 3, 4, 5])
    wrong_ans = ans + delta
    corrupted = [
        f"步骤1：先计算括号内 {a} + {b} = {inner}。",
        f"步骤2：再计算 {inner} × {c} = {wrong_ans}。",
        f"答案：{wrong_ans}。",
    ]
    error_type = "final_multiplication_error"
    return TraceTemplate(
        problem=f"计算 ({a} + {b}) × {c}。",
        correct_steps=[
            f"步骤1：先计算括号内 {a} + {b} = {inner}。",
            f"步骤2：再计算 {inner} × {c} = {ans}。",
            f"答案：{ans}。",
        ],
        corrupted_steps=corrupted,
        error_step_index=error_step,
        error_type=error_type,
        correct_answer=str(ans),
        corrupted_answer=str(wrong_ans),
    )


def make_linear_equation(rng: random.Random) -> TraceTemplate:
    x = rng.randint(2, 25)
    a = rng.randint(2, 9)
    b = rng.randint(3, 40)
    rhs = a * x + b
    after_sub = rhs - b
    error_step = 2
    wrong_x = x + rng.choice([-2, -1, 1, 2])
    corrupted = [
        f"步骤1：两边减去 {b}，得到 {a}x = {after_sub}。",
        f"步骤2：两边除以 {a}，得到 x = {wrong_x}。",
        f"答案：x = {wrong_x}。",
    ]
    corrupted_answer = str(wrong_x)
    error_type = "division_error"
    return TraceTemplate(
        problem=f"解方程 {a}x + {b} = {rhs}。",
        correct_steps=[
            f"步骤1：两边减去 {b}，得到 {a}x = {after_sub}。",
            f"步骤2：两边除以 {a}，得到 x = {x}。",
            f"答案：x = {x}。",
        ],
        corrupted_steps=corrupted,
        error_step_index=error_step,
        error_type=error_type,
        correct_answer=str(x),
        corrupted_answer=str(corrupted_answer),
    )


GENERATORS = [make_sum3, make_mul_add, make_parentheses, make_linear_equation]


def prefix_text(problem: str, steps: list[str], upto: int) -> str:
    visible_steps = "\n".join(steps[:upto])
    return f"题目：{problem}\n{visible_steps}"


def make_step_records(trace_id: str, template: TraceTemplate, version: str) -> list[dict[str, object]]:
    if version == "correct":
        steps = template.correct_steps
        error_step_index = -1
    elif version == "corrupted":
        steps = template.corrupted_steps
        error_step_index = template.error_step_index
    else:
        raise ValueError(version)

    full_trace = prefix_text(template.problem, steps, len(steps))
    rows = []
    for step_index in range(1, len(steps) + 1):
        if version == "correct":
            label = -1
            label_name = "no_error_control"
        elif step_index < error_step_index:
            label = 0
            label_name = "before_error"
        else:
            label = 1
            label_name = "at_or_after_error"

        rows.append(
            {
                "id": f"{trace_id}_{version}_step{step_index}",
                "trace_id": trace_id,
                "version": version,
                "problem": template.problem,
                "trace_text": full_trace,
                "step_index": step_index,
                "num_steps": len(steps),
                "step_text": steps[step_index - 1],
                "step_prefix_text": prefix_text(template.problem, steps, step_index),
                "error_step_index": error_step_index,
                "label": label,
                "label_name": label_name,
                "error_type": template.error_type,
                "correct_answer": template.correct_answer,
                "corrupted_answer": template.corrupted_answer,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_problems", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_file", default=str(DATA_OUT))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    for i in range(args.num_problems):
        generator = GENERATORS[i % len(GENERATORS)]
        template = generator(rng)
        trace_id = f"pe_v1_{i:04d}"
        rows.extend(make_step_records(trace_id, template, "correct"))
        rows.extend(make_step_records(trace_id, template, "corrupted"))

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    df.head(80).to_csv(PREVIEW_OUT, index=False, encoding="utf-8-sig")

    label_summary = df.groupby(["version", "label_name"]).size().rename("count").reset_index()
    error_summary = (
        df[df["version"] == "corrupted"]
        .groupby("error_type")
        .size()
        .rename("count")
        .reset_index()
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
        "# 可选实验B：Process Error 合成轨迹数据 v1",
        "",
        "## 定位",
        "",
        "该数据用于探索合成多步算术推理轨迹中，错误发生前后是否存在可解码的 step-level hidden-state signal（步骤级隐藏状态信号）。",
        "",
        "注意：这是受控合成数据，不是真实开放域 CoT 错误检测。",
        "",
        "## 输出",
        "",
        f"- 数据：`{out_file.relative_to(ROOT)}`",
        f"- 预览表：`{PREVIEW_OUT.relative_to(ROOT)}`",
        "",
        "## 数据规模",
        "",
        f"- problem 数：{args.num_problems}",
        f"- step-prefix 样本数：{len(df)}",
        "",
        "## 标签统计",
        "",
        md_table(label_summary),
        "",
        "## 错误类型统计",
        "",
        md_table(error_summary),
        "",
        "## 下一步",
        "",
        "1. 用 `82_extract_step_hidden_states.py` 抽取 corrupted trace 每一步末尾 hidden states。",
        "2. 用 `83_train_process_error_probe.py` 训练 before_error vs at_or_after_error probe。",
        "3. 补 step_index、length、TF-IDF 和 label shuffle control。",
    ]
    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", out_file)
    print("saved:", PREVIEW_OUT)
    print("saved:", SUMMARY_OUT)
    print("num rows:", len(df))
    print(label_summary.to_string(index=False))


if __name__ == "__main__":
    main()
