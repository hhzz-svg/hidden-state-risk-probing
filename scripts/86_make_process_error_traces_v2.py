# -*- coding: utf-8 -*-
"""
86_make_process_error_traces_v2.py

Optional Experiment B-v2: paired step-correct vs step-incorrect traces.

Unlike B-v1, labels are decoupled from step_index:
- For every problem and every step_index, create one correct prefix and one
  incorrect prefix.
- label=0: step_correct
- label=1: step_incorrect

This makes step_index alone uninformative by construction.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT / "data/experiment8_optionalB/process_error_traces_v2.jsonl"
REPORT_DIR = ROOT / "reports/experiment8_optionalB"
TABLE_DIR = REPORT_DIR / "tables"
PREVIEW_OUT = TABLE_DIR / "process_error_trace_v2_preview.csv"
SUMMARY_OUT = REPORT_DIR / "optionalB_process_error_v2_data_summary.md"


@dataclass
class PairedTrace:
    problem: str
    correct_steps: list[str]
    incorrect_steps_by_index: dict[int, str]
    error_type_by_index: dict[int, str]
    correct_answer: str


def delta(rng: random.Random, choices: list[int]) -> int:
    return rng.choice(choices)


def make_sum3(rng: random.Random) -> PairedTrace:
    a = rng.randint(12, 79)
    b = rng.randint(11, 68)
    c = rng.randint(6, 45)
    s1 = a + b
    ans = s1 + c
    wrong_s1 = s1 + delta(rng, [-3, -2, -1, 1, 2, 3])
    wrong_ans2 = ans + delta(rng, [-4, -3, -2, 2, 3, 4])
    wrong_ans3 = ans + delta(rng, [-5, -4, -3, 3, 4, 5])
    return PairedTrace(
        problem=f"计算 {a} + {b} + {c}。",
        correct_steps=[
            f"步骤1：先计算 {a} + {b} = {s1}。",
            f"步骤2：再计算 {s1} + {c} = {ans}。",
            f"答案：{ans}。",
        ],
        incorrect_steps_by_index={
            1: f"步骤1：先计算 {a} + {b} = {wrong_s1}。",
            2: f"步骤2：再计算 {s1} + {c} = {wrong_ans2}。",
            3: f"答案：{wrong_ans3}。",
        },
        error_type_by_index={
            1: "first_addition_error",
            2: "final_addition_error",
            3: "wrong_final_answer",
        },
        correct_answer=str(ans),
    )


def make_mul_add(rng: random.Random) -> PairedTrace:
    a = rng.randint(3, 18)
    b = rng.randint(4, 16)
    c = rng.randint(5, 60)
    prod = a * b
    ans = prod + c
    wrong_prod = prod + delta(rng, [-4, -3, -2, 2, 3, 4])
    wrong_ans2 = ans + delta(rng, [-5, -4, -3, 3, 4, 5])
    wrong_ans3 = ans + delta(rng, [-6, -5, -4, 4, 5, 6])
    return PairedTrace(
        problem=f"计算 {a} × {b} + {c}。",
        correct_steps=[
            f"步骤1：先计算 {a} × {b} = {prod}。",
            f"步骤2：再计算 {prod} + {c} = {ans}。",
            f"答案：{ans}。",
        ],
        incorrect_steps_by_index={
            1: f"步骤1：先计算 {a} × {b} = {wrong_prod}。",
            2: f"步骤2：再计算 {prod} + {c} = {wrong_ans2}。",
            3: f"答案：{wrong_ans3}。",
        },
        error_type_by_index={
            1: "multiplication_error",
            2: "addition_after_multiplication_error",
            3: "wrong_final_answer",
        },
        correct_answer=str(ans),
    )


def make_parentheses(rng: random.Random) -> PairedTrace:
    a = rng.randint(5, 30)
    b = rng.randint(4, 25)
    c = rng.randint(2, 9)
    inner = a + b
    ans = inner * c
    wrong_inner = inner + delta(rng, [-3, -2, -1, 1, 2, 3])
    wrong_ans2 = ans + delta(rng, [-6, -5, -4, 4, 5, 6])
    wrong_ans3 = ans + delta(rng, [-7, -6, -5, 5, 6, 7])
    return PairedTrace(
        problem=f"计算 ({a} + {b}) × {c}。",
        correct_steps=[
            f"步骤1：先计算括号内 {a} + {b} = {inner}。",
            f"步骤2：再计算 {inner} × {c} = {ans}。",
            f"答案：{ans}。",
        ],
        incorrect_steps_by_index={
            1: f"步骤1：先计算括号内 {a} + {b} = {wrong_inner}。",
            2: f"步骤2：再计算 {inner} × {c} = {wrong_ans2}。",
            3: f"答案：{wrong_ans3}。",
        },
        error_type_by_index={
            1: "parentheses_addition_error",
            2: "multiplication_after_parentheses_error",
            3: "wrong_final_answer",
        },
        correct_answer=str(ans),
    )


def make_linear_equation(rng: random.Random) -> PairedTrace:
    x = rng.randint(2, 25)
    a = rng.randint(2, 9)
    b = rng.randint(3, 40)
    rhs = a * x + b
    after_sub = rhs - b
    wrong_after_sub = after_sub + delta(rng, [-4, -3, -2, 2, 3, 4])
    wrong_x2 = x + delta(rng, [-2, -1, 1, 2])
    wrong_x3 = x + delta(rng, [-3, -2, -1, 1, 2, 3])
    return PairedTrace(
        problem=f"解方程 {a}x + {b} = {rhs}。",
        correct_steps=[
            f"步骤1：两边减去 {b}，得到 {a}x = {after_sub}。",
            f"步骤2：两边除以 {a}，得到 x = {x}。",
            f"答案：x = {x}。",
        ],
        incorrect_steps_by_index={
            1: f"步骤1：两边减去 {b}，得到 {a}x = {wrong_after_sub}。",
            2: f"步骤2：两边除以 {a}，得到 x = {wrong_x2}。",
            3: f"答案：x = {wrong_x3}。",
        },
        error_type_by_index={
            1: "subtraction_error",
            2: "division_error",
            3: "wrong_final_answer",
        },
        correct_answer=str(x),
    )


GENERATORS = [make_sum3, make_mul_add, make_parentheses, make_linear_equation]


def prefix_text(problem: str, steps: list[str]) -> str:
    return f"题目：{problem}\n" + "\n".join(steps)


def make_records(problem_id: str, trace: PairedTrace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step_index in range(1, len(trace.correct_steps) + 1):
        correct_prefix_steps = trace.correct_steps[:step_index]
        incorrect_prefix_steps = trace.correct_steps[: step_index - 1] + [
            trace.incorrect_steps_by_index[step_index]
        ]

        for label, label_name, prefix_steps, variant in [
            (0, "step_correct", correct_prefix_steps, "correct_step"),
            (1, "step_incorrect", incorrect_prefix_steps, "incorrect_step"),
        ]:
            rows.append(
                {
                    "id": f"{problem_id}_{variant}{step_index}",
                    "trace_id": problem_id,
                    "pair_id": f"{problem_id}_step{step_index}",
                    "version": "paired_step_v2",
                    "problem": trace.problem,
                    "step_index": step_index,
                    "num_steps": len(trace.correct_steps),
                    "step_text": prefix_steps[-1],
                    "step_prefix_text": prefix_text(trace.problem, prefix_steps),
                    "label": label,
                    "label_name": label_name,
                    "error_type": "none" if label == 0 else trace.error_type_by_index[step_index],
                    "correct_answer": trace.correct_answer,
                    "target_definition": "step_correct_vs_step_incorrect",
                }
            )
    return rows


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_problems", type=int, default=160)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_file", default=str(DATA_OUT))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    for i in range(args.num_problems):
        trace = GENERATORS[i % len(GENERATORS)](rng)
        rows.extend(make_records(f"pe_v2_{i:04d}", trace))

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    df.head(80).to_csv(PREVIEW_OUT, index=False, encoding="utf-8-sig")

    label_summary = df.groupby(["step_index", "label_name"]).size().rename("count").reset_index()
    error_summary = (
        df[df["label_name"] == "step_incorrect"]
        .groupby(["step_index", "error_type"])
        .size()
        .rename("count")
        .reset_index()
    )

    lines = [
        "# 可选实验B-v2：Step Correctness 合成轨迹数据",
        "",
        "## 定位",
        "",
        "B-v2 用于修正 B-v1 中标签与 step_index 强绑定的问题。每道题、每个 step_index 都包含一个正确 prefix 和一个错误 prefix，目标是 `step_correct` vs `step_incorrect`。",
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
        "## step_index × label 统计",
        "",
        md_table(label_summary),
        "",
        "## 错误类型统计",
        "",
        md_table(error_summary),
        "",
        "## 设计要点",
        "",
        "同一 step_index 下正负样本数量相同，因此 step_index baseline 理论上应接近随机。若 TF-IDF 仍很高，说明文本本身已经足以识别算术错误或仍存在数值模式混杂。",
    ]
    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", out_file)
    print("saved:", PREVIEW_OUT)
    print("saved:", SUMMARY_OUT)
    print("num rows:", len(df))
    print(label_summary.to_string(index=False))


if __name__ == "__main__":
    main()
