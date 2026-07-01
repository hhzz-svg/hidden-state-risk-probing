# -*- coding: utf-8 -*-
"""
61_routing_simulation.py

Experiment 6: Dynamic Routing Toy Simulation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    {
        "model": "Qwen2.5-0.5B-Instruct",
        "analysis_file": ROOT / "outputs/qwen05b/experiment4/experiment4_generation_behavior_analysis_answer_key/generation_behavior_merged_scores.csv",
        "out_dir": ROOT / "outputs/qwen05b/experiment6/experiment6_routing_simulation",
    },
    {
        "model": "Qwen2.5-1.5B-Instruct",
        "analysis_file": ROOT / "outputs/qwen15b/experiment4/experiment4_generation_behavior_analysis_answer_key/generation_behavior_merged_scores.csv",
        "out_dir": ROOT / "outputs/qwen15b/experiment6/experiment6_routing_simulation",
    },
    {
        "model": "Qwen2.5-7B-Instruct",
        "analysis_file": ROOT / "outputs/qwen7b/experiment4/experiment4_generation_behavior_analysis_answer_key/generation_behavior_merged_scores.csv",
        "out_dir": ROOT / "outputs/qwen7b/experiment6/experiment6_routing_simulation",
    },
]


HIGH_RISK_BEHAVIORS = {"hallucination", "refusal", "irrelevant"}


def sweep_thresholds(df: pd.DataFrame, score_col: str, conservative_cost: float = 3.0) -> pd.DataFrame:
    scores = df[score_col].astype(float).to_numpy()
    high_risk = df["final_behavior_label"].isin(HIGH_RISK_BEHAVIORS).astype(int).to_numpy()
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))

    rows = []
    for threshold in thresholds:
        conservative = (scores >= threshold).astype(int)
        direct = 1 - conservative
        direct_count = int(direct.sum())
        conservative_count = int(conservative.sum())
        high_risk_count = int(high_risk.sum())
        caught = int(((conservative == 1) & (high_risk == 1)).sum())
        missed = int(((direct == 1) & (high_risk == 1)).sum())
        direct_high_risk = int(((direct == 1) & (high_risk == 1)).sum())
        direct_total = max(direct_count, 1)

        rows.append(
            {
                "threshold": float(threshold),
                "coverage_direct_answer": direct_count / len(df),
                "conservative_route_rate": conservative_count / len(df),
                "high_risk_recall": caught / high_risk_count if high_risk_count else np.nan,
                "missed_high_risk_count": missed,
                "direct_high_risk_rate": direct_high_risk / direct_total,
                "estimated_avg_cost": (direct_count * 1.0 + conservative_count * conservative_cost) / len(df),
            }
        )
    return pd.DataFrame(rows)


def choose_operating_point(sweep: pd.DataFrame, min_recall: float = 0.90) -> pd.Series:
    candidates = sweep[sweep["high_risk_recall"] >= min_recall].copy()
    if candidates.empty:
        return sweep.sort_values(["high_risk_recall", "coverage_direct_answer"], ascending=[False, False]).iloc[0]
    return candidates.sort_values(["coverage_direct_answer", "estimated_avg_cost"], ascending=[False, True]).iloc[0]


def plot_tradeoff(sweep: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sweep["coverage_direct_answer"], sweep["high_risk_recall"], marker="o", markersize=3, label="High-risk recall")
    ax1.set_xlabel("Direct-answer coverage")
    ax1.set_ylabel("High-risk recall")
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(sweep["coverage_direct_answer"], sweep["estimated_avg_cost"], color="#F58518", label="Estimated cost")
    ax2.set_ylabel("Estimated average cost")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    summary_rows = []
    report_lines = [
        "# 第六次实验：动态路由 Toy Simulation 初步报告",
        "",
        "## 1. 实验目的",
        "",
        "本实验不是实现真正在线系统，也不是 Early Exit。它只是一个 toy simulation：根据生成前 hidden risk score 设置阈值，低风险样本直接回答，高风险样本转入保守策略。",
        "",
        "保守策略在这里不真正调用检索或强模型，只模拟为“需要额外处理”。默认成本设定：直接回答成本为 1，保守策略成本为 3。",
        "",
        "## 2. 主要结果",
        "",
        "| 模型 | 推荐阈值 | 直接回答覆盖率 | 高风险行为召回率 | 直接回答中的高风险率 | 平均成本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in MODELS:
        model = item["model"]
        df = pd.read_csv(item["analysis_file"], encoding="utf-8-sig")
        out_dir = item["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        sweep = sweep_thresholds(df, "hidden_risk_score_oof")
        operating = choose_operating_point(sweep, min_recall=0.90)

        sweep_path = out_dir / "routing_threshold_sweep.csv"
        curve_path = out_dir / "routing_tradeoff_curve.png"
        sweep.to_csv(sweep_path, index=False, encoding="utf-8-sig")
        plot_tradeoff(sweep, curve_path, f"{model}: Routing Tradeoff")

        summary_rows.append({"model": model, **operating.to_dict()})
        report_lines.append(
            f"| {model} | {operating['threshold']:.4f} | "
            f"{operating['coverage_direct_answer']:.1%} | "
            f"{operating['high_risk_recall']:.1%} | "
            f"{operating['direct_high_risk_rate']:.1%} | "
            f"{operating['estimated_avg_cost']:.2f} |"
        )

        single_report = [
            f"# {model} 动态路由模拟",
            "",
            f"- 推荐阈值：{operating['threshold']:.4f}",
            f"- 直接回答覆盖率：{operating['coverage_direct_answer']:.1%}",
            f"- 高风险行为召回率：{operating['high_risk_recall']:.1%}",
            f"- 直接回答中的高风险率：{operating['direct_high_risk_rate']:.1%}",
            f"- 估计平均成本：{operating['estimated_avg_cost']:.2f}",
            "",
            "该结果只表示阈值模拟下的覆盖率-风险取舍，不代表已经部署动态路由系统。",
        ]
        (out_dir / "routing_simulation_report.md").write_text("\n".join(single_report), encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows)
    out_root = ROOT / "reports/experiment6/tables"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "experiment6_routing_summary.csv"
    report_path = ROOT / "reports/experiment6/experiment6_routing_simulation_report.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report_lines.extend(
        [
            "",
            "## 3. 保守解释",
            "",
            "这个模拟说明 hidden risk score 可以作为轻量级路由信号的候选：提高阈值会增加直接回答覆盖率，但也更容易漏掉高风险行为；降低阈值会提高高风险召回率，但需要更多样本进入保守策略。",
            "",
            "当前不能声称已经实现动态路由系统或 Early Exit，只能说完成了基于已有分数的离线阈值模拟。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("saved:", summary_path)
    print("saved:", report_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
