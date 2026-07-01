# -*- coding: utf-8 -*-
"""
23_compare_probe_entropy.py

对比：
1. 最佳层 hidden-state probe 的 AUROC
2. entropy / top1 / margin baseline 的 AUROC
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_metrics", type=str, default="outputs/qwen05b/experiment2/experiment2_layerwise_probe/layerwise_probe_metrics.csv")
    parser.add_argument("--entropy_metrics", type=str, default="outputs/qwen05b/experiment2/experiment2_logit_baseline/entropy_baseline_metrics.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/qwen05b/experiment2/experiment2_compare")
    args = parser.parse_args()

    probe_path = Path(args.probe_metrics)
    entropy_path = Path(args.entropy_metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not probe_path.exists():
        raise FileNotFoundError(f"找不到 {probe_path}，请先运行 21_train_layerwise_probe.py")

    if not entropy_path.exists():
        raise FileNotFoundError(f"找不到 {entropy_path}，请先运行 22_extract_entropy_baseline.py")

    probe_df = pd.read_csv(probe_path)
    entropy_df = pd.read_csv(entropy_path)

    auroc_col = "auroc_mean" if "auroc_mean" in probe_df.columns else "AUROC_mean"
    best_idx = probe_df[auroc_col].idxmax()
    best_layer = int(probe_df.loc[best_idx, "layer"])
    best_probe_auc = float(probe_df.loc[best_idx, auroc_col])

    rows = [
        {
            "method": f"hidden_probe_layer_{best_layer}",
            "auroc": best_probe_auc,
        }
    ]

    for _, row in entropy_df.iterrows():
        rows.append(
            {
                "method": str(row["baseline"]),
                "auroc": float(row["auroc"]),
            }
        )

    compare_df = pd.DataFrame(rows)

    compare_path = out_dir / "probe_vs_entropy_compare.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.bar(compare_df["method"], compare_df["auroc"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUROC")
    plt.title("Hidden-state Probe vs Logit-level Baselines")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fig_path = out_dir / "probe_vs_entropy_compare.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("=" * 80)
    print("保存完成：")
    print(f"- {compare_path}")
    print(f"- {fig_path}")
    print("=" * 80)
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()
