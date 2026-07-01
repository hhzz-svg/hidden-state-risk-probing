# -*- coding: utf-8 -*-
"""
83_train_process_error_probe.py

Optional Experiment B: train layer-wise probes for synthetic process errors.

Evaluation uses GroupKFold by trace_id to avoid putting different steps from the
same corrupted trace into both train and test splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HIDDEN_KEYS = ["hidden", "hidden_states", "hiddens", "X"]


def find_key(d: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in d:
            return key
    raise KeyError(f"Cannot find any of {keys} in {list(d.keys())}")


def to_numpy_hidden(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float().numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"single hidden should be [L,H], got {arr.shape}")
    return arr.astype(np.float32)


def load_records(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, list) or not obj or not isinstance(obj[0], dict):
        raise TypeError("hidden file should contain list[dict]")

    hidden_list = []
    meta_rows = []
    labels = []
    groups = []
    for rec in obj:
        key = find_key(rec, HIDDEN_KEYS)
        hidden_list.append(to_numpy_hidden(rec[key]))
        labels.append(int(rec["label"]))
        groups.append(str(rec["trace_id"]))
        meta_rows.append({k: v for k, v in rec.items() if k != key})

    hidden = np.stack(hidden_list, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_arr = np.asarray(groups)
    meta = pd.DataFrame(meta_rows)
    return hidden, y, group_arr, meta


def evaluate_layer(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> dict[str, float]:
    splitter = GroupKFold(n_splits=n_splits)
    aucs, accs, ps, rs, f1s = [], [], [], [], []
    for train_idx, test_idx in splitter.split(X, y, groups):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed),
        )
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_idx], proba))
        accs.append(accuracy_score(y[test_idx], pred))
        p, r, f1, _ = precision_recall_fscore_support(y[test_idx], pred, average="binary", zero_division=0)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    return {
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "precision_mean": float(np.mean(ps)),
        "recall_mean": float(np.mean(rs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }


def plot_curve(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    x = df["layer"].to_numpy()
    y = df[f"{metric}_mean"].to_numpy()
    plt.plot(x, y, marker="o")
    std_col = f"{metric}_std"
    if std_col in df:
        e = df[std_col].to_numpy()
        plt.fill_between(x, y - e, y + e, alpha=0.2)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(f"Process Error Probe {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_file", default="outputs/qwen15b/experiment8_optionalB/hidden_probe/hidden_states_process_error.pt")
    parser.add_argument("--out_dir", default="outputs/qwen15b/experiment8_optionalB/hidden_probe/process_error_layerwise_probe")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hidden_file = Path(args.hidden_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden, y, groups, meta = load_records(hidden_file)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"binary labels expected, got {sorted(set(y))}")

    n, num_layers, hidden_size = hidden.shape
    group_count = len(set(groups))
    print("=" * 80)
    print("Optional Experiment B: Process Error Layer-wise Probe")
    print(f"hidden_file: {hidden_file}")
    print(f"hidden shape: N={n}, L={num_layers}, H={hidden_size}")
    print(f"labels: before_error={(y == 0).sum()}, at_or_after_error={(y == 1).sum()}")
    print(f"groups(trace_id): {group_count}")
    print("=" * 80)

    rows = []
    for layer in range(num_layers):
        metrics = evaluate_layer(hidden[:, layer, :], y, groups, args.n_splits, args.seed)
        rows.append({"layer": layer, **metrics})
        print(
            f"Layer {layer:02d} | AUROC={metrics['auroc_mean']:.4f}±{metrics['auroc_std']:.4f} "
            f"| ACC={metrics['accuracy_mean']:.4f} | F1={metrics['f1_mean']:.4f}"
        )

    df = pd.DataFrame(rows)
    metrics_path = out_dir / "layerwise_probe_metrics.csv"
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    best = df.loc[int(df["auroc_mean"].idxmax())]
    summary = {
        "hidden_file": str(hidden_file),
        "num_samples": int(n),
        "num_groups": int(group_count),
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "before_error_count": int((y == 0).sum()),
        "at_or_after_error_count": int((y == 1).sum()),
        "group_split": "GroupKFold(trace_id)",
        "n_splits": int(args.n_splits),
        "best_layer_by_auroc": int(best["layer"]),
        "best_auroc_mean": float(best["auroc_mean"]),
        "best_auroc_std": float(best["auroc_std"]),
        "best_accuracy_mean": float(best["accuracy_mean"]),
        "best_f1_mean": float(best["f1_mean"]),
    }
    summary_path = out_dir / "layerwise_probe_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    meta.to_csv(out_dir / "probe_metadata.csv", index=False, encoding="utf-8-sig")
    plot_curve(df, "auroc", out_dir / "layer_auc_curve.png", "AUROC")
    plot_curve(df, "f1", out_dir / "layer_f1_curve.png", "F1")

    print("=" * 80)
    print("saved:", metrics_path)
    print("saved:", summary_path)
    print("saved:", out_dir / "layer_auc_curve.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
