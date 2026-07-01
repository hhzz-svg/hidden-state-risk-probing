# -*- coding: utf-8 -*-
"""
84_process_error_controls.py

Optional Experiment B controls:
- step_index baseline
- prompt length baseline
- TF-IDF char n-gram baseline
- label-shuffle control for selected hidden layer

All cross-validation uses GroupKFold(trace_id).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
HIDDEN_KEYS = ["hidden", "hidden_states", "hiddens", "X"]


def find_key(d: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in d:
            return key
    raise KeyError(f"Cannot find any of {keys}")


def load_records(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, list) or not obj:
        raise TypeError("hidden file should contain list[dict]")
    key = find_key(obj[0], HIDDEN_KEYS)
    hidden = np.stack([rec[key].detach().cpu().float().numpy() for rec in obj], axis=0)
    meta = [{k: v for k, v in rec.items() if k != key} for rec in obj]
    return meta, hidden.astype(np.float32)


def eval_proba(y: np.ndarray, proba: np.ndarray, name: str) -> dict[str, float | str]:
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "method": name,
        "auroc_mean": float(roc_auc_score(y, proba)),
        "accuracy_mean": float(accuracy_score(y, pred)),
        "precision_mean": float(p),
        "recall_mean": float(r),
        "f1_mean": float(f1),
    }


def cv_numeric_score(score: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str, n_splits: int) -> dict[str, float | str]:
    # Direct score baseline. For monotonic baselines like step_index, AUROC is
    # meaningful. Median threshold is reported as a simple operating point.
    aucs, accs, ps, rs, f1s = [], [], [], [], []
    splitter = GroupKFold(n_splits=n_splits)
    for _, test_idx in splitter.split(score.reshape(-1, 1), y, groups):
        test_score = score[test_idx]
        y_test = y[test_idx]
        aucs.append(roc_auc_score(y_test, test_score))
        threshold = float(np.median(test_score))
        pred = (test_score >= threshold).astype(int)
        accs.append(accuracy_score(y_test, pred))
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    return {
        "method": name,
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)),
        "precision_mean": float(np.mean(ps)),
        "recall_mean": float(np.mean(rs)),
        "f1_mean": float(np.mean(f1s)),
    }


def cv_tfidf(texts: list[str], y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> dict[str, float | str]:
    aucs, accs, ps, rs, f1s = [], [], [], [], []
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(texts, y, groups):
        clf = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=1),
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed),
        )
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        clf.fit(train_texts, y[train_idx])
        proba = clf.predict_proba(test_texts)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_idx], proba))
        accs.append(accuracy_score(y[test_idx], pred))
        p, r, f1, _ = precision_recall_fscore_support(y[test_idx], pred, average="binary", zero_division=0)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    return {
        "method": "tfidf_char_ngram_step_prefix",
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)),
        "precision_mean": float(np.mean(ps)),
        "recall_mean": float(np.mean(rs)),
        "f1_mean": float(np.mean(f1s)),
    }


def cv_hidden_shuffle(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    repeats: int,
) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    aucs = []
    for repeat in range(repeats):
        shuffled = y.copy()
        rng.shuffle(shuffled)
        splitter = GroupKFold(n_splits=n_splits)
        fold_aucs = []
        for train_idx, test_idx in splitter.split(X, shuffled, groups):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed + repeat),
            )
            clf.fit(X[train_idx], shuffled[train_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(shuffled[test_idx], proba))
        aucs.append(float(np.mean(fold_aucs)))
    return {
        "method": "hidden_label_shuffle_selected_layer",
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "accuracy_mean": np.nan,
        "precision_mean": np.nan,
        "recall_mean": np.nan,
        "f1_mean": np.nan,
        "shuffle_repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_file", default="outputs/qwen15b/experiment8_optionalB/hidden_probe/hidden_states_process_error.pt")
    parser.add_argument("--probe_summary", default="outputs/qwen15b/experiment8_optionalB/hidden_probe/process_error_layerwise_probe/layerwise_probe_summary.json")
    parser.add_argument("--out_dir", default="outputs/qwen15b/experiment8_optionalB/controls")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_repeats", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows, hidden = load_records(Path(args.hidden_file))
    meta = pd.DataFrame(meta_rows)
    y = meta["label"].astype(int).to_numpy()
    groups = meta["trace_id"].astype(str).to_numpy()

    with Path(args.probe_summary).open("r", encoding="utf-8") as f:
        probe_summary = json.load(f)
    selected_layer = int(probe_summary["best_layer_by_auroc"])

    rows = [
        cv_numeric_score(meta["step_index"].astype(float).to_numpy(), y, groups, "step_index", args.n_splits),
        cv_numeric_score(meta["step_prefix_text"].astype(str).str.len().to_numpy(dtype=float), y, groups, "step_prefix_char_length", args.n_splits),
        cv_tfidf(meta["step_prefix_text"].astype(str).tolist(), y, groups, args.n_splits, args.seed),
        cv_hidden_shuffle(hidden[:, selected_layer, :], y, groups, args.n_splits, args.seed, args.shuffle_repeats),
    ]
    df = pd.DataFrame(rows)
    out_csv = out_dir / "process_error_controls.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {
        "hidden_file": args.hidden_file,
        "selected_layer_for_shuffle": selected_layer,
        "n_samples": int(len(y)),
        "n_groups": int(len(set(groups))),
        "before_error_count": int((y == 0).sum()),
        "at_or_after_error_count": int((y == 1).sum()),
        "controls_csv": str(out_csv),
    }
    summary_json = out_dir / "process_error_controls_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved:", out_csv)
    print("saved:", summary_json)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
