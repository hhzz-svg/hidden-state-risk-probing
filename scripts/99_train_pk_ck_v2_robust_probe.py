# -*- coding: utf-8 -*-
"""Train robust probes for Optional Experiment A v2 PK/CK behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
HIDDEN_FILE = ROOT / "outputs/qwen15b/experiment7_optionalA_v2/hidden_probe/hidden_states_pk_ck_v2.pt"
OUT_DIR = ROOT / "outputs/qwen15b/experiment7_optionalA_v2/hidden_probe/pk_ck_v2_robust_probe"
REPORT_DIR = ROOT / "reports/experiment7_optionalA"
TABLE_DIR = REPORT_DIR / "tables"
REPORT_MD = REPORT_DIR / "experiment7_optionalA_v2_probe_report.md"
HIDDEN_KEYS = ["hidden", "hidden_states", "hiddens", "X"]


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_key(d: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in d:
            return key
    raise KeyError(f"Cannot find any of {keys}")


def load_hidden(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    obj = torch_load(path)
    if not isinstance(obj, list) or not obj:
        raise TypeError("hidden file should contain list[dict]")
    key = find_key(obj[0], HIDDEN_KEYS)
    hidden = np.stack([rec[key].detach().cpu().float().numpy() for rec in obj], axis=0)
    meta = pd.DataFrame([{k: v for k, v in rec.items() if k != key} for rec in obj])
    y = meta["label"].astype(int).to_numpy()
    return hidden.astype(np.float32), y, meta


def split_indices(meta: pd.DataFrame, split: str):
    if split == "source_id_group5":
        groups = meta["source_id"].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=5)
        return list(splitter.split(np.zeros(len(meta)), meta["label"].astype(int).to_numpy(), groups)), groups
    if split == "prompt_style_heldout":
        groups = meta["prompt_style"].astype(str).to_numpy()
        splitter = LeaveOneGroupOut()
        return list(splitter.split(np.zeros(len(meta)), meta["label"].astype(int).to_numpy(), groups)), groups
    raise ValueError(split)


def metric_row(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "auroc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def fit_hidden(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed),
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def evaluate_hidden(hidden: np.ndarray, y: np.ndarray, meta: pd.DataFrame, split: str, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits, groups = split_indices(meta, split)
    rows = []
    n_layers = hidden.shape[1]
    for layer in range(n_layers):
        for fold, (train_idx, test_idx) in enumerate(splits):
            y_train, y_test = y[train_idx], y[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            proba = fit_hidden(hidden[train_idx, layer, :], y_train, hidden[test_idx, layer, :], seed)
            row = {
                "split": split,
                "layer": int(layer),
                "fold": int(fold),
                "heldout_group": str(sorted(set(groups[test_idx]))[0]),
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
                "test_positive": int(y_test.sum()),
            }
            row.update(metric_row(y_test, proba))
            rows.append(row)
    by_fold = pd.DataFrame(rows)
    summary = (
        by_fold.groupby(["split", "layer"])
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )
    return by_fold, summary


def evaluate_tfidf(meta: pd.DataFrame, y: np.ndarray, split: str, seed: int = 42) -> pd.DataFrame:
    splits, groups = split_indices(meta, split)
    texts = meta["prompt"].astype(str).tolist()
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        clf = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=1),
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed),
        )
        clf.fit([texts[i] for i in train_idx], y_train)
        proba = clf.predict_proba([texts[i] for i in test_idx])[:, 1]
        row = {
            "split": split,
            "method": "tfidf_char_ngram_prompt",
            "fold": int(fold),
            "heldout_group": str(sorted(set(groups[test_idx]))[0]),
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "test_positive": int(y_test.sum()),
        }
        row.update(metric_row(y_test, proba))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_length(meta: pd.DataFrame, y: np.ndarray, split: str) -> pd.DataFrame:
    splits, groups = split_indices(meta, split)
    scores = meta["prompt"].astype(str).str.len().to_numpy(dtype=float)
    rows = []
    for fold, (_, test_idx) in enumerate(splits):
        y_test = y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        row = {
            "split": split,
            "method": "prompt_length",
            "fold": int(fold),
            "heldout_group": str(sorted(set(groups[test_idx]))[0]),
            "train_samples": np.nan,
            "test_samples": int(len(test_idx)),
            "test_positive": int(y_test.sum()),
        }
        row.update(metric_row(y_test, scores[test_idx]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_controls(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "method"])
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
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


def fmt(x) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    hidden, y, meta = load_hidden(HIDDEN_FILE)

    hidden_folds = []
    hidden_summaries = []
    controls = []
    for split in ["source_id_group5", "prompt_style_heldout"]:
        fold_df, summary_df = evaluate_hidden(hidden, y, meta, split)
        hidden_folds.append(fold_df)
        hidden_summaries.append(summary_df)
        controls.append(evaluate_tfidf(meta, y, split))
        controls.append(evaluate_length(meta, y, split))

    hidden_by_fold = pd.concat(hidden_folds, ignore_index=True)
    hidden_summary = pd.concat(hidden_summaries, ignore_index=True)
    control_by_fold = pd.concat(controls, ignore_index=True)
    control_summary = summarize_controls(control_by_fold)

    hidden_by_fold_path = OUT_DIR / "pk_ck_v2_hidden_by_fold.csv"
    hidden_summary_path = OUT_DIR / "pk_ck_v2_hidden_layer_summary.csv"
    control_by_fold_path = OUT_DIR / "pk_ck_v2_controls_by_fold.csv"
    control_summary_path = OUT_DIR / "pk_ck_v2_controls_summary.csv"
    summary_json_path = OUT_DIR / "pk_ck_v2_probe_summary.json"
    hidden_by_fold.to_csv(hidden_by_fold_path, index=False, encoding="utf-8-sig")
    hidden_summary.to_csv(hidden_summary_path, index=False, encoding="utf-8-sig")
    control_by_fold.to_csv(control_by_fold_path, index=False, encoding="utf-8-sig")
    control_summary.to_csv(control_summary_path, index=False, encoding="utf-8-sig")

    report_hidden = TABLE_DIR / "pk_ck_v2_hidden_layer_summary.csv"
    report_controls = TABLE_DIR / "pk_ck_v2_controls_summary.csv"
    hidden_summary.to_csv(report_hidden, index=False, encoding="utf-8-sig")
    control_summary.to_csv(report_controls, index=False, encoding="utf-8-sig")

    best_rows = []
    for split, group in hidden_summary.groupby("split"):
        best = group.loc[group["auroc_mean"].idxmax()]
        best_rows.append(best.to_dict())
    best_df = pd.DataFrame(best_rows)
    summary_obj = {
        "hidden_file": str(HIDDEN_FILE),
        "num_samples": int(len(y)),
        "pk_follow_count": int((y == 0).sum()),
        "ck_follow_count": int((y == 1).sum()),
        "num_layers": int(hidden.shape[1]),
        "hidden_size": int(hidden.shape[2]),
        "best_by_split": best_rows,
    }
    summary_json_path.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    display_best = best_df[["split", "layer", "auroc_mean", "auroc_std", "accuracy_mean", "f1_mean"]].copy()
    for col in ["auroc_mean", "auroc_std", "accuracy_mean", "f1_mean"]:
        display_best[col] = display_best[col].map(fmt)
    display_controls = control_summary.copy()
    for col in ["auroc_mean", "auroc_std", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"]:
        display_controls[col] = display_controls[col].map(fmt)

    lines = [
        "# 可选实验 A v2 hidden-state probe 报告",
        "",
        "## 范围",
        "",
        f"- 二分类样本数：{len(y)}",
        f"- pk_follow：{int((y == 0).sum())}",
        f"- ck_follow：{int((y == 1).sum())}",
        "- mixed/refusal/other 未并入二分类目标。",
        "",
        "## 划分方式",
        "",
        "- `source_id_group5`：按原始事实 source_id 分组，避免同一事实的不同 prompt style 同时进入训练和测试。",
        "- `prompt_style_heldout`：每次留出一种 prompt style，检查跨提示模板外推。",
        "",
        "## Hidden probe 最佳层",
        "",
        md_table(display_best),
        "",
        "## 控制基线",
        "",
        md_table(display_controls),
        "",
        "## 保守解释",
        "",
        "v2 的样本量和划分方式比 v1 更稳，但标签仍是答案字符串辅助标签。正式写入论文前，应优先完成 PK/CK sample44 的人工复核，并把该实验定位为扩展或附录证据。",
        "",
        "## 输出文件",
        "",
        f"- `{hidden_by_fold_path.relative_to(ROOT)}`",
        f"- `{hidden_summary_path.relative_to(ROOT)}`",
        f"- `{control_by_fold_path.relative_to(ROOT)}`",
        f"- `{control_summary_path.relative_to(ROOT)}`",
        f"- `{summary_json_path.relative_to(ROOT)}`",
        f"- `{report_hidden.relative_to(ROOT)}`",
        f"- `{report_controls.relative_to(ROOT)}`",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", hidden_by_fold_path)
    print("saved:", hidden_summary_path)
    print("saved:", control_by_fold_path)
    print("saved:", control_summary_path)
    print("saved:", summary_json_path)
    print("saved:", REPORT_MD)
    print(display_best.to_string(index=False))
    print(display_controls.to_string(index=False))


if __name__ == "__main__":
    main()
