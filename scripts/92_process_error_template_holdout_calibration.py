# -*- coding: utf-8 -*-
"""
Calibration and threshold-stability addendum for Optional Experiment B-v2.

The existing held-out-template result reports AUROC and fixed-threshold metrics.
This script regenerates leave-one-template-out probabilities for selected layers
and asks whether a 0.5 threshold is stable across templates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
HIDDEN_FILE = ROOT / "outputs/qwen15b/experiment8_optionalB_v2/hidden_probe/hidden_states_process_error_v2.pt"
TEMPLATE_SUMMARY = ROOT / "outputs/qwen15b/experiment8_optionalB_v2/template_holdout/template_holdout_summary.json"
OUT_DIR = ROOT / "outputs/qwen15b/experiment8_optionalB_v2/calibration"
REPORT_DIR = ROOT / "reports/experiment8_optionalB"
TABLE_DIR = REPORT_DIR / "tables"
REPORT_MD = REPORT_DIR / "optionalB_process_error_v2_calibration_addendum.md"


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


def infer_template(problem: str) -> str:
    text = str(problem)
    if "方程" in text or "x" in text:
        return "linear_equation"
    if "(" in text or "（" in text:
        return "parentheses_mul"
    if "×" in text or "*" in text:
        return "mul_add"
    if "+" in text:
        return "sum3"
    return "unknown"


def load_hidden(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    obj = torch_load(path)
    if not isinstance(obj, list) or not obj:
        raise TypeError("hidden file should contain list[dict]")
    key = find_key(obj[0], HIDDEN_KEYS)
    hidden = np.stack([rec[key].detach().cpu().float().numpy() for rec in obj], axis=0)
    meta = pd.DataFrame([{k: v for k, v in rec.items() if k != key} for rec in obj])
    y = meta["label"].astype(int).to_numpy()
    meta["template_type"] = meta["problem"].astype(str).map(infer_template)
    return hidden.astype(np.float32), y, meta


def fit_lr(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed),
    )
    clf.fit(X_train, y_train)
    return clf


def best_threshold(y_true: np.ndarray, proba: np.ndarray, objective: str) -> tuple[float, float]:
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99), proba])
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        if objective == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        elif objective == "youden":
            tp = int(((pred == 1) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            score = tpr - fpr
        else:
            raise ValueError(objective)
        if score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t, best_score


def threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def ece_score(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (proba >= left) & (proba < right if right < 1.0 else proba <= right)
        if not mask.any():
            continue
        conf = float(proba[mask].mean())
        acc = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return ece


def evaluate_layer(hidden: np.ndarray, y: np.ndarray, meta: pd.DataFrame, layer: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    templates = sorted(t for t in meta["template_type"].unique() if t != "unknown")
    prediction_rows = []
    summary_rows = []

    for template in templates:
        test_idx = np.where(meta["template_type"].to_numpy() == template)[0]
        train_idx = np.where(meta["template_type"].to_numpy() != template)[0]
        clf = fit_lr(hidden[train_idx, layer, :], y[train_idx], seed)
        train_proba = clf.predict_proba(hidden[train_idx, layer, :])[:, 1]
        test_proba = clf.predict_proba(hidden[test_idx, layer, :])[:, 1]
        t_f1, train_f1 = best_threshold(y[train_idx], train_proba, "f1")
        t_youden, train_youden = best_threshold(y[train_idx], train_proba, "youden")
        t_oracle_f1, oracle_f1 = best_threshold(y[test_idx], test_proba, "f1")

        for threshold_name, threshold in [
            ("fixed_0.5", 0.5),
            ("train_best_f1", t_f1),
            ("train_youden", t_youden),
            ("test_oracle_f1_diagnostic_only", t_oracle_f1),
        ]:
            row = {
                "layer": int(layer),
                "heldout_template": template,
                "threshold_name": threshold_name,
                "threshold": float(threshold),
                "train_best_f1_at_threshold": float(train_f1) if threshold_name == "train_best_f1" else np.nan,
                "train_youden_at_threshold": float(train_youden) if threshold_name == "train_youden" else np.nan,
                "test_oracle_f1_at_threshold": float(oracle_f1) if threshold_name == "test_oracle_f1_diagnostic_only" else np.nan,
                "test_samples": int(len(test_idx)),
                "test_positive": int(y[test_idx].sum()),
                "auroc": float(roc_auc_score(y[test_idx], test_proba)),
                "brier": float(brier_score_loss(y[test_idx], test_proba)),
                "ece_10bin": float(ece_score(y[test_idx], test_proba)),
            }
            row.update(threshold_metrics(y[test_idx], test_proba, float(threshold)))
            summary_rows.append(row)

        for idx, proba in zip(test_idx, test_proba):
            rec = meta.iloc[int(idx)].to_dict()
            prediction_rows.append(
                {
                    "layer": int(layer),
                    "heldout_template": template,
                    "sample_index": int(idx),
                    "label": int(y[idx]),
                    "proba_step_incorrect": float(proba),
                    "pred_fixed_0_5": int(proba >= 0.5),
                    "trace_id": rec.get("trace_id", ""),
                    "step_index": rec.get("step_index", ""),
                    "label_name": rec.get("label_name", ""),
                    "problem": rec.get("problem", ""),
                    "step_prefix_text": rec.get("step_prefix_text", ""),
                }
            )

    return pd.DataFrame(prediction_rows), pd.DataFrame(summary_rows)


def md_table(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.4f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    hidden, y, meta = load_hidden(HIDDEN_FILE)
    if "unknown" in set(meta["template_type"]):
        raise ValueError("unknown template found")

    summary = json.loads(TEMPLATE_SUMMARY.read_text(encoding="utf-8"))
    layers = [
        int(summary["best_layer_by_auroc"]),
        int(summary["best_non_final_layer_by_auroc"]["layer"]),
    ]
    layers = sorted(set(layers))

    pred_frames = []
    summary_frames = []
    for layer in layers:
        preds, rows = evaluate_layer(hidden, y, meta, layer=layer, seed=42)
        pred_frames.append(preds)
        summary_frames.append(rows)

    pred_df = pd.concat(pred_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    agg = (
        summary_df.groupby(["layer", "threshold_name"])
        .agg(
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            brier_mean=("brier", "mean"),
            ece_10bin_mean=("ece_10bin", "mean"),
        )
        .reset_index()
    )

    pred_path = OUT_DIR / "template_holdout_calibration_predictions.csv"
    by_template_path = OUT_DIR / "template_holdout_calibration_by_template.csv"
    summary_path = OUT_DIR / "template_holdout_calibration_summary.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(by_template_path, index=False, encoding="utf-8-sig")
    agg.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report_table = TABLE_DIR / "process_error_v2_template_holdout_calibration_summary.csv"
    report_by_template = TABLE_DIR / "process_error_v2_template_holdout_calibration_by_template.csv"
    agg.to_csv(report_table, index=False, encoding="utf-8-sig")
    summary_df.to_csv(report_by_template, index=False, encoding="utf-8-sig")

    display = agg.copy()
    for col in [
        "threshold_mean",
        "threshold_std",
        "auroc_mean",
        "auroc_std",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "brier_mean",
        "ece_10bin_mean",
    ]:
        display[col] = display[col].map(fmt)

    best_layer = int(summary["best_layer_by_auroc"])
    best_rows = display[display["layer"] == best_layer]
    lines = [
        "# 可选实验 B-v2 校准与阈值稳定性补充",
        "",
        "## 目的",
        "",
        "已有 held-out template split 证明 hidden probe 的 AUROC 明显高于文本和步骤位置基线。本补充检查固定 0.5 阈值是否稳定，以及跨模板时是否需要重新校准。",
        "",
        "## 方法",
        "",
        "- 对每个 held-out template fold 重新训练 logistic probe。",
        "- 在测试模板上记录概率、AUROC、Brier score、10-bin ECE。",
        "- 比较固定 0.5 阈值、训练集最佳 F1 阈值、训练集 Youden 阈值。",
        "- `test_oracle_f1_diagnostic_only` 只作为诊断上界，不作为可部署结果。",
        "",
        "## 最佳层汇总",
        "",
        md_table(best_rows),
        "",
        "## 保守解释",
        "",
        "AUROC 高说明排序信号强；如果不同阈值策略的 accuracy/F1 波动较大，则说明跨模板概率校准不稳定。论文中应优先报告 AUROC，并把固定阈值 accuracy 写成诊断指标，而不是在线系统性能。",
        "",
        "## 输出文件",
        "",
        f"- `{pred_path.relative_to(ROOT)}`",
        f"- `{by_template_path.relative_to(ROOT)}`",
        f"- `{summary_path.relative_to(ROOT)}`",
        f"- `{report_table.relative_to(ROOT)}`",
        f"- `{report_by_template.relative_to(ROOT)}`",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", pred_path)
    print("saved:", by_template_path)
    print("saved:", summary_path)
    print("saved:", REPORT_MD)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
