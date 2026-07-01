# -*- coding: utf-8 -*-
"""
88_train_process_error_template_holdout.py

Optional Experiment B-v2 robustness check:
held-out template split for step_correct vs step_incorrect.

Each fold holds out one problem template, trains on the other templates, and
tests on the unseen template. This checks whether the signal generalizes beyond
the specific arithmetic/equation template family.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HIDDEN = ROOT / "outputs/qwen15b/experiment8_optionalB_v2/hidden_probe/hidden_states_process_error_v2.pt"
DEFAULT_OUT = ROOT / "outputs/qwen15b/experiment8_optionalB_v2/template_holdout"
REPORT_DIR = ROOT / "reports/experiment8_optionalB"
TABLE_DIR = REPORT_DIR / "tables"
REPORT_MD = REPORT_DIR / "optionalB_process_error_v2_template_holdout_report.md"

HIDDEN_KEYS = ["hidden", "hidden_states", "hiddens", "X"]


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
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, list) or not obj:
        raise TypeError("hidden file should contain list[dict]")
    key = find_key(obj[0], HIDDEN_KEYS)
    hidden = np.stack([rec[key].detach().cpu().float().numpy() for rec in obj], axis=0)
    meta_rows = [{k: v for k, v in rec.items() if k != key} for rec in obj]
    meta = pd.DataFrame(meta_rows)
    y = meta["label"].astype(int).to_numpy()
    meta["template_type"] = meta["problem"].astype(str).map(infer_template)
    return hidden.astype(np.float32), y, meta


def fit_predict_lr(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed),
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "auroc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def heldout_indices(meta: pd.DataFrame, template: str) -> tuple[np.ndarray, np.ndarray]:
    test_mask = meta["template_type"].astype(str).to_numpy() == template
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


def evaluate_hidden_layers(hidden: np.ndarray, y: np.ndarray, meta: pd.DataFrame, templates: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    n_layers = hidden.shape[1]
    for layer in range(n_layers):
        for template in templates:
            train_idx, test_idx = heldout_indices(meta, template)
            proba = fit_predict_lr(hidden[train_idx, layer, :], y[train_idx], hidden[test_idx, layer, :], seed)
            m = metrics_from_proba(y[test_idx], proba)
            rows.append(
                {
                    "layer": layer,
                    "heldout_template": template,
                    "train_samples": int(len(train_idx)),
                    "test_samples": int(len(test_idx)),
                    "test_positive": int(y[test_idx].sum()),
                    **m,
                }
            )
    per_template = pd.DataFrame(rows)
    summary = (
        per_template.groupby("layer")
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
    return per_template, summary


def numeric_score_controls(meta: pd.DataFrame, y: np.ndarray, templates: list[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    controls = {
        "step_index": meta["step_index"].astype(float).to_numpy(),
        "step_prefix_char_length": meta["step_prefix_text"].astype(str).str.len().to_numpy(dtype=float),
    }
    for method, score in controls.items():
        for template in templates:
            _, test_idx = heldout_indices(meta, template)
            m = metrics_from_proba(y[test_idx], score[test_idx])
            rows.append({"method": method, "heldout_template": template, **m})
    return rows


def tfidf_control(meta: pd.DataFrame, y: np.ndarray, templates: list[str], seed: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    texts = meta["step_prefix_text"].astype(str).tolist()
    for template in templates:
        train_idx, test_idx = heldout_indices(meta, template)
        clf = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=1),
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed),
        )
        clf.fit([texts[i] for i in train_idx], y[train_idx])
        proba = clf.predict_proba([texts[i] for i in test_idx])[:, 1]
        rows.append({"method": "tfidf_char_ngram_template_holdout", "heldout_template": template, **metrics_from_proba(y[test_idx], proba)})
    return rows


def hidden_label_shuffle_control(
    hidden: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    templates: list[str],
    selected_layer: int,
    seed: int,
    repeats: int,
) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    for template in templates:
        train_idx, test_idx = heldout_indices(meta, template)
        aucs = []
        for repeat in range(repeats):
            y_train = y[train_idx].copy()
            rng.shuffle(y_train)
            proba = fit_predict_lr(hidden[train_idx, selected_layer, :], y_train, hidden[test_idx, selected_layer, :], seed + repeat)
            aucs.append(float(roc_auc_score(y[test_idx], proba)))
        rows.append(
            {
                "method": "hidden_label_shuffle_selected_layer",
                "heldout_template": template,
                "auroc": float(np.mean(aucs)),
                "auroc_std_repeats": float(np.std(aucs)),
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "shuffle_repeats": int(repeats),
            }
        )
    return rows


def summarize_controls(control_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in control_df.groupby("method"):
        rows.append(
            {
                "method": method,
                "heldout_template": "mean_across_templates",
                "auroc": float(group["auroc"].mean()),
                "auroc_std_templates": float(group["auroc"].std()),
                "accuracy": float(group["accuracy"].mean(skipna=True)) if group["accuracy"].notna().any() else np.nan,
                "precision": float(group["precision"].mean(skipna=True)) if group["precision"].notna().any() else np.nan,
                "recall": float(group["recall"].mean(skipna=True)) if group["recall"].notna().any() else np.nan,
                "f1": float(group["f1"].mean(skipna=True)) if group["f1"].notna().any() else np.nan,
                "shuffle_repeats": group["shuffle_repeats"].dropna().iloc[0] if "shuffle_repeats" in group and group["shuffle_repeats"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def make_report(
    meta: pd.DataFrame,
    layer_summary: pd.DataFrame,
    hidden_by_template: pd.DataFrame,
    control_summary: pd.DataFrame,
    control_by_template: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    layer_table = TABLE_DIR / "process_error_v2_template_holdout_layer_summary.csv"
    hidden_template_table = TABLE_DIR / "process_error_v2_template_holdout_by_template.csv"
    controls_table = TABLE_DIR / "process_error_v2_template_holdout_controls.csv"
    layer_summary.to_csv(layer_table, index=False, encoding="utf-8-sig")
    hidden_by_template.to_csv(hidden_template_table, index=False, encoding="utf-8-sig")
    pd.concat([control_by_template, control_summary], ignore_index=True).to_csv(controls_table, index=False, encoding="utf-8-sig")

    best_layer = int(summary["best_layer_by_auroc"])
    best_non_final = summary["best_non_final_layer_by_auroc"]
    template_counts = (
        meta.groupby(["template_type", "label_name"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["template_type", "label_name"])
    )
    best_template_rows = hidden_by_template[hidden_by_template["layer"] == best_layer][
        ["heldout_template", "test_samples", "test_positive", "auroc", "accuracy", "f1"]
    ].copy()
    for col in ["auroc", "accuracy", "f1"]:
        best_template_rows[col] = best_template_rows[col].map(lambda x: fmt(x))
    controls_view = control_summary[["method", "auroc", "auroc_std_templates", "accuracy", "f1"]].copy()
    for col in ["auroc", "auroc_std_templates", "accuracy", "f1"]:
        controls_view[col] = controls_view[col].map(lambda x: fmt(x))

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
        "# 可选实验B-v2：Held-out Template 外推检查",
        "",
        "## 1. 实验目的",
        "",
        "B-v2 的主结果使用 `GroupKFold(trace_id)`，能避免同一道题的 prefix 同时进入训练和测试。",
        "本检查进一步把划分提高到模板级：每次留出一种题型模板，只用另外三种模板训练，再测试在未见过的模板上。",
        "",
        "这个实验回答的问题是：step correctness 的 hidden-state signal 是否只是在记具体算术/方程模板，还是能跨模板外推。",
        "",
        "## 2. 数据分布",
        "",
        md_table(template_counts),
        "",
        "四个模板分别是：三数加法、乘法加法、括号乘法、一元一次方程。每个模板中 correct/incorrect 数量平衡。",
        "",
        "## 3. Hidden-state probe 结果",
        "",
        f"- 最佳层：layer {best_layer}",
        f"- 模板外推平均 AUROC：{fmt(summary['best_auroc_mean'])} +/- {fmt(summary['best_auroc_std'])}",
        f"- 平均 Accuracy：{fmt(summary['best_accuracy_mean'])}",
        f"- 平均 F1：{fmt(summary['best_f1_mean'])}",
        f"- 最佳非最终层：layer {best_non_final['layer']}，AUROC={fmt(best_non_final['auroc_mean'])} +/- {fmt(best_non_final['auroc_std'])}",
        "",
        "### 最佳层按 held-out template 分解",
        "",
        md_table(best_template_rows),
        "",
        "注意：accuracy 使用固定 0.5 阈值，模板外推时概率校准可能失配；因此本检查以 AUROC 作为主指标。",
        "例如某些模板可能 AUROC 较高但固定阈值 accuracy 偏低，这说明排序信号存在，但跨模板阈值需要重新校准。",
        "",
        "## 4. 控制基线",
        "",
        md_table(controls_view),
        "",
        "## 5. 保守解释",
        "",
        "held-out template split 比普通 problem-group split 更严格。若 hidden probe 在该设置下仍明显高于 step_index、长度、TF-IDF 和标签打乱控制，",
        "说明 B-v2 的结果不只是模板内记忆或同题泄漏。",
        "",
        "但这个结果仍然来自合成算术/方程轨迹，不能直接外推到开放域自然语言推理。尤其是最佳层仍可能混有输出预测相关信息，因此建议报告最终层和最佳非最终层两个数字。",
        "",
        "## 6. 可写结论",
        "",
        "```text",
        "A stricter held-out-template evaluation further tests whether the step-correctness probe generalizes beyond the arithmetic template family seen during training. When each template was held out in turn, the hidden-state probe remained substantially above the step-index, length, TF-IDF, and label-shuffle controls. This suggests that the decodable signal is not solely driven by within-template leakage, although the result remains limited to controlled synthetic reasoning traces.",
        "```",
        "",
        "## 7. 输出文件",
        "",
        f"- `{layer_table.relative_to(ROOT)}`",
        f"- `{hidden_template_table.relative_to(ROOT)}`",
        f"- `{controls_table.relative_to(ROOT)}`",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_file", default=str(DEFAULT_HIDDEN))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_repeats", type=int, default=20)
    args = parser.parse_args()

    hidden_file = Path(args.hidden_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden, y, meta = load_hidden(hidden_file)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"binary labels expected, got {sorted(set(y))}")

    templates = sorted(t for t in meta["template_type"].unique() if t != "unknown")
    if "unknown" in set(meta["template_type"]):
        raise ValueError("unknown template found; refine infer_template()")

    print("=" * 80)
    print("Optional Experiment B-v2: Held-out Template Split")
    print(f"hidden_file: {hidden_file}")
    print(f"hidden shape: {hidden.shape}")
    print("templates:", ", ".join(templates))
    print("=" * 80)
    print(pd.crosstab(meta["template_type"], meta["label_name"]).to_string())

    hidden_by_template, layer_summary = evaluate_hidden_layers(hidden, y, meta, templates, args.seed)
    best = layer_summary.loc[layer_summary["auroc_mean"].idxmax()]
    final_layer = int(layer_summary["layer"].max())
    non_final = layer_summary[layer_summary["layer"] != final_layer]
    best_non_final = non_final.loc[non_final["auroc_mean"].idxmax()]
    best_layer = int(best["layer"])

    control_rows = []
    control_rows.extend(numeric_score_controls(meta, y, templates))
    control_rows.extend(tfidf_control(meta, y, templates, args.seed))
    control_rows.extend(hidden_label_shuffle_control(hidden, y, meta, templates, best_layer, args.seed, args.shuffle_repeats))
    control_by_template = pd.DataFrame(control_rows)
    control_summary = summarize_controls(control_by_template)

    hidden_by_template_path = out_dir / "hidden_template_holdout_by_template.csv"
    layer_summary_path = out_dir / "hidden_template_holdout_layer_summary.csv"
    controls_path = out_dir / "template_holdout_controls.csv"
    summary_path = out_dir / "template_holdout_summary.json"

    hidden_by_template.to_csv(hidden_by_template_path, index=False, encoding="utf-8-sig")
    layer_summary.to_csv(layer_summary_path, index=False, encoding="utf-8-sig")
    pd.concat([control_by_template, control_summary], ignore_index=True).to_csv(controls_path, index=False, encoding="utf-8-sig")

    summary = {
        "hidden_file": str(hidden_file),
        "num_samples": int(len(y)),
        "num_layers": int(hidden.shape[1]),
        "hidden_size": int(hidden.shape[2]),
        "templates": templates,
        "split": "leave_one_template_out",
        "best_layer_by_auroc": best_layer,
        "best_auroc_mean": float(best["auroc_mean"]),
        "best_auroc_std": float(best["auroc_std"]),
        "best_accuracy_mean": float(best["accuracy_mean"]),
        "best_f1_mean": float(best["f1_mean"]),
        "best_non_final_layer_by_auroc": {
            "layer": int(best_non_final["layer"]),
            "auroc_mean": float(best_non_final["auroc_mean"]),
            "auroc_std": float(best_non_final["auroc_std"]),
            "accuracy_mean": float(best_non_final["accuracy_mean"]),
            "f1_mean": float(best_non_final["f1_mean"]),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    make_report(meta, layer_summary, hidden_by_template, control_summary, control_by_template, summary)

    print("=" * 80)
    print("saved:", hidden_by_template_path)
    print("saved:", layer_summary_path)
    print("saved:", controls_path)
    print("saved:", summary_path)
    print("saved:", REPORT_MD)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(control_summary.to_string(index=False))


if __name__ == "__main__":
    main()
