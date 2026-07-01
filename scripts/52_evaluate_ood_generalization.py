# -*- coding: utf-8 -*-
"""
52_evaluate_ood_generalization.py

Experiment 5: train on controlled_v4, test on OOD_v1.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HIDDEN_KEYS = ["hidden_states", "hiddens", "all_hidden_states", "last_token_hidden_states", "hidden", "X"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_label(x: Any) -> int:
    if isinstance(x, (int, np.integer)) and int(x) in [0, 1]:
        return int(x)
    if isinstance(x, float) and int(x) in [0, 1]:
        return int(x)
    text = str(x).strip().lower()
    if text in ["known", "0", "low", "low-risk", "low_risk"]:
        return 0
    if text in ["unknown", "1", "high", "high-risk", "high_risk"]:
        return 1
    raise ValueError(f"无法识别标签：{x!r}")


def find_key(row: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in row:
            return key
    return None


def to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def normalize_single_hidden(h: Any) -> np.ndarray:
    arr = to_np(h)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[:, -1, :]
    elif arr.ndim == 4:
        arr = arr[0, :, -1, :]
    else:
        raise ValueError(f"不支持 hidden shape={arr.shape}")
    return arr.astype(np.float32)


def load_hidden(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        samples = []
        for i, row in enumerate(obj):
            key = find_key(row, HIDDEN_KEYS)
            if key is None:
                raise KeyError(f"record {i} 缺少 hidden 字段")
            samples.append(normalize_single_hidden(row[key]))
        return np.stack(samples, axis=0)
    if isinstance(obj, dict):
        key = find_key(obj, HIDDEN_KEYS)
        if key is None:
            raise KeyError(f"{path} 缺少 hidden 字段")
        arr = to_np(obj[key])
    else:
        arr = to_np(obj)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    if arr.ndim == 4:
        arr = arr[:, :, -1, :]
    return arr.astype(np.float32)


def evaluate_scores(y: np.ndarray, score: np.ndarray, name: str) -> Dict[str, float]:
    threshold = float(np.median(score))
    pred = (score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "method": name,
        "auroc": float(roc_auc_score(y, score)),
        "median_threshold": threshold,
        "accuracy_at_median": float(accuracy_score(y, pred)),
        "precision_at_median": float(p),
        "recall_at_median": float(r),
        "f1_at_median": float(f1),
    }


def plot_scores(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(df["method"], df["auroc"], color=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"])
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("OOD AUROC")
    plt.xlabel("Method")
    plt.title("Experiment 5 OOD Generalization")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/experiment3/prompts_controlled_v4.jsonl")
    parser.add_argument("--test_data", default="data/experiment5/prompts_ood_v1.jsonl")
    parser.add_argument("--train_hidden", default="outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/hidden_states.pt")
    parser.add_argument("--test_hidden", default="outputs/qwen05b/experiment5/experiment5_ood_v1/hidden_probe/hidden_states_ood.pt")
    parser.add_argument("--test_entropy", default="outputs/qwen05b/experiment5/experiment5_ood_v1/logit_baseline/entropy_baseline_scores.csv")
    parser.add_argument("--out_dir", default="outputs/qwen05b/experiment5/experiment5_ood_v1/ood_evaluation")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train_data))
    test_rows = load_jsonl(Path(args.test_data))
    train_y = np.array([normalize_label(r["label"]) for r in train_rows], dtype=int)
    test_y = np.array([normalize_label(r["label"]) for r in test_rows], dtype=int)
    train_texts = [r["prompt"] for r in train_rows]
    test_texts = [r["prompt"] for r in test_rows]

    train_hidden = load_hidden(Path(args.train_hidden))
    test_hidden = load_hidden(Path(args.test_hidden))
    if train_hidden.shape[1] <= args.layer or test_hidden.shape[1] <= args.layer:
        raise ValueError(f"layer={args.layer} 超出 hidden 层数")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=args.seed),
    )
    hidden_clf.fit(train_hidden[:, args.layer, :], train_y)
    hidden_score = hidden_clf.predict_proba(test_hidden[:, args.layer, :])[:, 1]

    length_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
    )
    train_len = np.array([[len(text), len(text.encode("utf-8"))] for text in train_texts])
    test_len = np.array([[len(text), len(text.encode("utf-8"))] for text in test_texts])
    length_clf.fit(train_len, train_y)
    length_score = length_clf.predict_proba(test_len)[:, 1]

    tfidf_clf = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=5000, min_df=1),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed),
    )
    tfidf_clf.fit(train_texts, train_y)
    tfidf_score = tfidf_clf.predict_proba(test_texts)[:, 1]

    entropy_df = pd.read_csv(args.test_entropy, encoding="utf-8-sig")

    metrics = [
        evaluate_scores(test_y, hidden_score, f"hidden_layer_{args.layer}"),
        evaluate_scores(test_y, length_score, "length_baseline"),
        evaluate_scores(test_y, tfidf_score, "tfidf_char_ngram"),
        evaluate_scores(test_y, entropy_df["score_entropy"].to_numpy(), "entropy"),
        evaluate_scores(test_y, entropy_df["score_neg_top1_prob"].to_numpy(), "negative_top1_prob"),
        evaluate_scores(test_y, entropy_df["score_neg_margin"].to_numpy(), "negative_margin"),
    ]
    metric_df = pd.DataFrame(metrics)

    score_rows = []
    for i, row in enumerate(test_rows):
        score_rows.append(
            {
                "id": row["id"],
                "label": int(test_y[i]),
                "label_name": "unknown_or_high_risk" if test_y[i] == 1 else "known_or_low_risk",
                "category": row.get("category", ""),
                "entity": row.get("entity", ""),
                "relation": row.get("relation", ""),
                "prompt": row["prompt"],
                "hidden_score": float(hidden_score[i]),
                "length_score": float(length_score[i]),
                "tfidf_score": float(tfidf_score[i]),
                "entropy_score": float(entropy_df.loc[i, "score_entropy"]),
                "neg_top1_prob_score": float(entropy_df.loc[i, "score_neg_top1_prob"]),
                "neg_margin_score": float(entropy_df.loc[i, "score_neg_margin"]),
            }
        )
    score_df = pd.DataFrame(score_rows)

    metric_path = out_dir / "ood_generalization_metrics.csv"
    score_path = out_dir / "ood_generalization_scores.csv"
    report_path = out_dir / "ood_generalization_report.md"
    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")
    score_df.to_csv(score_path, index=False, encoding="utf-8-sig")
    plot_scores(metric_df, out_dir / "ood_generalization_auroc.png")

    best_hidden = metric_df[metric_df["method"] == f"hidden_layer_{args.layer}"].iloc[0]
    report = f"""# Experiment 5: OOD 泛化实验初步报告

## 设置

- 训练数据：`{args.train_data}`，controlled_v4，600 条。
- 测试数据：`{args.test_data}`，OOD_v1，200 条。
- 模型：Qwen2.5-0.5B-Instruct。
- Hidden probe 使用层：Layer {args.layer}。
- 训练方式：只在 controlled_v4 hidden states 上训练 Logistic Regression probe，然后直接测试 OOD_v1。

## 结果

| 方法 | OOD AUROC | median 阈值 Accuracy | F1 |
| --- | ---: | ---: | ---: |
"""
    for _, row in metric_df.iterrows():
        report += f"| {row['method']} | {row['auroc']:.4f} | {row['accuracy_at_median']:.4f} | {row['f1_at_median']:.4f} |\n"

    report += f"""
## 初步解释

Hidden-state probe 在 OOD_v1 上的 AUROC 为 {best_hidden['auroc']:.4f}。
这个结果应被解释为“从 controlled_v4 学到的 query-level risk signal 在新实体/新关系上存在一定泛化”，
但还不能说明它已经能稳定检测真实幻觉。

如果后续要写论文，建议把 OOD_v1 作为分布外泛化的初步证据，并继续扩充不同主题的数据。
"""
    report_path.write_text(report, encoding="utf-8")

    print("saved:", metric_path)
    print("saved:", score_path)
    print("saved:", report_path)
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
