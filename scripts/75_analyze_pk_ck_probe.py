# -*- coding: utf-8 -*-
"""
75_analyze_pk_ck_probe.py

Optional Experiment A: summarize PK/CK behavior and probe results, including
simple text baselines for the qwen15b pk_follow vs ck_follow target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

QWEN05_LABELS = ROOT / "outputs/qwen05b/experiment7_optionalA/pk_ck_generation/pk_ck_behavior_labels.csv"
QWEN15_LABELS = ROOT / "outputs/qwen15b/experiment7_optionalA/pk_ck_generation/pk_ck_behavior_labels.csv"
QWEN15_PROBE_SUMMARY = ROOT / "outputs/qwen15b/experiment7_optionalA/hidden_probe/pk_ck_layerwise_probe/layerwise_probe_summary.json"
QWEN15_PROBE_METRICS = ROOT / "outputs/qwen15b/experiment7_optionalA/hidden_probe/pk_ck_layerwise_probe/layerwise_probe_metrics.csv"

REPORT_DIR = ROOT / "reports/experiment7_optionalA"
TABLE_DIR = REPORT_DIR / "tables"
BASELINE_CSV = TABLE_DIR / "pk_ck_qwen15b_text_baselines.csv"
PROBE_SUMMARY_CSV = TABLE_DIR / "pk_ck_probe_summary.csv"
REPORT_MD = REPORT_DIR / "optionalA_pk_ck_probe_report.md"

TARGET_MAP = {"pk_follow": 0, "ck_follow": 1}


def load_labels(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["model"] = model
    return df


def evaluate_scores(y: np.ndarray, score: np.ndarray, name: str) -> dict[str, float | str]:
    auroc = roc_auc_score(y, score)
    threshold = float(np.median(score))
    pred = (score >= threshold).astype(int)
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "method": name,
        "auroc_mean": float(auroc),
        "accuracy_mean": float(acc),
        "precision_mean": float(p),
        "recall_mean": float(r),
        "f1_mean": float(f1),
        "n_splits": 1,
    }


def cv_length_baseline(df: pd.DataFrame, text_col: str, y: np.ndarray, n_splits: int) -> dict[str, float | str]:
    score = df[text_col].astype(str).str.len().to_numpy(dtype=float)
    # If larger length points in the wrong direction, AUROC will expose that.
    return evaluate_scores(y, score, f"{text_col}_length")


def cv_tfidf_baseline(texts: list[str], y: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict[str, float | str]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, accs, ps, rs, f1s = [], [], [], [], []
    for train_idx, test_idx in skf.split(texts, y):
        x_train = [texts[i] for i in train_idx]
        x_test = [texts[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=1),
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed),
        )
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_test, proba))
        accs.append(accuracy_score(y_test, pred))
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    return {
        "method": "tfidf_char_ngram_prompt",
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)),
        "precision_mean": float(np.mean(ps)),
        "recall_mean": float(np.mean(rs)),
        "f1_mean": float(np.mean(f1s)),
        "n_splits": n_splits,
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    qwen05 = load_labels(QWEN05_LABELS, "Qwen2.5-0.5B-Instruct")
    qwen15 = load_labels(QWEN15_LABELS, "Qwen2.5-1.5B-Instruct")
    all_df = pd.concat([qwen05, qwen15], ignore_index=True)

    qwen15_target = qwen15[qwen15["behavior_label"].isin(TARGET_MAP)].copy()
    qwen15_target["target"] = qwen15_target["behavior_label"].map(TARGET_MAP).astype(int)
    y = qwen15_target["target"].to_numpy(dtype=int)

    baseline_rows = [
        cv_length_baseline(qwen15_target, "prompt", y, n_splits=5),
        cv_length_baseline(qwen15_target, "model_answer", y, n_splits=5),
        cv_tfidf_baseline(qwen15_target["prompt"].astype(str).tolist(), y, n_splits=5),
    ]
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(BASELINE_CSV, index=False, encoding="utf-8-sig")

    with QWEN15_PROBE_SUMMARY.open("r", encoding="utf-8") as f:
        probe_summary = json.load(f)
    probe_df = pd.DataFrame(
        [
            {
                "model": "Qwen2.5-1.5B-Instruct",
                "target": "pk_follow_vs_ck_follow",
                "num_samples": probe_summary["num_samples"],
                "pk_follow_count": int((y == 0).sum()),
                "ck_follow_count": int((y == 1).sum()),
                "best_layer": probe_summary["best_layer_by_auroc"],
                "best_auroc_mean": probe_summary["best_auroc_mean"],
                "best_auroc_std": probe_summary["best_auroc_std"],
                "best_accuracy_mean": probe_summary["best_accuracy_mean"],
                "best_f1_mean": probe_summary["best_f1_mean"],
            }
        ]
    )
    probe_df.to_csv(PROBE_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    behavior_summary = (
        all_df.groupby(["model", "behavior_label"])
        .size()
        .rename("count")
        .reset_index()
    )
    behavior_total = all_df.groupby("model").size().rename("total").reset_index()
    behavior_summary = behavior_summary.merge(behavior_total, on="model")
    behavior_summary["rate"] = behavior_summary["count"] / behavior_summary["total"]

    lines = [
        "# 可选实验A：PK/CK hidden-state probe 初步报告",
        "",
        "## 1. 定位",
        "",
        "本实验是主线实验之外的扩展探索：当 prompt 中的 CK（上下文知识）与 PK（参数知识）冲突时，生成前 hidden states 是否能预测模型最终更偏向 PK 还是 CK。",
        "",
        "它不替代主线实验 1-6，也不等同于真实 hallucination detection（幻觉检测）。",
        "",
        "## 2. 生成行为",
        "",
        "| 模型 | pk_follow | ck_follow | mixed/conflict | other |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for model in ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"]:
        sub = behavior_summary[behavior_summary["model"] == model].set_index("behavior_label")
        lines.append(
            f"| {model} | "
            f"{int(sub.loc['pk_follow', 'count']) if 'pk_follow' in sub.index else 0} | "
            f"{int(sub.loc['ck_follow', 'count']) if 'ck_follow' in sub.index else 0} | "
            f"{int(sub.loc['mixed_or_conflict_ack', 'count']) if 'mixed_or_conflict_ack' in sub.index else 0} | "
            f"{int(sub.loc['other', 'count']) if 'other' in sub.index else 0} |"
        )

    lines.extend(
        [
            "",
            "0.5B 的 pk_follow 只有 4 条，类别极不平衡，因此不适合直接做稳定二分类 probe；1.5B 有 pk_follow=38、ck_follow=22，可做小规模探索。",
            "",
            "## 3. Qwen2.5-1.5B hidden-state probe",
            "",
            "| 样本数 | pk_follow | ck_follow | 最佳层 | AUROC | Accuracy | F1 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {probe_summary['num_samples']} | {(y == 0).sum()} | {(y == 1).sum()} | "
            f"{probe_summary['best_layer_by_auroc']} | "
            f"{probe_summary['best_auroc_mean']:.4f} ± {probe_summary['best_auroc_std']:.4f} | "
            f"{probe_summary['best_accuracy_mean']:.4f} | {probe_summary['best_f1_mean']:.4f} |",
            "",
            "## 4. 文本基线",
            "",
            "| 方法 | AUROC | Accuracy | F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for _, row in baseline_df.iterrows():
        auroc_std = row.get("auroc_std", np.nan)
        auroc_text = f"{row['auroc_mean']:.4f}" if pd.isna(auroc_std) else f"{row['auroc_mean']:.4f} ± {auroc_std:.4f}"
        lines.append(
            f"| {row['method']} | {auroc_text} | {row['accuracy_mean']:.4f} | {row['f1_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 保守结论",
            "",
            "在这版 100 条 PK/CK 冲突数据上，0.5B 更倾向跟随上下文 CK；1.5B 行为更分化，既有 PK follow，也有 CK follow 和混合解释。对 1.5B 的 pk_follow/ck_follow 子集，生成前 hidden states 在中后层表现出较强可解码性。",
            "",
            "但该结论必须保守：样本数只有 60 条用于 probe，行为标签仍是答案字符串辅助标注，且 PK/CK prompt 模板较统一。后续若要写进正文，建议放在扩展实验或附录；若要强化，应扩大数据并加入人工复核。",
        ]
    )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", BASELINE_CSV)
    print("saved:", PROBE_SUMMARY_CSV)
    print("saved:", REPORT_MD)
    print(probe_df.to_string(index=False))
    print(baseline_df.to_string(index=False))


if __name__ == "__main__":
    main()
