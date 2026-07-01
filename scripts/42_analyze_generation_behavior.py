# -*- coding: utf-8 -*-
"""
42_analyze_generation_behavior.py

Experiment 4: analyze the relationship between pre-generation risk scores and
actual generation behavior.

注意：
1. query-level risk label 不是真实 hallucination label。
2. 如果人工标注表为空，本脚本只生成 provisional（初稿）行为标签，不能当作最终人工标注结果。

推荐运行：
python scripts/42_analyze_generation_behavior.py ^
  --data_file data/experiment3/prompts_controlled_v4.jsonl ^
  --generated_file outputs/qwen05b/experiment4/experiment4_generation_behavior/generated_answers.jsonl ^
  --manual_label_file outputs/qwen05b/experiment4/experiment4_generation_behavior/manual_behavior_labels_sample200.csv ^
  --hidden_file outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/hidden_states.pt ^
  --entropy_file outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/entropy_baseline_scores.csv ^
  --summary_file outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/group_split_entity_summary.json ^
  --out_dir outputs/qwen05b/experiment4/experiment4_generation_behavior_analysis
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABEL_KEYS = ["label", "risk_label", "known_unknown", "type", "category", "y"]
HIDDEN_KEYS = ["hidden_states", "hiddens", "all_hidden_states", "last_token_hidden_states", "hidden", "X"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_label(x: Any) -> int:
    if isinstance(x, (int, np.integer)) and int(x) in [0, 1]:
        return int(x)
    if isinstance(x, float) and int(x) in [0, 1]:
        return int(x)
    text = str(x).strip().lower()
    if text in ["known", "know", "k", "0", "false", "low", "low-risk", "low_risk"]:
        return 0
    if text in ["unknown", "unk", "u", "1", "true", "high", "high-risk", "high_risk"]:
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
    if isinstance(h, (list, tuple)) and h and any(isinstance(x, torch.Tensor) for x in h):
        layers = []
        for item in h:
            arr = to_np(item)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 2:
                arr = arr[-1]
            if arr.ndim != 1:
                raise ValueError(f"无法转换 hidden layer shape={arr.shape}")
            layers.append(arr.astype(np.float32))
        return np.stack(layers, axis=0)

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
        if arr.shape[0] != 1:
            raise ValueError(f"4D hidden 只支持 batch=1，当前 shape={arr.shape}")
        arr = arr[0, :, -1, :]
    else:
        raise ValueError(f"不支持的 hidden shape={arr.shape}")
    return arr.astype(np.float32)


def load_hidden_file(path: Path) -> Tuple[np.ndarray, List[str]]:
    obj = torch.load(path, map_location="cpu")
    ids: List[str] = []
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        hidden = []
        for i, row in enumerate(obj):
            hidden_key = find_key(row, HIDDEN_KEYS)
            if hidden_key is None:
                raise KeyError(f"hidden record {i} 缺少 hidden 字段")
            hidden.append(normalize_single_hidden(row[hidden_key]))
            ids.append(str(row.get("id", f"sample_{i:04d}")))
        return np.stack(hidden, axis=0), ids

    if isinstance(obj, dict):
        hidden_key = find_key(obj, HIDDEN_KEYS)
        if hidden_key is None:
            raise KeyError(f"{path} 缺少 hidden 字段")
        arr = to_np(obj[hidden_key])
    else:
        arr = to_np(obj)

    if arr.ndim == 2:
        arr = arr[:, None, :]
    if arr.ndim == 4:
        arr = arr[:, :, -1, :]
    if arr.ndim != 3:
        raise ValueError(f"hidden 最终应为 [N,L,H]，当前 shape={arr.shape}")
    ids = [f"sample_{i:04d}" for i in range(arr.shape[0])]
    return arr.astype(np.float32), ids


def get_best_layer(summary_file: Path, fallback: int) -> int:
    if not summary_file.exists():
        return fallback
    with summary_file.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return int(summary.get("best_layer", fallback))


def compute_oof_hidden_scores(
    hidden: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    layer: int,
    n_splits: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    X = hidden[:, layer, :]
    scores = np.zeros(len(labels), dtype=np.float32)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed),
    )

    unique_groups = len(set(groups.tolist()))
    if unique_groups >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X, labels, groups)
        split_name = "GroupKFold(entity)"
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = cv.split(X, labels)
        split_name = "StratifiedKFold"

    preds = np.zeros(len(labels), dtype=np.int64)
    for train_idx, test_idx in split_iter:
        clf.fit(X[train_idx], labels[train_idx])
        fold_scores = clf.predict_proba(X[test_idx])[:, 1]
        scores[test_idx] = fold_scores
        preds[test_idx] = (fold_scores >= 0.5).astype(np.int64)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    metrics = {
        "split": split_name,
        "layer": int(layer),
        "auroc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }
    return scores, metrics


def is_refusal(raw_answer: str, answer: str) -> bool:
    text = f"{answer}\n{raw_answer}".strip().lower()
    first = answer.strip().lower()
    refusal_patterns = [
        "不知道", "不清楚", "无法确定", "无法回答", "不能确定", "不能回答",
        "没有直接", "无直接", "不存在", "不适用", "未知", "无法提供",
        "not sure", "cannot", "can't", "unknown", "no direct",
    ]
    if first and len(first) <= 12 and not any(p in first for p in refusal_patterns):
        return False
    return any(pattern in text[:160] for pattern in refusal_patterns)


def is_irrelevant(raw_answer: str, answer: str) -> bool:
    first = answer.strip()
    if not first:
        return True
    irrelevant_patterns = ["Human:", "问题描述", "以下选项", "小天才", "A.", "B.", "C.", "D."]
    return any(pattern in first[:80] for pattern in irrelevant_patterns)


def provisional_behavior(row: pd.Series) -> Tuple[str, str]:
    label = int(row["label"])
    answer = str(row.get("model_answer", ""))
    raw = str(row.get("model_answer_raw", ""))

    if is_irrelevant(raw, answer):
        return "irrelevant", "auto: answer appears to drift away from the query"
    if is_refusal(raw, answer):
        return "refusal", "auto: refusal/uncertainty phrase detected"
    if label == 1:
        return "hallucination", "auto: high-risk entity-relation query received a specific answer"
    return "answered_known_unverified", "auto: known query received a specific answer; factual correctness not manually verified"


def load_manual_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["id", "behavior_label", "notes"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "behavior_label" not in df.columns:
        df["behavior_label"] = ""
    if "notes" not in df.columns:
        df["notes"] = ""
    return df


def merge_behavior_labels(generated_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    manual = manual_df[["id", "behavior_label", "notes"]].copy()
    manual["behavior_label"] = manual["behavior_label"].fillna("").astype(str).str.strip()
    manual["notes"] = manual["notes"].fillna("").astype(str)

    df = generated_df.merge(manual, on="id", how="left")
    df["behavior_label"] = df["behavior_label"].fillna("").astype(str).str.strip()
    df["notes"] = df["notes"].fillna("").astype(str)

    auto_labels, auto_notes = [], []
    for _, row in df.iterrows():
        label, note = provisional_behavior(row)
        auto_labels.append(label)
        auto_notes.append(note)
    df["auto_behavior_label"] = auto_labels
    df["auto_behavior_notes"] = auto_notes

    df["final_behavior_label"] = np.where(
        df["behavior_label"] != "",
        df["behavior_label"],
        df["auto_behavior_label"],
    )
    df["label_source"] = np.where(df["behavior_label"] != "", "manual", "auto_provisional")
    return df


def evaluate_binary_target(df: pd.DataFrame, target_col: str, score_cols: List[str]) -> pd.DataFrame:
    rows = []
    y = df[target_col].astype(int).to_numpy()
    if len(set(y.tolist())) < 2:
        return pd.DataFrame()
    for score_col in score_cols:
        score = df[score_col].astype(float).to_numpy()
        auroc = roc_auc_score(y, score)
        threshold = float(np.median(score))
        pred = (score >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        rows.append(
            {
                "target": target_col,
                "score": score_col,
                "auroc": float(auroc),
                "median_threshold": threshold,
                "accuracy_at_median": float(accuracy_score(y, pred)),
                "precision_at_median": float(p),
                "recall_at_median": float(r),
                "f1_at_median": float(f1),
            }
        )
    return pd.DataFrame(rows)


def plot_behavior_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["final_behavior_label"].value_counts().sort_index()
    plt.figure(figsize=(8, 4.8))
    counts.plot(kind="bar", color="#4C78A8")
    plt.xlabel("Behavior label")
    plt.ylabel("Count")
    plt.title("Generation Behavior Distribution")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_risk_by_behavior(df: pd.DataFrame, out_path: Path) -> None:
    labels = sorted(df["final_behavior_label"].dropna().unique().tolist())
    data = [df.loc[df["final_behavior_label"] == label, "hidden_risk_score_oof"].astype(float).to_numpy() for label in labels]
    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("Hidden risk score (OOF)")
    plt.xlabel("Behavior label")
    plt.title("Hidden Risk Score by Generation Behavior")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def df_to_markdown_simple(df: pd.DataFrame, include_index: bool = False) -> str:
    table = df.reset_index() if include_index else df.copy()
    if table.empty:
        return ""
    columns = [str(col) for col in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in table.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    out_path: Path,
    df: pd.DataFrame,
    hidden_metrics: Dict[str, float],
    behavior_metrics: pd.DataFrame,
    model_label: str,
) -> None:
    source_counts = df["label_source"].value_counts().to_dict()
    behavior_counts = df["final_behavior_label"].value_counts().to_dict()
    by_query = pd.crosstab(df["label_name"], df["final_behavior_label"])
    if source_counts.get("manual", 0) == len(df):
        label_note = "本报告使用外部标签文件中的 `behavior_label`。如果该文件来自 answer key，它属于答案表辅助标注；如果来自人工填写，则属于人工标注。"
    else:
        label_note = "本报告中仍有部分 `auto_provisional` 标签，这些是自动生成的行为标注初稿，不能等同于最终人工标注。"

    lines = [
        "# 第四次实验初步分析报告",
        "",
        "## 重要说明",
        "",
        label_note,
        "`label` 仍然只是 query-level risk label（问题级风险标签），不是真实 hallucination label（幻觉标签）。",
        "",
        "## 数据概况",
        "",
        f"- 模型：{model_label}",
        f"- 生成样本数：{len(df)}",
        f"- 标签来源统计：{source_counts}",
        f"- 行为标签分布：{behavior_counts}",
        "",
        "## Hidden risk score 设置",
        "",
        f"- 分割方式：{hidden_metrics.get('split')}",
        f"- 使用层：Layer {hidden_metrics.get('layer')}",
        f"- 对 query-level risk label 的 AUROC：{hidden_metrics.get('auroc'):.4f}",
        f"- Accuracy：{hidden_metrics.get('accuracy'):.4f}",
        f"- F1：{hidden_metrics.get('f1'):.4f}",
        "",
        "## 按 query-level label 的行为分布",
        "",
        df_to_markdown_simple(by_query, include_index=True),
        "",
        "## 行为预测指标",
        "",
        df_to_markdown_simple(behavior_metrics) if not behavior_metrics.empty else "当前目标只有一个类别，无法计算 AUROC。",
        "",
        "## 当前可写的保守结论",
        "",
        f"在当前答案表辅助标注下，{model_label} 对 high-risk entity-relation query 大多会给出具体答案，",
        "这些回答可被标记为 hallucination-like behavior（类幻觉行为）。",
        "同时，known query 中也存在一部分事实错误回答，因此第四次实验不再只是复现 query-level label。",
        "不过该标注仍建议抽样人工复核，论文中应写作“答案表辅助标注/初步行为标注”，避免说成严格人工金标准。",
        "",
        "## 下一步",
        "",
        "优先人工复核 `answer_key_behavior_labels.csv` 中的错误样本，尤其是别名、翻译名和有争议实体。",
        "复核后重新运行本脚本，以修订后的行为标签更新最终图表。",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data/experiment3/prompts_controlled_v4.jsonl")
    parser.add_argument("--generated_file", default="outputs/qwen05b/experiment4/experiment4_generation_behavior/generated_answers.jsonl")
    parser.add_argument("--manual_label_file", default="outputs/qwen05b/experiment4/experiment4_generation_behavior/manual_behavior_labels_sample200.csv")
    parser.add_argument("--hidden_file", default="outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/hidden_states.pt")
    parser.add_argument("--entropy_file", default="outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/entropy_baseline_scores.csv")
    parser.add_argument("--summary_file", default="outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/group_split_entity_summary.json")
    parser.add_argument("--out_dir", default="outputs/qwen05b/experiment4/experiment4_generation_behavior_analysis")
    parser.add_argument("--model_label", default="Qwen2.5 model")
    parser.add_argument("--best_layer", type=int, default=18)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_file = Path(args.data_file)
    generated_file = Path(args.generated_file)
    manual_label_file = Path(args.manual_label_file)
    hidden_file = Path(args.hidden_file)
    entropy_file = Path(args.entropy_file)
    summary_file = Path(args.summary_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_rows = load_jsonl(data_file)
    generated_rows = load_jsonl(generated_file)
    data_df = pd.DataFrame(data_rows)
    gen_df = pd.DataFrame(generated_rows)

    if len(data_df) != len(gen_df):
        raise ValueError(f"data rows={len(data_df)} 与 generated rows={len(gen_df)} 不一致")

    labels = np.array([normalize_label(x) for x in data_df["label"].tolist()], dtype=np.int64)
    groups = data_df["entity"].astype(str).to_numpy()
    layer = get_best_layer(summary_file, args.best_layer)

    hidden, _ = load_hidden_file(hidden_file)
    if hidden.shape[0] != len(data_df):
        raise ValueError(f"hidden samples={hidden.shape[0]} 与 data rows={len(data_df)} 不一致")

    hidden_scores, hidden_metrics = compute_oof_hidden_scores(
        hidden=hidden,
        labels=labels,
        groups=groups,
        layer=layer,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    entropy_df = pd.read_csv(entropy_file, encoding="utf-8-sig")
    if len(entropy_df) != len(data_df):
        raise ValueError(f"entropy rows={len(entropy_df)} 与 data rows={len(data_df)} 不一致")

    manual_df = load_manual_labels(manual_label_file)
    merged = merge_behavior_labels(gen_df, manual_df)
    merged["hidden_risk_score_oof"] = hidden_scores
    merged["entropy_score"] = entropy_df["score_entropy"].astype(float).to_numpy()
    merged["neg_top1_prob_score"] = entropy_df["score_neg_top1_prob"].astype(float).to_numpy()
    merged["neg_margin_score"] = entropy_df["score_neg_margin"].astype(float).to_numpy()

    high_risk_behaviors = {"hallucination", "refusal", "irrelevant"}
    merged["high_risk_behavior"] = merged["final_behavior_label"].isin(high_risk_behaviors).astype(int)
    merged["answered_without_refusal"] = (~merged["final_behavior_label"].isin({"refusal", "irrelevant"})).astype(int)

    behavior_metrics = evaluate_binary_target(
        merged,
        "high_risk_behavior",
        ["hidden_risk_score_oof", "entropy_score", "neg_top1_prob_score", "neg_margin_score"],
    )

    merged_path = out_dir / "generation_behavior_merged_scores.csv"
    provisional_path = out_dir / "behavior_labels_provisional.csv"
    distribution_path = out_dir / "behavior_distribution.csv"
    by_query_path = out_dir / "behavior_by_query_label.csv"
    metrics_path = out_dir / "behavior_prediction_metrics.csv"
    hidden_metrics_path = out_dir / "hidden_score_oof_metrics.json"

    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    merged[[
        "id", "final_behavior_label", "label_source", "auto_behavior_label",
        "auto_behavior_notes", "label", "label_name", "category", "entity",
        "relation", "model_answer", "model_answer_raw",
    ]].to_csv(provisional_path, index=False, encoding="utf-8-sig")
    merged["final_behavior_label"].value_counts().rename_axis("behavior_label").reset_index(name="count").to_csv(
        distribution_path, index=False, encoding="utf-8-sig"
    )
    pd.crosstab(merged["label_name"], merged["final_behavior_label"]).to_csv(by_query_path, encoding="utf-8-sig")
    behavior_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    hidden_metrics_path.write_text(json.dumps(hidden_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_behavior_distribution(merged, out_dir / "behavior_label_distribution.png")
    plot_risk_by_behavior(merged, out_dir / "risk_score_by_behavior.png")
    write_report(out_dir / "experiment4_preliminary_report.md", merged, hidden_metrics, behavior_metrics, args.model_label)

    print("=" * 80)
    print("Experiment 4 behavior analysis finished")
    print(f"out_dir: {out_dir}")
    print(f"merged rows: {len(merged)}")
    print(f"label sources: {merged['label_source'].value_counts().to_dict()}")
    print(f"behavior distribution: {merged['final_behavior_label'].value_counts().to_dict()}")
    print(f"hidden score AUROC for query-level label: {hidden_metrics['auroc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
