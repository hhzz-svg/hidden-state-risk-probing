# -*- coding: utf-8 -*-
"""
21_train_layerwise_probe.py
修正版：兼容 hidden_states_pilot.pt 保存为 list[dict] 的情况。
运行：
python scripts/21_train_layerwise_probe.py --hidden_file outputs/qwen05b/experiment1/experiment1_hidden_extraction/hidden_states_pilot.pt --data_file data/experiment1/prompts_pilot.jsonl --out_dir outputs/qwen05b/experiment2/experiment2_layerwise_probe
"""
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL_KEYS = ["label", "risk_label", "known_unknown", "type", "category", "y"]
HIDDEN_KEYS = ["hidden_states", "hiddens", "all_hidden_states", "last_token_hidden_states", "hidden", "X"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
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
    s = str(x).strip().lower()
    if s in ["known", "know", "k", "0", "false", "low", "low-risk", "low_risk"]:
        return 0
    if s in ["unknown", "unk", "u", "1", "true", "high", "high-risk", "high_risk"]:
        return 1
    raise ValueError(f"无法识别标签：{x!r}。请确保标签是 known/unknown 或 0/1。")


def extract_labels_from_jsonl(data_file: Path) -> Optional[np.ndarray]:
    rows = load_jsonl(data_file)
    if not rows:
        return None
    labels = []
    for i, row in enumerate(rows):
        found = None
        for k in LABEL_KEYS:
            if k in row:
                found = row[k]
                break
        if found is None:
            raise ValueError(f"{data_file} 第 {i+1} 行没有找到标签字段，可接受字段名：{LABEL_KEYS}")
        labels.append(normalize_label(found))
    return np.array(labels, dtype=np.int64)


def find_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def normalize_single_hidden(h: Any) -> np.ndarray:
    """把单条样本 hidden 统一成 [L, H]。"""
    # 常见：hidden_states 是 tuple/list，每个元素对应一层
    if isinstance(h, (list, tuple)) and len(h) > 0 and any(isinstance(x, torch.Tensor) for x in h):
        layer_vecs = []
        for item in h:
            arr = to_np(item)
            if arr.ndim == 3 and arr.shape[0] == 1:      # [1, seq, H]
                arr = arr[0]
            if arr.ndim == 2:                            # [seq, H]
                arr = arr[-1]
            if arr.ndim != 1:                            # [H]
                raise ValueError(f"无法把某层 hidden 转成 [H]，当前 shape={arr.shape}")
            layer_vecs.append(arr.astype(np.float32))
        return np.stack(layer_vecs, axis=0)

    arr = to_np(h)
    if arr.ndim == 1:                                    # [H]
        arr = arr[None, :]
    elif arr.ndim == 2:                                  # [L, H]
        pass
    elif arr.ndim == 3:
        if arr.shape[0] == 1:                            # [1, L, H]
            arr = arr[0]
        else:                                            # [L, seq, H]
            arr = arr[:, -1, :]
    elif arr.ndim == 4:                                  # [1, L, seq, H]
        if arr.shape[0] != 1:
            raise ValueError(f"4维 hidden 只支持 batch=1，当前 shape={arr.shape}")
        arr = arr[0, :, -1, :]
    else:
        raise ValueError(f"不支持的单样本 hidden 维度：shape={arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"单样本 hidden 最终应为 [L,H]，当前 shape={arr.shape}")
    return arr.astype(np.float32)


def normalize_batch_hidden(x: Any) -> np.ndarray:
    """把整体 hidden 统一成 [N, L, H]。"""
    if isinstance(x, (list, tuple)):
        samples = [normalize_single_hidden(item) for item in x]
        return np.stack(samples, axis=0).astype(np.float32)
    arr = to_np(x)
    if arr.ndim == 2:                                    # [N,H]
        arr = arr[:, None, :]
    elif arr.ndim == 3:                                  # [N,L,H]
        pass
    elif arr.ndim == 4:                                  # [N,L,seq,H]
        arr = arr[:, :, -1, :]
    else:
        raise ValueError(f"整体 hidden 维度应为 [N,H]/[N,L,H]/[N,L,S,H]，当前 shape={arr.shape}")
    return arr.astype(np.float32)


def load_list_of_dicts(records: List[Dict[str, Any]], data_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    hidden_list, labels = [], []
    complete_labels = True
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(f"records[{i}] 不是 dict，而是 {type(rec)}")
        hidden_key = find_key(rec, HIDDEN_KEYS)
        if hidden_key is None:
            raise KeyError(f"records[{i}] 找不到 hidden 字段。可接受字段名：{HIDDEN_KEYS}。当前字段：{list(rec.keys())}")
        hidden_list.append(normalize_single_hidden(rec[hidden_key]))
        label_key = find_key(rec, LABEL_KEYS)
        if label_key is None:
            complete_labels = False
        else:
            labels.append(normalize_label(rec[label_key]))
    hidden = np.stack(hidden_list, axis=0).astype(np.float32)
    if complete_labels and len(labels) == len(records):
        y = np.array(labels, dtype=np.int64)
    else:
        y = extract_labels_from_jsonl(data_file)
        if y is None:
            raise ValueError("pt 文件 records 里没有完整标签，且无法从 data_file 读取标签。")
    return hidden, y


def load_hidden_and_labels(hidden_file: Path, data_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    obj = torch.load(hidden_file, map_location="cpu")
    if isinstance(obj, dict):
        hidden_key = find_key(obj, HIDDEN_KEYS)
        if hidden_key is None:
            raise KeyError(f"{hidden_file} 是 dict，但找不到 hidden 字段。可接受字段：{HIDDEN_KEYS}。当前字段：{list(obj.keys())}")
        hidden = normalize_batch_hidden(obj[hidden_key])
        label_key = find_key(obj, ["labels", "risk_labels", "known_unknown_labels", "y", "label"])
        if label_key is not None:
            labels = np.array([normalize_label(v) for v in obj[label_key]], dtype=np.int64)
        else:
            labels = extract_labels_from_jsonl(data_file)
            if labels is None:
                raise KeyError(f"{hidden_file} 中没有 labels，且无法从 {data_file} 读取标签。")
    elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        hidden, labels = load_list_of_dicts(obj, data_file)
    else:
        hidden = normalize_batch_hidden(obj)
        labels = extract_labels_from_jsonl(data_file)
        if labels is None:
            raise ValueError(f"{hidden_file} 没有标签；同时 {data_file} 不存在或为空。")

    if hidden.ndim == 2:
        hidden = hidden[:, None, :]
    if hidden.ndim != 3:
        raise ValueError(f"hidden states 最终维度应为 [N,L,H]，当前 shape={hidden.shape}")
    n = len(labels)
    if hidden.shape[0] != n and hidden.shape[1] == n:
        hidden = np.transpose(hidden, (1, 0, 2))
    if hidden.shape[0] != n:
        raise ValueError(f"样本数不匹配：hidden.shape[0]={hidden.shape[0]}，labels={n}。")
    return hidden.astype(np.float32), labels.astype(np.int64)


def evaluate_one_layer(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int, max_iter: int) -> Dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="liblinear", random_state=seed),
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, accs, ps, rs, f1s = [], [], [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_test, proba))
        accs.append(accuracy_score(y_test, pred))
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        ps.append(p); rs.append(r); f1s.append(f1)
    return {
        "auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "precision_mean": float(np.mean(ps)), "precision_std": float(np.std(ps)),
        "recall_mean": float(np.mean(rs)), "recall_std": float(np.std(rs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
    }


def plot_metric_curve(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["layer"], df[f"{metric}_mean"], marker="o")
    if f"{metric}_std" in df.columns:
        y = df[f"{metric}_mean"].to_numpy()
        e = df[f"{metric}_std"].to_numpy()
        x = df["layer"].to_numpy()
        plt.fill_between(x, y - e, y + e, alpha=0.2)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(f"Layer-wise Probe {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_file", type=str, default="outputs/qwen05b/experiment1/experiment1_hidden_extraction/hidden_states_pilot.pt")
    parser.add_argument("--data_file", type=str, default="data/experiment1/prompts_pilot.jsonl")
    parser.add_argument("--out_dir", type=str, default="outputs/qwen05b/experiment2/experiment2_layerwise_probe")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=2000)
    args = parser.parse_args()

    hidden_file = Path(args.hidden_file)
    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden, labels = load_hidden_and_labels(hidden_file, data_file)
    n, num_layers, hidden_size = hidden.shape
    known_count = int((labels == 0).sum())
    unknown_count = int((labels == 1).sum())

    print("=" * 80)
    print("Experiment 2: Layer-wise Hidden-State Probe")
    print(f"hidden_file: {hidden_file}")
    print(f"data_file:   {data_file}")
    print(f"hidden shape: N={n}, L={num_layers}, H={hidden_size}")
    print(f"labels: known={known_count}, unknown={unknown_count}")
    print("=" * 80)

    if min(known_count, unknown_count) < args.n_splits:
        raise ValueError(f"每类样本数必须 >= n_splits。当前 known={known_count}, unknown={unknown_count}, n_splits={args.n_splits}")

    rows = []
    for layer in range(num_layers):
        metrics = evaluate_one_layer(hidden[:, layer, :], labels, args.n_splits, args.seed, args.max_iter)
        rows.append({"layer": layer, **metrics})
        print(f"Layer {layer:02d} | AUROC={metrics['auroc_mean']:.4f}±{metrics['auroc_std']:.4f} | ACC={metrics['accuracy_mean']:.4f} | F1={metrics['f1_mean']:.4f}")

    df = pd.DataFrame(rows)
    metrics_path = out_dir / "layerwise_probe_metrics.csv"
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    best_idx = int(df["auroc_mean"].idxmax())
    best_row = df.loc[best_idx].to_dict()
    summary = {
        "hidden_file": str(hidden_file), "data_file": str(data_file),
        "num_samples": n, "num_layers": num_layers, "hidden_size": hidden_size,
        "known_count": known_count, "unknown_count": unknown_count,
        "best_layer_by_auroc": int(best_row["layer"]),
        "best_auroc_mean": float(best_row["auroc_mean"]),
        "best_auroc_std": float(best_row["auroc_std"]),
        "best_accuracy_mean": float(best_row["accuracy_mean"]),
        "best_f1_mean": float(best_row["f1_mean"]),
        "n_splits": args.n_splits, "seed": args.seed,
    }
    summary_path = out_dir / "layerwise_probe_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_metric_curve(df, "auroc", out_dir / "layer_auc_curve.png", "AUROC")
    plot_metric_curve(df, "f1", out_dir / "layer_f1_curve.png", "F1")

    print("=" * 80)
    print("保存完成：")
    print(f"- {metrics_path}")
    print(f"- {summary_path}")
    print(f"- {out_dir / 'layer_auc_curve.png'}")
    print(f"- {out_dir / 'layer_f1_curve.png'}")
    print("=" * 80)
    print("最佳层摘要：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
