import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_hidden_records(path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, list) or not obj or not isinstance(obj[0], dict):
        raise ValueError("expected hidden file to be list[dict]")
    hidden = torch.stack([item["hidden"] for item in obj]).float().numpy()
    metadata = pd.DataFrame([{k: v for k, v in item.items() if k != "hidden"} for item in obj])
    return hidden, metadata


def evaluate_cv(X, y, groups, model, n_splits, name):
    cv = GroupKFold(n_splits=n_splits)
    aucs, accs, f1s = [], [], []

    for train_idx, test_idx in cv.split(X, y, groups):
        if isinstance(X, list):
            X_train = [X[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        prob = fitted.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_test, prob))
        accs.append(accuracy_score(y_test, pred))
        f1s.append(f1_score(y_test, pred))

    return {
        "name": name,
        "AUROC_mean": float(np.mean(aucs)),
        "AUROC_std": float(np.std(aucs)),
        "Accuracy_mean": float(np.mean(accs)),
        "Accuracy_std": float(np.std(accs)),
        "F1_mean": float(np.mean(f1s)),
        "F1_std": float(np.std(f1s)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_file", default="outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/hidden_states.pt")
    parser.add_argument("--data_file", default="data/experiment3/prompts_controlled_v3.jsonl")
    parser.add_argument("--out_dir", default="outputs/qwen05b/experiment3/experiment3_v4/hidden_probe")
    parser.add_argument("--group_col", default="entity", choices=["entity", "category", "relation"])
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.data_file)
    texts = [row["prompt"] for row in rows]
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    data_meta = pd.DataFrame(rows)

    hidden, hidden_meta = load_hidden_records(args.hidden_file)
    if hidden.shape[0] != len(rows):
        raise ValueError(f"sample mismatch: hidden={hidden.shape[0]}, data={len(rows)}")

    groups = data_meta[args.group_col].astype(str).to_numpy()
    group_counts = data_meta.groupby(args.group_col)["label"].agg(["count", "nunique"])
    group_counts.to_csv(out_dir / f"group_split_{args.group_col}_group_counts.csv", encoding="utf-8-sig")

    length_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    X_len = data_meta["prompt"].map(len).to_numpy()[:, None]
    length_result = evaluate_cv(
        X_len, y, groups, length_model, args.n_splits, f"group_{args.group_col}_length_baseline"
    )

    tfidf_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    max_features=5000,
                    min_df=1,
                ),
            ),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    tfidf_result = evaluate_cv(
        texts, y, groups, tfidf_model, args.n_splits, f"group_{args.group_col}_tfidf_baseline"
    )

    hidden_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )

    layer_rows = []
    for layer in range(hidden.shape[1]):
        result = evaluate_cv(
            hidden[:, layer, :],
            y,
            groups,
            hidden_model,
            args.n_splits,
            f"group_{args.group_col}_hidden_layer_{layer}",
        )
        layer_rows.append({"layer": layer, **result})
        print(
            f"Layer {layer:02d} | AUROC={result['AUROC_mean']:.4f}±{result['AUROC_std']:.4f} "
            f"| ACC={result['Accuracy_mean']:.4f} | F1={result['F1_mean']:.4f}"
        )

    layer_df = pd.DataFrame(layer_rows)
    layer_path = out_dir / f"group_split_{args.group_col}_layerwise_probe_metrics.csv"
    layer_df.to_csv(layer_path, index=False, encoding="utf-8-sig")

    best = layer_df.loc[layer_df["AUROC_mean"].idxmax()].to_dict()
    summary = {
        "group_col": args.group_col,
        "n_splits": args.n_splits,
        "num_samples": int(len(rows)),
        "num_groups": int(len(np.unique(groups))),
        "length_baseline": length_result,
        "tfidf_baseline": tfidf_result,
        "best_layer": int(best["layer"]),
        "best_hidden_auroc_mean": float(best["AUROC_mean"]),
        "best_hidden_auroc_std": float(best["AUROC_std"]),
        "best_hidden_accuracy_mean": float(best["Accuracy_mean"]),
        "best_hidden_f1_mean": float(best["F1_mean"]),
    }

    summary_path = out_dir / f"group_split_{args.group_col}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame([length_result, tfidf_result]).to_csv(
        out_dir / f"group_split_{args.group_col}_text_baselines.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)
    print("[OUT]", layer_path)
    print("[OUT]", summary_path)


if __name__ == "__main__":
    main()
