import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_text(row):
    for key in ["prompt", "query", "question", "text", "input"]:
        if key in row:
            return str(row[key])
    raise KeyError(f"找不到 prompt/query/question/text/input 字段，当前字段为: {list(row.keys())}")


def get_label(row):
    for key in ["label", "risk_label", "y"]:
        if key in row:
            v = row[key]
            if isinstance(v, str):
                v_low = v.lower()
                if v_low in ["known", "low-risk", "low_risk", "0"]:
                    return 0
                if v_low in ["unknown", "high-risk", "high_risk", "1"]:
                    return 1
            return int(v)
    raise KeyError(f"找不到 label/risk_label/y 字段，当前字段为: {list(row.keys())}")


def cv_binary_scores(X, y, model, n_splits=5, name="model"):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs, accs, f1s = [], [], []

    for train_idx, test_idx in cv.split(X, y):
        if isinstance(X, list):
            X_train = [X[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
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


def load_hidden_tensor(path: str):
    """
    读取 hidden_states_pilot.pt。

    兼容格式：
    1. Tensor: [N, L, D]
    2. List[Tensor]: N 个 [L, D]
    3. List[Dict]: 每个 dict 里有 hidden_states / hidden / hidden_state 等字段
    4. Dict: dict 里有 hidden_states / hidden / samples 等字段
    """
    obj = torch.load(path, map_location="cpu")

    print("[DEBUG] loaded pt type:", type(obj))

    # 情况 1：整体就是 tensor
    if isinstance(obj, torch.Tensor):
        print("[DEBUG] tensor shape:", obj.shape)
        if obj.ndim == 3:
            return obj
        raise ValueError(f"tensor 维度不对，期望 [N,L,D]，实际 {obj.shape}")

    hidden_keys = [
        "hidden_states",
        "hidden_state",
        "hidden",
        "layer_hidden_states",
        "features",
        "all_hidden_states",
        "X",
    ]

    # 情况 2：外层是 list
    if isinstance(obj, list):
        print("[DEBUG] list len:", len(obj))
        first = obj[0]
        print("[DEBUG] first item type:", type(first))

        # 2.1 List[Tensor]
        if isinstance(first, torch.Tensor):
            print("[DEBUG] first tensor shape:", first.shape)
            hidden = torch.stack(obj)
            print("[DEBUG] stacked hidden shape:", hidden.shape)
            if hidden.ndim == 3:
                return hidden
            raise ValueError(f"stack 后维度不对，实际 {hidden.shape}")

        # 2.2 List[Dict]
        if isinstance(first, dict):
            print("[DEBUG] first item keys:", list(first.keys()))

            for key in hidden_keys:
                if key in first:
                    print("[DEBUG] use hidden key:", key)

                    tensors = []
                    for item in obj:
                        h = item[key]

                        if isinstance(h, torch.Tensor):
                            tensors.append(h)
                        else:
                            tensors.append(torch.tensor(h))

                    hidden = torch.stack(tensors)
                    print("[DEBUG] stacked hidden shape:", hidden.shape)

                    if hidden.ndim == 3:
                        return hidden

                    raise ValueError(f"stack 后维度不对，实际 {hidden.shape}")

            raise ValueError(
                "list 里的元素是 dict，但没找到 hidden 字段。"
                f"可用字段为：{list(first.keys())}"
            )

        raise ValueError(f"list 里的元素类型暂不支持：{type(first)}")

    # 情况 3：外层是 dict
    if isinstance(obj, dict):
        print("[DEBUG] dict keys:", list(obj.keys()))

        for key in hidden_keys:
            if key in obj:
                value = obj[key]
                print("[DEBUG] use dict key:", key, "type:", type(value))

                if isinstance(value, torch.Tensor):
                    print("[DEBUG] value shape:", value.shape)
                    if value.ndim == 3:
                        return value

                if isinstance(value, list):
                    hidden = torch.stack([
                        v if isinstance(v, torch.Tensor) else torch.tensor(v)
                        for v in value
                    ])
                    print("[DEBUG] stacked value shape:", hidden.shape)
                    if hidden.ndim == 3:
                        return hidden

        for sample_key in ["samples", "records", "data", "items", "examples"]:
            if sample_key in obj and isinstance(obj[sample_key], list):
                samples = obj[sample_key]
                first = samples[0]
                print("[DEBUG] sample key:", sample_key)
                print("[DEBUG] first sample keys:", list(first.keys()) if isinstance(first, dict) else "not dict")

                if isinstance(first, dict):
                    for key in hidden_keys:
                        if key in first:
                            tensors = []
                            for item in samples:
                                h = item[key]
                                tensors.append(h if isinstance(h, torch.Tensor) else torch.tensor(h))

                            hidden = torch.stack(tensors)
                            print("[DEBUG] stacked sample hidden shape:", hidden.shape)

                            if hidden.ndim == 3:
                                return hidden

    raise ValueError("无法从 pt 文件中读取 hidden states。请把 [DEBUG] 输出发我。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/experiment1/prompts_pilot.jsonl")
    parser.add_argument("--hidden", default="outputs/qwen05b/experiment1/experiment1_hidden_extraction/hidden_states_pilot.pt")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out_dir", default="outputs/qwen05b/experiment3/experiment3_pilot_controls/hidden_controls")
    parser.add_argument("--shuffle_repeats", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.data)
    texts = [get_text(r) for r in rows]
    y = np.array([get_label(r) for r in rows], dtype=int)

    print(f"[INFO] 样本数: {len(texts)}")
    print(f"[INFO] label=0 数量: {(y == 0).sum()}, label=1 数量: {(y == 1).sum()}")

    # 1. 长度统计
    print("[STEP 1] 计算 prompt 字符长度和 token 长度...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    char_lens = np.array([len(t) for t in texts])
    token_lens = np.array([
        len(tokenizer(t, add_special_tokens=False)["input_ids"])
        for t in texts
    ])

    length_df = pd.DataFrame({
        "text": texts,
        "label": y,
        "char_len": char_lens,
        "token_len": token_lens,
    })
    length_df.to_csv(out_dir / "length_stats.csv", index=False, encoding="utf-8-sig")

    summary = length_df.groupby("label")[["char_len", "token_len"]].agg(
        ["mean", "std", "min", "max"]
    )
    summary.to_csv(out_dir / "length_summary.csv", encoding="utf-8-sig")

    print(summary)

    # 2. 只用长度做分类
    print("[STEP 2] 训练 length baseline（只用长度分类）...")
    X_len = length_df[["char_len", "token_len"]].values

    length_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    length_result = cv_binary_scores(
        X_len, y, length_model, n_splits=5, name="length_baseline"
    )
    pd.DataFrame([length_result]).to_csv(
        out_dir / "length_baseline_results.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(length_result)

    # 3. TF-IDF 文本基线
    print("[STEP 3] 训练 TF-IDF text baseline（只看文本，不看模型 hidden states）...")

    tfidf_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=5000,
            min_df=1,
        )),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    tfidf_result = cv_binary_scores(
        texts, y, tfidf_model, n_splits=5, name="tfidf_char_ngram_baseline"
    )
    pd.DataFrame([tfidf_result]).to_csv(
        out_dir / "tfidf_baseline_results.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(tfidf_result)

    # 训练全量 TF-IDF，导出最强词面特征
    tfidf_model.fit(texts, y)
    vec = tfidf_model.named_steps["tfidf"]
    clf = tfidf_model.named_steps["clf"]
    feats = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]

    top_unknown = pd.DataFrame({
        "feature": feats[np.argsort(coefs)[-30:][::-1]],
        "coef": np.sort(coefs)[-30:][::-1],
        "direction": "towards_unknown_label_1",
    })
    top_known = pd.DataFrame({
        "feature": feats[np.argsort(coefs)[:30]],
        "coef": np.sort(coefs)[:30],
        "direction": "towards_known_label_0",
    })
    pd.concat([top_unknown, top_known]).to_csv(
        out_dir / "tfidf_top_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 4. label shuffle control
    print("[STEP 4] 训练 label shuffle control（标签打乱控制）...")

    hidden = load_hidden_tensor(args.hidden)
    if hidden.ndim != 3:
        raise ValueError(f"hidden shape 应为 [N, L, D]，当前为 {hidden.shape}")

    X_hidden = hidden.float().numpy()
    n, num_layers, hidden_dim = X_hidden.shape

    if n != len(y):
        raise ValueError(f"hidden 样本数 {n} 与 data 样本数 {len(y)} 不一致")

    shuffle_records = []
    rng = np.random.default_rng(42)

    for repeat in range(args.shuffle_repeats):
        y_shuffle = rng.permutation(y)

        for layer in range(num_layers):
            X_layer = X_hidden[:, layer, :]

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear"
                )),
            ])

            result = cv_binary_scores(
                X_layer,
                y_shuffle,
                model,
                n_splits=5,
                name=f"shuffle_repeat_{repeat}_layer_{layer}"
            )

            shuffle_records.append({
                "repeat": repeat,
                "layer": layer,
                "AUROC": result["AUROC_mean"],
                "Accuracy": result["Accuracy_mean"],
                "F1": result["F1_mean"],
            })

    shuffle_df = pd.DataFrame(shuffle_records)
    shuffle_df.to_csv(
        out_dir / "label_shuffle_control_all_layers.csv",
        index=False,
        encoding="utf-8-sig"
    )

    shuffle_summary = shuffle_df.groupby("layer")[["AUROC", "Accuracy", "F1"]].agg(
        ["mean", "std", "max"]
    )
    shuffle_summary.to_csv(
        out_dir / "label_shuffle_control_summary.csv",
        encoding="utf-8-sig"
    )

    print("[DONE] 混杂因素控制完成。输出目录：", out_dir)
    print("\n关键结果：")
    print("1. length baseline:", length_result)
    print("2. TF-IDF baseline:", tfidf_result)
    print("3. label shuffle AUROC overall mean:",
          float(shuffle_df["AUROC"].mean()))
    print("4. label shuffle AUROC overall max:",
          float(shuffle_df["AUROC"].max()))


if __name__ == "__main__":
    main()
