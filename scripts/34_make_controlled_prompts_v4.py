import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_v3_domains():
    script_path = Path(__file__).with_name("32_make_controlled_prompts_v3.py")
    spec = importlib.util.spec_from_file_location("controlled_v3", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PAIRED_DOMAINS


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_prompt(entity, relation):
    return f"请回答下面查询。\n对象：{entity}\n属性：{relation}\n答案："


def add_domain_records(records, category, known_relation, risk_relation, entities, start_index):
    idx = start_index
    for entity in entities:
        records.append(
            {
                "id": f"ctrlv4_{idx:04d}",
                "prompt": make_prompt(entity, known_relation),
                "label": 0,
                "type": "known_controlled_v4",
                "category": category,
                "entity": entity,
                "relation": known_relation,
            }
        )
        idx += 1
        records.append(
            {
                "id": f"ctrlv4_{idx:04d}",
                "prompt": make_prompt(entity, risk_relation),
                "label": 1,
                "type": "unknown_controlled_v4",
                "category": category,
                "entity": entity,
                "relation": risk_relation,
            }
        )
        idx += 1
    return idx


def build_records(seed=42):
    records = []
    idx = 0
    for pair in load_v3_domains():
        idx = add_domain_records(
            records,
            pair["a_category"],
            pair["a_relation"],
            pair["b_relation"],
            pair["a_entities"],
            idx,
        )
        idx = add_domain_records(
            records,
            pair["b_category"],
            pair["b_relation"],
            pair["a_relation"],
            pair["b_entities"],
            idx,
        )

    rng = random.Random(seed)
    rng.shuffle(records)
    for i, record in enumerate(records):
        record["id"] = f"ctrlv4_{i:04d}"
    return records


def evaluate_cv(X, y, model, cv, groups=None, name="model"):
    aucs, accs, f1s = [], [], []
    split_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)

    for train_idx, test_idx in split_iter:
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


def get_token_lens(texts, tokenizer_name):
    if tokenizer_name.lower() in {"none", "no", "skip"}:
        return np.array([len(text) for text in texts]), "char_fallback"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return np.array(
            [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
        ), tokenizer_name
    except Exception as exc:
        print("[WARN] tokenizer failed; token_len uses char-length fallback:", repr(exc))
        return np.array([len(text) for text in texts]), "char_fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/experiment3/prompts_controlled_v4.jsonl")
    parser.add_argument("--out_dir", default="outputs/qwen05b/experiment3/experiment3_v4/text_controls")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(seed=args.seed)
    write_jsonl(records, args.out)

    texts = [record["prompt"] for record in records]
    y = np.array([record["label"] for record in records], dtype=int)
    groups = np.array([record["entity"] for record in records], dtype=object)

    char_lens = np.array([len(text) for text in texts])
    token_lens, token_len_source = get_token_lens(texts, args.tokenizer)

    df = pd.DataFrame(records)
    df["char_len"] = char_lens
    df["token_len"] = token_lens
    df["token_len_source"] = token_len_source
    df.to_csv(out_dir / "controlled_v4_full.csv", index=False, encoding="utf-8-sig")
    df.head(40).to_csv(out_dir / "controlled_v4_preview.csv", index=False, encoding="utf-8-sig")

    length_summary = df.groupby("label")[["char_len", "token_len"]].agg(["mean", "std", "min", "max"])
    length_summary.to_csv(out_dir / "controlled_v4_length_summary.csv", encoding="utf-8-sig")

    length_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
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

    stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grouped = GroupKFold(n_splits=5)
    X_len = df[["char_len", "token_len"]].values

    results = [
        evaluate_cv(X_len, y, length_model, stratified, name="v4_random_length_baseline"),
        evaluate_cv(texts, y, tfidf_model, stratified, name="v4_random_tfidf_baseline"),
        evaluate_cv(X_len, y, length_model, grouped, groups=groups, name="v4_entity_group_length_baseline"),
        evaluate_cv(texts, y, tfidf_model, grouped, groups=groups, name="v4_entity_group_tfidf_baseline"),
    ]

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / "controlled_v4_text_baseline_results.csv", index=False, encoding="utf-8-sig")

    print("[INFO] saved:", args.out)
    print("[INFO] num_samples:", len(records))
    print("[INFO] known:", int((y == 0).sum()))
    print("[INFO] unknown:", int((y == 1).sum()))
    print(length_summary)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
