import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PAIRED_DOMAINS = [
    {
        "a_category": "country",
        "a_relation": "首都",
        "a_entities": [
            "法国", "日本", "加拿大", "巴西", "埃及", "澳大利亚", "墨西哥", "印度", "希腊", "泰国",
            "越南", "韩国", "德国", "意大利", "西班牙", "葡萄牙", "俄罗斯", "阿根廷", "智利", "挪威",
            "瑞典", "芬兰", "波兰", "瑞士", "土耳其", "英国", "美国", "南非", "尼日利亚", "肯尼亚",
        ],
        "b_category": "book",
        "b_relation": "作者",
        "b_entities": [
            "《红楼梦》", "《三国演义》", "《水浒传》", "《西游记》", "《骆驼祥子》", "《边城》", "《围城》", "《雷雨》", "《呐喊》", "《朝花夕拾》",
            "《哈姆雷特》", "《一九八四》", "《傲慢与偏见》", "《老人与海》", "《百年孤独》", "《悲惨世界》", "《变形记》", "《复活》", "《战争与和平》", "《童年》",
            "《茶馆》", "《家》", "《子夜》", "《活着》", "《平凡的世界》", "《呼啸山庄》", "《小王子》", "《局外人》", "《神曲》", "《浮士德》",
        ],
    },
    {
        "a_category": "chemical_compound",
        "a_relation": "化学式",
        "a_entities": [
            "水", "二氧化碳", "一氧化碳", "氯化钠", "氨气", "甲烷", "乙醇", "葡萄糖", "硫酸", "盐酸",
            "硝酸", "氢氧化钠", "碳酸钠", "氯化钾", "氧气", "氢气", "氮气", "臭氧", "过氧化氢", "乙酸",
            "碳酸氢钠", "二氧化硫", "氯气", "氧化铝", "硫化氢", "氢氧化钙", "碳酸钙", "硫酸铜", "氯化银", "二氧化硅",
        ],
        "b_category": "organization",
        "b_relation": "总部所在地",
        "b_entities": [
            "联合国", "世界银行", "国际货币基金组织", "世界卫生组织", "联合国教科文组织", "国际奥委会", "欧盟委员会", "北约", "世界贸易组织", "国际法院",
            "红十字国际委员会", "国际刑警组织", "石油输出国组织", "国际原子能机构", "国际海事组织", "国际劳工组织", "联合国儿童基金会", "世界气象组织", "国际足联", "亚洲开发银行",
            "非洲联盟", "东盟", "阿拉伯联盟", "国际电信联盟", "世界知识产权组织", "国际标准化组织", "欧洲中央银行", "世界自然基金会", "国际能源署", "国际民航组织",
        ],
    },
    {
        "a_category": "river",
        "a_relation": "主要流经大洲",
        "a_entities": [
            "尼罗河", "亚马孙河", "长江", "黄河", "密西西比河", "恒河", "多瑙河", "莱茵河", "湄公河", "伏尔加河",
            "刚果河", "赞比西河", "印度河", "幼发拉底河", "底格里斯河", "泰晤士河", "塞纳河", "叶尼塞河", "鄂毕河", "勒拿河",
            "墨累河", "奥里诺科河", "拉普拉塔河", "珠江", "黑龙江", "易北河", "罗讷河", "塔里木河", "额尔齐斯河", "阿姆河",
        ],
        "b_category": "element",
        "b_relation": "化学符号",
        "b_entities": [
            "氢", "氧", "钠", "铝", "硫", "碳", "氮", "氯", "钾", "钙",
            "铁", "铜", "银", "金", "氦", "锂", "硼", "氟", "镁", "硅",
            "磷", "锌", "汞", "铅", "碘", "氖", "氩", "铬", "锰", "镍",
        ],
    },
    {
        "a_category": "person",
        "a_relation": "出生国家",
        "a_entities": [
            "爱因斯坦", "牛顿", "达尔文", "莎士比亚", "贝多芬", "莫扎特", "拿破仑", "林肯", "华盛顿", "居里夫人",
            "特斯拉", "达芬奇", "伽利略", "苏格拉底", "柏拉图", "亚里士多德", "马克思", "哥白尼", "门捷列夫", "巴赫",
            "肖邦", "安徒生", "雨果", "托尔斯泰", "泰戈尔", "笛卡尔", "康德", "黑格尔", "弗洛伊德", "丘吉尔",
        ],
        "b_category": "physical_quantity",
        "b_relation": "国际单位",
        "b_entities": [
            "力", "能量", "功率", "电压", "电流", "电阻", "电荷量", "频率", "压强", "速度",
            "加速度", "质量", "长度", "时间", "温度", "物质的量", "光强", "磁通量", "电容", "电感",
            "磁感应强度", "照度", "放射性活度", "动量", "角速度", "功", "热量", "面积", "体积", "密度",
        ],
    },
    {
        "a_category": "programming_language",
        "a_relation": "主要设计者",
        "a_entities": [
            "Python", "Java", "C语言", "C++", "JavaScript", "Ruby", "Go", "Rust", "PHP", "Perl",
            "Swift", "Kotlin", "Scala", "R语言", "MATLAB", "Lua", "Haskell", "Erlang", "Fortran", "Pascal",
            "TypeScript", "Objective-C", "Dart", "Julia", "C#", "Visual Basic", "Lisp", "Prolog", "Smalltalk", "BASIC",
        ],
        "b_category": "mountain",
        "b_relation": "所在山脉",
        "b_entities": [
            "珠穆朗玛峰", "乔戈里峰", "干城章嘉峰", "洛子峰", "马卡鲁峰", "卓奥友峰", "道拉吉里峰", "马纳斯卢峰", "南迦帕尔巴特峰", "安纳布尔纳峰",
            "勃朗峰", "乞力马扎罗山", "阿空加瓜山", "德纳里峰", "厄尔布鲁士山", "惠特尼山", "富士山", "玉山", "泰山", "华山",
            "黄山", "庐山", "峨眉山", "五台山", "武夷山", "阿尔卑斯山主峰", "奥林匹斯山", "维苏威火山", "喀喇昆仑山主峰", "唐古拉山主峰",
        ],
    },
]


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_prompt(entity, relation):
    return f"请回答：{entity}的{relation}是什么？"


def add_domain_records(records, category, known_relation, risk_relation, entities, start_index):
    idx = start_index
    for entity in entities:
        records.append(
            {
                "id": f"ctrlv3_{idx:04d}",
                "prompt": make_prompt(entity, known_relation),
                "label": 0,
                "type": "known_controlled_v3",
                "category": category,
                "entity": entity,
                "relation": known_relation,
            }
        )
        idx += 1
        records.append(
            {
                "id": f"ctrlv3_{idx:04d}",
                "prompt": make_prompt(entity, risk_relation),
                "label": 1,
                "type": "unknown_controlled_v3",
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

    for pair in PAIRED_DOMAINS:
        if len(pair["a_entities"]) != len(pair["b_entities"]):
            raise ValueError(f"unbalanced pair: {pair['a_category']} vs {pair['b_category']}")

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
        record["id"] = f"ctrlv3_{i:04d}"

    return records


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
        token_lens = np.array(
            [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
        )
        return token_lens, tokenizer_name
    except Exception as exc:
        print("[WARN] tokenizer failed; token_len uses char-length fallback:", repr(exc))
        return np.array([len(text) for text in texts]), "char_fallback"


def export_top_tfidf_features(model, texts, y, out_path, top_k=50):
    fitted = clone(model)
    fitted.fit(texts, y)
    vec = fitted.named_steps["tfidf"]
    clf = fitted.named_steps["clf"]
    feats = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]

    top_unknown_idx = np.argsort(coefs)[-top_k:][::-1]
    top_known_idx = np.argsort(coefs)[:top_k]

    top_unknown = pd.DataFrame(
        {
            "feature": feats[top_unknown_idx],
            "coef": coefs[top_unknown_idx],
            "direction": "towards_unknown_label_1",
        }
    )
    top_known = pd.DataFrame(
        {
            "feature": feats[top_known_idx],
            "coef": coefs[top_known_idx],
            "direction": "towards_known_label_0",
        }
    )
    pd.concat([top_unknown, top_known]).to_csv(out_path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/experiment3/prompts_controlled_v3.jsonl")
    parser.add_argument("--out_dir", default="outputs/qwen05b/experiment3/experiment3_v3/text_controls")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(seed=args.seed)
    write_jsonl(records, args.out)

    texts = [record["prompt"] for record in records]
    y = np.array([record["label"] for record in records], dtype=int)

    print("[INFO] saved:", args.out)
    print("[INFO] num_samples:", len(records))
    print("[INFO] known:", int((y == 0).sum()))
    print("[INFO] unknown:", int((y == 1).sum()))

    char_lens = np.array([len(text) for text in texts])
    token_lens, token_len_source = get_token_lens(texts, args.tokenizer)

    df = pd.DataFrame(records)
    df["char_len"] = char_lens
    df["token_len"] = token_lens
    df["token_len_source"] = token_len_source

    df.to_csv(out_dir / "controlled_v3_full.csv", index=False, encoding="utf-8-sig")
    df.head(40).to_csv(out_dir / "controlled_v3_preview.csv", index=False, encoding="utf-8-sig")

    length_summary = df.groupby("label")[["char_len", "token_len"]].agg(["mean", "std", "min", "max"])
    length_summary.to_csv(out_dir / "controlled_v3_length_summary.csv", encoding="utf-8-sig")

    category_summary = df.groupby(["category", "label"]).size().reset_index(name="count")
    category_summary.to_csv(out_dir / "controlled_v3_category_summary.csv", index=False, encoding="utf-8-sig")

    relation_summary = df.groupby(["relation", "label"]).size().reset_index(name="count")
    relation_summary.to_csv(out_dir / "controlled_v3_relation_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[STEP 1] length summary")
    print(length_summary)

    print("\n[STEP 2] length baseline")
    X_len = df[["char_len", "token_len"]].values
    length_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    length_result = cv_binary_scores(
        X_len,
        y,
        length_model,
        n_splits=5,
        name="controlled_v3_length_baseline",
    )
    print(length_result)

    print("\n[STEP 3] TF-IDF char n-gram baseline")
    tfidf_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    max_features=12000,
                    min_df=1,
                ),
            ),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    tfidf_result = cv_binary_scores(
        texts,
        y,
        tfidf_model,
        n_splits=5,
        name="controlled_v3_tfidf_char_ngram_baseline",
    )
    print(tfidf_result)

    pd.DataFrame([length_result, tfidf_result]).to_csv(
        out_dir / "controlled_v3_text_baseline_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    export_top_tfidf_features(
        tfidf_model,
        texts,
        y,
        out_dir / "controlled_v3_tfidf_top_features.csv",
        top_k=50,
    )

    print("\n[DONE] controlled_v3 dataset and text baselines completed.")
    print("[OUT DATA]", args.out)
    print("[OUT DIR]", out_dir)
    print("[CHECK] target length AUROC <= 0.65; target TF-IDF AUROC <= 0.75")


if __name__ == "__main__":
    main()
