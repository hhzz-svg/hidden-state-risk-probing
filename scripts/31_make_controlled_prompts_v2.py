import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


CATEGORIES = [
    {
        "name": "capital",
        "template": "请回答：{x}的首都是哪里？",
        "known": [
            "法国", "日本", "加拿大", "巴西", "埃及",
            "澳大利亚", "墨西哥", "印度", "希腊", "泰国",
            "越南", "韩国", "德国", "意大利", "西班牙",
            "葡萄牙", "俄罗斯", "阿根廷", "智利", "挪威",
            "瑞典", "芬兰", "波兰", "瑞士", "土耳其",
        ],
        "unknown": [
            "洛维尼亚", "卡尔多亚", "贝纳西亚", "托兰尼亚", "艾森尼亚",
            "米拉瓦亚", "诺瓦利亚", "塔苏尼亚", "阿文尼亚", "格林达亚",
            "索兰尼亚", "维塔尼亚", "多米兰亚", "兰西维亚", "奥贝西亚",
            "卡里诺亚", "帕尔文亚", "塞利达亚", "费伦尼亚", "米索尼亚",
            "阿尔维达", "利奥尼亚", "巴索兰亚", "维斯塔亚", "诺卡西亚",
        ],
    },
    {
        "name": "river_continent",
        "template": "请回答：{x}主要流经哪个大洲？",
        "known": [
            "尼罗河", "亚马孙河", "长江", "黄河", "密西西比河",
            "恒河", "多瑙河", "莱茵河", "湄公河", "伏尔加河",
            "刚果河", "赞比西河", "印度河", "幼发拉底河", "底格里斯河",
            "泰晤士河", "塞纳河", "叶尼塞河", "鄂毕河", "勒拿河",
            "墨累河", "奥里诺科河", "拉普拉塔河", "湄南河", "珠江",
        ],
        "unknown": [
            "萨林河", "莫维河", "卡尔文河", "诺兰河", "贝苏河",
            "拉提河", "米兰河", "塔维河", "奥森河", "维洛河",
            "帕森河", "索迪河", "阿贝河", "利塔河", "格兰河",
            "多维河", "费尔河", "洛宁河", "塞曼河", "巴伦河",
            "科维河", "兰托河", "依萨河", "西文河", "纳洛河",
        ],
    },
    {
        "name": "book_author",
        "template": "请回答：《{x}》的作者是谁？",
        "known": [
            "红楼梦", "三国演义", "水浒传", "西游记", "骆驼祥子",
            "边城", "围城", "雷雨", "呐喊", "朝花夕拾",
            "哈姆雷特", "一九八四", "傲慢与偏见", "老人与海", "百年孤独",
            "悲惨世界", "变形记", "复活", "战争与和平", "童年",
            "茶馆", "家", "子夜", "飘", "欧也妮葛朗台",
        ],
        "unknown": [
            "雾港来信", "星河旧梦", "蓝钟旅人", "北岸之书", "银灯年代",
            "石桥晚歌", "风塔笔记", "白沙花园", "深巷回声", "乌木日记",
            "海棠迷宫", "暮色邮差", "灰鸽学院", "远山账本", "铜雀旅店",
            "静水剧场", "青岚手札", "月下船票", "雪原长椅", "南城钟楼",
            "黑檀纪事", "橘色荒原", "旧港钟声", "林间证词", "春河档案",
        ],
    },
    {
        "name": "chemical_formula",
        "template": "请回答：{x}的化学式是什么？",
        "known": [
            "水", "二氧化碳", "一氧化碳", "氯化钠", "氨气",
            "甲烷", "乙醇", "葡萄糖", "硫酸", "盐酸",
            "硝酸", "氢氧化钠", "碳酸钙", "氯化钾", "氧气",
            "氢气", "氮气", "臭氧", "过氧化氢", "乙酸",
            "碳酸氢钠", "二氧化硫", "氯气", "氧化铁", "硫化氢",
        ],
        "unknown": [
            "纳维酸", "洛米醇", "泽兰酮", "卡芬盐", "米索烷",
            "贝洛酸", "塔宁醇", "奥维酮", "兰泽盐", "帕米烷",
            "索利酸", "维诺醇", "科兰酮", "阿索盐", "费米烷",
            "利贝酸", "诺卡醇", "格维酮", "多兰盐", "塞洛烷",
            "巴索酸", "艾米醇", "霍兰酮", "依文盐", "西洛烷",
        ],
    },
    {
        "name": "person_country",
        "template": "请回答：{x}出生在哪个国家？",
        "known": [
            "爱因斯坦", "牛顿", "达尔文", "莎士比亚", "贝多芬",
            "莫扎特", "拿破仑", "林肯", "华盛顿", "居里夫人",
            "特斯拉", "达芬奇", "伽利略", "苏格拉底", "柏拉图",
            "亚里士多德", "马克思", "哥白尼", "门捷列夫", "巴赫",
            "肖邦", "安徒生", "雨果", "托尔斯泰", "泰戈尔",
        ],
        "unknown": [
            "阿维森", "洛塔尔", "米兰德", "贝尔文", "卡索恩",
            "诺维克", "塔兰德", "索米尔", "维森特", "帕洛文",
            "奥贝林", "格兰托", "费尔曼", "利文索", "多米克",
            "塞伦塔", "巴维尔", "艾洛斯", "科维安", "兰伯索",
            "霍森特", "依维诺", "西塔文", "纳洛夫", "莫尔迪",
        ],
    },
    {
        "name": "physical_unit",
        "template": "请回答：{x}的国际单位是什么？",
        "known": [
            "力", "功", "能量", "功率", "电压",
            "电流", "电阻", "电荷量", "频率", "压强",
            "速度", "加速度", "质量", "长度", "时间",
            "温度", "物质的量", "光强", "磁通量", "电容",
            "电感", "磁感应强度", "照度", "放射性活度", "动量",
        ],
        "unknown": [
            "澜势", "维量", "洛压", "迁率", "塔能",
            "索阻", "纳动度", "贝势差", "米通量", "卡感度",
            "奥容率", "帕温差", "利光度", "格流势", "多频量",
            "塞场强", "巴荷度", "艾能率", "科动压", "兰磁率",
            "霍折度", "依速势", "西相量", "莫转度", "费稳率",
        ],
    },
    {
        "name": "element_symbol",
        "template": "请回答：{x}的化学符号是什么？",
        "known": [
            "氢", "氦", "锂", "铍", "硼",
            "碳", "氮", "氧", "氟", "氖",
            "钠", "镁", "铝", "硅", "磷",
            "硫", "氯", "氩", "钾", "钙",
            "铁", "铜", "锌", "银", "金",
        ],
        "unknown": [
            "纳维", "洛铂", "泽钨", "卡镁", "米钛",
            "贝铬", "塔铟", "奥锂", "兰钴", "帕镍",
            "索钒", "维锡", "科铱", "阿钽", "费锆",
            "利锗", "诺铯", "格钯", "多锶", "塞铷",
            "巴铪", "艾镓", "霍镧", "依铌", "西钼",
        ],
    },
    {
        "name": "organization_headquarters",
        "template": "请回答：{x}的总部位于哪里？",
        "known": [
            "联合国", "世界银行", "国际货币基金组织", "世界卫生组织", "联合国教科文组织",
            "国际奥委会", "欧盟委员会", "北约", "世界贸易组织", "国际法院",
            "红十字国际委员会", "国际刑警组织", "石油输出国组织", "国际原子能机构", "国际海事组织",
            "国际劳工组织", "联合国儿童基金会", "世界气象组织", "国际足联", "亚洲开发银行",
            "非洲联盟", "东盟", "阿拉伯联盟", "国际电信联盟", "世界知识产权组织",
        ],
        "unknown": [
            "蓝岸研究会", "诺维基金会", "格兰协作组织", "塔林发展署", "米洛标准委员会",
            "贝森安全联盟", "索兰文化组织", "维塔经济论坛", "卡文贸易协会", "奥森医学中心",
            "帕罗科学联盟", "费尔数据组织", "利塔教育基金会", "多米能源署", "塞洛海事协会",
            "巴索儿童基金会", "艾文气象中心", "科兰体育联合会", "兰迪专利组织", "霍维金融集团",
            "依萨合作委员会", "西文治理联盟", "纳洛研究院", "莫尔工业联盟", "阿贝技术论坛",
        ],
    },
]


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_records(seed=42):
    records = []
    idx = 0

    for cat in CATEGORIES:
        known = cat["known"]
        unknown = cat["unknown"]
        template = cat["template"]

        assert len(known) == len(unknown), cat["name"]

        for x in known:
            records.append({
                "id": f"ctrl_{idx:04d}",
                "prompt": template.format(x=x),
                "label": 0,
                "type": "known_controlled_v2",
                "category": cat["name"],
                "entity": x,
            })
            idx += 1

        for x in unknown:
            records.append({
                "id": f"ctrl_{idx:04d}",
                "prompt": template.format(x=x),
                "label": 1,
                "type": "unknown_controlled_v2",
                "category": cat["name"],
                "entity": x,
            })
            idx += 1

    rng = random.Random(seed)
    rng.shuffle(records)

    # 重新编号，避免 id 暴露原始顺序
    for i, r in enumerate(records):
        r["id"] = f"ctrl_{i:04d}"

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


def get_token_lens(texts, tokenizer_name):
    if tokenizer_name.lower() in ["none", "no", "skip"]:
        return None

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return np.array([
            len(tokenizer(t, add_special_tokens=False)["input_ids"])
            for t in texts
        ])
    except Exception as e:
        print("[WARN] tokenizer 加载失败，跳过 token_len。错误：", repr(e))
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/experiment3/prompts_controlled_v2.jsonl")
    parser.add_argument("--out_dir", default="outputs/qwen05b/experiment3/experiment3_v2/text_controls")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(seed=args.seed)
    write_jsonl(records, args.out)

    texts = [r["prompt"] for r in records]
    y = np.array([r["label"] for r in records], dtype=int)

    print("[INFO] saved:", args.out)
    print("[INFO] num_samples:", len(records))
    print("[INFO] known:", int((y == 0).sum()))
    print("[INFO] unknown:", int((y == 1).sum()))

    char_lens = np.array([len(t) for t in texts])
    token_lens = get_token_lens(texts, args.tokenizer)

    df = pd.DataFrame(records)
    df["char_len"] = char_lens

    if token_lens is not None:
        df["token_len"] = token_lens

    df.to_csv(out_dir / "controlled_v2_full.csv", index=False, encoding="utf-8-sig")
    df.head(30).to_csv(out_dir / "controlled_v2_preview.csv", index=False, encoding="utf-8-sig")

    length_cols = ["char_len"]
    if token_lens is not None:
        length_cols.append("token_len")

    length_summary = df.groupby("label")[length_cols].agg(["mean", "std", "min", "max"])
    length_summary.to_csv(out_dir / "controlled_v2_length_summary.csv", encoding="utf-8-sig")

    category_summary = df.groupby(["category", "label"]).size().reset_index(name="count")
    category_summary.to_csv(out_dir / "controlled_v2_category_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[STEP 1] 长度统计：")
    print(length_summary)

    print("\n[STEP 2] length baseline：")

    if token_lens is not None:
        X_len = df[["char_len", "token_len"]].values
    else:
        X_len = df[["char_len"]].values

    length_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    length_result = cv_binary_scores(
        X_len,
        y,
        length_model,
        n_splits=5,
        name="controlled_v2_length_baseline",
    )

    print(length_result)

    print("\n[STEP 3] TF-IDF char n-gram baseline：")

    tfidf_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=8000,
            min_df=1,
        )),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])

    tfidf_result = cv_binary_scores(
        texts,
        y,
        tfidf_model,
        n_splits=5,
        name="controlled_v2_tfidf_char_ngram_baseline",
    )

    print(tfidf_result)

    results = pd.DataFrame([length_result, tfidf_result])
    results.to_csv(out_dir / "controlled_v2_text_baseline_results.csv", index=False, encoding="utf-8-sig")

    # 导出 TF-IDF 最强词面特征，方便检查有没有明显混杂词
    tfidf_model.fit(texts, y)
    vec = tfidf_model.named_steps["tfidf"]
    clf = tfidf_model.named_steps["clf"]

    feats = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]

    top_unknown_idx = np.argsort(coefs)[-40:][::-1]
    top_known_idx = np.argsort(coefs)[:40]

    top_unknown = pd.DataFrame({
        "feature": feats[top_unknown_idx],
        "coef": coefs[top_unknown_idx],
        "direction": "towards_unknown_label_1",
    })

    top_known = pd.DataFrame({
        "feature": feats[top_known_idx],
        "coef": coefs[top_known_idx],
        "direction": "towards_known_label_0",
    })

    pd.concat([top_unknown, top_known]).to_csv(
        out_dir / "controlled_v2_tfidf_top_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[DONE] controlled prompts v2 生成与文本混杂检查完成。")
    print("[OUT DATA]", args.out)
    print("[OUT DIR]", out_dir)

    print("\n关键判断标准：")
    print("1. length baseline AUROC 最好 <= 0.65")
    print("2. TF-IDF baseline AUROC 最好 <= 0.75，至少不能接近 0.95")
    print("3. 如果 TF-IDF 仍然很高，说明还要继续改数据集，而不是急着抽 hidden states")


if __name__ == "__main__":
    main()
