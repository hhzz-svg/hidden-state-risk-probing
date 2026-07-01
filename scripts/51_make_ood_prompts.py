# -*- coding: utf-8 -*-
"""
51_make_ood_prompts.py

Experiment 5: OOD generalization dataset.

构造一批不复用 controlled_v4 实体/主题的查询，用于测试 hidden-state probe
是否能从原数据分布泛化到新实体与新关系。
"""

import argparse
import json
import random
from pathlib import Path


PAIRS = [
    {
        "a_category": "city",
        "a_relation": "所在国家",
        "a_entities": ["北京", "上海", "纽约", "伦敦", "巴黎", "东京", "悉尼", "开罗", "罗马", "莫斯科"],
        "b_category": "film",
        "b_relation": "导演",
        "b_entities": ["《泰坦尼克号》", "《盗梦空间》", "《阿凡达》", "《千与千寻》", "《星球大战》", "《教父》", "《肖申克的救赎》", "《霸王别姬》", "《卧虎藏龙》", "《寄生虫》"],
    },
    {
        "a_category": "company",
        "a_relation": "创始人",
        "a_entities": ["苹果公司", "微软", "特斯拉公司", "亚马逊", "Meta", "阿里巴巴", "腾讯", "谷歌", "华为", "小米"],
        "b_category": "planet",
        "b_relation": "所属恒星",
        "b_entities": ["水星", "金星", "地球", "火星", "木星", "土星", "天王星", "海王星", "谷神星", "冥王星"],
    },
    {
        "a_category": "animal",
        "a_relation": "所属纲",
        "a_entities": ["老虎", "家犬", "家猫", "蓝鲸", "蝙蝠", "鸡", "鹰", "鲤鱼", "青蛙", "蛇"],
        "b_category": "artist",
        "b_relation": "代表作品",
        "b_entities": ["毕加索", "莫奈", "梵高", "达利", "马蒂斯", "伦勃朗", "米开朗基罗", "齐白石", "徐悲鸿", "达芬奇"],
    },
    {
        "a_category": "software_framework",
        "a_relation": "主要编程语言",
        "a_entities": ["Django", "Flask", "React", "Vue", "Angular", "Spring", "Laravel", "Ruby on Rails", "TensorFlow", "PyTorch"],
        "b_category": "university",
        "b_relation": "所在国家",
        "b_entities": ["哈佛大学", "斯坦福大学", "牛津大学", "剑桥大学", "东京大学", "清华大学", "北京大学", "巴黎大学", "多伦多大学", "墨尔本大学"],
    },
    {
        "a_category": "disease",
        "a_relation": "常见病原体",
        "a_entities": ["流感", "新冠感染", "肺结核", "疟疾", "霍乱", "麻疹", "狂犬病", "乙型肝炎", "艾滋病", "登革热"],
        "b_category": "award",
        "b_relation": "颁发机构",
        "b_entities": ["诺贝尔文学奖", "普利策奖", "奥斯卡金像奖", "图灵奖", "菲尔兹奖", "格莱美奖", "金球奖", "雨果奖", "布克奖", "茅盾文学奖"],
    },
]


def make_prompt(entity: str, relation: str) -> str:
    return f"请回答下面查询。\n对象：{entity}\n属性：{relation}\n答案："


def add_records(records, category, known_relation, risk_relation, entities, idx):
    for entity in entities:
        records.append(
            {
                "id": f"ood_{idx:04d}",
                "prompt": make_prompt(entity, known_relation),
                "label": 0,
                "type": "known_ood",
                "category": category,
                "entity": entity,
                "relation": known_relation,
            }
        )
        idx += 1
        records.append(
            {
                "id": f"ood_{idx:04d}",
                "prompt": make_prompt(entity, risk_relation),
                "label": 1,
                "type": "unknown_ood",
                "category": category,
                "entity": entity,
                "relation": risk_relation,
            }
        )
        idx += 1
    return idx


def build_records(seed: int):
    records = []
    idx = 0
    for pair in PAIRS:
        idx = add_records(records, pair["a_category"], pair["a_relation"], pair["b_relation"], pair["a_entities"], idx)
        idx = add_records(records, pair["b_category"], pair["b_relation"], pair["a_relation"], pair["b_entities"], idx)

    rng = random.Random(seed)
    rng.shuffle(records)
    for i, record in enumerate(records):
        record["id"] = f"ood_{i:04d}"
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/experiment5/prompts_ood_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = build_records(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("saved:", out)
    print("num_samples:", len(records))
    print("known:", sum(1 for r in records if r["label"] == 0))
    print("unknown:", sum(1 for r in records if r["label"] == 1))


if __name__ == "__main__":
    main()
