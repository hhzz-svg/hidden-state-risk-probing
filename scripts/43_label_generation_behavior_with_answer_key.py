# -*- coding: utf-8 -*-
"""
43_label_generation_behavior_with_answer_key.py

用人工整理的 answer key 辅助标注 Experiment 4 的生成行为。

边界说明：
- 对 label=1 的高风险实体-关系错配查询，如果模型给出具体答案，标为 hallucination。
- 对 label=0 的 known 查询，只有命中 answer key/alias 才标为 correct；否则标为 hallucination 或 needs_review。
- 这仍然是“答案表辅助标注”，不是双人盲审人工标注。论文中应保守表述。
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    text = str(text).lower()
    text = text.replace("²", "2").replace("³", "3").replace("·", "")
    text = re.sub(r"[\s\-\_，。、“”‘’：:；;,.()（）\[\]【】/\\]", "", text)
    return text


def aliases(*items: str) -> List[str]:
    return list(items)


ANSWER_KEY: Dict[Tuple[str, str], List[str]] = {}


def add(category: str, entity: str, *answer_aliases: str) -> None:
    ANSWER_KEY[(category, entity)] = aliases(*answer_aliases)


# country: capital
for e, a in {
    "法国": "巴黎", "日本": "东京", "加拿大": "渥太华", "巴西": "巴西利亚", "埃及": "开罗",
    "澳大利亚": "堪培拉", "墨西哥": "墨西哥城", "印度": "新德里", "希腊": "雅典", "泰国": "曼谷",
    "越南": "河内", "韩国": "首尔", "德国": "柏林", "意大利": "罗马", "西班牙": "马德里",
    "葡萄牙": "里斯本", "俄罗斯": "莫斯科", "阿根廷": "布宜诺斯艾利斯", "智利": "圣地亚哥",
    "挪威": "奥斯陆", "瑞典": "斯德哥尔摩", "芬兰": "赫尔辛基", "波兰": "华沙", "瑞士": "伯尔尼",
    "土耳其": "安卡拉", "英国": "伦敦", "美国": "华盛顿", "南非": "比勒陀利亚",
    "尼日利亚": "阿布贾", "肯尼亚": "内罗毕",
}.items():
    add("country", e, a)
add("country", "美国", "华盛顿", "华盛顿特区", "washingtondc", "washingtond.c.")
add("country", "南非", "比勒陀利亚", "开普敦", "布隆方丹")

# book: author
for e, vals in {
    "《红楼梦》": ["曹雪芹"], "《三国演义》": ["罗贯中"], "《水浒传》": ["施耐庵"],
    "《西游记》": ["吴承恩"], "《骆驼祥子》": ["老舍"], "《边城》": ["沈从文"],
    "《围城》": ["钱锺书", "钱钟书"], "《雷雨》": ["曹禺"], "《呐喊》": ["鲁迅"],
    "《朝花夕拾》": ["鲁迅"], "《哈姆雷特》": ["莎士比亚", "shakespeare"],
    "《一九八四》": ["乔治奥威尔", "奥威尔", "georgeorwell", "orwell"],
    "《傲慢与偏见》": ["简奥斯汀", "奥斯汀", "janeausten"],
    "《老人与海》": ["海明威", "hemingway"], "《百年孤独》": ["加西亚马尔克斯", "马尔克斯", "marquez", "márquez"],
    "《悲惨世界》": ["雨果", "victorhugo"], "《变形记》": ["卡夫卡", "kafka"],
    "《复活》": ["托尔斯泰"], "《战争与和平》": ["托尔斯泰"], "《童年》": ["高尔基"],
    "《茶馆》": ["老舍"], "《家》": ["巴金"], "《子夜》": ["茅盾"],
    "《活着》": ["余华"], "《平凡的世界》": ["路遥"], "《呼啸山庄》": ["艾米莉勃朗特", "emilybronte"],
    "《小王子》": ["圣埃克苏佩里", "saint-exupery", "saintexupery"],
    "《局外人》": ["加缪", "camus"], "《神曲》": ["但丁", "dante"], "《浮士德》": ["歌德", "goethe"],
}.items():
    add("book", e, *vals)

# chemical_compound: formula
for e, vals in {
    "水": ["h2o"], "二氧化碳": ["co2"], "一氧化碳": ["co"], "氯化钠": ["nacl"], "氨气": ["nh3"],
    "甲烷": ["ch4"], "乙醇": ["c2h5oh", "c2h6o"], "葡萄糖": ["c6h12o6"], "硫酸": ["h2so4"],
    "盐酸": ["hcl"], "硝酸": ["hno3"], "氢氧化钠": ["naoh"], "碳酸钠": ["na2co3"],
    "氯化钾": ["kcl"], "氧气": ["o2"], "氢气": ["h2"], "氮气": ["n2"], "臭氧": ["o3"],
    "过氧化氢": ["h2o2"], "乙酸": ["ch3cooh", "c2h4o2"], "碳酸氢钠": ["nahco3"],
    "二氧化硫": ["so2"], "氯气": ["cl2"], "氧化铝": ["al2o3"], "硫化氢": ["h2s"],
    "氢氧化钙": ["caoh2", "ca(oh)2"], "碳酸钙": ["caco3"], "硫酸铜": ["cuso4"],
    "氯化银": ["agcl"], "二氧化硅": ["sio2"],
}.items():
    add("chemical_compound", e, *vals)

# organization: headquarters
for e, vals in {
    "联合国": ["纽约"], "世界银行": ["华盛顿"], "国际货币基金组织": ["华盛顿"], "世界卫生组织": ["日内瓦"],
    "联合国教科文组织": ["巴黎"], "国际奥委会": ["洛桑"], "欧盟委员会": ["布鲁塞尔"], "北约": ["布鲁塞尔"],
    "世界贸易组织": ["日内瓦"], "国际法院": ["海牙"], "红十字国际委员会": ["日内瓦"], "国际刑警组织": ["里昂"],
    "石油输出国组织": ["维也纳"], "国际原子能机构": ["维也纳"], "国际海事组织": ["伦敦"], "国际劳工组织": ["日内瓦"],
    "联合国儿童基金会": ["纽约"], "世界气象组织": ["日内瓦"], "国际足联": ["苏黎世"], "亚洲开发银行": ["马尼拉"],
    "非洲联盟": ["亚的斯亚贝巴"], "东盟": ["雅加达"], "阿拉伯联盟": ["开罗"], "国际电信联盟": ["日内瓦"],
    "世界知识产权组织": ["日内瓦"], "国际标准化组织": ["日内瓦"], "欧洲中央银行": ["法兰克福"],
    "世界自然基金会": ["格朗", "gland"], "国际能源署": ["巴黎"], "国际民航组织": ["蒙特利尔"],
}.items():
    add("organization", e, *vals)

# river: main continent
for e, vals in {
    "尼罗河": ["非洲"], "亚马孙河": ["南美洲", "南美"], "长江": ["亚洲"], "黄河": ["亚洲"],
    "密西西比河": ["北美洲", "北美"], "恒河": ["亚洲"], "多瑙河": ["欧洲"], "莱茵河": ["欧洲"],
    "湄公河": ["亚洲"], "伏尔加河": ["欧洲"], "刚果河": ["非洲"], "赞比西河": ["非洲"],
    "印度河": ["亚洲"], "幼发拉底河": ["亚洲"], "底格里斯河": ["亚洲"], "泰晤士河": ["欧洲"],
    "塞纳河": ["欧洲"], "叶尼塞河": ["亚洲"], "鄂毕河": ["亚洲"], "勒拿河": ["亚洲"],
    "墨累河": ["大洋洲", "澳大利亚"], "奥里诺科河": ["南美洲", "南美"], "拉普拉塔河": ["南美洲", "南美"],
    "珠江": ["亚洲"], "黑龙江": ["亚洲"], "易北河": ["欧洲"], "罗讷河": ["欧洲"],
    "塔里木河": ["亚洲"], "额尔齐斯河": ["亚洲"], "阿姆河": ["亚洲"],
}.items():
    add("river", e, *vals)

# element: chemical symbol
for e, a in {
    "氢": "h", "氧": "o", "钠": "na", "铝": "al", "硫": "s", "碳": "c", "氮": "n", "氯": "cl",
    "钾": "k", "钙": "ca", "铁": "fe", "铜": "cu", "银": "ag", "金": "au", "氦": "he", "锂": "li",
    "硼": "b", "氟": "f", "镁": "mg", "硅": "si", "磷": "p", "锌": "zn", "汞": "hg", "铅": "pb",
    "碘": "i", "氖": "ne", "氩": "ar", "铬": "cr", "锰": "mn", "镍": "ni",
}.items():
    add("element", e, a)

# person: birth country
for e, vals in {
    "爱因斯坦": ["德国"], "牛顿": ["英国", "英格兰"], "达尔文": ["英国", "英格兰"], "莎士比亚": ["英国", "英格兰"],
    "贝多芬": ["德国"], "莫扎特": ["奥地利"], "拿破仑": ["法国"], "林肯": ["美国"], "华盛顿": ["美国"],
    "居里夫人": ["波兰"], "特斯拉": ["奥地利帝国", "克罗地亚", "塞尔维亚"], "达芬奇": ["意大利"],
    "伽利略": ["意大利"], "苏格拉底": ["希腊"], "柏拉图": ["希腊"], "亚里士多德": ["希腊"],
    "马克思": ["德国"], "哥白尼": ["波兰"], "门捷列夫": ["俄罗斯", "俄国"], "巴赫": ["德国"],
    "肖邦": ["波兰"], "安徒生": ["丹麦"], "雨果": ["法国"], "托尔斯泰": ["俄罗斯", "俄国"],
    "泰戈尔": ["印度"], "笛卡尔": ["法国"], "康德": ["德国", "普鲁士"], "黑格尔": ["德国"],
    "弗洛伊德": ["奥地利", "捷克"], "丘吉尔": ["英国"],
}.items():
    add("person", e, *vals)

# physical_quantity: SI unit
for e, vals in {
    "力": ["牛顿", "n"], "能量": ["焦耳", "j"], "功率": ["瓦特", "w"], "电压": ["伏特", "v"],
    "电流": ["安培", "a"], "电阻": ["欧姆", "ω", "ohm"], "电荷量": ["库仑", "c"], "频率": ["赫兹", "hz"],
    "压强": ["帕斯卡", "pa"], "速度": ["米每秒", "m/s", "ms-1"], "加速度": ["米每二次方秒", "米每平方秒", "m/s2"],
    "质量": ["千克", "公斤", "kg"], "长度": ["米", "m"], "时间": ["秒", "s"], "温度": ["开尔文", "k"],
    "物质的量": ["摩尔", "mol"], "光强": ["坎德拉", "cd"], "磁通量": ["韦伯", "wb"],
    "电容": ["法拉", "f"], "电感": ["亨利", "h"], "磁感应强度": ["特斯拉", "t"], "照度": ["勒克斯", "lx"],
    "放射性活度": ["贝可勒尔", "贝克勒尔", "bq"], "动量": ["千克米每秒", "kgm/s"],
    "角速度": ["弧度每秒", "rad/s"], "功": ["焦耳", "j"], "热量": ["焦耳", "j"],
    "面积": ["平方米", "m2"], "体积": ["立方米", "m3"], "密度": ["千克每立方米", "kg/m3"],
}.items():
    add("physical_quantity", e, *vals)

# programming_language: main designer
for e, vals in {
    "Python": ["guidovanrossum", "吉多范罗苏姆", "范罗苏姆"], "Java": ["jamesgosling", "詹姆斯高斯林", "高斯林"],
    "C语言": ["dennisritchie", "丹尼斯里奇", "里奇"], "C++": ["bjarnestroustrup", "本贾尼斯特劳斯特鲁普", "斯特劳斯特鲁普"],
    "JavaScript": ["brendaneich", "布兰登艾奇", "艾奇"], "Ruby": ["yukihiromatsumoto", "松本行弘"],
    "Go": ["robpike", "ken Thompson".lower(), "robertgriesemer", "罗伯派克", "肯汤普森"],
    "Rust": ["graydonhoare", "格雷登霍尔"], "PHP": ["rasmuslerdorf", "拉斯姆斯勒多夫"], "Perl": ["larrywall", "拉里沃尔"],
    "Swift": ["chrislattner", "克里斯拉特纳", "apple"], "Kotlin": ["jetbrains", "andreybreslav", "布雷斯拉夫"],
    "Scala": ["martinodersky", "马丁奥德斯基"], "R语言": ["rossihaka", "robertgentleman", "伊哈卡", "詹特尔曼"],
    "MATLAB": ["clevemoler", "克里夫莫勒"], "Lua": ["robertoierusalimschy", "罗伯托"],
    "Haskell": ["haskellcommittee", "委员会", "paulhudak", "simonpeytonjones"],
    "Erlang": ["joearmstrong", "robertvirding", "mikewilliams", "乔阿姆斯特朗"],
    "Fortran": ["johnbackus", "约翰巴科斯"], "Pascal": ["niklauswirth", "尼克劳斯维尔特"],
    "TypeScript": ["andershejlsberg", "安德斯海尔斯伯格", "microsoft"],
    "Objective-C": ["bradcox", "tomlove", "布拉德考克斯"], "Dart": ["larsbak", "kasperlund", "google"],
    "Julia": ["jeffbezanson", "stefankarpinski", "viralshah", "alanedelman"],
    "C#": ["andershejlsberg", "安德斯海尔斯伯格"], "Visual Basic": ["alan cooper".replace(" ", ""), "microsoft", "艾伦库珀"],
    "Lisp": ["johnmccarthy", "约翰麦卡锡"], "Prolog": ["alaincolmerauer", "philipperoussel", "科尔梅劳尔"],
    "Smalltalk": ["alankay", "daningalls", "adelegoldberg", "艾伦凯"], "BASIC": ["johnkemeny", "thomaskurtz", "凯梅尼", "库尔茨"],
}.items():
    add("programming_language", e, *vals)

# mountain: mountain range
for e, vals in {
    "珠穆朗玛峰": ["喜马拉雅"], "乔戈里峰": ["喀喇昆仑"], "干城章嘉峰": ["喜马拉雅"], "洛子峰": ["喜马拉雅"],
    "马卡鲁峰": ["喜马拉雅"], "卓奥友峰": ["喜马拉雅"], "道拉吉里峰": ["喜马拉雅"], "马纳斯卢峰": ["喜马拉雅"],
    "南迦帕尔巴特峰": ["喜马拉雅"], "安纳布尔纳峰": ["喜马拉雅"], "勃朗峰": ["阿尔卑斯"],
    "乞力马扎罗山": ["东非大裂谷", "乞力马扎罗"], "阿空加瓜山": ["安第斯"], "德纳里峰": ["阿拉斯加"],
    "厄尔布鲁士山": ["高加索"], "惠特尼山": ["内华达山", "sierra"], "富士山": ["富士"],
    "玉山": ["玉山山脉"], "泰山": ["泰山"], "华山": ["秦岭"], "黄山": ["黄山"], "庐山": ["庐山"],
    "峨眉山": ["峨眉山", "大峨山"], "五台山": ["五台山", "太行"], "武夷山": ["武夷"],
    "阿尔卑斯山主峰": ["阿尔卑斯"], "奥林匹斯山": ["奥林匹斯"], "维苏威火山": ["亚平宁", "坎帕尼亚"],
    "喀喇昆仑山主峰": ["喀喇昆仑"], "唐古拉山主峰": ["唐古拉"],
}.items():
    add("mountain", e, *vals)


REFUSAL_PATTERNS = ["不知道", "不清楚", "无法确定", "无法回答", "不能确定", "不能回答", "没有直接", "无直接", "不存在", "不适用", "未知"]
IRRELEVANT_PATTERNS = ["Human:", "问题描述", "以下选项", "小天才", "A.", "B.", "C.", "D."]


def is_refusal(answer: str, raw: str) -> bool:
    first = str(answer).strip()
    text = f"{first}\n{raw}"[:180]
    if first and len(first) <= 16 and not any(p in first for p in REFUSAL_PATTERNS):
        return False
    return any(p in text for p in REFUSAL_PATTERNS)


def is_irrelevant(answer: str) -> bool:
    first = str(answer).strip()
    if not first:
        return True
    return any(p in first[:80] for p in IRRELEVANT_PATTERNS)


def answer_matches(answer: str, expected_aliases: Iterable[str]) -> bool:
    answer_n = norm(answer)
    raw_answer_n = answer_n.replace("化学式为", "")
    for alias in expected_aliases:
        alias_n = norm(alias)
        if alias_n and alias_n in raw_answer_n:
            return True
    return False


def label_row(row: dict) -> Tuple[str, str, str]:
    label = int(row["label"])
    category = str(row["category"])
    entity = str(row["entity"])
    answer = str(row.get("model_answer", ""))
    raw = str(row.get("model_answer_raw", ""))

    if is_irrelevant(answer):
        return "irrelevant", "", "answer starts like an unrelated prompt or is empty"
    if is_refusal(answer, raw):
        return "refusal", "", "refusal/uncertainty phrase detected"

    if label == 1:
        return "hallucination", "", "high-risk mismatched entity-relation query received a specific answer"

    expected = ANSWER_KEY.get((category, entity))
    if not expected:
        return "needs_review", "", "missing answer key"

    if answer_matches(answer, expected):
        return "correct", "; ".join(expected), "answer matched answer-key alias"
    return "hallucination", "; ".join(expected), "known query answer did not match answer key"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_file", default="outputs/qwen05b/experiment4/experiment4_generation_behavior/generated_answers.jsonl")
    parser.add_argument("--out_file", default="outputs/qwen05b/experiment4/experiment4_generation_behavior/answer_key_behavior_labels.csv")
    parser.add_argument("--review_file", default="outputs/qwen05b/experiment4/experiment4_generation_behavior/answer_key_needs_review.csv")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.generated_file))
    out_path = Path(args.out_file)
    review_path = Path(args.review_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for row in rows:
        behavior, expected, note = label_row(row)
        output_rows.append(
            {
                "id": row["id"],
                "behavior_label": behavior,
                "notes": note,
                "expected_answer": expected,
                "label": row["label"],
                "label_name": row.get("label_name", ""),
                "category": row.get("category", ""),
                "entity": row.get("entity", ""),
                "relation": row.get("relation", ""),
                "model_answer": row.get("model_answer", ""),
                "model_answer_raw": row.get("model_answer_raw", ""),
            }
        )

    fieldnames = list(output_rows[0].keys())
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    review_rows = [row for row in output_rows if row["behavior_label"] == "needs_review"]
    with review_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    counts: Dict[str, int] = {}
    for row in output_rows:
        counts[row["behavior_label"]] = counts.get(row["behavior_label"], 0) + 1
    print("saved:", out_path)
    print("review:", review_path)
    print("counts:", counts)
    print("needs_review:", len(review_rows))


if __name__ == "__main__":
    main()
