# 第五次实验：OOD 泛化实验初步报告

## 1. 实验目的

第五次实验用于检查：在 controlled_v4 上训练得到的 query-level risk probe，换到新实体、新关系、新主题后是否仍有泛化能力。
OOD（out-of-distribution，分布外泛化）比同分布交叉验证更严格，因此结果即使下降也很正常。

## 2. 数据与设置

- 训练数据：`data/experiment3/prompts_controlled_v4.jsonl`，600 条。
- 测试数据：`data/experiment5/prompts_ood_v1.jsonl`，200 条，known/high-risk 各 100。
- OOD 主题：城市-国家、电影-导演、公司-创始人、行星-所属恒星、动物-所属纲、艺术家-代表作品、软件框架-主要语言、高校-所在国家、疾病-病原体、奖项-颁发机构。
- 评估：只在 controlled_v4 hidden states 上训练 Logistic Regression probe，然后直接测试 OOD_v1。

## 3. 结果

| 模型 | hidden probe AUROC | entropy AUROC | TF-IDF AUROC | length AUROC |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 0.7512 | 0.7801 | 0.5094 | 0.5000 |
| Qwen2.5-1.5B-Instruct | 0.9016 | 0.7912 | 0.5094 | 0.5000 |
| Qwen2.5-7B-Instruct | 0.8389 | 0.5796 | 0.5093 | 0.5000 |

## 4. 初步解释

1. 文本表面基线在 OOD_v1 上接近随机：length AUROC=0.500，TF-IDF AUROC≈0.509。这说明新数据没有被简单文本特征轻易区分。
2. Qwen2.5-0.5B 的 hidden probe OOD AUROC≈0.751，说明有一定分布外泛化，但强度中等；entropy baseline 在这批数据上略高。
3. Qwen2.5-1.5B 的 hidden probe OOD AUROC≈0.902，高于 entropy baseline≈0.791，说明更大模型的中间层风险信号在 OOD 设置下更稳定。
4. Qwen2.5-7B 的 hidden probe OOD AUROC≈0.839，高于 entropy baseline≈0.580；它支持 7B 复现中 hidden-state risk signal 仍能跨主题泛化，但强度低于 1.5B 本次结果。
5. 这一步可以作为“该信号并非完全局限于 controlled_v4 原分布”的初步证据，但不能说已经解决泛化或真实幻觉检测。

## 5. 可写结论

在新实体和新关系构成的 OOD_v1 数据上，文本基线接近随机，而 hidden-state probe 仍保持高于随机的区分能力；其中 1.5B 模型表现更强。
这支持 query-level risk signal 具有一定跨主题泛化能力，但 0.5B 上的下降也提示该信号仍受数据分布和模型规模影响。