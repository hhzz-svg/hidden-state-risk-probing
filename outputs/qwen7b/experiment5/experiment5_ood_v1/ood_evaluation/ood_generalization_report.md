# Experiment 5: OOD 泛化实验初步报告

## 设置

- 训练数据：`data/experiment3/prompts_controlled_v4.jsonl`，controlled_v4，600 条。
- 测试数据：`data/experiment5/prompts_ood_v1.jsonl`，OOD_v1，200 条。
- 模型：Qwen2.5-0.5B-Instruct。
- Hidden probe 使用层：Layer 10。
- 训练方式：只在 controlled_v4 hidden states 上训练 Logistic Regression probe，然后直接测试 OOD_v1。

## 结果

| 方法 | OOD AUROC | median 阈值 Accuracy | F1 |
| --- | ---: | ---: | ---: |
| hidden_layer_10 | 0.8389 | 0.7800 | 0.7800 |
| length_baseline | 0.5000 | 0.5000 | 0.6667 |
| tfidf_char_ngram | 0.5093 | 0.5100 | 0.5149 |
| entropy | 0.5796 | 0.5700 | 0.5700 |
| negative_top1_prob | 0.5548 | 0.5700 | 0.5700 |
| negative_margin | 0.5447 | 0.5600 | 0.5600 |

## 初步解释

Hidden-state probe 在 OOD_v1 上的 AUROC 为 0.8389。
这个结果应被解释为“从 controlled_v4 学到的 query-level risk signal 在新实体/新关系上存在一定泛化”，
但还不能说明它已经能稳定检测真实幻觉。

如果后续要写论文，建议把 OOD_v1 作为分布外泛化的初步证据，并继续扩充不同主题的数据。
