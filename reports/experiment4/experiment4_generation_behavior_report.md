# 第四次实验：真实生成行为标注初步报告

## 1. 实验目的

前三次实验证明，在 controlled_v4 和 entity-group split 设置下，query-level risk label 可以从中后层 hidden states（隐藏状态）中被高度线性解码。
第四次实验进一步检查：这个生成前风险信号是否与模型真实生成行为存在关联。

注意：这里的 query-level risk label 仍然不是 hallucination label（幻觉标签）。本实验通过模型真实生成答案，再用答案表辅助标注 correct/refusal/hallucination/irrelevant。

## 2. 设置

- 数据：`data/experiment3/prompts_controlled_v4.jsonl`，600 条，known/low-risk 300 条，unknown/high-risk 300 条。
- 模型：Qwen2.5-0.5B-Instruct 与 Qwen2.5-1.5B-Instruct。
- 生成方式：greedy decoding，`max_new_tokens=32`，取第一行作为主要答案，同时保留 raw answer。
- 行为标注：使用 `scripts/43_label_generation_behavior_with_answer_key.py` 的答案表辅助标注；仍建议抽样人工复核。

## 3. 生成行为分布

| 模型 | known 正确 | known 错误/类幻觉 | high-risk 类幻觉 | high-risk 拒答 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 182/300 (60.7%) | 117/300 (39.0%) | 300/300 (100.0%) | 0/300 (0.0%) |
| Qwen2.5-1.5B-Instruct | 233/300 (77.7%) | 67/300 (22.3%) | 298/300 (99.3%) | 2/300 (0.7%) |
| Qwen2.5-7B-Instruct | 279/300 (93.0%) | 20/300 (6.7%) | 279/300 (93.0%) | 20/300 (6.7%) |

## 4. 风险分数预测真实行为

这里的目标是 `high_risk_behavior`，即标注为 hallucination/refusal/irrelevant 的生成行为。

| 模型 | hidden risk score AUROC | entropy AUROC | negative top1 AUROC | negative margin AUROC |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 0.8816 | 0.9163 | 0.9077 | 0.8818 |
| Qwen2.5-1.5B-Instruct | 0.9479 | 0.9316 | 0.9252 | 0.8948 |
| Qwen2.5-7B-Instruct | 0.9701 | 0.4264 | 0.3910 | 0.3696 |

## 5. 初步结论

1. 两个模型在 high-risk entity-relation query 上都很少拒答，更多是给出具体但不可靠的答案；这支持继续研究生成前风险信号与实际生成行为的关系。
2. Qwen2.5-1.5B 的 known 正确率高于 0.5B，说明更大模型在同一数据上生成质量更好，但 high-risk 查询仍主要表现为类幻觉回答。
3. 对真实行为标签而言，0.5B 上 entropy baseline 的 AUROC 略高于 hidden risk score；1.5B 上 hidden risk score 略高于 entropy baseline。这个结果要诚实写，不应只强调 hidden probe。
4. 本实验的标注仍是答案表辅助标注，不是严格双人独立人工标注；最终论文中应把它作为初步证据，而不是最终幻觉检测结论。

## 6. 后续建议

- 抽样复核 `answer_key_behavior_labels.csv` 中 known 错误样本，重点看别名、翻译名、争议事实。
- 下一步可以进入 Experiment 5: OOD 泛化实验，检查 probe 换一批新实体/新关系后是否还能工作。