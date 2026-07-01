# 第四次实验初步分析报告

## 重要说明

本报告使用外部标签文件中的 `behavior_label`。如果该文件来自 answer key，它属于答案表辅助标注；如果来自人工填写，则属于人工标注。
`label` 仍然只是 query-level risk label（问题级风险标签），不是真实 hallucination label（幻觉标签）。

## 数据概况

- 模型：Qwen2.5-7B-Instruct
- 生成样本数：600
- 标签来源统计：{'manual': 600}
- 行为标签分布：{'hallucination': 299, 'correct': 279, 'refusal': 20, 'irrelevant': 2}

## Hidden risk score 设置

- 分割方式：GroupKFold(entity)
- 使用层：Layer 10
- 对 query-level risk label 的 AUROC：1.0000
- Accuracy：0.9983
- F1：0.9983

## 按 query-level label 的行为分布

| label_name | correct | hallucination | irrelevant | refusal |
| --- | --- | --- | --- | --- |
| known_or_low_risk | 279 | 20 | 1 | 0 |
| unknown_or_high_risk | 0 | 279 | 1 | 20 |

## 行为预测指标

| target | score | auroc | median_threshold | accuracy_at_median | precision_at_median | recall_at_median | f1_at_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_risk_behavior | hidden_risk_score_oof | 0.9700867584497369 | 0.36734478175640106 | 0.965 | 1.0 | 0.9345794392523364 | 0.966183574879227 |
| high_risk_behavior | entropy_score | 0.42635581013633467 | 0.24423386901617045 | 0.41833333333333333 | 0.4533333333333333 | 0.4236760124610592 | 0.43800322061191627 |
| high_risk_behavior | neg_top1_prob_score | 0.3910438928527562 | -0.9545186460018158 | 0.405 | 0.44 | 0.411214953271028 | 0.4251207729468599 |
| high_risk_behavior | neg_margin_score | 0.3696334260096696 | -0.9237272450700402 | 0.3883333333333333 | 0.42333333333333334 | 0.3956386292834891 | 0.40901771336553944 |

## 当前可写的保守结论

在当前答案表辅助标注下，Qwen2.5-7B-Instruct 对 high-risk entity-relation query 大多会给出具体答案，
这些回答可被标记为 hallucination-like behavior（类幻觉行为）。
同时，known query 中也存在一部分事实错误回答，因此第四次实验不再只是复现 query-level label。
不过该标注仍建议抽样人工复核，论文中应写作“答案表辅助标注/初步行为标注”，避免说成严格人工金标准。

## 下一步

优先人工复核 `answer_key_behavior_labels.csv` 中的错误样本，尤其是别名、翻译名和有争议实体。
复核后重新运行本脚本，以修订后的行为标签更新最终图表。