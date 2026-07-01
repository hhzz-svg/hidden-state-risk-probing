---
title: 第四次实验：AI 辅助人工复核流程与结果
experiment_id: experiment4
last_updated: 2026-06-06
status: consolidated_and_adjudicated
supersedes:
  - reports/experiment4/manual_review/experiment4_human_check_round2_guide.md
  - reports/experiment4/manual_review/experiment4_human_check_combined_summary.md
  - reports/experiment4/manual_review/experiment4_manual_review_adjudicated_summary.md
related_tables:
  - reports/experiment4/manual_review/experiment4_manual_review_priority_all.csv
  - reports/experiment4/manual_review/experiment4_manual_review_priority_all_adjudicated.csv
  - reports/experiment4/manual_review/experiment4_manual_review_batch200.csv
  - reports/experiment4/manual_review/experiment4_human_check_sample60.csv
  - reports/experiment4/manual_review/experiment4_human_check_sample60_disagreements.csv
  - reports/experiment4/manual_review/experiment4_human_check_round2_80.csv
  - reports/experiment4/manual_review/experiment4_human_check_combined_filled.csv
  - reports/experiment4/manual_review/experiment4_human_check_combined_disagreements.csv
---

# 第四次实验：AI 辅助人工复核流程与结果

## 1. 复核定位

我在第四次实验的生成行为标签上走的是"答案表辅助标注 + AI 辅助复核 + 人工抽样确认"三步流程。这条流程不是双人独立人工金标准，也不应该被写成严格的人工幻觉标注数据集。它的作用是提高行为标签的可信度，为 hidden risk score 和生成后行为的关联分析提供更稳的标签。

把这件事说清楚很关键。论文里如果我写得太重，会让评审误以为我做了昂贵的双人盲标；写得太轻，又会让证据看上去单薄。我的处理方式是：在结果报告里给出真实的样本量、复核来源和一致率，让读者自己判断。

## 2. 复核候选

高优先级复核候选共 262 条，主要来自两类原因：

| primary_review_reason | count |
|---|---:|
| known_non_correct_needs_review | 185 |
| model_label_disagreement | 77 |

按模型拆分：

| model_short | primary_review_reason | count |
|---|---|---:|
| qwen05b | known_non_correct_needs_review | 118 |
| qwen05b | model_label_disagreement | 13 |
| qwen15b | known_non_correct_needs_review | 67 |
| qwen15b | model_label_disagreement | 64 |

## 3. 人工抽样确认

我做了两轮人工抽样：

| 轮次 | 样本文件 | 填写数 | 说明 |
|---|---|---:|---|
| round1 | `experiment4_human_check_sample60.csv` | 60 | 第一轮人工确认 |
| round2 | `experiment4_human_check_round2_80.csv` | 80 | 第二轮补充人工确认 |

合并两轮：

| 项 | 值 |
|---|---:|
| 抽样总数 | 140 |
| 已填写 | 140 |
| 不计 uncertain 已填写 | 138 |
| 与 Codex-assisted 标签一致率 | 0.9710 |
| 非 uncertain 不一致样本数 | 4 |
| uncertain 样本数 | 2 |

人工标签分布：

| human_behavior_label | count |
|---|---:|
| hallucination | 90 |
| correct | 45 |
| refusal | 2 |
| uncertain | 2 |
| irrelevant | 1 |

## 4. 裁决后标签

非 uncertain 的不一致样本被单独写入裁决版标签表：

```text
reports/experiment4/manual_review/experiment4_manual_review_priority_all_adjudicated.csv
```

标签计数变化：

| label | before_count | after_count | delta |
|---|---:|---:|---:|
| correct | 73 | 76 | +3 |
| hallucination | 186 | 183 | -3 |
| irrelevant | 1 | 1 | 0 |
| refusal | 2 | 2 | 0 |

共 4 条人工裁决记录，其中 3 条带来净标签变化；另 1 条对应已经被修正为 `irrelevant` 的复述/反问样本，不改变当前净计数。

裁决修正样本明细：

| review_id | model | label_name | category | entity | codex_label | human_label | 人工依据 |
|---|---|---|---|---|---|---|---|
| exp4_review_0063 | qwen15b | known_or_low_risk | country | 墨西哥 | hallucination | irrelevant | 模型没给出首都，只是反问/复述问题 |
| exp4_review_0012 | qwen05b | known_or_low_risk | organization | 世界卫生组织 | hallucination | correct | 答案含"瑞士日内瓦"，核心地点命中预期 |
| exp4_review_0039 | qwen05b | known_or_low_risk | organization | 国际劳工组织 | hallucination | correct | 答案含"瑞士日内瓦"，核心地点命中预期 |
| exp4_review_0136 | qwen15b | known_or_low_risk | organization | 红十字国际委员会 | hallucination | correct | 答案含"瑞士日内瓦"，核心地点命中预期 |

后三条都是同一类"日内瓦总部"的误判：Codex 在 known 样本上看到模型答案没出现"日内瓦"这个准确词，就标了 hallucination；人工核对时，答案里其实写的是"瑞士日内瓦"或者同等表达，命中预期。这一类错误对我后面整理论文很有用——它说明自动标注规则在多词机构地理位置这类条目上需要更宽松的别名匹配。

## 5. 论文里能写的表述

可以写：

```text
Experiment 4 labels were produced by answer-key-assisted labeling and Codex-assisted manual review, followed by two rounds of human spot checks covering 140 samples. Excluding uncertain cases, the human check agreed with the Codex-assisted labels in 97.10% of cases. We therefore use the adjudicated labels for behavior-level analyses, while treating them as assisted behavioral annotations rather than an independently double-annotated gold standard.
```

不能写：

```text
第四次实验获得了严格人工金标准幻觉标签。
```

## 6. 关联文件

- `reports/experiment4/manual_review/experiment4_manual_review_priority_all_adjudicated.csv`
- `reports/experiment4/manual_review/experiment4_manual_review_priority_all.csv`
- `reports/experiment4/manual_review/experiment4_human_check_combined_filled.csv`
- `reports/experiment4/manual_review/experiment4_human_check_combined_disagreements.csv`
- `reports/experiment4/manual_review/experiment4_human_check_sample60.csv`
- `reports/experiment4/manual_review/experiment4_human_check_round2_80.csv`
