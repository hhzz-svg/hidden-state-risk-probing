---
title: 可选实验 A：PK/CK 知识冲突综合报告
experiment_id: experiment7_optionalA
last_updated: 2026-06-06
status: consolidated_with_v2_expansion
supersedes:
  - reports/experiment7_optionalA/optionalA_pk_ck_conflict_data_summary.md
  - reports/experiment7_optionalA/optionalA_pk_ck_behavior_report.md
  - reports/experiment7_optionalA/optionalA_pk_ck_probe_report.md
  - reports/experiment7_optionalA/experiment7_optionalA_expansion_plan_v2.md
  - reports/experiment7_optionalA/experiment7_optionalA_v2_generation_summary.md
  - reports/experiment7_optionalA/experiment7_optionalA_v2_probe_report.md
  - reports/experiment7_optionalA/pk_ck_v2_manual_review_guide.md
  - reports/experiment7_optionalA/pk_ck_v2_manual_review_summary.md
related_outputs:
  - outputs/qwen05b/experiment7_optionalA/pk_ck_generation/generated_answers.jsonl
  - outputs/qwen05b/experiment7_optionalA/pk_ck_generation/pk_ck_behavior_labels.csv
  - outputs/qwen15b/experiment7_optionalA/pk_ck_generation/generated_answers.jsonl
  - outputs/qwen15b/experiment7_optionalA/pk_ck_generation/pk_ck_behavior_labels.csv
  - outputs/qwen15b/experiment7_optionalA/hidden_probe/hidden_states_pk_ck.pt
  - outputs/qwen15b/experiment7_optionalA/hidden_probe/pk_ck_layerwise_probe/layerwise_probe_metrics.csv
  - outputs/qwen15b/experiment7_optionalA/hidden_probe/pk_ck_layerwise_probe/layerwise_probe_summary.json
  - outputs/qwen15b/experiment7_optionalA_v2/pk_ck_generation/generated_answers.jsonl
  - outputs/qwen15b/experiment7_optionalA_v2/pk_ck_generation/pk_ck_behavior_labels.csv
  - outputs/qwen15b/experiment7_optionalA_v2/hidden_probe/pk_ck_v2_robust_probe/pk_ck_v2_hidden_layer_summary.csv
  - outputs/qwen15b/experiment7_optionalA_v2/hidden_probe/pk_ck_v2_robust_probe/pk_ck_v2_controls_summary.csv
related_tables:
  - reports/experiment7_optionalA/tables/pk_ck_conflict_v1_preview.csv
  - reports/experiment7_optionalA/tables/pk_ck_behavior_summary.csv
  - reports/experiment7_optionalA/tables/pk_ck_probe_summary.csv
  - reports/experiment7_optionalA/tables/pk_ck_qwen15b_text_baselines.csv
  - reports/experiment7_optionalA/tables/pk_ck_conflict_v2_expanded_preview.csv
  - reports/experiment7_optionalA/tables/pk_ck_v2_qwen15b_behavior_summary.csv
  - reports/experiment7_optionalA/tables/pk_ck_v2_qwen15b_by_prompt_style.csv
  - reports/experiment7_optionalA/tables/pk_ck_v2_qwen15b_by_category.csv
  - reports/experiment7_optionalA/tables/pk_ck_v2_hidden_layer_summary.csv
  - reports/experiment7_optionalA/tables/pk_ck_v2_controls_summary.csv
---

# 可选实验 A：PK/CK 知识冲突综合报告

## 1. 我为什么做这次实验

主线实验里，我研究的是模型 hidden states 中的查询风险表征。可选实验 A 想换一个角度：当 prompt 里给出的 contextual knowledge（上下文知识，CK）和模型自己的 parametric knowledge（参数知识，PK）冲突时，模型最后会跟随哪一种？生成前 hidden states 能不能预测这个选择？

这不是主线实验 1-6 的替代品，也不是真实 hallucination detection。我把它放在扩展位置上：它只能告诉我 hidden-state probing 框架能不能扩展到知识来源选择这种任务。

## 2. 三个研究问题

1. 在 PK/CK 冲突设置中，两个 Qwen2.5 模型最终回答更倾向于 PK 还是 CK？
2. 当模型既出现 pk_follow 又出现 ck_follow 时，生成前 hidden states 能不能预测这个偏好？
3. 文本长度和 TF-IDF baseline 是不是已经能解释这种行为差异？

## 3. 数据设计

### 3.1 PK / CK 定义

- `pk_answer`：通常事实答案，对应参数知识方向。
- `ck_answer`：prompt 背景里人为给出的冲突答案，对应上下文知识方向。
- `prompt`：模型实际看到的完整输入。
- `expected_later_labels`：模型生成后再标注的行为标签。

我特别想强调一点：这里的任务不是主线的 known / unknown 风险标签，也不是 hallucination 标签。pk_follow 与 ck_follow 必须来自模型真实生成答案后的行为标注，不能从 prompt 本身推出来。

### 3.2 v1 数据

数据文件：

```text
data/experiment7_optionalA/pk_ck_conflict_v1.jsonl
```

规模：100 条，5 类各 20 条。

| category | count |
|---|---:|
| chemical_formula | 20 |
| country_capital | 20 |
| country_continent | 20 |
| element_symbol | 20 |
| river_continent | 20 |

预览：

```text
reports/experiment7_optionalA/tables/pk_ck_conflict_v1_preview.csv
```

行为标签集合：

```text
pk_follow
ck_follow
mixed_or_conflict_ack
refusal
other
```

## 4. v1 生成行为分布

### 4.1 Qwen2.5-0.5B

| 模型 | pk_follow | ck_follow | mixed/conflict | refusal | other |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 4 | 81 | 15 | 0 | 0 |

0.5B 几乎只跟随上下文，pk_follow 只有 4 条。类别极不平衡，不能直接做稳定的二分类 probe，所以 0.5B 在 v1 之后退出。

### 4.2 Qwen2.5-1.5B

| 模型 | pk_follow | ck_follow | mixed/conflict | refusal | other |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 38 | 22 | 39 | 0 | 1 |

1.5B 同时有 pk_follow=38 和 ck_follow=22，能凑出一个小规模的二分类子集。

## 5. v1 Hidden-state probe 结果

### 5.1 子集

```text
pk_follow = 38
ck_follow = 22
num_samples = 60
```

### 5.2 Layer-wise probe

| 样本数 | pk_follow | ck_follow | 最佳层 | AUROC | Accuracy | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 38 | 22 | 20 | 0.9829 ± 0.0343 | 0.9333 | 0.9232 |

在这个 60 条子集上，hidden-state probe 可以较高精度地区分 pk_follow 与 ck_follow。

### 5.3 v1 文本基线对比

| 方法 | AUROC | Accuracy | F1 |
|---|---:|---:|---:|
| prompt_length | 0.5843 | 0.5667 | 0.5000 |
| model_answer_length | 0.5915 | 0.4333 | 0.5641 |
| tfidf_char_ngram_prompt | 0.8175 ± 0.1010 | 0.7833 | 0.6002 |

hidden probe 高于长度基线，也高于 TF-IDF prompt baseline。但 TF-IDF 已经能拿到 0.8175 ± 0.1010，说明 prompt 模板或类别还在给文本模型留口子，不能把这一结果直接解释成"模型内部纯粹的知识选择机制"。

## 6. v1 阶段的局限

v1 跑完后我自己看到三个明显的问题：

1. 二分类子集只有 60 条。
2. prompt 模板只有一种，TF-IDF baseline 因此偏高。
3. PK / CK 标签来源于答案字符串匹配，没有经过人工复核。

v2 就是为了回应这三点。

## 7. v2 扩样动机与设计

v2 没有把同一事实的不同模板重复计数当成独立样本来虚增证据，而是引入 `source_id` 和 `prompt_style` 两层结构，让我可以做 source 分组划分和 prompt style 留出划分。

数据文件：

```text
data/experiment7_optionalA/pk_ck_conflict_v2_expanded.jsonl
```

规模：

```text
100 个 source_id × 4 种 prompt style = 400 条
```

prompt style 和指令强度的对应：

| prompt_style | instruction_strength |
|---|---|
| brief_qa | neutral_short |
| conflict_explicit | explicit_context_priority |
| quoted_context_then_question | neutral |
| use_material_first | context_priority |

类别覆盖与 v1 一致：chemical_formula、country_capital、country_continent、element_symbol、river_continent，每类 20 个 source_id × 4 style = 80 条。

预览：

```text
reports/experiment7_optionalA/tables/pk_ck_conflict_v2_expanded_preview.csv
```

## 8. v2 生成行为分布

### 8.1 全量行为分布（Qwen2.5-1.5B-Instruct）

| behavior_label | count | rate |
|---|---:|---:|
| mixed_or_conflict_ack | 162 | 0.405 |
| ck_follow | 126 | 0.315 |
| pk_follow | 111 | 0.278 |
| other | 1 | 0.003 |

`mixed_or_conflict_ack` 占比最高，意味着模型在多数情况下不会干净地选一边，而是同时提到 PK 和 CK 两种说法。这一类我不并入二分类目标。

### 8.2 二分类子集

| 子集 | 样本数 |
|---|---:|
| pk_follow | 111 |
| ck_follow | 126 |
| 合计 | 237 |

237 条已经明显超过 v1 的 60 条子集，能支撑更稳的 hidden-state probe。

## 9. v2 PK/CK 人工复核

### 9.1 抽样

| 项 | 值 |
|---|---:|
| 抽样总数 | 44 |
| 已填写 | 44 |
| 不计 uncertain 已填写 | 44 |
| 规则修正后自动标签一致率 | 1.0000 |
| 不一致样本数 | 0 |

### 9.2 人工标签分布

| human_pkck_label | count |
|---|---:|
| mixed_or_conflict_ack | 20 |
| pk_follow | 17 |
| ck_follow | 6 |
| other | 1 |

### 9.3 Auto × Human 混淆表

| auto_pkck_label | human_pkck_label | count |
|---|---|---:|
| ck_follow | ck_follow | 6 |
| mixed_or_conflict_ack | mixed_or_conflict_ack | 20 |
| other | other | 1 |
| pk_follow | pk_follow | 17 |

44 条样本里，规则修正后的自动标签和人工判断完全一致。我没法把这等同于双人独立金标准，但可以说在这一抽样下，规则和人工对得起。论文里我会继续写成"自动标注 + 抽样人工核对"。

## 10. v2 Hidden-state probe 结果

### 10.1 划分方式

```text
source_id_group5         按 source_id 5 折，防止同一事实跨训练/测试
prompt_style_heldout     每次留出一种 prompt style，检查跨模板外推
```

### 10.2 Hidden probe 最佳层

| split | layer | AUROC | AUROC std | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| source_id_group5 | 21 | 0.9910 | 0.0073 | 0.9578 | 0.9598 |
| prompt_style_heldout | 28 | 0.9103 | 0.0118 | 0.8008 | 0.5329 |

### 10.3 控制基线

| split | method | AUROC | AUROC std | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| source_id_group5 | prompt_length | 0.8835 | 0.0412 | 0.5313 | 0.6930 |
| source_id_group5 | tfidf_char_ngram_prompt | 0.9726 | 0.0201 | 0.8986 | 0.8973 |
| prompt_style_heldout | prompt_length | 0.6553 | 0.0888 | 0.1904 | 0.2934 |
| prompt_style_heldout | tfidf_char_ngram_prompt | 0.9174 | 0.0476 | 0.6413 | 0.4267 |

### 10.4 我的解读

source_id_group5 下，hidden probe 的 AUROC 是 0.9910，TF-IDF 是 0.9726，hidden 只比文本基线略高一点。换到 prompt_style 留出这种更严格的划分，hidden probe 反而以 0.9103 略低于 TF-IDF 的 0.9174。这意味着 v2 数据里仍然残留相当多的模板和词汇层面的可学信号，hidden probe 没能稳稳拉开和文本基线的差距。

## 11. v1 + v2 整体阶段性结论

v1 给了一个初步线索：1.5B 在 PK/CK 冲突的小样本上，hidden probe 可以以较高 AUROC 预测最终知识来源偏好。

v2 在 237 条二分类样本和两种严格划分上重新审视这条线索。source 分组下 hidden probe 仍保持高 AUROC，但 TF-IDF 文本基线同样很高；prompt style 留出下，hidden probe 与 TF-IDF 接近甚至略低。

合起来看，当前 PK/CK 数据中，prompt 模板和实体词汇仍然承载相当一部分可解码信号。hidden states 的额外贡献既没有被严格证伪，也没有被严格证实。

## 12. 实验边界

1. v1 二分类子集只有 60 条，模板单一，只能算初步探索。
2. v2 扩样到 237 条二分类样本，但 TF-IDF baseline 偏高，hidden probe 的边际优势不稳定。
3. PK/CK 标签来源于答案字符串规则，只有 v2 上做了 44 条小样本人工核对。
4. 只评估了 Qwen2.5-1.5B；0.5B 在 v1 已经因严重不平衡退出二分类。
5. 这仍是知识冲突场景下的可解释性探索，不能解释为模型具备显式知识来源选择能力。

## 13. 与项目主线的关系

可选实验 A 适合放在附录或扩展实验位置。v1 给出"hidden-state probing 可以触及知识来源偏好"的初步线索；v2 在更大样本和多模板下验证了这条线索，同时也把 TF-IDF baseline 不可忽视的事实摆了出来。

在论文里，可选实验 A 承担"扩展加局限性"的角色：说明 hidden-state probing 的适用范围可以从 query-level risk 推到知识来源选择，但它替代不了实验 3-6 的主证据链，也不能写成知识冲突任务上的稳定结论。
