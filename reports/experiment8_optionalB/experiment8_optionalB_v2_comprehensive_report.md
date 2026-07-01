---
title: 可选实验 B v2：过程错误/步骤正确性综合报告
experiment_id: experiment8_optionalB
last_updated: 2026-06-06
status: consolidated_with_calibration_addendum
supersedes:
  - reports/experiment8_optionalB/optionalB_process_error_v2_data_summary.md
  - reports/experiment8_optionalB/optionalB_process_error_v2_report.md
  - reports/experiment8_optionalB/optionalB_process_error_v2_template_holdout_report.md
  - reports/experiment8_optionalB/optionalB_process_error_v2_calibration_addendum.md
related_outputs:
  - outputs/qwen15b/experiment8_optionalB_v2/hidden_probe/hidden_states_process_error_v2.pt
  - outputs/qwen15b/experiment8_optionalB_v2/hidden_probe/process_error_layerwise_probe/layerwise_probe_metrics.csv
  - outputs/qwen15b/experiment8_optionalB_v2/controls/process_error_controls.csv
  - outputs/qwen15b/experiment8_optionalB_v2/template_holdout/hidden_template_holdout_layer_summary.csv
  - outputs/qwen15b/experiment8_optionalB_v2/template_holdout/template_holdout_controls.csv
  - outputs/qwen15b/experiment8_optionalB_v2/calibration/template_holdout_calibration_predictions.csv
  - outputs/qwen15b/experiment8_optionalB_v2/calibration/template_holdout_calibration_by_template.csv
  - outputs/qwen15b/experiment8_optionalB_v2/calibration/template_holdout_calibration_summary.csv
related_tables:
  - reports/experiment8_optionalB/tables/process_error_trace_v2_preview.csv
  - reports/experiment8_optionalB/tables/process_error_v2_probe_summary.csv
  - reports/experiment8_optionalB/tables/process_error_v2_controls.csv
  - reports/experiment8_optionalB/tables/process_error_v2_template_holdout_layer_summary.csv
  - reports/experiment8_optionalB/tables/process_error_v2_template_holdout_controls.csv
  - reports/experiment8_optionalB/tables/process_error_v2_template_holdout_by_template.csv
  - reports/experiment8_optionalB/tables/process_error_v2_template_holdout_calibration_summary.csv
  - reports/experiment8_optionalB/tables/process_error_v2_template_holdout_calibration_by_template.csv
---

# 可选实验 B v2：过程错误/步骤正确性综合报告

## 1. 我为什么做这次实验

B-v2 是 B-v1 的修正版。我在 B-v1 里把 `before_error` 与 `at_or_after_error` 这两个标签和 step_index、文本长度强绑定到一起，所以 hidden-state probe 的满分根本不能解释成"模型内部检测到了错误"。任何能看到位置或者长度的特征都能拿满分。

v2 我把目标改成：

```text
step_correct vs step_incorrect
```

每道题、每个 step_index 都同时构造一个正确 prefix 和一个错误 prefix，让 step_index 在设计上就携带不了标签信息。如果 hidden probe 还能区分对错，那就不能用"它只是在看第几步"打发掉。

这次的数据仍然是受控合成轨迹，不等同于开放域链式推理错误检测。我把它放在机制可解释性扩展的位置，不放在主证据链里。

## 2. 三个研究问题

1. 同一 step_index 下配对 correct / incorrect prefix 之后，hidden states 还能不能区分步骤正确性？
2. step_index、prefix 长度、TF-IDF 和 label shuffle control 会不会复现 hidden probe 的高分？
3. 如果把一种题型模板整个留出去训练，hidden-state probe 在没见过的模板上还有外推能力吗？

## 3. 数据设计修正

### 3.1 v1 与 v2 的关键区别

v1 让标签和 step_index 强绑定；v2 让每道题、每个 step_index 都包含一个正确 prefix 和一个错误 prefix，把目标改成 `step_correct` vs `step_incorrect`。这样 step_index 自己不再携带标签信息。

### 3.2 v2 数据规模

数据文件：

```text
data/experiment8_optionalB/process_error_traces_v2.jsonl
```

预览：

```text
reports/experiment8_optionalB/tables/process_error_trace_v2_preview.csv
```

规模：

- problem 数：160
- step-prefix 样本数：960
- problem groups：160

### 3.3 每个 step_index 下 correct/incorrect 各 160

| step_index | label_name | count |
|---|---|---:|
| 1 | step_correct | 160 |
| 1 | step_incorrect | 160 |
| 2 | step_correct | 160 |
| 2 | step_incorrect | 160 |
| 3 | step_correct | 160 |
| 3 | step_incorrect | 160 |

错误类型：

| step_index | error_type | count |
|---|---|---:|
| 1 | first_addition_error | 40 |
| 1 | multiplication_error | 40 |
| 1 | parentheses_addition_error | 40 |
| 1 | subtraction_error | 40 |
| 2 | addition_after_multiplication_error | 40 |
| 2 | division_error | 40 |
| 2 | final_addition_error | 40 |
| 2 | multiplication_after_parentheses_error | 40 |
| 3 | wrong_final_answer | 160 |

同一 step_index 下正负样本数量相同，step_index baseline 理论上应该接近随机。

## 4. Hidden-state probe 结果（GroupKFold by problem）

评估划分：

```text
GroupKFold(trace_id)
```

同一道题的所有 prefix 不跨训练和测试集。

| 模型 | 目标 | 样本数 | groups | step_correct | step_incorrect | best layer | AUROC | Accuracy | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | step_correct_vs_step_incorrect | 960 | 160 | 480 | 480 | 28 | 0.9908 ± 0.0018 | 0.9531 | 0.9524 |

最佳层是 layer 28，AUROC=0.9908 ± 0.0018，Accuracy=0.9531，F1=0.9524。

我也单独看了不要最后一层的情况，最佳非最终层：

```text
layer 26
AUROC = 0.9141 ± 0.0130
```

报告非最终层很重要。最终层有可能混入更直接的输出预测或答案判别信息，不能把它当作"中间表征里就有错误检测信号"的唯一证据。

## 5. 控制实验

| method | AUROC | Accuracy | F1 |
|---|---:|---:|---:|
| step_index | 0.5000 ± 0.0000 | 0.5000 | 0.5714 |
| step_prefix_char_length | 0.5005 ± 0.0006 | 0.5000 | 0.5131 |
| tfidf_char_ngram_step_prefix | 0.5564 ± 0.0215 | 0.5656 | 0.6023 |
| hidden_label_shuffle_selected_layer | 0.4951 ± 0.0209 |  |  |

v2 和 v1 最大的区别就在这一节：简单控制项再也不会复现 hidden probe 的高分。`step_index` 是严格的随机水平，`step_prefix_char_length` 接近随机，TF-IDF 字符 n-gram 只略高于随机，label shuffle 也接近随机。

## 6. Held-out template 外推

### 6.1 实验设置

§4 的主结果用的是 `GroupKFold(trace_id)`，已经能避免同一道题的 prefix 同时进入训练和测试。但这还不够严格，因为同一种题型模板下的不同题目可能共享数值或结构。所以我把划分提到模板级：每次留出一种题型模板，只用另外三种模板训练，再到没见过的模板上测试。

四种模板：

```text
sum3
mul_add
parentheses_mul
linear_equation
```

每种模板里 correct / incorrect 平衡：

| template_type | step_correct | step_incorrect |
|---|---:|---:|
| linear_equation | 120 | 120 |
| mul_add | 120 | 120 |
| parentheses_mul | 120 | 120 |
| sum3 | 120 | 120 |

### 6.2 Hidden probe 外推结果

| 指标 | 结果 |
|---|---:|
| best final layer | layer 28 |
| held-out template AUROC | 0.9130 ± 0.0406 |
| Accuracy | 0.7167 |
| F1 | 0.7602 |
| best non-final layer | layer 25, AUROC=0.8253 ± 0.0642 |

按 held-out template 拆开：

| heldout_template | test_samples | test_positive | AUROC | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| linear_equation | 240 | 120 | 0.9053 | 0.5000 | 0.6667 |
| mul_add | 240 | 120 | 0.9706 | 0.9000 | 0.9016 |
| parentheses_mul | 240 | 120 | 0.9008 | 0.7292 | 0.7451 |
| sum3 | 240 | 120 | 0.8754 | 0.7375 | 0.7273 |

### 6.3 Held-out template 控制基线

| method | AUROC | AUROC std across templates | Accuracy | F1 |
|---|---:|---:|---:|---:|
| hidden_label_shuffle_selected_layer | 0.5049 | 0.0184 |  |  |
| step_index | 0.5000 | 0.0000 | 0.5000 | 0.6667 |
| step_prefix_char_length | 0.5008 | 0.0005 | 0.5000 | 0.6667 |
| tfidf_char_ngram_template_holdout | 0.5260 | 0.0229 | 0.5271 | 0.5674 |

### 6.4 解读

held-out template 这种划分比普通 problem-group 更严格。hidden probe 仍然明显高于 step_index、长度、TF-IDF 和标签打乱控制，这把"结果来自模板内记忆或同题泄漏"的可能性进一步降低了。

要小心的是，accuracy 是基于固定 0.5 阈值的，模板外推时概率校准可能失配。所以我把 AUROC 当作主指标。例如 `linear_equation` 的 AUROC 是 0.9053，但固定阈值 accuracy 只有 0.5000——排序信号是存在的，跨模板的阈值需要单独校准。

### 6.5 概率校准与阈值稳定性补充

#### 6.5.1 我为什么做这个补充

held-out template 那一步证明了 hidden probe 在 AUROC 意义上明显高于文本和位置基线。但 §6.4 里固定 0.5 阈值的 accuracy 在不同模板上波动很大，说明排序虽然稳，但概率本身可能没校准好。我想把这个问题拆开来看。

#### 6.5.2 方法

- 对每个 held-out template fold 重新训练 logistic probe。
- 在测试模板上记录概率、AUROC、Brier score、10-bin ECE（Expected Calibration Error，期望校准误差）。
- 比较三种阈值策略：固定 0.5、训练集最佳 F1、训练集 Youden，外加一个 `test_oracle_f1_diagnostic_only`。
- `test_oracle_f1_diagnostic_only` 只是诊断上界，告诉我"如果在测试集上挑最优阈值能拿到多少"，不能写成可部署结果。

#### 6.5.3 最佳层（layer 28）阈值比较

| threshold_name | threshold_mean | threshold_std | AUROC | Accuracy | F1 | Brier | ECE 10-bin |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_0.5 | 0.5000 | 0.0000 | 0.9130 | 0.7167 | 0.7602 | 0.2558 | 0.2598 |
| train_best_f1 | 0.4325 | 0.0340 | 0.9130 | 0.7167 | 0.7607 | 0.2558 | 0.2598 |
| train_youden | 0.4325 | 0.0340 | 0.9130 | 0.7167 | 0.7607 | 0.2558 | 0.2598 |
| test_oracle_f1_diagnostic_only | 0.4166 | 0.4992 | 0.9130 | 0.8271 | 0.8480 | 0.2558 | 0.2598 |

#### 6.5.4 我的解读

AUROC 在四种阈值下都是 0.9130，排序信号本身稳定。但固定 0.5 阈值和训练集衍生阈值在 accuracy / F1 上几乎没差别，并且和诊断上界（test_oracle_f1）有看得见的缺口。Brier score 和 ECE 都偏高（约 0.26），说明跨模板时概率输出本身没被校准好。

论文里我会优先报告 AUROC，把固定阈值下的 accuracy / F1 写成诊断指标。校准与阈值选择是后续工作。

## 7. 阶段性结论

在配对构造的过程错误切片里，同一道题、同一推理步位置都包含正确和错误 prefix，再用按题目分组的交叉验证评估。Qwen2.5-1.5B 的 hidden-state probe 能高精度区分 step_correct 与 step_incorrect，而 step_index、prefix 长度、TF-IDF 文本基线和标签打乱控制都明显落在 hidden probe 下面。

held-out template split 进一步表明这一结果不是简单的同题泄漏或模板内记忆。但所有数据仍然限定在受控合成算术 / 方程轨迹里，不能直接外推到开放域自然语言推理。

## 8. 实验边界

1. 数据仍然是模板化合成算术 / 方程任务，可能残留数值模式或模板模式。
2. 最强结果出现在最终层，最终层可能混入更直接的输出预测或答案判别信息，所以我同时报告了非最终层。
3. 跨模板的固定阈值 accuracy 不稳定，AUROC 更适合作为模板外推的主指标。
4. 这一结果不能直接外推为开放域自然语言推理错误检测能力。

## 9. 与项目主线的关系

可选实验 B v2 适合放在附录或正文的机制扩展分析位置。它说明 hidden-state probing 不只能研究 query-level risk label，也可以扩展到受控合成轨迹里的 step-level correctness（步骤级正确性）信号。

我在论文结构里会把 B-v2 放在主线实验之后，用来展示 hidden-state probing 框架的扩展性，而不是替代实验 3-6 的主证据链。
