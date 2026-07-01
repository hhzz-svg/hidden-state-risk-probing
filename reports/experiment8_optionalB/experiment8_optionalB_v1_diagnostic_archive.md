---
title: 可选实验 B v1：诊断归档（已弃用）
experiment_id: experiment8_optionalB
last_updated: 2026-06-06
status: consolidated_diagnostic_only
supersedes:
  - reports/experiment8_optionalB/optionalB_process_error_data_summary.md
  - reports/experiment8_optionalB/optionalB_process_error_preliminary_report.md
related_outputs:
  - outputs/qwen15b/experiment8_optionalB/hidden_probe/hidden_states_process_error.pt
  - outputs/qwen15b/experiment8_optionalB/hidden_probe/process_error_layerwise_probe/layerwise_probe_metrics.csv
  - outputs/qwen15b/experiment8_optionalB/controls/process_error_controls.csv
related_tables:
  - reports/experiment8_optionalB/tables/process_error_trace_preview.csv
  - reports/experiment8_optionalB/tables/process_error_probe_summary.csv
  - reports/experiment8_optionalB/tables/process_error_controls.csv
---

# 可选实验 B v1：诊断归档（已弃用）

> **状态：已诊断弃用。正式结果见 `experiment8_optionalB_v2_comprehensive_report.md`。**
> **保留这份文件是因为它记录了我的一次设计失败，可以在论文 Discussion 里作为"自查并修正混杂"的例子。**

## 1. 文件性质说明

B-v1 是我对合成多步算术推理轨迹做的一个 step-level hidden-state signal（步骤级隐藏状态信号）探索。我后来跑混杂控制发现，v1 的标签和步骤位置、文本长度强绑定，所以这一版不能作为正向证据使用。

我把 v1 的数据、结果和失败诊断单独留下来。它的价值不在"这条路走通了"，而在"我自己走出去发现错了，并据此把数据重做了一版"。

## 2. v1 实验设计

v1 想看：在合成多步算术或方程推理轨迹中，错误发生前后的 hidden states 是否存在可解码差异？

二分类目标：

```text
before_error vs at_or_after_error
```

这本来就是受控合成数据，不等同于开放域链式推理错误检测。

## 3. v1 数据规模

数据文件：

```text
data/experiment8_optionalB/process_error_traces_v1.jsonl
```

预览：

```text
reports/experiment8_optionalB/tables/process_error_trace_preview.csv
```

规模：

- problem 数：160
- step-prefix 样本数：960

标签：

| version | label_name | count |
|---|---|---:|
| correct | no_error_control | 480 |
| corrupted | at_or_after_error | 320 |
| corrupted | before_error | 160 |

错误类型：

| error_type | count |
|---|---:|
| division_error | 120 |
| final_addition_error | 240 |
| final_multiplication_error | 120 |

v1 的二分类 probe 只用 corrupted step-prefix 样本：

```text
before_error = 160
at_or_after_error = 320
num_samples = 480
```

## 4. v1 失败诊断

### 4.1 Hidden probe

评估方式：`GroupKFold(trace_id)`，避免同一题的不同步骤同时出现在训练和测试。

| 模型 | 样本数 | groups | 最佳层 | AUROC | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 480 | 160 | 1 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 |

### 4.2 step_index 基线

```text
step_index AUROC = 1.0000 ± 0.0000
Accuracy = 1.0000
F1 = 1.0000
```

### 4.3 长度基线

```text
step_prefix_char_length AUROC = 1.0000 ± 0.0000
Accuracy = 0.8562
F1 = 0.8784
```

### 4.4 TF-IDF 基线

```text
tfidf_char_ngram_step_prefix AUROC = 1.0000 ± 0.0000
Accuracy = 1.0000
F1 = 1.0000
```

### 4.5 我的诊断

hidden-state probe 满分，但 step_index、长度和 TF-IDF 也都满分。这说明 v1 的标签设计本身和位置、文本长度高度绑定：`before_error` 总在第 1 步，`at_or_after_error` 总在第 2 步或答案步。任何能看到"这条样本是第几步"的特征都能拿满分。

所以 v1 不能写成"模型内部已经检测到推理过程错误"。它只是 sanity check 失败的样子——我设计的数据让标签可以被位置直接预测。

## 5. 我的改进方向

v2 要做的就是把错误标签和 step_index 解耦：

1. 让错误可能出现在第 1、2、3 步，不要固定在第 2 步。
2. 同一 step_index 下既有正确 prefix 也有错误 prefix。
3. 比较同一题、同一步位置的 correct prefix 与 corrupted prefix。
4. 把主目标从 `before_error vs after_error` 改成 `step_correct vs step_incorrect`。
5. 继续报告 step_index、length、TF-IDF 和 label shuffle 控制基线。

## 6. 教训记录

可写的保守表述：

```text
In a first synthetic process-error slice, hidden-state probes can perfectly separate before-error and at/after-error step prefixes. However, simple step-index and text baselines achieve the same performance, indicating strong positional and textual confounds. We therefore treat this result as a diagnostic failure case and propose a stricter follow-up design that decouples error labels from step position.
```

这段话可以放进论文 Discussion，说明我没有只挑漂亮结果发表，也会把因标签设计造成的混杂记录下来。
