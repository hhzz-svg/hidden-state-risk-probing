---
title: 第二次实验：Layer-wise Hidden-State Probe（层级隐藏状态探针）
experiment_id: experiment2
last_updated: 2026-06-06
status: consolidated
supersedes:
  - reports/experiment2/README(第二次实验).md
  - reports/experiment2/第二次实验ai文档.md
related_outputs:
  - outputs/qwen05b/experiment1/experiment1_hidden_extraction/hidden_states_pilot.pt
  - outputs/qwen05b/experiment2/experiment2_layerwise_probe/layerwise_probe_metrics.csv
  - outputs/qwen05b/experiment2/experiment2_layerwise_probe/layerwise_probe_summary.json
  - outputs/qwen05b/experiment2/experiment2_logit_baseline/entropy_baseline_metrics.csv
  - outputs/qwen05b/experiment2/experiment2_compare/probe_vs_entropy_compare.csv
  - outputs/qwen05b/experiment2/experiment2_compare/probe_vs_entropy_compare.png
related_tables: []
---

# 第二次实验：Layer-wise Hidden-State Probe（层级隐藏状态探针）

## 1. 我为什么做这次实验

第一次实验只完成了 hidden states 抽取的工程闭环：对 known / unknown 两类 query，在生成答案之前的 prefill 阶段抽取 query 最后一个 token 的全层 hidden states，存到 `outputs/qwen05b/experiment1/experiment1_hidden_extraction/hidden_states_pilot.pt`。

第二次实验我把它转成一个可以验证的问题：生成前 hidden states 里到底有没有 query-level risk signal（查询级风险信号），如果有，它能不能被一个简单的线性分类器读出来？

这一阶段我只评估 hidden-state probing 的可行性。它不是幻觉检测，也不构成 Early Exit 或线上拦截流程。我把它当作后续混杂控制、行为关联和分流模拟的前置测量。

## 2. 三个研究问题

1. 不同层的 hidden states 能不能区分 known query 与 unknown query？
2. 哪一层的可解码信号最强？
3. hidden-state probe 是不是优于简单的 logit-level uncertainty baseline？

## 3. 数据

### 3.1 数据来源

我直接用第一次实验的 200 条 pilot prompts：

- known：100 条
- unknown：100 条

数据文件是 `data/experiment1/prompts_pilot.jsonl`。模型输入、标签和 hidden states 抽取结果都来自上一阶段，这一步没有重新生成数据。

### 3.2 标签定义与边界

二分类标签：

```text
known = 0
unknown = 1
```

这是 query-level risk label，不是 hallucination label。它反映的是 prompt 在我设定下属于 known / unknown 风险划分，不是模型回答是否真的出现幻觉。所以本实验只能讨论 hidden states 里有没有跟这个风险划分相关的表征，不能把结果解释成真实幻觉检测。

## 4. 方法

### 4.1 Hidden states 抽取

每条 query 用第一次实验保存的 hidden states：

```text
hidden_states_pilot.pt
```

张量形状：

```text
[N, L, H]
```

N 为样本数，L 为层数，H 为 hidden size。对每一层单独取：

```text
X_layer = hidden_states[:, layer, :]
```

作为该层的样本表征。

### 4.2 Layer-wise probe

对每一层分别训练一个 Logistic Regression：

```text
hidden state -> Logistic Regression -> known / unknown
```

评估用 5-fold Stratified Cross Validation，让 known / unknown 类别比例稳定，也避免单次 train / test 划分的偶然性。输出 AUROC、Accuracy 和 F1，用来比较不同层的线性可分性。

### 4.3 Logit-level baseline

作为对照，我在每条 query 最后一个 prompt token 上算 next-token logits，提取三类 logit-level uncertainty signal：

```text
entropy
top1 probability
top1-top2 margin
```

这些基线在回答一个直接的问题：如果我只看输出分布的不确定性，是不是也能区分 known / unknown？hidden-state probe 与它们比较，可以判断风险信号是不是主要来自生成前的内部表征，而不是简单的 next-token 置信度。

## 5. 实验命令

### 5.1 Layer-wise probe

```bash
python scripts/21_train_layerwise_probe.py --hidden_file outputs/qwen05b/experiment1/experiment1_hidden_extraction/hidden_states_pilot.pt --data_file data/experiment1/prompts_pilot.jsonl --out_dir outputs/qwen05b/experiment2/experiment2_layerwise_probe
```

### 5.2 Entropy baseline

```bash
python scripts/22_extract_entropy_baseline.py --model_name <你的模型名或本地路径> --data_file data/experiment1/prompts_pilot.jsonl --out_dir outputs/qwen05b/experiment2/experiment2_logit_baseline
```

### 5.3 Probe vs entropy 对比

```bash
python scripts/23_compare_probe_entropy.py --probe_metrics outputs/qwen05b/experiment2/experiment2_layerwise_probe/layerwise_probe_metrics.csv --entropy_metrics outputs/qwen05b/experiment2/experiment2_logit_baseline/entropy_baseline_metrics.csv --out_dir outputs/qwen05b/experiment2/experiment2_compare
```

## 6. 结果

### 6.1 Layer-wise probe

| 项 | 数值 |
|---|---:|
| 样本数 | 200 |
| 层数 | 25 |
| hidden size | 896 |
| known | 100 |
| unknown | 100 |
| best layer | Layer 20 |
| best AUROC | 0.999 ± 0.002 |
| Accuracy | 0.99 |
| F1 | 0.99 |

在 pilot 数据集上，known / unknown query-level risk label 可以从生成前 hidden states 中被高精度线性解码。最佳层是 Layer 20，AUROC=0.999 ± 0.002，Accuracy 和 F1 都是 0.99。

### 6.2 Logit-level baseline

`outputs/qwen05b/experiment2/experiment2_logit_baseline/entropy_baseline_metrics.csv` 的结果：

| baseline | AUROC | median threshold accuracy | F1 |
|---|---:|---:|---:|
| entropy | 0.6535 | 0.61 | 0.61 |
| negative_top1_prob | 0.5853 | 0.57 | 0.57 |
| negative_margin | 0.5418 | 0.55 | 0.55 |

三个 logit-level baseline 都明显低于 hidden-state probe。entropy 的 AUROC 是 0.6535，negative_top1_prob 是 0.5853，negative_margin 是 0.5418。仅靠 next-token 输出分布的不确定性，达不到 hidden states probe 的区分能力。

### 6.3 Probe vs baseline 对比

hidden-state probe 的最佳 AUROC 是 0.999 ± 0.002，高于三种 logit-level baseline。对比表和图：

- `outputs/qwen05b/experiment2/experiment2_compare/probe_vs_entropy_compare.csv`
- `outputs/qwen05b/experiment2/experiment2_compare/probe_vs_entropy_compare.png`

## 7. 阶段性结论

第二次实验给出一个清晰的初步发现：在 200 条 pilot prompts 上，模型生成前 hidden states 中存在很强的 known / unknown query-level risk 可解码信号。这个结果支持我沿 hidden-state probing 路线继续推进。

但 AUROC 接近 1.0 这件事，本身就让我警觉。它可能说明 hidden states 里确实有稳定的风险相关表征，也可能受 prompt 长度、实体词、关系词、模板差异或数据构造方式的影响。所以我把第二次实验的价值定位为"发现了一个需要被混杂控制重新检验的高分现象"，而不是"我已经拿到稳定结论"。

## 8. 实验边界

我不能说：

```text
模型已经能识别真实幻觉。
模型具备人类式未知判断能力。
该 probe 可以直接用于线上回答拦截。
```

更准确的说法是：

```text
在 pilot 数据上，known / unknown 查询级风险标签可以从生成前隐藏状态中被高精度线性解码。
```

我没有验证回答事实性，也没有评估模型生成后的错误类型。query-level risk label 必须与 hallucination label 区分开；probe 分数只是后续分析里的内部表征指标，不能写成可部署的检测工具。

## 9. 与项目主线的关系

实验 2 是项目从"能不能抽取 hidden states"转向"hidden states 里有没有可读信号"的第一步。它把 layer-wise probe 与 logit baseline 的对比框架搭起来，也给出了一个需要重新审查的高分结果。

在整条研究线上，实验 2 不作为最强证据。它的角色是"发现现象、引出实验 3"。后续主线要回答的是：这个高分能不能在更严格的控制数据上保留？它能不能与模型生成行为、跨模型迁移和离线分流模拟形成一致证据？

## 10. 下一步

我下一步进入第三次实验：检查实验 2 的高 AUROC 是不是来自 prompt 长度、实体表面词、关系词或训练流程泄漏，并构造更严格的 controlled prompts。只有这些混杂因素被系统控制之后，hidden-state probe 才能作为项目主线里的更可靠证据。
