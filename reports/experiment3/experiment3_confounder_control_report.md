---
title: 第三次实验：混杂因素控制 + 模型规模复现
experiment_id: experiment3
last_updated: 2026-06-06
status: consolidated
supersedes:
  - reports/experiment3/README(第三次实验).md
  - reports/experiment3/experiment3_final_summary.md
  - reports/experiment3/experiment3_controlled_v3_report.md
  - reports/experiment3/experiment3_controlled_v4_report.md
related_outputs:
  - outputs/qwen05b/experiment3/experiment3_v4/text_controls/controlled_v4_text_baseline_results.csv
  - outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/hidden_states.pt
  - outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/group_split_entity_layerwise_probe_metrics.csv
  - outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/entropy_baseline_metrics.csv
  - outputs/qwen05b/experiment3/experiment3_v4/hidden_controls/label_shuffle_control_summary.csv
  - outputs/qwen15b/experiment3/experiment3_v4/hidden_probe/hidden_states.pt
  - outputs/qwen15b/experiment3/experiment3_v4/hidden_probe/group_split_entity_layerwise_probe_metrics.csv
  - outputs/qwen15b/experiment3/experiment3_v4/hidden_probe/entropy_baseline_metrics.csv
  - outputs/qwen15b/experiment3/experiment3_v4/hidden_controls/label_shuffle_control_summary.csv
related_tables: []
---

# 第三次实验：混杂因素控制 + 模型规模复现

## 1. 我为什么做这次实验

实验 2 在 pilot 数据上跑出 AUROC 接近 1.0 的 hidden-state probe（探针/轻量分类器）结果。我看到结果时第一反应不是高兴，而是怀疑：这个高分到底来自模型 hidden states（隐藏状态）里真实的查询风险表征，还是 known / unknown prompt 在表面上就已经能被字符串特征区分？

第三次实验是为了回答这个怀疑。我要把研究问题收紧到：

```text
实验 2 的高分是否可能由 prompt 长度、实体表面词、关系词组合或训练流程泄漏造成？
```

我仍然只研究 query-level risk label（查询级风险标签）在生成前 hidden states 中的可解码性，没有把这个标签改写成真实 hallucination label（幻觉标签）。这一阶段的目标是建立一组文本混杂更低、划分更严格、且能跨模型复现的主控数据。

## 2. 三个研究问题

1. 原始 pilot 的高 AUROC 在多大程度上来自 prompt 表面混杂？
2. 在 entity-group split（按实体分组划分）下，长度基线与 TF-IDF 文本基线能否降到接近随机？
3. 在文本混杂降低后，Qwen2.5-0.5B 与 Qwen2.5-1.5B 的 hidden states 是否仍能解码 query-level risk label？

## 3. 我的数据迭代过程

控制数据不是一次设计好的，是我跑出问题、再修正、再跑出新问题的过程。我把这一过程完整保留下来，避免论文里只展示干净的 controlled_v4 而隐瞒中间的失败。

### 3.1 原始 pilot

原始 pilot 让我第一次看到 hidden probe 高分，但同样暴露出强文本混杂：

```text
length AUROC = 0.867
TF-IDF AUROC = 0.978
```

分类器可以靠 prompt 长度、关键词或模板差异完成区分。所以原始 pilot 我只把它当作发现问题的入口，不当作最终证据。

### 3.2 controlled_v2

我在 controlled_v2 统一了部分格式，文本混杂下降：

```text
length AUROC = 0.692
TF-IDF AUROC = 0.819
```

两个基线仍明显高于随机，说明数据里还有可被文本模型利用的表面线索。我把 controlled_v2 留作中间迭代，没有用它做主控。

### 3.3 controlled_v3

到了 controlled_v3，我让同一实体和同一关系词都同时出现在 label=0 与 label=1 里，目的是让 known / unknown 的差异尽量来自"实体与关系是否语义匹配"，而不是某个词本身。

统一格式：

```text
请回答：{entity}的{relation}是什么？
```

示例：

```text
请回答：法国的首都是什么？ label=0
请回答：法国的作者是什么？ label=1
请回答：《红楼梦》的作者是什么？ label=0
请回答：《红楼梦》的首都是什么？ label=1
```

普通 split 下，文本基线已经接近随机：

```text
length baseline AUROC = 0.441
TF-IDF char n-gram baseline AUROC = 0.410
```

更保守的控制脚本给出：

```text
length baseline AUROC = 0.441
TF-IDF char n-gram baseline AUROC = 0.626
label shuffle AUROC overall mean = 0.502
label shuffle AUROC overall max = 0.604
```

但我加入 entity-group split 之后又翻车了：

```text
v3 entity-group TF-IDF AUROC = 0.950
```

"实体 + 的 + 关系"这种局部字符片段仍然让 TF-IDF 抓到实体类型和关系词的组合模式。v3 因此不能拿来当主控数据。

### 3.4 controlled_v4

我在 controlled_v4 把 entity 和 relation 拆到不同字段行，让 char n-gram 没法跨越实体和关系的边界形成稳定预测：

```text
请回答下面查询。
对象：法国
属性：首都
答案：
```

数据文件：

```text
data/experiment3/prompts_controlled_v4.jsonl
```

数据规模：

```text
600 条
known = 300
unknown/high-risk = 300
```

在 entity-group split 下，文本基线终于落到接近随机：

```text
length baseline AUROC = 0.482
TF-IDF char n-gram baseline AUROC = 0.457
```

我把 controlled_v4 选作第三次实验的正式主控数据。

## 4. controlled_v4 的设计要点

controlled_v4 与前三版最大的区别，是把实体和关系拆到不同字段行，而不是继续用"X 的 Y 是什么"这种短句模板。这样可以减少 char n-gram 跨越实体与关系边界、形成稳定预测片段的风险。

我把 entity-group split 选作主划分方式。原因是每个实体会成对出现（known 与 unknown 各一条），普通随机划分会让同一实体的另一条样本进入训练集，没法充分检验实体表面词的泄漏。entity-group split 更贴合这一阶段的控制目标。

## 5. 文本基线

### 5.1 长度基线

controlled_v4 在 entity-group split 下的 length AUROC：

```text
0.482
```

### 5.2 TF-IDF char n-gram 基线

controlled_v4 在 entity-group split 下的 TF-IDF AUROC：

```text
0.457
```

两个基线都落在随机水平附近，说明 controlled_v4 中的 known / unknown 划分不再容易被简单文本表面特征区分。

## 6. Hidden-state probe 结果

### 6.1 Qwen2.5-0.5B

hidden states 文件：

```text
outputs/qwen05b/experiment3/experiment3_v4/hidden_probe/hidden_states.pt
hidden shape per sample = [25, 896]
```

entity-group layer-wise probe：

```text
best layer = Layer 18
AUROC = 0.9999 ± 0.0001
Accuracy = 0.9933
F1 = 0.9933
```

### 6.2 Qwen2.5-1.5B

hidden states 文件：

```text
outputs/qwen15b/experiment3/experiment3_v4/hidden_probe/hidden_states.pt
hidden shape per sample = [29, 1536]
```

entity-group layer-wise probe：

```text
best layer = Layer 21
AUROC = 1.0000 ± 0.0000
Accuracy = 0.9983
F1 = 0.9983
```

两个模型都在中后层给出很高的线性可解码性。与实验 2 不同，这一次的高分是在文本基线接近随机、且使用 entity-group split 的条件下拿到的，所以我把它作为项目主线的主要证据。

## 7. Logit-level baseline 对比

Qwen2.5-0.5B：

```text
entropy AUROC = 0.7984
negative top1 probability AUROC = 0.7803
negative margin AUROC = 0.7438
```

Qwen2.5-1.5B：

```text
entropy AUROC = 0.8781
negative top1 probability AUROC = 0.8565
negative margin AUROC = 0.8186
```

logit-level uncertainty baseline 高于随机，但仍低于 hidden-state probe。我的解读是：风险划分信号不只体现在 next-token 输出分布里，生成前的内部表征本身提供了更强的可解码信息。

## 8. Label shuffle control

Qwen2.5-0.5B：

```text
mean AUROC = 0.5053
max AUROC = 0.5936
```

Qwen2.5-1.5B：

```text
mean AUROC = 0.4984
max AUROC = 0.5792
```

label shuffle 接近随机。这一步不能单独证明 hidden-state signal 的因果来源，但可以降低"训练与评估流程本身制造高分"的可能性。

## 9. 跨模型对比与阶段性结论

在 controlled_v4 + entity-group split 设置下，Qwen2.5-0.5B 与 Qwen2.5-1.5B 的中后层 hidden states 都能高精度区分人工构造的 query-level risk label。这个信号明显强于长度基线、TF-IDF 文本基线和 logit-level uncertainty baseline；label shuffle 接近随机。

这是我目前能拿到的最强阶段性证据。它把实验 2 的"高分但可能有混杂"，推进到"在强控制数据和跨模型复现下仍然保持高可解码性"。

## 10. 实验边界

我不能写：

```text
模型已经能检测真实幻觉。
模型具备人类式未知判断能力。
该 probe 已经可以作为可靠的线上回答拦截器。
```

更准确的写法：

```text
在 controlled_v4 中，query-level risk label 可以从生成前 hidden states 中被高精度线性解码。
```

我没有直接评估回答事实性，也没有说明模型为什么形成这种表征。我能支持的是：hidden-state probing 可以作为机制可解释性线索；我不能支持的是：把 probe 结果直接等同于真实幻觉判断或因果机制结论。

## 11. 与项目主线的关系

第三次实验是正文里的核心方法学证据。它把三件事串了起来：识别并修正原始 pilot 与 v2 / v3 的文本混杂；在 controlled_v4 中建立更严格的数据与划分；在 0.5B 与 1.5B 两个模型上复现 hidden-state probe 的高可解码性。

实验 3 也是后续行为关联（实验 4）、OOD 泛化（实验 5）和离线路由模拟（实验 6）的基础。所有后续实验都默认 controlled_v4 是可信的主控数据。

## 12. 下一步

我下一步进入实验 4：让模型在 controlled_v4 查询上真实生成答案，再检查生成前的 hidden risk score 是否与 correct / hallucination / refusal / irrelevant 等实际行为存在关联。实验 4 的重点不再是"标签本身能不能被解码"，而是"风险分数与模型输出行为之间是否存在可观察关系"。
