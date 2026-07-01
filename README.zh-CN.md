# Hidden State Risk Probing 中文说明

[English README](README.md) | 中文

Hidden State Risk Probing 是一个可复现实验仓库，用于研究指令微调语言模型的 hidden states 是否携带与风险类别和后续行为相关的信号。

项目内容覆盖受控提示构造、hidden-state probe、生成行为标注、分布外评估、离线路由模拟，以及关于知识冲突和过程错误的扩展实验。

## 仓库内容

- `data/`：受控提示、OOD 评估、知识冲突提示和过程错误轨迹等输入数据。
- `scripts/`：数据生成、hidden-state 抽取、probe、行为分析、路由和扩展实验脚本。
- `outputs/`：生成的指标、摘要、图表、模型输出和张量清单。
- `reports/`：实验报告、结果表、人工复核记录和审计摘要。
- `env/`：依赖清单和环境说明。
- `docs/`：补充复现说明。

## 目录结构

```text
data/       实验输入数据。
scripts/    可复现实验和分析脚本。
outputs/    生成结果和图表。
reports/    报告和复核记录。
env/        依赖和环境文件。
docs/       复现说明。
```

## 快速开始

```powershell
python -m pip install -r env/requirements.txt
```

随后可以查看 `data/` 中的输入数据，按需运行 `scripts/` 中带编号的脚本，并将生成结果与 `outputs/`、`reports/` 中的表格和报告进行对照。

## 实验索引

| 实验 | 主题 |
|---|---|
| 1 | hidden-state 抽取试验。 |
| 2 | layer-wise probe 与 entropy baseline。 |
| 3 | 受控提示与混杂控制。 |
| 4 | 生成行为分析与人工复核。 |
| 5 | 分布外泛化。 |
| 6 | 离线路由模拟。 |
| 7 Optional A | PK/CK 知识冲突扩展实验。 |
| 8 Optional B | 合成过程错误扩展实验。 |

## 如何阅读结果

建议先阅读 `reports/` 中的 Markdown 报告，再查看对应的 CSV/XLSX 表格以核对具体数值和人工复核记录。按模型生成的结果位于 `outputs/{model}/experiment*/`。

## 大型张量文件

Hidden-state 张量文件（`*.pt`）通过 `outputs/TENSOR_MANIFEST.tsv` 记录其目标路径、字节数和 SHA256。需要检查张量级结果时，可使用对应抽取脚本重新生成。