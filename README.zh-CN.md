# Hidden State Risk Probing 中文说明

[English README](README.md) | 中文

本仓库是一个轻量级公开实验复现包，用于整理和核验一组试探性实验：大语言模型的 hidden states 中是否包含与风险相关的可探测信号。

这个仓库用于实验审阅和复现，不是论文投稿包；其中不包含论文正文、海报、投稿文件，也不包含本地写作过程文件。

## 仓库结构

```text
data/       实验输入数据，包括受控数据、OOD、知识冲突和过程错误实验。
scripts/    可复现实验脚本，包括数据构造、hidden-state 抽取、probe、行为分析、路由和扩展实验。
outputs/    轻量结果文件，包括 CSV、JSON、JSONL、PNG、Markdown 和张量清单。
reports/    Markdown 报告、结果表、人工复核记录和审计文件。
env/        依赖清单和环境说明。
docs/       公开复现包的发布范围说明。
```

## 包含内容

- `data/` 下的实验输入数据。
- `scripts/` 下的可复现实验和分析脚本。
- `outputs/` 下的轻量输出结果。
- `reports/` 下的 Markdown 报告、表格和人工复核证据。
- `env/` 下的环境依赖说明。

## 有意排除的内容

- `outputs/**/*.pt`：hidden-state 张量体积较大且可重建。被移除张量的路径、字节数和 SHA256 记录在 `outputs/TENSOR_MANIFEST.tsv`。
- `reports/**/*.docx`：本地生成的报告和 Q&A 草稿不作为公开证据包的一部分。
- 本地编辑器配置、bytecode cache、过程笔记和历史规划文件。
- 论文正文、海报、投稿文件和许可证文件。

## 如何使用

1. 根据 `env/requirements.txt` 安装依赖。
2. 查看 `data/` 中的输入数据。
3. 如果要复现实验，按 `scripts/` 中文件名前缀的数字顺序运行对应脚本。
4. 查看 `outputs/` 中的轻量生成结果。
5. 使用 `reports/` 中的 Markdown 报告和人工复核记录核验实验结论。

## 当前发布范围

当前仓库适合作为轻量级公开实验材料包，用于同行查看和复现实验流程。正式论文发布前，还需要补充许可证、引用元数据、最终作者信息和稳定归档地址。

## 说明

英文版 `README.md` 作为 GitHub 默认展示页，中文版 `README.zh-CN.md` 用于对照阅读。两个文件描述的是同一个公开复现包范围。
