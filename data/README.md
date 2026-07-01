# data 目录说明

这里只放实验输入数据集，不放模型输出、报告或脚本。

```text
data/
  experiment1/             # 第一次实验：pilot prompts
  experiment3/             # 第三次实验：controlled v2 / v3 / v4 prompts（主控数据）
  experiment5/             # 第五次实验：OOD v1 prompts
  experiment7_optionalA/   # 可选实验 A：PK/CK 冲突 v1 与 v2 扩样
  experiment8_optionalB/   # 可选实验 B：过程错误 v1 与 v2 配对轨迹
```

规则：

1. 后续实验可以复用前面实验的数据，但原始数据仍放在最早产生它的实验目录里。
2. 第四、五、六次实验都复用 `data/experiment3/prompts_controlled_v4.jsonl`。
3. 新数据集命名建议带版本号，例如 `prompts_controlled_v5.jsonl` 或 `prompts_ood_v2.jsonl`。
