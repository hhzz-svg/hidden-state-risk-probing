# outputs directory

This directory contains experiment outputs. It does not contain raw input data, scripts, manuscript text, or formal report drafts.

## Layout

```text
outputs/
  TENSOR_MANIFEST.tsv       Manifest for removed hidden-state tensor files.
  qwen05b/                  Results for Qwen2.5-0.5B-Instruct.
  qwen15b/                  Results for Qwen2.5-1.5B-Instruct.
  qwen7b/                   Results for Qwen2.5-7B-Instruct.
```

Model directories are further organized by experiment and stage, for example `hidden_probe`, `logit_baseline`, `generation`, `ood_evaluation`, `routing_simulation`, or `calibration`.

## Tensor policy

The public package does not include `*.pt` hidden-state tensors. They are large and reconstructible, so this package keeps only:

- `TENSOR_MANIFEST.tsv`: relative path, byte size, and SHA256 for each removed tensor.
- Lightweight result files: CSV, JSON, JSONL, PNG, and Markdown.

To rebuild tensors, run the corresponding hidden-state extraction scripts from `scripts/` and use the paths listed in `TENSOR_MANIFEST.tsv`.

## Write rules

1. The first directory level should be a model name, such as `qwen05b`, `qwen15b`, or `qwen7b`.
2. The second level should be an experiment name, such as `experiment4` or `experiment7_optionalA_v2`.
3. Stage-level outputs should be placed below the experiment directory.
4. Cross-model summary tables belong in `reports/<experiment>/tables/`, not in the outputs root.
