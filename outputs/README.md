# outputs directory

This directory contains generated experiment outputs: metrics, summaries, plots, model generations, and tensor manifests.

## Layout

```text
outputs/
  TENSOR_MANIFEST.tsv       Paths, byte sizes, and SHA256 hashes for hidden-state tensors.
  qwen05b/                  Results for Qwen2.5-0.5B-Instruct.
  qwen15b/                  Results for Qwen2.5-1.5B-Instruct.
  qwen7b/                   Results for Qwen2.5-7B-Instruct.
```

Model directories are further organized by experiment and stage, for example `hidden_probe`, `logit_baseline`, `generation`, `ood_evaluation`, `routing_simulation`, or `calibration`.

## Large tensor files

The repository keeps lightweight result files in Git. Hidden-state tensor files (`*.pt`) are tracked through `TENSOR_MANIFEST.tsv`; regenerate them with the corresponding extraction scripts when needed.

## Write rules

1. The first directory level should be a model name, such as `qwen05b`, `qwen15b`, or `qwen7b`.
2. The second level should be an experiment name, such as `experiment4` or `experiment7_optionalA_v2`.
3. Stage-level outputs should be placed below the experiment directory.
4. Cross-model summary tables belong in `reports/<experiment>/tables/`, not in the outputs root.