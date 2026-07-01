# Reproducibility notes

This document summarizes how to navigate the experiment materials in this repository.

## Environment

Install the main dependencies with:

```powershell
python -m pip install -r env/requirements.txt
```

Some scripts require local model downloads or model weights. Keep those outside the repository and point the scripts to the local model location when needed.

## Data

Input datasets are stored under `data/` and grouped by experiment. The same input files can be reused across multiple model runs.

## Scripts

Scripts are numbered by experiment and step. For a given experiment, start with the lower-numbered data-construction or extraction scripts, then run the analysis or summary scripts.

## Outputs

Generated outputs are grouped by model under `outputs/{model}/experiment*/`. Result tables and figures are kept in Git so the reported values can be inspected without rerunning every model call.

## Hidden-state tensors

Large hidden-state tensors use the `*.pt` format. Their expected paths, byte sizes, and SHA256 hashes are listed in `outputs/TENSOR_MANIFEST.tsv`. Regenerate them with the corresponding extraction scripts when tensor-level analysis is required.

## Reports

Markdown reports in `reports/` provide the main reading path. CSV/XLSX files in `reports/*/tables/` and `reports/*/manual_review/` provide the exact values and review records behind the summaries.