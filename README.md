# Hidden State Risk Probing

[中文说明](README.zh-CN.md) | English

Hidden State Risk Probing is a reproducible experiment repository for studying whether hidden states from instruction-tuned language models carry signals related to risk categories and downstream behavior.

The project combines controlled prompt construction, hidden-state probing, behavior labeling, out-of-distribution evaluation, routing simulation, and extension experiments on knowledge conflict and process errors.

## Repository contents

- `data/`: input datasets for controlled prompts, OOD evaluation, knowledge-conflict prompts, and process-error traces.
- `scripts/`: reproducible scripts for data generation, hidden-state extraction, probes, behavior analysis, routing, and extensions.
- `outputs/`: generated metrics, summaries, plots, model outputs, and tensor manifests.
- `reports/`: experiment reports, result tables, manual review records, and audit summaries.
- `env/`: dependency lists and environment notes.
- `docs/`: additional reproducibility notes.

## Repository layout

```text
data/       Experiment input data.
scripts/    Reproducible experiment and analysis scripts.
outputs/    Generated results and figures.
reports/    Human-readable reports and review records.
env/        Dependency and environment files.
docs/       Reproducibility notes.
```

## Quick start

```powershell
python -m pip install -r env/requirements.txt
```

Then inspect the input data under `data/`, run the relevant numbered scripts under `scripts/`, and compare the generated results with the tables and reports under `outputs/` and `reports/`.

## Experiment map

| Experiment | Focus |
|---|---|
| 1 | Pilot hidden-state extraction. |
| 2 | Layer-wise probe and entropy baseline. |
| 3 | Controlled prompts and confounder checks. |
| 4 | Generation behavior analysis and manual review. |
| 5 | Out-of-distribution generalization. |
| 6 | Offline routing simulation. |
| 7 Optional A | PK/CK knowledge-conflict extension. |
| 8 Optional B | Synthetic process-error extension. |

## Reading the results

Start with the Markdown reports in `reports/`, then use the corresponding CSV/XLSX tables for exact values and manual review records. Model-specific generated outputs are organized under `outputs/{model}/experiment*/`.

## Large tensor files

Hidden-state tensor files (`*.pt`) are represented by `outputs/TENSOR_MANIFEST.tsv`, which records their expected paths, byte sizes, and SHA256 hashes. Regenerate them with the extraction scripts when tensor-level inspection is needed.