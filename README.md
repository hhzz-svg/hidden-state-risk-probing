# Hidden State Risk Probing

This repository is a lightweight public replication package for pilot experiments on whether LLM hidden states contain risk-related signals.

The package is intended for experiment review and reproducibility. It is not a paper submission package, and it does not include the manuscript, poster, submission files, or local writing process files.

## Repository layout

```text
data/       Input datasets for the controlled, OOD, knowledge-conflict, and process-error experiments.
scripts/    Reproducible scripts for data generation, hidden-state extraction, probes, behavior analysis, routing, and extensions.
outputs/    Lightweight result files: CSV, JSON, JSONL, PNG, Markdown, and a tensor manifest.
reports/    Markdown reports, result tables, manual review records, and audit files.
env/        Dependency lists and environment notes.
docs/       Release-scope notes for the public replication package.
```

## What is included

- Experiment input data under `data/`.
- Reproducible analysis scripts under `scripts/`.
- Lightweight outputs under `outputs/`.
- Markdown reports, tables, and manual review evidence under `reports/`.
- Environment requirements under `env/`.

## What is intentionally excluded

- `outputs/**/*.pt`: hidden-state tensors are large and reconstructible. Their paths, byte sizes, and SHA256 hashes are listed in `outputs/TENSOR_MANIFEST.tsv`.
- `reports/**/*.docx`: generated local report and Q&A drafts are not part of the public evidence package.
- Local editor state, bytecode caches, process notes, and historical planning files.
- Manuscript, poster, submission files, and license files.

## How to use

1. Install dependencies from `env/requirements.txt`.
2. Inspect input data under `data/`.
3. Run scripts in numeric order from `scripts/` when reproducing a specific experiment.
4. Check `outputs/` for lightweight generated results.
5. Use `reports/` for the human-readable experiment summaries and manual review records.

## Current release scope

This repository is suitable as a lightweight public experiment package. Before formal paper release, add a license, citation metadata, final author information, and a stable archive identifier.
