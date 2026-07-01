# Public replication package manifest

## Repository

This manifest describes the public GitHub repository `hhzz-svg/hidden-state-risk-probing`.

Only the `hidden_state_pilot` experiment package is published. The manuscript, poster, and submission package are not published in this repository.

## Included

- `data/`: input data for controlled, OOD, knowledge-conflict, and process-error experiments.
- `scripts/`: data construction, hidden-state extraction, probing, behavior analysis, OOD evaluation, routing, and extension scripts.
- `env/`: dependency lists and environment notes.
- `outputs/`: CSV, JSON, JSONL, PNG, Markdown outputs, and `TENSOR_MANIFEST.tsv`.
- `reports/`: Markdown reports, result tables, and manual review records.
- `docs/`: release-scope documentation.

## Excluded

- `outputs/**/*.pt`: 12 hidden-state tensor files, about 888 MB in total. They are listed in `outputs/TENSOR_MANIFEST.tsv` with relative paths, byte sizes, and SHA256 hashes, and can be rebuilt from the extraction scripts.
- `reports/**/*.docx`: generated local report and Q&A drafts.
- Local editor settings, Python bytecode caches, historical planning notes, and process-status files.
- Manuscript text, poster files, submission files, and license files.

## Release principle

The package keeps the materials needed to inspect experimental numbers and rerun lightweight analyses, while excluding reconstructible large tensors and local writing-process artifacts. Before formal paper release, add a license, citation metadata, final author information, and a stable archive identifier.
