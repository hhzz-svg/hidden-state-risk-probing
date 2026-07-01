# scripts directory

This directory contains reproducible experiment scripts and helper scripts. It does not contain data, model outputs, manuscript text, or formal report drafts.

## Naming convention

```text
<experiment-number><step-number>_<short_description>.py
```

Examples:

```text
31_make_controlled_prompts_v2.py
41_generate_answers.py
52_evaluate_ood_generalization.py
61_routing_simulation.py
```

## Script groups

```text
11-12     Experiment 1: hidden-state extraction and file checks.
21-24     Experiment 2: probe, entropy baseline, comparison, and Windows runner.
31-35     Experiment 3: controlled data v2/v3/v4 and confounder controls.
41-50     Experiment 4: generation behavior, answer-key labeling, and manual review.
51-53     Experiment 5: OOD data, OOD evaluation, and summaries.
61        Experiment 6: offline routing simulation.
71-75     Optional A v1: PK/CK conflict data, generation, behavior labels, and probe.
81-88     Optional B: process-error v1/v2 and template holdout.
91-103    Later review and extension scripts: Experiment 4 round 2, PK/CK v2, calibration, adjudication, and review workbook sync.
```

## Notes

1. Scripts use the project layout `data/experimentN`, `outputs/{model}/experimentN`, and `reports/experimentN`.
2. Historical DOCX generation scripts were removed from the public package.
3. Public verification should rely on Markdown reports, CSV/XLSX review tables, and lightweight output files.
