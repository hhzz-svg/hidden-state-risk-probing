# data directory

This directory contains experiment input datasets. Files are organized by experiment rather than by model, because the same prompts or traces can be reused across multiple model runs.

## Layout

```text
data/
  experiment1/             Pilot hidden-state prompts.
  experiment3/             Controlled prompt sets for confounder checks.
  experiment5/             Out-of-distribution prompt set.
  experiment7_optionalA/   PK/CK knowledge-conflict prompt sets.
  experiment8_optionalB/   Synthetic process-error traces.
```

## Notes

- Keep source data in this directory only.
- Do not write model outputs or reports here.
- Generated results should go to `outputs/`.
- Cross-model summary tables should go to `reports/<experiment>/tables/`.
