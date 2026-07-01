# reports directory

This directory keeps the human-readable evidence for the public replication package: Markdown reports, result tables, manual review files, and audit records.

Generated DOCX reports, Q&A drafts, historical planning notes, and process-status notes are intentionally excluded from the public package.

## Layout

```text
reports/
  experiment2/
  experiment3/
  experiment4/
  experiment5/
  experiment6/
  experiment7_optionalA/
  experiment8_optionalB/
```

## Main report index

| Experiment | Main public material |
|---|---|
| 2 | `experiment2/experiment2_layerwise_probe_report.md` |
| 3 | `experiment3/experiment3_confounder_control_report.md` |
| 4 | `experiment4/experiment4_generation_behavior_report.md` and `experiment4/manual_review/experiment4_manual_review_pipeline_and_results.md` |
| 5 | `experiment5/experiment5_ood_generalization_report.md` |
| 6 | `experiment6/experiment6_routing_simulation_report.md` |
| 7 Optional A | `experiment7_optionalA/experiment7_optionalA_pk_ck_comprehensive_report.md` |
| 8 Optional B | `experiment8_optionalB/experiment8_optionalB_v1_diagnostic_archive.md` and `experiment8_optionalB/experiment8_optionalB_v2_comprehensive_report.md` |

## Tables and review files

```text
experiment4/tables/                                  Cross-model behavior and prediction metrics.
experiment4/manual_review/*.csv                      Manual and assisted review records.
experiment5/tables/                                  OOD summary tables.
experiment6/tables/                                  Offline routing summary tables.
experiment7_optionalA/tables/                        PK/CK behavior, probe, and control-baseline tables.
experiment7_optionalA/pk_ck_v2_manual_review_*.csv   Optional A v2 manual review files.
experiment8_optionalB/tables/                        Process-error v1/v2, template holdout, and calibration tables.
```

## Use rules

1. Use Markdown reports and CSV/XLSX tables as the public evidence source.
2. Keep `tables/*`, `manual_review/*.csv`, and `manual_review/*.xlsx` unchanged unless rerunning the corresponding experiment.
3. Optional A and Optional B are extension experiments. They do not replace the main evidence chain from Experiments 3-6.
4. Deleted DOCX, Q&A, and planning files are not part of the public evidence package.
