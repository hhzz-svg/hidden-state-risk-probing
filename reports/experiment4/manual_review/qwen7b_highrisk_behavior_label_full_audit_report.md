# Qwen2.5-7B Experiment 4 high-risk behavior label audit

Scope: all 300 high-risk controlled_v4 generations for Qwen2.5-7B-Instruct.

This is a Codex-assisted audit, not a double-blind human annotation. It is intended to identify label-boundary issues before writing the paper.

## Summary

- original_behavior_label / hallucination: 279
- original_behavior_label / irrelevant: 1
- original_behavior_label / refusal: 20
- audit_bucket / explanatory_specific_answer: 13
- audit_bucket / refusal_or_not_applicable_style: 273
- audit_bucket / short_specific_answer: 13
- audit_bucket / specific_answer: 1
- codex_review_decision / flag_boundary_case: 252
- codex_review_decision / keep_original: 48
- codex_review_label / hallucination: 27
- codex_review_label / irrelevant: 1
- codex_review_label / refusal: 20
- codex_review_label / refusal_or_not_applicable_boundary: 252

## Interpretation

- The original answer-key labels are adequate for a coarse non-correct/non-direct behavior target, because both hallucination-like answers and refusals indicate that the query should enter a conservative route.
- The word hallucination is too strong for many 7B high-risk rows. Many answers explicitly state that the requested attribute is not applicable or ask for clarification.
- Paper wording should therefore use phrases such as high-risk behavior, non-direct answer, or refusal/not-applicable boundary cases, rather than treating all high-risk labels as true hallucinations.