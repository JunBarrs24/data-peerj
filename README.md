# Evaluation data and code: ValladolIA municipal legal assistant

Supplement for the article "From Citizen Queries to Expert Review: Evaluating a
Retrieval-Augmented Municipal Legal Assistant"

This bundle contains only the data and code needed to reproduce the evaluation reported in
the article. It is a standalone subset of a larger private project repository; the runtime
application, secrets, and other files are deliberately excluded.

## Contents

```
data/
  judge_benchmark_anonymized.csv          Layer 1 expert benchmark (47 questions, 4 judges, consolidated answer + status)
  production_interactions_deidentified.csv Layer 3/4 production interactions (664 rows; derived fields only; no user text, no IDs)
  production_summary.json                  Aggregates (n=664) that reproduce Tables 4, 5, 6
validator_output/
  baseline_report.json                     Layer 2 baseline run (5.46)
  report_test_2026-03-02_035001.json       Layer 2 optimized run (7.56)
  latest_report.json                       Layer 2 later 47-case run (8.41)
  rerun_2026-06-10_run1.json               Layer 2 current-system clean re-run (8.18)
  retrieval_metrics_2026-06-10.json        Retrieval recall@k / MRR (proxy gold)
  llm_responses_2026-06-11_052339.json     System answers scored in the clean re-run
code/
  run_validation.py                        LLM-as-judge harness (judge model: llama3 via Ollama)
  retrieval_metrics.py                     recall@k / precision@k / MRR against judge-cited articles
  thresholds.json                          Validator pass/fail thresholds
```

## File-to-claim map

| Article element | File(s) |
|---|---|
| Table 2 (benchmark, states 34/12/1) | `data/judge_benchmark_anonymized.csv` |
| Table 3 (5.46 / 7.56 / 8.18 / 8.41, worse counts, pass=false) | `validator_output/*.json`, `code/thresholds.json` |
| Retrieval recall@5 0.768, MRR 0.498 (and 31-case variant) | `validator_output/retrieval_metrics_2026-06-10.json`, `code/retrieval_metrics.py` |
| Table 4 (type / regulation distribution) | `data/production_interactions_deidentified.csv`, `data/production_summary.json` |
| Table 5 (feedback 49/1/614, categories) | same as Table 4 |
| Table 6 (latency, tokens, chunks) | same as Table 4 |
| Figure 2 (validator bars) | `validator_output/*.json` |
| Figure 3 (demand by type) | `data/production_summary.json` |

## Reproduce

> Both `production_summary.json` and `production_interactions_deidentified.csv` were
> regenerated to the 664-interaction cut (source `new-data-jun21.csv`, to 2026-06-21) after
> the 2026-06-16 Reglamento de Tránsito reform drove a ×2.51 surge over the earlier 265 cut.
> The 250/265 figures are retained only as historical baselines in the article prose.

- Tables 4 to 6 and Figures 3: read `production_interactions_deidentified.csv` (one row per
  interaction, derived fields only) or `production_summary.json` (precomputed aggregates).
- Table 3 and Figure 2: read the values directly from the `validator_output/*.json` reports
  (`summary.averages.overall`, `summary.counts`). The judge runs at temperature 0, so the
  reports are the deterministic record of each run.
- Retrieval metrics: `retrieval_metrics_2026-06-10.json` holds the recall@k / MRR values;
  `retrieval_metrics.py` is the script that produced them from the judge-cited articles.

Note on re-running `run_validation.py`: the harness is included for transparency. It expects a
reference file in the original column layout; `judge_benchmark_anonymized.csv` renames the four
judges to A to D, so re-running requires mapping those columns back. The committed
`validator_output/*.json` reports are the primary, directly-checkable record of the reported
numbers.

## Privacy and de-identification

The production interactions come from a public deployment. Before release the raw logs were
reduced to non-identifying derived fields. Removed: user/session identifiers (even hashed),
the raw question text, the generated answer text, the enhancer JSON, and the interaction id.
Kept: question type, classified regulation, latency, token counts, retrieved-chunk count,
feedback (like/dislike/none), feedback category, and the date. This preserves every
distribution reported in the article while exposing no user-generated content. The expert
benchmark judges are anonymized to A to D.

## License

- Data (`data/`, `validator_output/`): CC BY 4.0.
- Code (`code/`): MIT.
