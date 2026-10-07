# Batch testing (CDSS pipeline)

Run everything from the `integration` folder with the venv: `.venv\Scripts\python.exe -B -m ...`.
Before a run with Vector RAG, start Qdrant (`<qdrant folder>\start_qdrant.ps1`) and keep its window open.

## 1. Input: one case per line (JSONL)

```json
{"case_id": "mq_001", "set": "medqa|rare|medsafety|clarifier|other", "input_text": "...",
 "gold_diagnosis": "...", "gold_synonyms": ["..."], "gold_differential": ["..."],
 "gold_med_flags": [{"type": "contraindication|drug_clash", "drug": "...", "target": "...",
                     "severity": "major|moderate|minor (drug_clash) or \"\" (contraindication)"}],
 "gold_entities": {"symptoms": [], "absent": [], "medications": [], "diseases": []},
 "notes": "..."}
```

- `case_id` must be unique (it is also the file name of the saved result). `set` must be one of the five values.
- Unused fields may be empty (`""`, `[]`). Example with 3 cases: `cdss_pipeline\cases\example_cases.jsonl`.
- `medsafety` cases with an EMPTY `gold_med_flags` list are **clean** cases (no known problem).
- `gold_med_flags.target` is a disease for `contraindication` and another drug for `drug_clash`.
- `gold_med_flags.severity`: fill it only for `drug_clash` (DDInter levels). Leave it `""` for `contraindication`:
  KG contraindication warnings always have severity `unknown` (DrugBank gives none), so it is never scored.

## 2. Run

```
.venv\Scripts\python.exe -B -m cdss_pipeline.run_batch --input cases.jsonl --out results\full_run_1
.venv\Scripts\python.exe -B -m cdss_pipeline.run_batch --input cases.jsonl --out results\no_rag_1 --disable rag
```

Options: `--disable rag,kg,evidence,optimizer` (ablation; rag and kg cannot both be off), `--limit N`.

The results folder gets `run_manifest.json` (date, switches, every model and where it came from, commits of all
agents, Groq route, Qdrant counts or "rag disabled"), `cases\<case_id>.json` (the same data as
`run_pipeline --json`, plus the input case and per-case seconds / Groq calls / OpenRouter calls / errors),
`summary.csv` and, if any case crashed, `failed_cases.txt`.

**Resume:** run the same command again. Saved cases are skipped (no quota spent again); crashed cases are retried.
A results folder made with different switches, models or commits is never mixed: the run stops and asks for a
new `--out`.

## 3. Score

```
.venv\Scripts\python.exe -B -m cdss_pipeline.score --run results\full_run_1
.venv\Scripts\python.exe -B -m cdss_pipeline.score --run results\full_run_1 --compare results\no_rag_1
```

Writes `scores.json`, `scores.txt` and `unmatched_for_review.csv` into the run folder.

### Disease name matching
1. **exact**: both names normalised (lowercase, `&` -> and, punctuation removed, simple singular form) and equal;
2. **part**: a `/` part or the bracket content of a name equals the other, e.g. `Heart attack (Myocardial infarction)`;
3. **synonym**: the predicted name (or a part) equals one of `gold_synonyms`.

The rule used is recorded. Everything else goes to `unmatched_for_review.csv` (columns `reviewer_1`,
`reviewer_2` for two people to mark manually).

### Who is counted
- `analysis_status` not `analysed` (Clarifier failed / no clinical items): **never counted as wrong**, reported
  separately, left out of Tests 1-3 (Test 4 keeps `no_clinical_items`, drops `clarifier_error`).
- `evidence_fallback_used` (Groq fallback summaries): affects only the evidence summaries, C1 and the optimizer
  loop, so these cases are left out of **Test 1, the loops table and the confidence analysis only**; they stay in
  Tests 2, 3 and 4. The number left out is reported.
- A test needing a disabled agent says so (e.g. RAG top-1 with `--disable rag`).

### Measures (each proportion with a 95% Wilson interval)
- **Test 1 (Vector RAG)**: RAG top-1 accuracy; correct anywhere in the structured lists (all RAG candidates,
  KG top 10, supported_by_both); secondary: gold named anywhere in the free-text Fusion report.
- **Test 2 (KG)**: hit@1, hit@3, hit@5, hit@10 and MRR on the KG top-10 ranking (the pipeline itself uses the top 5;
  the top 10 is saved read-only for scoring).
- **Test 3 (medication safety, set `medsafety`)**, using KG warnings of type contraindication / drug_clash
  (a drug_clash matches in either drug order):
  - flag-level catch rate = caught gold flags / all gold flags;
  - case-level false alarm rate = clean medsafety cases with at least one warning / all clean medsafety cases;
  - warning-level precision = warnings matching a gold flag of their case / all warnings (all medsafety cases);
  - severity correct (drug_clash only) = caught drug_clash flags where a matching warning has the gold severity /
    caught drug_clash flags (contraindication flags are left out: their warnings never have a severity);
  - warnings by severity: major / moderate / minor / unknown, each / all warnings.
- **Test 4 (Clarifier)**: precision and recall (micro, over all cases) for symptoms, medications and diseases;
  absent-correct = gold `absent` items found as absent / all gold absent items; also absent items leaked into the
  present lists.
- RAG top-1 accuracy by optimizer loops (0 / 1 / 2); mean C1, C2, fused and graph_confidence for top-1 correct vs
  wrong; average seconds, Groq calls and OpenRouter calls per case.
- `--compare`: McNemar exact test (two-sided, binomial on the discordant pairs) on the SAME case_ids, for RAG top-1,
  correct anywhere and KG hit@k.
