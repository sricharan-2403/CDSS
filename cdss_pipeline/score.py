r"""Score a batch run (made by run_batch). Definitions: see TESTING.md.

From the integration folder:
    .venv\Scripts\python.exe -B -m cdss_pipeline.score --run results\<run_name> [--compare results\<other_run>]

Writes into --run: scores.json, scores.txt (readable) and unmatched_for_review.csv (names the automatic matching
did not accept, for manual checking by two people). Cases whose analysis_status is not "analysed" are never
counted as wrong; evidence-fallback cases are left out of Test 1, the loops table and the confidence analysis
only (the KG candidates, KG safety list and Clarifier do not depend on the evidence summaries).
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

from cdss_pipeline.run_batch import saved_result
from cdss_pipeline.wrappers import match_parts

Z = 1.96  # 95%
SEVERITIES = ("major", "moderate", "minor", "unknown")
FLAG_TYPES = ("contraindication", "drug_clash")
ENTITY_KINDS = ("symptoms", "medications", "diseases")
HIT_KS = (1, 3, 5, 10)


# ---------------------------------------------------------------- statistics

def wilson(k, n):
    """Proportion k/n with its 95% Wilson score interval (None values when n = 0)."""
    if not n:
        return {"k": k, "n": 0, "p": None, "low": None, "high": None}
    p = k / n
    centre = p + Z * Z / (2 * n)
    spread = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    denom = 1 + Z * Z / n
    return {"k": k, "n": n, "p": round(p, 4), "low": round((centre - spread) / denom, 4),
            "high": round((centre + spread) / denom, 4)}


def mcnemar_exact(b, c):
    """Exact (binomial) two-sided McNemar p-value from the discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 6)


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return {"mean": round(sum(values) / len(values), 4) if values else None, "n": len(values)}


# ---------------------------------------------------------------- name matching

def _singular(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def norm(text):
    """Lowercase, '&' -> 'and', punctuation -> space, simple singular form of every word."""
    text = str(text or "").lower().replace("&", " and ")
    return " ".join(_singular(w) for w in re.sub(r"[^a-z0-9]+", " ", text).split())


def _forms(name):
    """The whole name plus its '/' parts and bracket content, all normalised."""
    forms = {norm(name)} | {norm(p) for p in match_parts(name)}
    return {f for f in forms if f}


def match_rule(predicted, gold, synonyms=()):
    """-> "exact" | "part" | "synonym" | None. part = a '/' part or bracket content equals the gold name."""
    if not predicted or not gold:
        return None
    if norm(predicted) == norm(gold):
        return "exact"
    if _forms(predicted) & _forms(gold):
        return "part"
    if _forms(predicted) & {f for s in synonyms or [] for f in _forms(s)}:
        return "synonym"
    return None


def first_match(names, gold, synonyms=()):
    """-> (1-based rank, rule) of the first name matching gold, or (None, None)."""
    for rank, name in enumerate(names or [], start=1):
        rule = match_rule(name, gold, synonyms)
        if rule:
            return rank, rule
    return None, None


def mentioned_in_text(text, gold, synonyms=()):
    """Secondary check: gold (or a synonym) appears as whole words in free text."""
    body = f" {norm(text)} "
    return any(f" {form} " in body for name in [gold, *(synonyms or [])] for form in _forms(name))


# ---------------------------------------------------------------- loading

def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    items = [item for item in (saved_result(p) for p in sorted((run_dir / "cases").glob("*.json"))) if item]
    if not items:
        raise SystemExit(f"\nSTOPPED: no saved cases in {run_dir / 'cases'}")
    return manifest, items


def _case(item):
    return item.get("case") or {}


def _fo(item):
    return item.get("final_output") or {}


def _analysed(item):
    return _fo(item).get("analysis_status") == "analysed"


def _fallback(item):
    return bool(_fo(item).get("evidence_fallback_used"))


def _disabled(item, switch):
    return switch in (_fo(item).get("disabled_agents") or [])


# ---------------------------------------------------------------- tests

def test1(items, review):
    """RAG top-1, correct anywhere in the structured lists, and (secondary) in the Fusion report text."""
    top1, anywhere, fusion_text, per_case = [], [], [], {}
    excluded = {"not_analysed": 0, "evidence_fallback": 0, "no_gold_diagnosis": 0, "rag_disabled_top1": 0}
    for item in items:
        case, fo = _case(item), _fo(item)
        gold, synonyms = case.get("gold_diagnosis"), case.get("gold_synonyms") or []
        if not gold:
            excluded["no_gold_diagnosis"] += 1
            continue
        if not _analysed(item):
            excluded["not_analysed"] += 1
            continue
        if _fallback(item):
            excluded["evidence_fallback"] += 1
            continue
        record = {"case_id": case.get("case_id"), "loops": item.get("loop_count"), "fo": fo}
        if _disabled(item, "rag"):
            excluded["rag_disabled_top1"] += 1
        else:
            first = (fo.get("rag_candidates") or [None])[0]
            rule = match_rule(first, gold, synonyms)
            record["top1"], record["top1_rule"] = bool(rule), rule
            top1.append(bool(rule))
            if not rule:
                review.append(("test1_rag_top1", case, 1, first))
        lists = (fo.get("rag_candidates") or []) + (fo.get("kg_ranking_top10") or []) + (
            fo.get("supported_by_both") or [])
        rank, rule = first_match(lists, gold, synonyms)
        record["anywhere"], record["anywhere_rule"] = rank is not None, rule
        anywhere.append(rank is not None)
        fusion_text.append(mentioned_in_text(fo.get("fusion_report"), gold, synonyms))
        per_case[record["case_id"]] = record
    return {"rag_top1_accuracy": wilson(sum(top1), len(top1)),
            "correct_anywhere_structured": wilson(sum(anywhere), len(anywhere)),
            "secondary_mentioned_in_fusion_report": wilson(sum(fusion_text), len(fusion_text)),
            "match_rules": _rule_counts(per_case.values(), ("top1_rule", "anywhere_rule")),
            "excluded": excluded}, per_case


def _rule_counts(records, keys):
    counts = {}
    for record in records:
        for key in keys:
            if record.get(key):
                counts.setdefault(key, {}).setdefault(record[key], 0)
                counts[key][record[key]] += 1
    return counts


def test2(items, review):
    """KG hit@1/3/5/10 and MRR on kg_ranking_top10 (evidence-fallback cases kept)."""
    ranks, per_case, rules = [], {}, {}
    excluded = {"not_analysed": 0, "no_gold_diagnosis": 0, "kg_disabled": 0}
    for item in items:
        case, fo = _case(item), _fo(item)
        gold, synonyms = case.get("gold_diagnosis"), case.get("gold_synonyms") or []
        if not gold:
            excluded["no_gold_diagnosis"] += 1
            continue
        if not _analysed(item):
            excluded["not_analysed"] += 1
            continue
        if _disabled(item, "kg"):
            excluded["kg_disabled"] += 1
            continue
        ranking = fo.get("kg_ranking_top10") or []
        rank, rule = first_match(ranking, gold, synonyms)
        ranks.append(rank)
        per_case[case.get("case_id")] = rank
        if rule:
            rules[rule] = rules.get(rule, 0) + 1
        else:
            for position, name in enumerate(ranking, start=1):
                review.append(("test2_kg_ranking", case, position, name))
    out = {f"hit@{k}": wilson(sum(1 for r in ranks if r and r <= k), len(ranks)) for k in HIT_KS}
    out["mrr"] = round(sum(1 / r for r in ranks if r) / len(ranks), 4) if ranks else None
    out["n"], out["match_rules"], out["excluded"] = len(ranks), rules, excluded
    return out, per_case


def _flag_matches(warning, flag):
    if warning.get("type") != flag.get("type"):
        return False
    drug, target = norm(warning.get("drug")), norm(warning.get("target"))
    if flag.get("type") == "drug_clash":  # either order
        return {drug, target} == {norm(flag.get("drug")), norm(flag.get("target"))}
    return drug == norm(flag.get("drug")) and match_rule(warning.get("target"), flag.get("target")) is not None


def _severity(value):
    value = str(value or "").strip().lower()
    return value if value in SEVERITIES else "unknown"


def test3(items, review):
    """Medication safety on set "medsafety" (evidence-fallback cases kept). Definitions in TESTING.md."""
    gold_total = caught = severity_ok = clash_caught = 0
    clean_cases = clean_with_warning = 0
    warnings_total = warnings_matching = 0
    by_severity = {s: 0 for s in SEVERITIES}
    excluded = {"not_analysed": 0, "kg_disabled": 0}
    cases = 0
    for item in items:
        case, fo = _case(item), _fo(item)
        if case.get("set") != "medsafety":
            continue
        if not _analysed(item):
            excluded["not_analysed"] += 1
            continue
        if _disabled(item, "kg"):
            excluded["kg_disabled"] += 1
            continue
        cases += 1
        flags = case.get("gold_med_flags") or []
        warnings = [w for w in (fo.get("medication_safety") or {}).get("warnings") or []
                    if w.get("type") in FLAG_TYPES]
        for flag in flags:
            gold_total += 1
            hits = [w for w in warnings if _flag_matches(w, flag)]
            if hits:
                caught += 1
                # severity only for drug_clash: KG contraindications never carry a severity (DrugBank has none)
                if flag.get("type") != "drug_clash":
                    continue
                clash_caught += 1
                severity_ok += any(_severity(w.get("severity")) == _severity(flag.get("severity")) for w in hits)
            else:
                review.append(("test3_missed_gold_flag", case, None,
                               f"{flag.get('type')}: {flag.get('drug')} -> {flag.get('target')}"))
        for w in warnings:
            warnings_total += 1
            by_severity[_severity(w.get("severity"))] += 1
            if any(_flag_matches(w, flag) for flag in flags):
                warnings_matching += 1
            else:
                review.append(("test3_unmatched_warning", case, None,
                               f"{w.get('type')}: {w.get('drug')} -> {w.get('target')} ({w.get('severity')})"))
        if not flags:
            clean_cases += 1
            clean_with_warning += bool(warnings)
    return {"cases": cases,
            "flag_catch_rate": wilson(caught, gold_total),
            "case_false_alarm_rate": wilson(clean_with_warning, clean_cases),
            "warning_precision": wilson(warnings_matching, warnings_total),
            "severity_correct_among_caught_drug_clash": wilson(severity_ok, clash_caught),
            "warnings_by_severity": {s: wilson(by_severity[s], warnings_total) for s in SEVERITIES},
            "excluded": excluded}


def test4(items):
    """Clarifier precision / recall per entity kind, absent-correct and absent-leaked (fallback cases kept)."""
    counts = {k: {"tp": 0, "pred": 0, "gold": 0} for k in ENTITY_KINDS}
    absent = {"gold": 0, "found_absent": 0, "leaked_present": 0}
    cases, excluded = 0, {"clarifier_failed": 0}
    for item in items:
        case = _case(item)
        gold = case.get("gold_entities") or {}
        if case.get("set") != "clarifier" and not any(gold.get(k) for k in (*ENTITY_KINDS, "absent")):
            continue
        output = item.get("clarifier_output") or {}
        if _fo(item).get("analysis_status") == "clarifier_error" or not output:
            excluded["clarifier_failed"] += 1
            continue
        cases += 1
        for kind in ENTITY_KINDS:
            if case.get("set") != "clarifier" and not gold.get(kind):
                continue
            g, p = {norm(x) for x in gold.get(kind) or []}, {norm(x) for x in output.get(kind) or []}
            counts[kind]["tp"] += len(g & p)
            counts[kind]["pred"] += len(p)
            counts[kind]["gold"] += len(g)
        gold_absent = {norm(x) for x in gold.get("absent") or []}
        entities = (item.get("clarifier_details") or {}).get("entities") or []
        pred_absent = {norm(e.get("text")) for e in entities if e.get("status") == "absent"}
        present = {norm(x) for kind in ENTITY_KINDS for x in output.get(kind) or []}
        absent["gold"] += len(gold_absent)
        absent["found_absent"] += len(gold_absent & pred_absent)
        absent["leaked_present"] += len(gold_absent & present)
    out = {"cases": cases, "excluded": excluded}
    for kind, c in counts.items():
        out[kind] = {"precision": wilson(c["tp"], c["pred"]), "recall": wilson(c["tp"], c["gold"])}
    out["absent_correct"] = wilson(absent["found_absent"], absent["gold"])
    out["absent_leaked_as_present"] = wilson(absent["leaked_present"], absent["gold"])
    return out


def by_loops(per_case_test1):
    """RAG top-1 accuracy by number of optimizer loops (Test 1 cases only: fallback already excluded)."""
    groups = {}
    for record in per_case_test1.values():
        if "top1" in record:
            groups.setdefault(record.get("loops") or 0, []).append(record["top1"])
    return {str(loops): wilson(sum(v), len(v)) for loops, v in sorted(groups.items())}


def confidence_analysis(per_case_test1):
    """Mean C1 / C2 / fused / graph_confidence for top-1 correct vs wrong (Test 1 cases only)."""
    out = {}
    for label, wanted in (("top1_correct", True), ("top1_wrong", False)):
        records = [r for r in per_case_test1.values() if r.get("top1") is wanted]
        conf = [(r["fo"].get("confidence") or {}) for r in records]
        out[label] = {"C1": _mean(c.get("C1") for c in conf), "C2": _mean(c.get("C2") for c in conf),
                      "fused": _mean(c.get("fused") for c in conf),
                      "graph_confidence": _mean(r["fo"].get("graph_confidence") for r in records)}
    return out


def cost(items):
    batch = [item.get("batch") or {} for item in items]
    return {"seconds_per_case": _mean(b.get("seconds") for b in batch),
            "groq_calls_per_case": _mean(b.get("groq_calls") for b in batch),
            "openrouter_calls_per_case": _mean(b.get("openrouter_calls") for b in batch)}


def exclusions(items):
    statuses = {}
    for item in items:
        status = _fo(item).get("analysis_status")
        if status != "analysed":
            statuses[status] = statuses.get(status, 0) + 1
    return {"cases_saved": len(items), "not_analysed_by_status": statuses,
            "evidence_fallback_cases": sum(1 for item in items if _analysed(item) and _fallback(item)),
            "evidence_fallback_rule": "excluded from Test 1, the loops table and the confidence analysis; "
                                      "kept in Test 2, Test 3 and Test 4"}


def score_items(items):
    review = []
    t1, per_case_t1 = test1(items, review)
    t2, per_case_t2 = test2(items, review)
    scores = {"cases": exclusions(items), "test1_rag": t1, "test2_kg": t2, "test3_medication_safety":
              test3(items, review), "test4_clarifier": test4(items), "accuracy_by_loops": by_loops(per_case_t1),
              "confidence_analysis": confidence_analysis(per_case_t1), "cost": cost(items)}
    binary = {cid: {"test1_top1": r.get("top1"), "test1_anywhere": r.get("anywhere")} for cid, r in per_case_t1.items()}
    for cid, rank in per_case_t2.items():
        binary.setdefault(cid, {}).update({f"test2_hit@{k}": bool(rank and rank <= k) for k in HIT_KS})
    return scores, review, binary


def compare(binary_a, binary_b):
    """McNemar exact test on the SAME case_ids, per binary outcome (both runs must have a value)."""
    out = {}
    metrics = sorted({m for d in (binary_a, binary_b) for v in d.values() for m in v})
    for metric in metrics:
        pairs = [(a[metric], binary_b[cid][metric]) for cid, a in binary_a.items()
                 if a.get(metric) is not None and binary_b.get(cid, {}).get(metric) is not None]
        b = sum(1 for x, y in pairs if x and not y)
        c = sum(1 for x, y in pairs if y and not x)
        out[metric] = {"paired_cases": len(pairs), "run_correct": wilson(sum(x for x, _ in pairs), len(pairs)),
                       "other_correct": wilson(sum(y for _, y in pairs), len(pairs)),
                       "only_run_correct": b, "only_other_correct": c, "mcnemar_exact_p": mcnemar_exact(b, c)}
    return out


# ---------------------------------------------------------------- output

def _pct(w):
    if w["p"] is None:
        return "n/a (0 cases)"
    return f"{w['k']}/{w['n']} = {100 * w['p']:.1f}% (95% CI {100 * w['low']:.1f}-{100 * w['high']:.1f}%)"


def scores_text(scores, manifest, run_dir):
    s, lines = scores, []
    add = lines.append
    add(f"SCORES for {run_dir}")
    add(f"disabled agents: {', '.join(manifest.get('disabled_agents') or []) or 'none (full system)'}; "
        f"groq route: {manifest.get('groq_route')}; commits: {manifest.get('commits')}")
    c = s["cases"]
    add(f"\nCases saved: {c['cases_saved']}; not analysed (never counted as wrong): {c['not_analysed_by_status'] or 0}; "
        f"evidence fallback: {c['evidence_fallback_cases']} ({c['evidence_fallback_rule']})")
    t1 = s["test1_rag"]
    add("\nTest 1 - Vector RAG")
    add(f"  RAG top-1 accuracy:               {_pct(t1['rag_top1_accuracy'])}")
    add(f"  correct anywhere (structured):    {_pct(t1['correct_anywhere_structured'])}")
    add(f"  secondary: named in Fusion report {_pct(t1['secondary_mentioned_in_fusion_report'])}")
    add(f"  match rules: {t1['match_rules']}; excluded: {t1['excluded']}")
    t2 = s["test2_kg"]
    add("\nTest 2 - Knowledge Graph ranking")
    for k in HIT_KS:
        add(f"  hit@{k:<2}  {_pct(t2[f'hit@{k}'])}")
    add(f"  MRR {t2['mrr']} (n {t2['n']}); match rules: {t2['match_rules']}; excluded: {t2['excluded']}")
    t3 = s["test3_medication_safety"]
    add(f"\nTest 3 - Medication safety ({t3['cases']} medsafety cases)")
    add(f"  flag catch rate:                  {_pct(t3['flag_catch_rate'])}")
    add(f"  case false alarm rate (clean):    {_pct(t3['case_false_alarm_rate'])}")
    add(f"  warning precision:                {_pct(t3['warning_precision'])}")
    add(f"  severity correct (drug_clash):    {_pct(t3['severity_correct_among_caught_drug_clash'])}")
    for sev, w in t3["warnings_by_severity"].items():
        add(f"  warnings {sev:<9}               {_pct(w)}")
    add(f"  excluded: {t3['excluded']}")
    t4 = s["test4_clarifier"]
    add(f"\nTest 4 - Clarifier ({t4['cases']} cases; excluded {t4['excluded']})")
    for kind in ENTITY_KINDS:
        add(f"  {kind:<12} precision {_pct(t4[kind]['precision'])}")
        add(f"  {'':<12} recall    {_pct(t4[kind]['recall'])}")
    add(f"  absent correct:            {_pct(t4['absent_correct'])}")
    add(f"  absent leaked as present:  {_pct(t4['absent_leaked_as_present'])}")
    add("\nRAG top-1 accuracy by optimizer loops")
    for loops, w in s["accuracy_by_loops"].items():
        add(f"  {loops} loop(s): {_pct(w)}")
    if not s["accuracy_by_loops"]:
        add("  n/a (no Test 1 cases)")
    add(f"\nConfidence (Test 1 cases): {json.dumps(s['confidence_analysis'])}")
    co = s["cost"]
    add(f"\nCost per case: {co['seconds_per_case']['mean']} s, Groq {co['groq_calls_per_case']['mean']}, "
        f"OpenRouter {co['openrouter_calls_per_case']['mean']} (n {co['seconds_per_case']['n']})")
    if "compare" in s:
        add(f"\nCompared with {s['compare']['other_run']} (McNemar exact, same case_ids)")
        for metric, m in s["compare"]["metrics"].items():
            add(f"  {metric:<16} this {_pct(m['run_correct'])} | other {_pct(m['other_correct'])} | "
                f"only this {m['only_run_correct']}, only other {m['only_other_correct']}, p = {m['mcnemar_exact_p']}")
    return "\n".join(lines) + "\n"


def write_review(path, run_dir, review):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "case_id", "test", "rank", "predicted", "gold_diagnosis", "gold_synonyms",
                         "reviewer_1", "reviewer_2"])
        for test, case, rank, predicted in review:
            writer.writerow([str(run_dir), case.get("case_id"), test, rank or "", predicted or "",
                             case.get("gold_diagnosis") or "", "; ".join(case.get("gold_synonyms") or []), "", ""])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help=r"results\<run_name> made by run_batch")
    parser.add_argument("--compare", default=None, help=r"another results\<run_name> (same case_ids) for McNemar")
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    manifest, items = load_run(run_dir)
    scores, review, binary = score_items(items)
    if args.compare:
        _, other_items = load_run(args.compare)
        _, _, other_binary = score_items(other_items)
        scores["compare"] = {"other_run": str(args.compare), "metrics": compare(binary, other_binary)}
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    text = scores_text(scores, manifest, run_dir)
    (run_dir / "scores.txt").write_text(text, encoding="utf-8")
    write_review(run_dir / "unmatched_for_review.csv", run_dir, review)
    print(text)
    print(f"written: {run_dir / 'scores.json'}, {run_dir / 'scores.txt'}, {run_dir / 'unmatched_for_review.csv'} "
          f"({len(review)} rows to review)")
    return scores


if __name__ == "__main__":
    main()
