r"""Check real-mode agents: load time, time per call, peak RAM, and compare with replay.

From the integration folder:
    .venv\Scripts\python.exe -B -m cdss_pipeline.check_real --agents knowledge_graph
    .venv\Scripts\python.exe -B -m cdss_pipeline.check_real --agents clarifier,knowledge_graph ^
        --cases cdss_pipeline\cases\cases_example.json --out step13_combined_check.txt

The selected agents are switched to real mode for THIS run only; all others stay replay.
The Clarifier (Varshit's new version) loads with the other agents and makes 1 Groq call per text;
in this check it is called once for the first case (reused for the KG check) plus once per case run.
A RAM guard stops the check if free system RAM drops below 0.5 GB.
"""

import argparse
import json
import os
import sys
import threading
import time

import psutil

from cdss_pipeline import config

GB = 1024 ** 3
STOP_BELOW_GB = 0.5
WARN_BELOW_GB = 1.0
START_MIN_GB = 1.5

CASE1_TEXT = ("56 year old man with chest pain, sweating and shortness of breath. "
              "History of myocardial infarction. Currently taking sildenafil.")
CLARIFIER_KEYS = ["clinical_text", "diseases", "symptoms", "medications", "tests"]


class RamMonitor(threading.Thread):
    """Samples this process's RAM and free system RAM every 0.2 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.proc = psutil.Process()
        self.peak_rss = 0
        self.min_free = psutil.virtual_memory().available
        self.running = True

    def run(self):
        while self.running:
            rss = self.proc.memory_info().rss
            free = psutil.virtual_memory().available
            self.peak_rss = max(self.peak_rss, rss)
            self.min_free = min(self.min_free, free)
            if free < STOP_BELOW_GB * GB:
                print(f"\n!!! STOPPED: free RAM {free / GB:.2f} GB is below {STOP_BELOW_GB} GB", flush=True)
                for child in self.proc.children(recursive=True):
                    child.kill()
                os._exit(3)
            time.sleep(0.2)

    def snapshot(self):
        return {"process_rss_gb": round(self.proc.memory_info().rss / GB, 2),
                "peak_process_rss_gb": round(self.peak_rss / GB, 2),
                "free_now_gb": round(psutil.virtual_memory().available / GB, 2),
                "lowest_free_gb": round(self.min_free / GB, 2)}


def _summary(kg):
    pred = (kg.get("disease_prediction") or {}).get("candidate_diseases") or []
    safety = kg.get("medication_safety") or {}
    return {
        "status": kg.get("status"),
        "known_diseases": kg.get("known_diseases"),
        "predicted_diseases": [c["disease"] for c in pred],
        "warnings_summary": safety.get("summary"),
        "medications_matched": safety.get("medications_matched"),
    }


def compare_kg(real, replay, lines):
    from cdss_pipeline.wrappers import make_kg_safety_messages
    rs, ps = _summary(real), _summary(replay)
    lines.append("KG real vs kg_case1_output.json")
    for key in rs:
        same = rs[key] == ps[key]
        if not same and key in ("known_diseases", "medications_matched"):
            same_ci = str(rs[key]).lower() == str(ps[key]).lower()
            tag = "SAME (ignoring capitals)" if same_ci else "DIFFERENT"
        else:
            tag = "SAME" if same else "DIFFERENT"
        lines.append(f"  {key}: {tag}")
        if tag != "SAME":
            lines.append(f"    real:   {rs[key]}")
            lines.append(f"    replay: {ps[key]}")

    real_items, replay_items = make_kg_safety_messages(real), make_kg_safety_messages(replay)
    lines.append(f"  kg_safety_messages: real {len(real_items)} items, replay {len(replay_items)} items")
    if real_items == replay_items:
        lines.append("  kg_safety_messages: SAME (every item identical)")
    else:
        lower = lambda items: [{k: str(v).lower() for k, v in i.items()} for i in items]
        same_ci = lower(real_items) == lower(replay_items)
        lines.append("  kg_safety_messages: " + ("SAME (ignoring capitals)" if same_ci else "DIFFERENT"))
        for a, b in zip(real_items, replay_items):
            if a != b:
                lines.append(f"    real:   {a}")
                lines.append(f"    replay: {b}")
        for extra in real_items[len(replay_items):]:
            lines.append(f"    only in real:   {extra}")
        for extra in replay_items[len(real_items):]:
            lines.append(f"    only in replay: {extra}")
    contra = [i for i in real_items if i["type"] == "contraindication"]
    lines.append(f"  contraindications (real): {[(i['drug'], i['target']) for i in contra]}")


def check_clarifier(out, sample, lines):
    lines.append("Clarifier real output:")
    for key in CLARIFIER_KEYS:
        lines.append(f"  {key}: {out.get(key)!r}")
    keys_ok = sorted(out) == sorted(CLARIFIER_KEYS)
    lines.append(f"  keys exactly the 5 expected: {'YES' if keys_ok else 'NO -> ' + str(sorted(out))}")
    bad = [(k, v) for k in CLARIFIER_KEYS[1:] for v in out.get(k) or []
           if not isinstance(v, str) or v != v.lower()]
    lines.append(f"  all items lowercase strings: {'YES' if not bad else 'NO -> ' + str(bad)}")
    lines.append("Compared with case 1 sample (samples/clarifier_case1.json):")
    for key in CLARIFIER_KEYS[1:]:
        real, want = set(out.get(key) or []), set(sample.get(key) or [])
        if real == want:
            lines.append(f"  {key}: SAME {sorted(real)}")
        else:
            lines.append(f"  {key}: DIFFERENT  missing {sorted(want - real)}  extra {sorted(real - want)}")


def _sum_counts(records, key, only=None, exclude=None):
    return sum(n for r in records for src, n in r.get(key, {}).items()
               if (only is None or src == only) and src != exclude)


def _service_counts(wrappers):
    ev = wrappers.EVIDENCE_STATS["calls"]
    return {"OpenRouter (RAG)": wrappers.RAG_STATS["openrouter_calls"],
            "OpenRouter (RAG) retries": wrappers.RAG_STATS["retries"],
            "OpenRouter (RAG) failures": wrappers.RAG_STATS["failures"],
            "Groq": _sum_counts(ev, "requests", only="groq"),
            "Groq (Clarifier)": wrappers.CLARIFIER_STATS["groq_requests"],
            "OpenRouter (Fusion)": wrappers.FUSION_STATS["openrouter_calls"],
            "OpenRouter (Fusion) failures": wrappers.FUSION_STATS["openrouter_failures"],
            "OpenRouter (Fusion) retries": wrappers.FUSION_STATS["openrouter_retries"]}


def report_fusion(state, passes, rag_calls, lines):
    """Per pass: C1, C2, fused, rag_top_k, graph-score default, fallback; loop reason; Sildenafil check."""
    from cdss_pipeline import config as cfg
    lines.append("Fusion (Medico):")
    for i, p in enumerate(passes):
        top_k = rag_calls[i].get("top_k") if i < len(rag_calls) else "?"
        lines.append(f"  pass {i + 1} (rag_top_k {top_k}): C1 {p.get('C1')}, C2 {p.get('C2')}, "
                     f"fused {p.get('fused')}, C2 graph part {p.get('graph_score')}"
                     f"{' (0.90 DEFAULT used)' if p.get('graph_score_default_0_90') else ' (found in KG text)'}, "
                     f"fallback_used {p.get('fallback_used')}, {p.get('seconds')} s, "
                     f"OpenRouter calls {p.get('calls')}, retries {p.get('retries', 0)}"
                     + (f", FAILURES {p['failures']}" if p.get("failures") else ""))
    loops, conf = state.get("loop_count", 0), state.get("rag_confidence")
    rag_ok = bool(state.get("rag_output")) and (state.get("rag_output") or {}).get("status") != "error"
    if not rag_ok:
        why = "stopped: Vector RAG unavailable (no loop)"
    elif conf is None:
        why = "stopped: no confidence score"
    elif conf >= cfg.THRESHOLD:
        why = f"stopped: C1 {conf} >= {cfg.THRESHOLD}"
    else:
        why = f"stopped: C1 {conf} < {cfg.THRESHOLD} but max {cfg.MAX_LOOPS} loops reached"
    lines.append(f"  loops: {loops} ({why}); each loop happened because C1 < {cfg.THRESHOLD}")
    report = (state.get("fusion_output") or {}).get("fusion_report") or ""
    verbatim_at = report.find("KNOWLEDGE GRAPH SAFETY MESSAGES (verbatim)")
    ai_part = report if verbatim_at < 0 else report[:verbatim_at]
    has = lambda text: "sildenafil" in text.lower() and "contraindicated" in text.lower()
    lines.append(f"  Sildenafil contraindication in fusion_report: {'YES' if has(report) else 'NO'}"
                 f" (in AI text: {'yes' if has(ai_part) else 'no'}; "
                 f"verbatim safety block appended by her code: {'yes' if verbatim_at >= 0 else 'no'})")
    lines.append(f"  final fallback_used: {(state.get('fusion_output') or {}).get('fallback_used')}")


def report_evidence(state, records, pass_starts, lines):
    """Per loop: diseases scanned / new. Per disease: items, top item, counts."""
    lines.append("Evidence Scanner:")
    bounds = pass_starts + [len(records)]
    total = 0
    for i in range(len(pass_starts)):
        scanned = [r["disease"] for r in records[bounds[i]:bounds[i + 1]]]
        total += len(scanned)
        label = "first pass" if i == 0 else f"loop {i}"
        lines.append(f"  {label}: {len(scanned)} new disease(s) scanned {scanned}")
    lines.append(f"  total diseases scanned: {total}")
    evidence = state.get("evidence_output") or {}
    for r in records:
        result = evidence.get(r["disease"]) or {}
        items = result.get("evidence") or []
        lines.append(f"  - {r['disease']}: {r.get('seconds')} s, {len(items)} items"
                     + (", broadened" if r.get("broadened") else "")
                     + ("  <-- ZERO RESULTS" if not items else "")
                     + (f"  ERROR {r['error']}" if r.get("error") else ""))
        if items:
            top = items[0]
            lines.append(f"      top: [{top.get('source')}] {top.get('tier_label')} score {top.get('score')}: "
                         f"{top.get('title')}")
        if r.get("paused_seconds"):
            lines.append(f"      PAUSED {r['paused_seconds']} s before this disease ({r.get('pause_reason')})")
        lim = r.get("groq_limits") or {}
        if lim:
            lines.append(f"      Groq after scan: tokens left {lim.get('x-ratelimit-remaining-tokens')}, "
                         f"requests left today {lim.get('x-ratelimit-remaining-requests')}")
        lines.append(f"      requests {r['requests']}, 429s {r['rate_limited'] or 0}, "
                     f"other HTTP errors {r['http_errors'] or 0}, failed {r['failed_requests'] or 0}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="knowledge_graph",
                        help="comma-separated agents to run in real mode")
    parser.add_argument("--cases", help="JSON file [{id, patient_text}]; default = case 1 text only")
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--out", default=str(config.INTEGRATION_DIR / "step13_kg_check.txt"))
    args = parser.parse_args()

    real_agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    config.AGENT_MODES.update({a: ("real" if a in real_agents else "replay") for a in config.AGENTS})
    if args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases = json.load(f)
    else:
        cases = [{"id": "case1", "patient_text": CASE1_TEXT}]

    from cdss_pipeline import wrappers
    from cdss_pipeline.run_pipeline import run, startup

    fake = config.CLARIFIER_FAKE and "clarifier" in real_agents
    lines = []
    if fake:
        lines.append("!" * 64 + f"\n!!! {config.FAKE_WARNING}\n" + "!" * 64)
    lines += [f"STEP 13 real-mode check - {time.strftime('%Y-%m-%d %H:%M')}",
              f"modes: {config.AGENT_MODES}",
              f"python: {sys.version.split()[0]}",
              f"cases: {[c['id'] for c in cases]}"]
    mon = RamMonitor()
    before = mon.snapshot()
    lines.append(f"before loading: {before}")
    if real_agents and before["free_now_gb"] < START_MIN_GB:
        print(config.redact("\n".join(lines)))
        print(f"NOT STARTED: only {before['free_now_gb']} GB free, need at least {START_MIN_GB} GB. "
              "Close some programs and try again.")
        sys.exit(2)
    mon.start()

    # Same startup as run_pipeline: DB check, then every real agent loads once
    t = time.perf_counter()
    loaded = startup()
    lines.append(f"startup: {round(time.perf_counter() - t, 2)} s total")
    for agent, info in loaded.items():
        lines.append(f"  {agent}: {info}")
    lines.append(f"after startup (main process): {mon.snapshot()}")

    clarifier_output = None
    if "clarifier" in real_agents:
        try:
            clarifier_output, details = wrappers.run_clarifier_full(cases[0]["patient_text"])
            lines.append(f"patient_text ({cases[0]['id']}): {cases[0]['patient_text']}")
            check_clarifier(clarifier_output, wrappers._sample("clarifier_case1.json"), lines)
            lines.append(f"  clarifier_details: {details}")
        except Exception as exc:
            lines.append(f"Clarifier FAILED: {config.redact(exc)}")
            lines.append(f"  clarifier_details: {getattr(exc, 'details', None)}")

    if "knowledge_graph" in real_agents and "error" not in loaded.get("knowledge_graph", {}):
        if clarifier_output is None:  # reuse the Clarifier result above: no extra Groq call
            clarifier_output = wrappers.run_clarifier(cases[0]["patient_text"])
        times, real = [], None
        for _ in range(args.calls):
            t = time.perf_counter()
            real = wrappers.run_knowledge_graph(clarifier_output)
            times.append(round(time.perf_counter() - t, 2))
        lines.append(f"run_kg_agent time per call (s): {times}")
        lines.append(f"clarifier_output used by KG: {clarifier_output}")
        compare_kg(real, wrappers._load_json(str(config.KG_SAMPLE)), lines)

    # Mark where each evidence_scanner pass starts (graph is rebuilt inside run())
    from cdss_pipeline import graph as graph_mod
    pass_starts, original_node = [], graph_mod.evidence_scanner_node

    def marked_node(state):
        pass_starts.append(len(wrappers.EVIDENCE_STATS["calls"]))
        return original_node(state)
    graph_mod.evidence_scanner_node = marked_node

    for case in cases:
        t = time.perf_counter()
        calls_before = len(wrappers.RAG_STATS["calls"])
        fusion_before = len(wrappers.FUSION_STATS["passes"])
        counts_before = _service_counts(wrappers)
        case_start = time.perf_counter()
        ev_before, passes_before = len(wrappers.EVIDENCE_STATS["calls"]), len(pass_starts)
        state, _ = run(case["patient_text"])
        fo = state.get("final_output") or {}
        if "vector_rag" in real_agents:
            lines.append(f"Vector RAG {case['id']}:")
            for c in wrappers.RAG_STATS["calls"][calls_before:]:
                lines.append(f"  call top_k={c.get('top_k')}: total {c.get('total_seconds')} s = "
                             f"embedding {c.get('embed_seconds')} s + search {c.get('symptom_search_seconds')} s "
                             f"(symptom) + {c.get('literature_search_seconds')} s (literature) "
                             f"+ OpenRouter {c.get('openrouter_seconds')} s" + (
                                 f" ({c['retries']} retries: {', '.join(map(str, c['retry_codes']))}; "
                                 f"waited {c['retry_wait_seconds']} s)" if c.get("retries") else ""))
            rag = state.get("rag_output") or {}
            meta = rag.get("metadata") or {}
            lines.append(f"  status: {rag.get('status')}, total_evidence_count: {meta.get('total_evidence_count')} "
                         f"(symptom {meta.get('symptom_evidence_count')}, "
                         f"literature {meta.get('literature_evidence_count')})")
            for d in rag.get("diagnostic_candidates") or []:
                lines.append(f"  #{d.get('rank')} {d.get('condition')} (evidence {d.get('supporting_evidence')})")
                lines.append(f"      why: {d.get('justification')}")
            lines.append("  relevance scores: " + ", ".join(
                f"{e.get('relevance_score'):.3f} [{e.get('source')}]" for e in rag.get("evidence") or []))
            lines.append(f"  limitations: {rag.get('limitations')}")
        if "fusion" in real_agents:
            report_fusion(state, wrappers.FUSION_STATS["passes"][fusion_before:],
                          wrappers.RAG_STATS["calls"][calls_before:], lines)
            after = _service_counts(wrappers)
            lines.append(f"  case total time: {round(time.perf_counter() - case_start, 1)} s; calls this case: "
                         + ", ".join(f"{k} {after[k] - counts_before[k]}" for k in after))
        if "evidence_scanner" in real_agents:
            report_evidence(state, wrappers.EVIDENCE_STATS["calls"][ev_before:],
                            [p - ev_before for p in pass_starts[passes_before:]], lines)
        lines.append(f"pipeline {case['id']}: {round(time.perf_counter() - t, 2)} s, "
                     f"loops {state.get('loop_count')}, errors {state.get('errors')}")
        lines.append(f"  analysis_status: {fo.get('analysis_status')}")
        lines.append(f"  clarifier_output: {state.get('clarifier_output')}")
        lines.append(f"  KG predicted: {[c['disease'] for c in fo.get('rare_serious_conditions') or []]}")
        lines.append(f"  contraindications: "
                     f"{[(i['drug'], i['target']) for i in state.get('kg_safety_messages') or [] if i['type'] == 'contraindication']}")
        lines.append(f"  supported_by_both: {fo.get('supported_by_both')}, "
                     f"rag_matches_known_disease: {fo.get('rag_matches_known_disease')}")

    if "evidence_scanner" in real_agents:
        recs = wrappers.EVIDENCE_STATS["calls"]
        lines.append(f"Evidence totals: {len(recs)} diseases scanned, "
                     f"web requests {_sum_counts(recs, 'requests', exclude='groq')}, "
                     f"Groq calls {_sum_counts(recs, 'requests', only='groq')}, "
                     f"rate-limited (429) {_sum_counts(recs, 'rate_limited')}, "
                     f"other HTTP errors {_sum_counts(recs, 'http_errors')}, "
                     f"failed requests {_sum_counts(recs, 'failed_requests')}, "
                     f"total Groq pause {round(sum(r.get('paused_seconds') or 0 for r in recs), 1)} s")
        lines.append(f"Groq requests left today (end): {wrappers.groq_requests_left_text()}")
    if "fusion" in real_agents:
        fs = wrappers.FUSION_STATS
        lines.append(f"OpenRouter calls made: {fs['openrouter_calls']} (failures {fs['openrouter_failures']}, "
                     f"retries {fs['openrouter_retries']})")
    if "vector_rag" in real_agents:
        lines.append(f"OpenRouter (RAG Answerer) calls made: {wrappers.RAG_STATS['openrouter_calls']} "
                     f"(retries: {wrappers.RAG_STATS['retries']}, failures: {wrappers.RAG_STATS['failures']})")
    mon.running = False
    snap = mon.snapshot()
    lines.append(f"RAM at end (main process): {snap}")
    if snap["lowest_free_gb"] < WARN_BELOW_GB:
        lines.append(f"WARNING: free RAM went below {WARN_BELOW_GB} GB (lowest {snap['lowest_free_gb']} GB)")
    if fake:
        lines.append(f"!!! {config.FAKE_WARNING}")

    text = config.redact("\n".join(lines))  # safety net: keys and user ids hidden
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
