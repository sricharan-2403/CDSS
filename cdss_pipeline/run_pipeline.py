r"""Run one or more cases through the CDSS pipeline and print the final output.

From the integration folder:
    .venv\Scripts\python.exe -B -m cdss_pipeline.run_pipeline
    .venv\Scripts\python.exe -B -m cdss_pipeline.run_pipeline --text "..." --text "..."
    .venv\Scripts\python.exe -B -m cdss_pipeline.run_pipeline --cases cdss_pipeline\cases\cases_example.json
    default: SHORT view (answer + warnings); add --verbose for everything
    add --json for raw final_output JSON (with clarifier_details next to it)
    add --disable rag,kg,evidence,optimizer for an ablation run (or CDSS_DISABLE in .env; --disable wins)

Startup order: 1. read-only Qdrant check (Vector RAG; start Qdrant first with start_qdrant.ps1)
               2. every real agent loads once in this process (the Clarifier is a Groq client)
               3. each case runs through the graph (1 Clarifier Groq call per case)
"""

import argparse
import json

from cdss_pipeline import config, wrappers
from cdss_pipeline.graph import build_graph
from cdss_pipeline.state import initial_state

DEFAULT_TEXT = ("56 year old man with chest pain, sweating and shortness of breath. "
                "History of myocardial infarction. Currently taking sildenafil.")


def run(patient_text, disabled=()):
    """One case. disabled = ablation switches (rag, kg, evidence, optimizer); [] = full system."""
    disabled = config.parse_disabled(disabled)
    wrappers.reset_case_warnings()
    graph = build_graph(disabled)
    path, final = [], None
    for mode, chunk in graph.stream(initial_state(patient_text, config.TOP_K_START, disabled),
                                    stream_mode=["updates", "values"]):
        if mode == "updates":
            path.extend(chunk.keys())
        else:
            final = chunk
    return final, path


def json_result(case_id, state):
    """The --json result for one case (run_batch saves exactly this, plus its own fields)."""
    item = {"id": case_id, "final_output": state.get("final_output"),
            "clarifier_details": state.get("clarifier_details"),
            "clarifier_output": state.get("clarifier_output"),
            "loop_count": state.get("loop_count")}
    if config.CLARIFIER_FAKE and config.AGENT_MODES.get("clarifier") == "real":
        item["warning"] = config.FAKE_WARNING
    return item


def _fake_banner():
    if config.CLARIFIER_FAKE and config.AGENT_MODES.get("clarifier") == "real":
        print("\n" + "!" * 64 + f"\n!!! {config.FAKE_WARNING}\n" + "!" * 64)


def _section(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def print_clarifier_details(details):
    _section("Clarifier details")
    if not details or details.get("mode") == "replay":
        print(details.get("mode") or "none")
        return
    entities = details.get("entities") or []
    counts = {s: sum(1 for e in entities if e.get("status") == s) for s in ("present", "absent", "possible")}
    print(f"mode {details.get('mode')}, model {details.get('model')}, commit {details.get('commit')}, "
          f"attempts {details.get('attempts')}, {details.get('seconds')} s"
          + (f", paused {details['paused_seconds']} s" if details.get("paused_seconds") else ""))
    print(f"entities: {counts['present']} present, {counts['absent']} absent, {counts['possible']} possible; "
          f"removed by grounding check: {details.get('removed_by_grounding_check')}")
    print(f"demographics: {details.get('demographics')}")
    for e in entities:
        extra = ", ".join(f"{k} {e[k]}" for k in ("body_site", "duration", "severity", "value") if e.get(k))
        print(f"- {e.get('text')} [{e.get('type')}, {e.get('status')}] evidence: {e.get('evidence_text')!r}"
              + (f" ({extra})" if extra else ""))
    if details.get("error"):
        print(f"error: {details['error']}")
    if details.get("groq_error_message"):
        print(f"groq error: {details['groq_error_message']}")


def _value(v):
    return "not available" if v is None else v


def confidence_line(out):
    c = out.get("confidence") or {}
    return (f"C1 {_value(c.get('C1'))} | C2 {_value(c.get('C2'))} | fused {_value(c.get('fused'))} | "
            f"graph_confidence (KG) {_value(out.get('graph_confidence'))}")


def warning_lines(state):
    """One-line warnings for anything important, shown even in the short view."""
    out = state.get("final_output") or {}
    lines = []
    status = out.get("analysis_status")
    if status == "clarifier_error":
        error = (state.get("clarifier_details") or {}).get("error") or "see errors"
        lines.append(f"Clarifier failed - case not analysed ({error})")
    elif status == "no_clinical_items":
        lines.append("Clarifier found no present clinical items - case not analysed")
    if out.get("evidence_fallback_used"):
        eg = out.get("evidence_groq") or {}
        lines.append(f"Evidence Scanner used Groq fallbacks for {eg.get('fallbacks')} of {eg.get('groq_calls')} calls"
                     " - evidence quality reduced")
    if any(n.startswith("Fusion AI report unavailable") for n in out.get("notes") or []):
        lines.append("Fusion AI report unavailable, fallback used")
    if out.get("unmatched_symptoms"):
        lines.append(f"Symptoms not found in the KG: {', '.join(out['unmatched_symptoms'])}")
    if state.get("errors"):
        lines.append(f"{len(state['errors'])} agent error(s) - see Errors")
    return lines


def print_short(state, case_text):
    """Default view: the clinical answer plus anything that went wrong."""
    out = state.get("final_output") or {}
    print(f"\nCase: {case_text}")
    for line in warning_lines(state):
        print(f"! {line}")
    print(f"Analysis status: {out.get('analysis_status')}")
    if out.get("disabled_agents"):
        print(f"Disabled agents (ablation): {', '.join(out['disabled_agents'])}")
    d = out.get("most_likely_diagnosis")
    print(f"Most likely diagnosis (Vector RAG): {d['condition']} - {d['justification']}" if d else
          "Most likely diagnosis (Vector RAG): none")
    print("Rare / serious conditions (KG):" + ("" if out.get("rare_serious_conditions") else " none"))
    for c in out.get("rare_serious_conditions") or []:
        print(f"  - {c['disease']} (coverage {c['coverage']}, matched: {', '.join(c['matched_symptoms'] or [])})")
    warnings = (out.get("medication_safety") or {}).get("warnings") or []
    print("Medication safety:" + ("" if warnings else " no warnings"))
    for w in warnings:
        print(f"  - [{w['type']}, {w['severity']}] {w['drug']} -> {w['target']}: {w['message']}")
    print(f"Supported by both: {', '.join(out.get('supported_by_both') or []) or 'none'}")
    print(f"Confidence: {confidence_line(out)}")
    print("Notes:" + ("" if out.get("notes") else " none"))
    for n in out.get("notes") or []:
        print(f"  - {n}")
    print("Errors:" + ("" if state.get("errors") else " none"))
    for e in state.get("errors") or []:
        print(f"  - {e['node']}: {e['message']}")


def print_symptom_matching(kg_output):
    _section("KG symptom matching")
    matching = (kg_output or {}).get("symptom_matching")
    if not matching:
        print("not available (replay mode or KG unavailable)")
        return
    for m in matching:
        site = f", body_site {m['body_site']}" if m.get("body_site") else ""
        case_site = f", case site {m['case_body_site']}" if m.get("case_body_site") else ""
        print(f"- {m['symptom']}{site}{case_site} -> {m['graph_term'] or 'NOT FOUND'} ({m['rule']})")
    details = (kg_output or {}).get("graph_confidence_details")
    print(f"graph_confidence: {_value((kg_output or {}).get('graph_confidence'))} {details}")


def print_report(state, path):
    """--verbose: everything."""
    out = state.get("final_output") or {}
    for line in warning_lines(state):
        print(f"! {line}")
    _section("Agent modes / keys")
    print("modes:", config.AGENT_MODES)
    print("disabled agents (ablation):", out.get("disabled_agents") or "none (full system)")
    print("keys: ", config.key_status())
    _section("Clarifier output")
    print(state.get("clarifier_output"))
    print_clarifier_details(state.get("clarifier_details") or {})
    _section("Analysis status")
    print(out.get("analysis_status"))
    _section("Path")
    print(" -> ".join(path))
    print(f"loops: {state.get('loop_count')}   final rag_top_k: {state.get('rag_top_k')}")

    _section("Most likely diagnosis (Vector RAG)")
    d = out.get("most_likely_diagnosis")
    print(f"{d['condition']}\n  why: {d['justification']}" if d else "none")

    _section("Rare / serious conditions (KG)")
    for c in out.get("rare_serious_conditions") or []:
        print(f"- {c['disease']}  (coverage {c['coverage']}, matched: {', '.join(c['matched_symptoms'] or [])})")
    print_symptom_matching(state.get("kg_output"))

    _section("Medication safety")
    ms = out.get("medication_safety") or {}
    for w in ms.get("warnings", []):
        print(f"- [{w['type']}, {w['severity']}] {w['drug']} -> {w['target']}: {w['message']}")
    for t in ms.get("treatment_options", []):
        print(f"  {t['disease']}: no known clash: {'; '.join(t['no_known_clash']) or 'none'}")
        for c in t["clash_moderate_or_major"]:
            print(f"    clash: {c['drug']} with {', '.join(c['clashes_with'])}")
        if t["clash_minor"]:
            print(f"    minor clashes: {len(t['clash_minor'])} drugs")
        if t["clash_unknown_severity"]:
            print(f"    clashes of unknown severity (not rated in DDInter): {len(t['clash_unknown_severity'])} drugs")

    _section("Supported by both")
    print(", ".join(out.get("supported_by_both") or []) or "none")

    _section("Also a known diagnosis from the clinical text")
    print(", ".join(out.get("rag_matches_known_disease") or []) or "none")

    _section("Confidence")
    print(confidence_line(out))
    print(out.get("confidence"))

    _section("Evidence scanned (each disease once)")
    for name, ev in (state.get("evidence_output") or {}).items():
        print(f"- {name}: {len(ev.get('evidence') or [])} items" + (" (replay: no sample)" if ev.get("replay") else ""))

    _section("Evidence Scanner Groq")
    eg = out.get("evidence_groq") or {}
    print(f"Groq calls {eg.get('groq_calls', 0)}, fallbacks {eg.get('fallbacks', 0)}, "
          f"evidence_fallback_used {out.get('evidence_fallback_used')}")
    for name, g in (state.get("evidence_groq") or {}).items():
        print(f"- {name}: Groq calls {g['groq_llm_calls']}, fallbacks {g['groq_fallbacks']}"
              + (f", Groq says: {g['groq_error_message']}" if g.get("groq_error_message") else ""))

    _section("Fusion report (Medico)")
    print(out.get("fusion_report") or "none")
    _section("Optimizer note (Medico)")
    print(out.get("optimizer_note") or "none")

    _section("Notes")
    for n in out.get("notes") or []:
        print("-", n)

    _section("Errors")
    for e in state.get("errors") or []:
        print(f"- {e['node']}: {e['message']}")
    if not state.get("errors"):
        print("none")


def load_cases(args):
    cases = []
    if args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases = [{"id": c.get("id") or f"case{i + 1}", "patient_text": c["patient_text"]}
                     for i, c in enumerate(json.load(f))]
    cases += [{"id": f"text{i + 1}", "patient_text": t} for i, t in enumerate(args.text or [])]
    return cases or [{"id": "case1", "patient_text": DEFAULT_TEXT}]


def startup(disabled=()):
    """Read-only Qdrant check first (Vector RAG, unless disabled), then every enabled real agent loads once."""
    disabled = config.parse_disabled(disabled)
    report = {"disabled_agents": disabled or "none (full system)"}
    if config.AGENT_MODES.get("vector_rag") == "real" and "rag" not in disabled:
        # Cheap read-only Qdrant check FIRST, so a missing/wrong database stops the run before anything loads
        try:
            wrappers.check_qdrant()
        except wrappers.RagDatabaseError as exc:
            raise SystemExit(f"\nSTOPPED: Vector RAG database check failed.\n  {exc}\n"
                             "Nothing was created or changed. Fix Qdrant and run again.")
    report.update(wrappers.init_agents(disabled=disabled))
    err = (report.get("vector_rag") or {}).get("error", "")
    if err.startswith("RagDatabaseError"):
        raise SystemExit(f"\nSTOPPED: Vector RAG database check failed.\n  {err}")
    return report


def _summary_row(case_id, state):
    out = state.get("final_output") or {}
    d = out.get("most_likely_diagnosis") or {}
    c = out.get("confidence") or {}
    return [case_id, str(out.get("analysis_status")), d.get("condition") or "none",
            ", ".join(out.get("supported_by_both") or []) or "none",
            ", ".join(out.get("rag_matches_known_disease") or []) or "none",
            str(c.get("C1")), str(state.get("loop_count")), str(len(state.get("errors") or []))]


def print_summary(rows):
    head = ["case", "status", "diagnosis", "supported by both", "known diagnosis", "C1", "loops", "errors"]
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    _section("Summary")
    for r in [head, ["-" * w for w in widths]] + rows:
        print(" | ".join(v.ljust(w) for v, w in zip(r, widths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="append", help="patient text (can repeat)")
    parser.add_argument("--cases", help="JSON file: [{id, patient_text}, ...]")
    parser.add_argument("--json", action="store_true", help="print raw final_output JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="print everything (startup report, Clarifier details, path, evidence, Fusion report, ...)")
    parser.add_argument("--disable", default=None,
                        help="ablation: comma-separated agents to switch off: rag, kg, evidence, optimizer")
    args = parser.parse_args()
    try:
        disabled = config.parse_disabled(args.disable if args.disable is not None else config.DISABLED_FROM_ENV)
    except config.SwitchError as exc:
        raise SystemExit(f"\nSTOPPED: {exc}")
    cases = load_cases(args)

    if not args.json:
        _fake_banner()
    loaded = startup(disabled)
    if not args.json:
        if args.verbose:
            _section("Startup loading (real agents)")
            print(loaded or "no real agents")
        else:
            for agent, info in loaded.items():
                if isinstance(info, dict) and info.get("error"):
                    print(f"! {agent} failed to load: {info['error']}")

    results, rows = [], []
    for case in cases:
        state, path = run(case["patient_text"], disabled)
        rows.append(_summary_row(case["id"], state))
        if args.json:
            results.append(json_result(case["id"], state))
        elif args.verbose:
            print("\n" + "#" * 64 + f"\n# CASE {case['id']}\n# {case['patient_text']}\n" + "#" * 64)
            print_report(state, path)
        else:
            print("\n" + "#" * 64 + f"\n# CASE {case['id']}")
            print_short(state, case["patient_text"])

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False))
    else:
        if len(cases) > 1:
            print_summary(rows)
        _fake_banner()


if __name__ == "__main__":
    main()
