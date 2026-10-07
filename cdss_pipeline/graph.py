"""LangGraph flow for the CDSS pipeline (Step 12).

clarifier -> (vector_rag || knowledge_graph) -> evidence_scanner -> fusion -> optimizer
clarifier -> final_output (case not analysed: Clarifier failed, or found no present items)
optimizer -> vector_rag   (if rag_confidence < 0.65 and loop_count < 2 and Vector RAG worked)
optimizer -> final_output (otherwise)

How evidence_scanner waits for both on the first pass but only vector_rag on loops:
- vector_rag and knowledge_graph are both started by clarifier, so they run in the
  same LangGraph step. evidence_scanner has a plain edge from each and runs once,
  in the next step, after both finished.
- On a loop the optimizer sends the flow to vector_rag only. Nothing starts
  knowledge_graph again, so evidence_scanner is started by vector_rag alone.
- A join edge add_edge(["vector_rag", "knowledge_graph"], ...) is NOT used: it would
  wait for KG on every loop and silently never run evidence_scanner again.
- defer=True on evidence_scanner is extra safety: it waits until all other pending
  nodes are done.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from cdss_pipeline import config, wrappers
from cdss_pipeline.state import CDSSState
from cdss_pipeline.wrappers import clean_name, match_parts, name_parts


FUSION_FALLBACK_NOTE = "Fusion AI report unavailable, fallback used"

# final_output["analysis_status"]: a batch runner counts the two "not analysed" kinds separately
# and never scores them as a wrong diagnosis.
ANALYSED = "analysed"
CLARIFIER_ERROR = "clarifier_error"
NO_CLINICAL_ITEMS = "no_clinical_items"
NOT_ANALYSED_NOTES = {
    CLARIFIER_ERROR: "Clarifier failed - case not analysed",
    NO_CLINICAL_ITEMS: "Clarifier found no present clinical items - case not analysed",
}


def _error(node, exc):
    message = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None:  # e.g. "OpenRouter request failed." hides the real reason
        message += f" (cause: {type(cause).__name__}: {str(cause)[:300]})"
    return {"node": node, "message": config.redact(message)}


def _rag_available(rag_output):
    return bool(rag_output) and rag_output.get("status") != "error"


def _kg_available(kg_output):
    return bool(kg_output) and kg_output.get("status") != "error"


def _rag_candidates(rag_output):
    if not _rag_available(rag_output):
        return []
    cands = rag_output.get("diagnostic_candidates") or []
    return sorted(cands, key=lambda c: c.get("rank", 999))


def _kg_candidates(kg_output):
    if not _kg_available(kg_output):
        return []
    return (kg_output.get("disease_prediction") or {}).get("candidate_diseases") or []


def _unique(names):
    seen = []
    for n in names:
        if n and n not in seen:
            seen.append(n)
    return seen


def _rag_disease_names(state):
    top = _rag_candidates(state.get("rag_output"))[:config.EVIDENCE_TOP_RAG]
    return _unique(clean_name(c.get("condition")) for c in top)


def _kg_disease_names(state):
    top = _kg_candidates(state.get("kg_output"))[:config.EVIDENCE_TOP_KG]
    return _unique(clean_name(c.get("disease")) for c in top)


# ---------------------------------------------------------------- nodes

def clarifier_node(state):
    try:
        output, details = wrappers.run_clarifier_full(state.get("patient_text", ""))
        return {"clarifier_output": output, "clarifier_details": details}
    except Exception as exc:
        details = dict(getattr(exc, "details", None) or {})
        details.setdefault("error", config.redact(f"{type(exc).__name__}: {exc}"))
        return {"clarifier_output": {}, "clarifier_details": details,
                "errors": [_error("clarifier", exc)]}


def analysis_status(state):
    """analysed | clarifier_error | no_clinical_items (all 4 main lists empty, not an error)."""
    if any(e.get("node") == "clarifier" for e in state.get("errors") or []):
        return CLARIFIER_ERROR
    out = state.get("clarifier_output") or {}
    if not any(out.get(key) for key in wrappers.CLARIFIER_LISTS):
        return NO_CLINICAL_ITEMS
    return ANALYSED


def _disabled(state):
    return set(state.get("disabled_agents") or [])


def route_after_clarifier(state):
    if analysis_status(state) != ANALYSED:
        return "final_output"  # skip every other agent: no silent empty input
    off = _disabled(state)
    return [node for switch, node in (("rag", "vector_rag"), ("kg", "knowledge_graph")) if switch not in off]


def vector_rag_node(state):
    loop = state.get("loop_count", 0)
    try:
        out = wrappers.run_vector_rag(state.get("clarifier_output") or {}, state["rag_top_k"])
        return {"rag_output": out}
    except Exception as exc:
        error = {**_error("vector_rag", exc), "loop": loop}
        if loop > 0 and _rag_available(state.get("rag_output")):
            return {"errors": [error]}  # loop failed: keep the previous pass's rag_output
        return {"rag_output": None, "errors": [error]}


def _rag_failed_this_pass(state):
    loop = state.get("loop_count", 0)
    return any(e.get("node") == "vector_rag" and e.get("loop") == loop for e in state.get("errors") or [])


def _rag_loop_failed(state):
    return any(e.get("node") == "vector_rag" and (e.get("loop") or 0) > 0 for e in state.get("errors") or [])


def knowledge_graph_node(state):
    try:
        entities = (state.get("clarifier_details") or {}).get("entities") or []  # body_site for matching
        kg_output = wrappers.run_knowledge_graph(state.get("clarifier_output") or {}, entities)
    except Exception as exc:
        return {"kg_output": None, "kg_safety_messages": [],
                "errors": [_error("knowledge_graph", exc)]}
    try:
        messages = wrappers.make_kg_safety_messages(kg_output)
    except Exception as exc:
        return {"kg_output": kg_output, "kg_safety_messages": [],
                "errors": [_error("knowledge_graph (safety list)", exc)]}
    return {"kg_output": kg_output, "kg_safety_messages": messages}


def evidence_scanner_node(state):
    already = state.get("evidence_output") or {}
    # Parts of every disease already scanned (earlier loops); grows during this pass too
    seen = {p for key in already for p in name_parts(key)}
    diseases = _unique(_rag_disease_names(state) + _kg_disease_names(state))
    new, errors = {}, []
    first_record = len(wrappers.EVIDENCE_STATS["calls"])  # real-mode records added in this pass
    for disease in diseases:
        parts = name_parts(disease)
        if not parts or seen & set(parts):
            continue  # skipped: one of its parts was already scanned
        seen |= set(parts)
        try:
            new[disease] = wrappers.run_evidence_scanner(disease)
        except Exception as exc:
            errors.append(_error(f"evidence_scanner ({disease})", exc))
    update = {"evidence_output": new, "evidence_groq": {
        r["disease"]: {"groq_llm_calls": r.get("groq_llm_calls", 0),
                       "groq_fallbacks": r.get("groq_fallbacks", 0),
                       "groq_error_message": r.get("groq_error_message")}
        for r in wrappers.EVIDENCE_STATS["calls"][first_record:]}}
    if errors:
        update["errors"] = errors
    return update


def _find_evidence(evidence, name):
    """Evidence entry sharing any "/" part with name (first scanned wins), or None."""
    parts = set(name_parts(name))
    for key, result in evidence.items():  # dict keeps scan order
        if parts & set(name_parts(key)):
            return key, result
    return None


def _hypothesis_results(evidence, names):
    found, results = set(), []
    for name in names:
        hit = _find_evidence(evidence, name)
        if hit and hit[0] not in found:
            found.add(hit[0])
            results.append(hit[1])
    return {"hypothesis_results": results}


def fusion_node(state):
    evidence = state.get("evidence_output") or {}
    medico_state = {
        "raw_vector_evidence": state.get("rag_output") or {},
        "raw_vector_web_evidence": _hypothesis_results(evidence, _rag_disease_names(state)),
        # graph_confidence is the ONLY confidence her C2 can read (structural part); none when it is None
        "raw_graph_evidence": wrappers.medico_kg_view(state.get("kg_output")),
        "raw_graph_web_evidence": _hypothesis_results(evidence, _kg_disease_names(state)),
        "kg_safety_messages": wrappers._scrub_scores(state.get("kg_safety_messages") or []),
        "attempt": state.get("loop_count", 0) + 1,
    }
    try:
        out = wrappers.run_fusion(medico_state)
        c1 = out["confidence"]["confidence_1"]["score"]
        # RAG disabled: her C1 is an invented placeholder (0.5 similarity guess), never used as rag_confidence
        rag_confidence = None if "rag" in _disabled(state) else float(c1)
        update = {"fusion_output": out, "rag_confidence": rag_confidence}
        if out.get("fallback_used"):
            reason = wrappers.fusion_failure_reason() if config.AGENT_MODES.get("fusion") == "real" else None
            update["errors"] = [{"node": "fusion", "message": FUSION_FALLBACK_NOTE + (
                f": {reason}" if reason else " (reason not available)")}]
        return update
    except Exception as exc:
        return {"fusion_output": None, "rag_confidence": None, "errors": [_error("fusion", exc)]}


def optimizer_node(state):
    conf = state.get("rag_confidence")
    loops = state.get("loop_count", 0)
    max_loops = 0 if "optimizer" in _disabled(state) else config.MAX_LOOPS
    if (_rag_available(state.get("rag_output")) and not _rag_failed_this_pass(state)
            and conf is not None
            and conf < config.THRESHOLD and loops < max_loops):
        return Command(goto="vector_rag", update={
            "rag_top_k": state["rag_top_k"] + config.TOP_K_STEP,
            "loop_count": loops + 1,
        })
    return Command(goto="final_output")


# ---------------------------------------------------------------- final output

def _medication_safety(state):
    options = []
    by_disease = ((state.get("kg_output") or {}).get("medication_safety") or {}).get("by_disease") or []
    for entry in by_disease:
        t = entry.get("treatment_options") or {}
        if not any(t.get(k) for k in t):
            continue
        options.append({
            "disease": entry["disease"],
            "already_taken_by_patient": [e["drug"] for e in t.get("already_taken_by_patient", [])],
            "no_known_clash": [e["drug"] for e in t.get("no_known_clash", [])],
            "clash_moderate_or_major": [
                {"drug": e["drug"],
                 "clashes_with": [f"{c['medication']} ({c.get('severity', 'unknown')})"
                                  for c in e.get("clashes_with", [])]}
                for e in t.get("clash_major", []) + t.get("clash_moderate", [])],
            "clash_minor": [e["drug"] for e in t.get("clash_minor_or_unknown", [])
                            if any(c.get("severity") == "minor" for c in e.get("clashes_with", []))],
            "clash_unknown_severity": [e["drug"] for e in t.get("clash_minor_or_unknown", [])
                                       if any(c.get("severity") != "minor" for c in e.get("clashes_with", []))],
        })
    return {"warnings": state.get("kg_safety_messages") or [], "treatment_options": options}


def _all_parts(names):
    """Match parts for supported_by_both / rag_matches_known_disease: bracket content counts too."""
    return {p for n in names for p in match_parts(n)}


def _rag_names(rag_output):
    return _all_parts(c.get("condition") for c in _rag_candidates(rag_output))


def supported_by_both(rag_output, kg_output):
    """Shared name parts between Vector RAG candidates and KG PREDICTED diseases.

    Names are cleaned and split on "/"; bracket content counts as an extra part (match_parts).
    """
    kg = _all_parts(c.get("disease") for c in _kg_candidates(kg_output))
    return sorted(_rag_names(rag_output) & kg)


def rag_matches_known_disease(rag_output, clarifier_output):
    """Shared name parts between Vector RAG candidates and known diseases from the Clarifier."""
    known = _all_parts((clarifier_output or {}).get("diseases") or [])
    return sorted(_rag_names(rag_output) & known)


RAG_OFF_LABEL = ("Note: Vector RAG disabled - 'Score 1' / C1 below is a placeholder, not a measured value.")


def _label_if_rag_off(text, off):
    """Her fusion report / optimizer note still print her invented C1 when RAG is off: label it (text kept)."""
    if text and "rag" in off:
        return f"{RAG_OFF_LABEL}\n{text}"
    return text


def ablation_note(off):
    return f"Ablation run: disabled {', '.join(sorted(off))} - not the full system"


def evidence_groq_summary(state):
    """Totals of her Groq jobs and fallbacks for this case (all loops)."""
    per = state.get("evidence_groq") or {}
    messages = [v["groq_error_message"] for v in per.values() if v.get("groq_error_message")]
    return {"groq_calls": sum(v.get("groq_llm_calls", 0) for v in per.values()),
            "fallbacks": sum(v.get("groq_fallbacks", 0) for v in per.values()),
            "diseases_with_fallback": [d for d, v in per.items() if v.get("groq_fallbacks")],
            "groq_error_message": messages[-1] if messages else None}


def evidence_fallback_note(summary):
    note = (f"Evidence Scanner: Groq unavailable, used fallback for {summary['fallbacks']} of "
            f"{summary['groq_calls']} calls - evidence quality reduced")
    return note + (f" (Groq says: {summary['groq_error_message']})" if summary["groq_error_message"] else "")


def _confidence(fusion_output, disabled=()):
    """Her scores for final_output. A score that depends on a DISABLED input is hidden (None) because her code
    invents a placeholder for it (C1: 0.5 similarity guess; C2: near-zero); her raw numbers stay in fusion_output."""
    if not fusion_output:
        return None
    c = fusion_output.get("confidence") or {}
    c1 = (c.get("confidence_1") or {}).get("score")
    c2 = (c.get("confidence_2") or {}).get("score")
    fused, passes = c.get("final_score"), c.get("passes_threshold")
    hidden = []
    if "rag" in disabled:
        c1, hidden = None, hidden + ["C1 (Vector RAG disabled)"]
    if "kg" in disabled:
        c2, hidden = None, hidden + ["C2 (Knowledge Graph disabled)"]
    if hidden:
        fused, passes = None, None
    out = {
        "C1": c1,
        "C2": c2,
        "fused": fused,
        "passes_threshold": passes,
        "C1_passes_threshold": c1 is not None and c1 >= config.THRESHOLD,
        "threshold": config.THRESHOLD,
    }
    if hidden:
        out["not_available"] = hidden + ["fused (needs C1 and C2)"]
    return out


def final_output_node(state):
    rag_output, kg_output = state.get("rag_output"), state.get("kg_output")
    conf = state.get("rag_confidence")
    loops = state.get("loop_count", 0)
    status = analysis_status(state)
    evidence_groq = evidence_groq_summary(state)
    notes = []

    top = _rag_candidates(rag_output)[:1]
    diagnosis = ({"condition": top[0].get("condition"), "justification": top[0].get("justification")}
                 if top else None)

    off = _disabled(state)
    if off:
        notes.append(ablation_note(off))
    if status != ANALYSED:
        notes.append(NOT_ANALYSED_NOTES[status])  # other agents were skipped on purpose
    else:
        if "rag" in off:
            notes.append("Vector RAG disabled: C1 and fused confidence not available")
        elif not _rag_available(rag_output):
            notes.append("Vector RAG unavailable")
        elif _rag_loop_failed(state):
            notes.append("Vector RAG loop failed, previous result kept")
        if "kg" in off:
            notes.append("Knowledge Graph disabled: no KG candidates and no medication safety list; "
                         "C2 and fused confidence not available")
        elif not _kg_available(kg_output):
            notes.append("Knowledge Graph unavailable")
        if "evidence" in off:
            notes.append("Evidence Scanner disabled: C1/C2 computed without web evidence "
                         "(web term 0, default agreement)")
        if "optimizer" in off:
            notes.append("Optimizer loop disabled: single pass")
        if evidence_groq["fallbacks"]:
            notes.append(evidence_fallback_note(evidence_groq))
        if (state.get("fusion_output") or {}).get("fallback_used"):
            notes.append(FUSION_FALLBACK_NOTE)
        if not state.get("fusion_output"):
            notes.append("Fusion unavailable, no confidence score")
        elif conf is not None and conf < config.THRESHOLD and loops >= config.MAX_LOOPS:
            notes.append(f"confidence stayed below {config.THRESHOLD} after {loops} loops")
        elif conf is not None and conf >= config.THRESHOLD:
            notes.append(f"confidence reached {config.THRESHOLD} after {loops} loop(s)")
    if config.CLARIFIER_FAKE and config.AGENT_MODES.get("clarifier") == "real":
        notes.insert(0, config.FAKE_WARNING)
    replay = [a for a, m in config.AGENT_MODES.items() if m == "replay"]
    if replay:
        notes.append("replay mode (saved samples) for: " + ", ".join(replay))
    if state.get("errors"):
        notes.append(f"{len(state['errors'])} agent error(s), see errors")

    return {"final_output": {
        "analysis_status": status,
        "disabled_agents": sorted(off),  # [] = full system; saved results always show which version made them
        "most_likely_diagnosis": diagnosis,
        "rare_serious_conditions": [
            {"disease": c.get("disease"), "matched_symptoms": c.get("matched_symptoms"),
             "coverage": round(c.get("coverage_fraction") or 0, 3)}
            for c in _kg_candidates(kg_output)],
        "medication_safety": _medication_safety(state),
        "supported_by_both": supported_by_both(rag_output, kg_output),
        "rag_matches_known_disease": rag_matches_known_disease(
            rag_output, state.get("clarifier_output")),
        "confidence": _confidence(state.get("fusion_output"), off),
        # KG's own score for its top disease (also the structural part of her C2); None = not available
        "graph_confidence": (kg_output or {}).get("graph_confidence") if _kg_available(kg_output) else None,
        "unmatched_symptoms": (kg_output or {}).get("unmatched_symptoms") or [],
        # full ranked lists (for scoring "correct anywhere" and KG hit@k)
        "rag_candidates": [c.get("condition") for c in _rag_candidates(rag_output)],
        "kg_ranking_top10": [r.get("disease") for r in ((kg_output or {}).get("ranking_top10") or [])]
                            if _kg_available(kg_output) else [],
        "fusion_report": _label_if_rag_off((state.get("fusion_output") or {}).get("fusion_report"), off),
        "optimizer_note": _label_if_rag_off((state.get("fusion_output") or {}).get("optimizer_response"), off),
        # a batch runner can exclude or re-run cases whose evidence used her Groq fallbacks
        "evidence_fallback_used": evidence_groq["fallbacks"] > 0,
        "evidence_groq": evidence_groq,
        "notes": notes,
    }}


# ---------------------------------------------------------------- graph

def build_graph(disabled=()):
    """Full graph, or an ablation graph without the disabled agents (initial_state must get the same list)."""
    off = set(config.parse_disabled(disabled))
    rag, kg, evidence = "rag" not in off, "kg" not in off, "evidence" not in off
    g = StateGraph(CDSSState)
    g.add_node("clarifier", clarifier_node)
    if rag:
        g.add_node("vector_rag", vector_rag_node)
    if kg:
        g.add_node("knowledge_graph", knowledge_graph_node)
    if evidence:
        g.add_node("evidence_scanner", evidence_scanner_node, defer=True)
    # without the scanner, fusion is the join point of RAG and KG (defer: wait for both)
    g.add_node("fusion", fusion_node, defer=not evidence)
    g.add_node("optimizer", optimizer_node,
               destinations=("vector_rag", "final_output") if rag else ("final_output",))
    g.add_node("final_output", final_output_node)

    g.add_edge(START, "clarifier")
    # RAG and KG (the enabled ones) in the same step, or straight to final_output
    g.add_conditional_edges("clarifier", route_after_clarifier,
                            [n for n, on in (("vector_rag", rag), ("knowledge_graph", kg)) if on] + ["final_output"])
    after = "evidence_scanner" if evidence else "fusion"
    if rag:
        g.add_edge("vector_rag", after)
    if kg:
        g.add_edge("knowledge_graph", after)
    if evidence:
        g.add_edge("evidence_scanner", "fusion")
    g.add_edge("fusion", "optimizer")
    g.add_edge("final_output", END)
    return g.compile()
