r"""Flow tests for the CDSS pipeline. No API calls: all agents run in replay mode
or are replaced with small fakes that record how they were called.

Run from the integration folder:
    .venv\Scripts\python.exe -B -m pytest cdss_pipeline\tests -q -p no:cacheprovider
"""

import copy
import json
import sys
from pathlib import Path

import pytest

from cdss_pipeline import config, wrappers
from cdss_pipeline.graph import build_graph, rag_matches_known_disease, supported_by_both
from cdss_pipeline.state import initial_state

_REAL_RUN_EVIDENCE = wrappers.run_evidence_scanner


@pytest.fixture
def calls(monkeypatch):
    """Force replay mode and record calls to each wrapper."""
    monkeypatch.setattr(config, "AGENT_MODES", {a: "replay" for a in config.AGENTS})
    log = {"vector_rag_top_k": [], "knowledge_graph": 0, "evidence": [], "fusion": 0}

    real_rag, real_kg, real_ev = (wrappers.run_vector_rag, wrappers.run_knowledge_graph,
                                  wrappers.run_evidence_scanner)

    def rag(clarifier_output, top_k):
        log["vector_rag_top_k"].append(top_k)
        return real_rag(clarifier_output, top_k)

    def kg(clarifier_output, entities=None):
        log["knowledge_graph"] += 1
        return real_kg(clarifier_output, entities)

    def ev(disease):
        log["evidence"].append(disease)
        return real_ev(disease)

    monkeypatch.setattr(wrappers, "run_vector_rag", rag)
    monkeypatch.setattr(wrappers, "run_knowledge_graph", kg)
    monkeypatch.setattr(wrappers, "run_evidence_scanner", ev)
    return log


def set_c1(monkeypatch, log, c1):
    """Fusion fake that always reports confidence_1.score = c1."""
    def fusion(medico_state):
        log["fusion"] += 1
        out = wrappers._sample("fusion_case1.json")
        out["confidence"]["confidence_1"]["score"] = c1
        out["fallback_used"] = False  # the real sample was made with the fallback
        return out
    monkeypatch.setattr(wrappers, "run_fusion", fusion)


def run(text="test patient"):
    path, final = [], None
    for mode, chunk in build_graph().stream(initial_state(text),
                                            stream_mode=["updates", "values"]):
        if mode == "updates":
            path.extend(chunk.keys())
        else:
            final = chunk
    return final, path


# ---------------------------------------------------------------- loop behaviour

def test_kg_runs_once_even_with_two_loops(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    state, _ = run()
    assert state["loop_count"] == 2
    assert calls["knowledge_graph"] == 1


def test_low_confidence_loops_at_most_twice(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    state, _ = run()
    assert state["loop_count"] == 2
    assert calls["fusion"] == 3  # first pass + 2 loops
    assert "confidence stayed below 0.65 after 2 loops" in state["final_output"]["notes"]


def test_rag_top_k_goes_5_10_15(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    state, _ = run()
    assert calls["vector_rag_top_k"] == [5, 10, 15]
    assert state["rag_top_k"] == 15


def test_high_confidence_goes_straight_to_final(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    state, path = run()
    assert state["loop_count"] == 0
    assert calls["vector_rag_top_k"] == [5]
    assert path[-2:] == ["optimizer", "final_output"]
    assert state["final_output"]["confidence"]["C1"] == 0.9


def test_node_order_first_pass_and_loops(monkeypatch, calls):
    """First pass: KG and RAG (same step, either order) BEFORE one evidence_scanner.
    Loops: only vector_rag, then evidence_scanner, fusion, optimizer."""
    set_c1(monkeypatch, calls, 0.3)
    _, path = run()
    assert path[0] == "clarifier"
    assert set(path[1:3]) == {"knowledge_graph", "vector_rag"}
    loop = ["vector_rag", "evidence_scanner", "fusion", "optimizer"]
    assert path[3:] == ["evidence_scanner", "fusion", "optimizer"] + loop + loop + ["final_output"]
    assert path.count("knowledge_graph") == 1
    assert path.count("evidence_scanner") == 3


def test_each_disease_scanned_once(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    state, _ = run()
    assert len(calls["evidence"]) == len(set(calls["evidence"]))
    assert len(calls["evidence"]) <= config.EVIDENCE_TOP_RAG + config.EVIDENCE_TOP_KG
    assert set(state["evidence_output"]) == set(calls["evidence"])


def test_loop_scans_only_new_diseases(monkeypatch, calls):
    """A bigger RAG result on loop 1 brings a new disease: only that one is scanned."""
    set_c1(monkeypatch, calls, 0.3)
    base_rag = wrappers.run_vector_rag

    def rag(clarifier_output, top_k):
        out = base_rag(clarifier_output, top_k)
        if top_k > 5:
            out["diagnostic_candidates"][2] = {"condition": "Pericarditis", "rank": 3}
        return out
    monkeypatch.setattr(wrappers, "run_vector_rag", rag)
    run()
    assert calls["evidence"].count("pericarditis") == 1
    assert calls["evidence"][-1] == "pericarditis"


# ---------------------------------------------------------------- errors

def test_kg_error_is_recorded_and_flow_finishes(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)

    def boom(clarifier_output, entities=None):
        raise RuntimeError("graph file missing")
    monkeypatch.setattr(wrappers, "run_knowledge_graph", boom)
    state, _ = run()
    assert state["final_output"] is not None
    assert any(e["node"] == "knowledge_graph" for e in state["errors"])
    assert "Knowledge Graph unavailable" in state["final_output"]["notes"]


def test_vector_rag_crash_does_not_loop(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)

    def boom(clarifier_output, top_k):
        raise RuntimeError("qdrant not found")
    monkeypatch.setattr(wrappers, "run_vector_rag", boom)
    state, _ = run()
    assert state["loop_count"] == 0
    assert "Vector RAG unavailable" in state["final_output"]["notes"]
    assert any(e["node"] == "vector_rag" for e in state["errors"])


def test_errors_are_added_not_overwritten(monkeypatch, calls):
    """Two nodes in the SAME step (vector_rag and KG) both fail: both errors kept."""
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr(wrappers, "run_vector_rag",
                        lambda c, k: (_ for _ in ()).throw(RuntimeError("rag down")))
    monkeypatch.setattr(wrappers, "run_knowledge_graph",
                        lambda c, e=None: (_ for _ in ()).throw(RuntimeError("kg down")))
    state, _ = run()
    nodes = [e["node"] for e in state["errors"]]
    assert "vector_rag" in nodes and "knowledge_graph" in nodes


def test_error_message_hides_key_values(monkeypatch, calls):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-value-123456")
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr(wrappers, "run_vector_rag",
                        lambda c, k: (_ for _ in ()).throw(RuntimeError("bad key fake-test-value-123456")))
    state, _ = run()
    msg = [e["message"] for e in state["errors"] if e["node"] == "vector_rag"][0]
    assert "fake-test-value-123456" not in msg and "***" in msg


def test_real_mode_not_loaded_is_recorded(monkeypatch, calls):
    """Real mode selected but init_agents() never loaded it: recorded, flow finishes."""
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "fusion": "real"})
    monkeypatch.delitem(wrappers._REAL, "fusion", raising=False)
    state, _ = run()
    assert any(e["node"] == "fusion" and "not loaded" in e["message"] for e in state["errors"])
    assert state["final_output"] is not None


# ---------------------------------------------------------------- names and safety list

def test_clean_name():
    assert wrappers.clean_name("Myocardial Infarction (Heart Attack)") == "myocardial infarction"
    assert wrappers.clean_name("  Type 2   Diabetes ") == "type 2 diabetes"
    assert wrappers.clean_name("Asthma [severe]") == "asthma"


def test_supported_by_both_ignores_brackets_and_case(calls):
    rag = {"status": "success", "diagnostic_candidates": [
        {"condition": "Myocardial Infarction (Heart Attack)", "rank": 1},
        {"condition": "Heart Arrhythmia", "rank": 2}]}
    kg = {"status": "success", "known_diseases": [],
          "disease_prediction": {"candidate_diseases": [{"disease": "myocardial  infarction"}]}}
    assert supported_by_both(rag, kg) == ["myocardial infarction"]


def test_supported_by_both_ignores_kg_known_diseases():
    """A KG KNOWN disease is not a KG prediction, so it does not count."""
    rag = {"status": "success", "diagnostic_candidates": [
        {"condition": "Myocardial Infarction (Heart Attack)", "rank": 1}]}
    kg = {"status": "success", "known_diseases": ["Myocardial Infarction"],
          "disease_prediction": {"candidate_diseases": [{"disease": "thymic carcinoma"}]}}
    assert supported_by_both(rag, kg) == []


def test_rag_matches_known_disease_ignores_brackets_and_case():
    rag = {"status": "success", "diagnostic_candidates": [
        {"condition": "Myocardial Infarction (Heart Attack)", "rank": 1},
        {"condition": "Heart Arrhythmia", "rank": 2}]}
    clarifier = {"diseases": ["  myocardial   infarction "]}
    assert rag_matches_known_disease(rag, clarifier) == ["myocardial infarction"]
    assert rag_matches_known_disease(rag, {}) == []


def test_case1_supported_by_both_and_known_disease(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    state, _ = run()
    assert state["final_output"]["supported_by_both"] == []
    assert state["final_output"]["rag_matches_known_disease"] == ["myocardial infarction"]


def test_kg_safety_list_from_case1():
    kg = copy.deepcopy(wrappers._load_json(str(config.KG_SAMPLE)))
    items = wrappers.make_kg_safety_messages(kg)
    for item in items:
        assert set(item) == {"type", "drug", "target", "severity", "message"}
    contra = [i for i in items if i["type"] == "contraindication"]
    assert contra == [{"type": "contraindication", "drug": "Sildenafil",
                       "target": "myocardial infarction", "severity": "unknown",
                       "message": "Sildenafil is recorded as contraindicated in myocardial infarction."}]
    clashes = [i for i in items if i["type"] == "treatment_clash"]
    assert {i["drug"] for i in clashes} == {"Dicoumarol", "Dexamethasone", "Procarbazine"}
    assert all(i["severity"] in ("moderate", "major") and i["target"] == "Sildenafil" for i in clashes)
    summaries = [i for i in items if i["type"] == "treatment_clash_summary"]
    assert [(i["drug"], i["target"], i["severity"]) for i in summaries] == [
        ("15 treatment options", "Sildenafil", "unknown"),
        ("10 treatment options", "Sildenafil", "unknown")]
    assert "for myocardial infarction" in summaries[0]["message"]
    assert "for Hodgkins lymphoma" in summaries[1]["message"]
    for i in summaries:
        assert "interaction of unknown severity (not rated in DDInter)" in i["message"]
        assert "minor or" not in i["message"]


def _clash(drug, *pairs):
    return {"drug": drug, "link_type": "indication",
            "clashes_with": [{"medication": m, "severity": s} for m, s in pairs]}


def test_summary_keeps_minor_and_unknown_separate():
    """One disease with minor AND unknown clashes, and two current medicines."""
    kg = {"medication_safety": {"warnings": [], "by_disease": [{
        "disease": "asthma",
        "treatment_options": {"clash_minor_or_unknown": [
            _clash("Drug A", ("Metoprolol", "minor")),
            _clash("Drug B", ("Metoprolol", "minor")),
            _clash("Drug C", ("Metoprolol", "unknown")),
            _clash("Drug D", ("Aspirin", "unknown"), ("Metoprolol", "unknown")),
        ]}}]}}
    items = wrappers.make_kg_safety_messages(kg)
    assert items == [
        {"type": "treatment_clash_summary", "drug": "1 treatment option", "target": "Aspirin",
         "severity": "unknown",
         "message": "1 treatment option for asthma has an interaction of unknown severity "
                    "(not rated in DDInter) with Aspirin: Drug D."},
        {"type": "treatment_clash_summary", "drug": "2 treatment options", "target": "Metoprolol",
         "severity": "minor",
         "message": "2 treatment options for asthma have a minor interaction with Metoprolol: "
                    "Drug A, Drug B."},
        {"type": "treatment_clash_summary", "drug": "2 treatment options", "target": "Metoprolol",
         "severity": "unknown",
         "message": "2 treatment options for asthma have an interaction of unknown severity "
                    "(not rated in DDInter) with Metoprolol: Drug C, Drug D."},
    ]


def test_kg_safety_list_drug_clash():
    kg = {"medication_safety": {"warnings": [{
        "type": "current_medication_interaction", "medications": ["Warfarin", "Aspirin"],
        "severity": "major", "message": "Warfarin and Aspirin have a recorded interaction."}],
        "by_disease": []}}
    assert wrappers.make_kg_safety_messages(kg) == [{
        "type": "drug_clash", "drug": "Warfarin", "target": "Aspirin", "severity": "major",
        "message": "Warfarin and Aspirin have a recorded interaction."}]


# ---------------------------------------------------------------- real mode (fake agent, nothing loaded)

def test_real_kg_output_uses_same_safety_list(monkeypatch, calls):
    """Real-mode KG output goes through the same safety-list maker as replay."""
    set_c1(monkeypatch, calls, 0.9)
    sample = copy.deepcopy(wrappers._load_json(str(config.KG_SAMPLE)))
    seen = {}

    def fake_run_kg_agent(clarifier_output, top_k=5):
        seen["top_k"] = top_k
        return copy.deepcopy(sample)
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "knowledge_graph": "real"})
    monkeypatch.setitem(wrappers._REAL, "knowledge_graph", fake_run_kg_agent)
    state, _ = run()
    assert seen["top_k"] == 5
    assert state["kg_safety_messages"] == wrappers.make_kg_safety_messages(sample)
    assert not state["errors"]


def test_real_kg_error_status_is_recorded(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "knowledge_graph": "real"})
    monkeypatch.setitem(wrappers._REAL, "knowledge_graph",
                        lambda c, top_k=5: {"status": "error", "error": "load_resources: missing pkl"})
    state, _ = run()
    assert any(e["node"] == "knowledge_graph" and "missing pkl" in e["message"] for e in state["errors"])
    assert "Knowledge Graph unavailable" in state["final_output"]["notes"]


def test_real_mode_without_init_is_recorded(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "knowledge_graph": "real"})
    monkeypatch.delitem(wrappers._REAL, "knowledge_graph", raising=False)
    state, _ = run()
    assert any("init_agents" in e["message"] for e in state["errors"])


def test_init_agents_records_load_failure(monkeypatch):
    monkeypatch.setattr(config, "AGENT_MODES", {a: "replay" for a in config.AGENTS} | {"knowledge_graph": "real"})
    monkeypatch.delitem(wrappers._REAL, "knowledge_graph", raising=False)
    monkeypatch.setitem(wrappers._LOADERS, "knowledge_graph",
                        lambda: (_ for _ in ()).throw(OSError("pkl not found")))
    report = wrappers.init_agents()
    assert "pkl not found" in report["knowledge_graph"]["error"]


# ---------------------------------------------------------------- Clarifier (new version; fake Groq server, no network)

import json as _json
from types import SimpleNamespace

import httpx

from cdss_pipeline.graph import ANALYSED, CLARIFIER_ERROR, NO_CLINICAL_ITEMS, NOT_ANALYSED_NOTES


def _entity(text, type_, status="present", evidence=None, **extra):
    return {"text": text, "type": type_, "status": status, "evidence_text": evidence or text,
            "body_site": None, "duration": None, "severity": None, "value": None, **extra}


def _groq_ok(payload, headers=None):
    """A Groq chat completion whose message content is the JSON payload."""
    return httpx.Response(200, headers=headers or {}, json={
        "id": "t", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": _json.dumps(payload)}}]})


def _groq_error(status, headers=None):
    return httpx.Response(status, headers=headers or {}, json={"error": {"message": "test error"}})


def _first_word_reply(text):
    """Default reply: one present symptom grounded on the first word of the text."""
    return _groq_ok({"entities": [_entity("fake symptom", "symptom", evidence=text.split()[0])],
                     "demographics": {"age": None, "sex": None}})


@pytest.fixture
def clarifier(monkeypatch, tmp_path):
    """Real-mode Clarifier (his real code) talking to a scripted fake Groq server.

    Put httpx.Response objects or exceptions in .replies; requests land in .requests.
    """
    monkeypatch.setattr(config, "GROQ_LIMITS_FILE", tmp_path / "groq_limits.json")
    monkeypatch.setattr(wrappers, "GROQ_LIMITS", {})
    monkeypatch.setattr(wrappers, "CLARIFIER_STATS", {"groq_requests": 0, "calls": []})
    monkeypatch.setattr(wrappers, "_REAL", dict(wrappers._REAL))
    monkeypatch.setattr(wrappers, "_clarifier_commit", lambda: "51dacaf")
    monkeypatch.setattr(config, "CLARIFIER_FAKE", False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-123")
    monkeypatch.setattr(config, "GROQ_GATEWAY_BASE", "")  # direct route unless a test sets the gateway
    script = SimpleNamespace(replies=[], requests=[], waits=[])
    monkeypatch.setattr(wrappers, "_sleep", script.waits.append)

    def handler(request):
        body = _json.loads(request.content)
        script.requests.append(body)
        reply = script.replies.pop(0) if script.replies else _first_word_reply(body["messages"][-1]["content"])
        if isinstance(reply, Exception):
            raise reply
        return reply
    wrappers._load_clarifier(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "clarifier": "real"})
    return script


def _downstream_not_run(calls):
    return (calls["vector_rag_top_k"] == [] and calls["knowledge_graph"] == 0
            and calls["evidence"] == [] and calls["fusion"] == 0)


def test_clarifier_success_passes_exactly_5_keys_and_details(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    text = "56 year old man with chest pain, denies fever. Takes aspirin."
    clarifier.replies.append(_groq_ok({
        "entities": [_entity("chest pain", "symptom", body_site="chest"),
                     _entity("fever", "symptom", status="absent", evidence="denies fever"),
                     _entity("aspirin", "medication"),
                     _entity("made up", "disease", evidence="not in the text")],  # removed by grounding
        "demographics": {"age": 56, "sex": "male"}}))
    state, path = run(text)
    out = state["clarifier_output"]
    assert sorted(out) == sorted(wrappers.CLARIFIER_KEYS)
    assert out == {"clinical_text": text, "diseases": [], "symptoms": ["chest pain"],
                   "medications": ["aspirin"], "tests": []}
    d = state["clarifier_details"]
    assert [(e["text"], e["status"]) for e in d["entities"]] == [
        ("chest pain", "present"), ("fever", "absent"), ("aspirin", "present")]
    assert d["entities"][0]["body_site"] == "chest" and d["entities"][1]["evidence_text"] == "denies fever"
    assert d["demographics"] == {"age": 56, "sex": "male"}
    assert d["removed_by_grounding_check"] == 1
    assert d["commit"] == "51dacaf" and d["attempts"] == 1 and d["model"]
    assert state["final_output"]["analysis_status"] == ANALYSED
    assert "clarifier_details" not in state["final_output"]
    assert calls["knowledge_graph"] == 1 and "vector_rag" in path
    assert not state["errors"]
    assert clarifier.requests[0]["messages"][-1]["content"] == text


def test_clarifier_error_status_recorded_and_case_skipped(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(_groq_error(400))  # bad request: no retry
    state, path = run()
    assert [e["node"] for e in state["errors"]] == ["clarifier"]
    assert "Groq bad request (HTTP 400)" in state["errors"][0]["message"]
    assert state["clarifier_output"] == {}
    assert path == ["clarifier", "final_output"] and _downstream_not_run(calls)
    fo = state["final_output"]
    assert fo["analysis_status"] == CLARIFIER_ERROR
    assert NOT_ANALYSED_NOTES[CLARIFIER_ERROR] in fo["notes"]
    assert "Vector RAG unavailable" not in fo["notes"]
    assert state["clarifier_details"]["attempts"] == 1 and "HTTP 400" in state["clarifier_details"]["error"]


def test_clarifier_missing_entities_is_an_error(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(_groq_ok({"demographics": {}}))
    state, _ = run()
    assert any(e["node"] == "clarifier" and "missing the entities field" in e["message"]
               for e in state["errors"])
    assert state["final_output"]["analysis_status"] == CLARIFIER_ERROR
    assert _downstream_not_run(calls)


def test_clarifier_no_present_items_skips_without_error(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(_groq_ok({
        "entities": [_entity("fever", "symptom", status="absent", evidence="no fever"),
                     _entity("pneumonia", "disease", status="possible", evidence="possible pneumonia")],
        "demographics": {}}))
    state, path = run("no fever, possible pneumonia")
    assert not state["errors"]
    assert path == ["clarifier", "final_output"] and _downstream_not_run(calls)
    fo = state["final_output"]
    assert fo["analysis_status"] == NO_CLINICAL_ITEMS
    assert NOT_ANALYSED_NOTES[NO_CLINICAL_ITEMS] in fo["notes"]
    assert [e["status"] for e in state["clarifier_details"]["entities"]] == ["absent", "possible"]


def test_clarifier_retries_once_on_rate_limit(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(_groq_error(429, {"retry-after": "2"}))
    state, _ = run()
    assert len(clarifier.requests) == 2
    assert clarifier.waits == [5]  # retry-after 2 s, but at least CLARIFIER_RETRY_WAIT
    d = state["clarifier_details"]
    assert d["attempts"] == 2 and "rate limit" in d["attempt_errors"][0]
    assert not state["errors"] and state["final_output"]["analysis_status"] == ANALYSED


def test_clarifier_retries_only_once(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies += [_groq_error(429), _groq_error(429)]
    state, _ = run()
    assert len(clarifier.requests) == 2
    assert state["final_output"]["analysis_status"] == CLARIFIER_ERROR
    assert state["clarifier_details"]["attempts"] == 2


@pytest.mark.parametrize("exc", [httpx.ConnectError("down"), httpx.ReadTimeout("slow")])
def test_clarifier_retries_once_on_connection_or_timeout(monkeypatch, calls, clarifier, exc):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(exc)
    state, _ = run()
    assert len(clarifier.requests) == 2 and state["clarifier_details"]["attempts"] == 2
    assert state["final_output"]["analysis_status"] == ANALYSED


@pytest.mark.parametrize("status", [401, 403, 404, 400])
def test_clarifier_no_retry_for_auth_bad_request_or_model(monkeypatch, calls, clarifier, status):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(_groq_error(status))
    state, _ = run()
    assert len(clarifier.requests) == 1 and clarifier.waits == []
    assert state["clarifier_details"]["attempts"] == 1
    assert state["final_output"]["analysis_status"] == CLARIFIER_ERROR


def test_clarifier_error_message_hides_key(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr(wrappers._REAL["clarifier"], "process",
                        lambda text: (_ for _ in ()).throw(RuntimeError("bad key test-groq-key-123")))
    state, _ = run()
    assert state["final_output"]["analysis_status"] == CLARIFIER_ERROR
    assert "test-groq-key-123" not in str(state["errors"]) + str(state["clarifier_details"])


def test_pacing_sees_clarifier_rate_limit_headers(monkeypatch, calls, clarifier):
    """Clarifier responses feed the SAME tracker as the Evidence Scanner (and the limits file)."""
    headers = {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "7s",
               "x-ratelimit-remaining-requests": "321", "x-ratelimit-limit-requests": "1000",
               "x-ratelimit-reset-requests": "1m0s"}
    clarifier.replies.append(_groq_ok({"entities": [_entity("fake symptom", "symptom", evidence="test")],
                                       "demographics": {}}, headers))
    wrappers.run_clarifier("test patient")
    assert wrappers.GROQ_LIMITS["remaining_tokens"] == 100
    assert wrappers.GROQ_LIMITS["remaining_requests"] == 321
    saved = _json.loads(config.GROQ_LIMITS_FILE.read_text(encoding="utf-8"))
    assert saved["remaining_requests"] == 321
    assert wrappers.CLARIFIER_STATS["groq_requests"] == 1
    # tokens are low now: the next Clarifier call (or Evidence Scanner disease) pauses first
    wrappers.run_clarifier("test patient")
    assert len(clarifier.waits) == 1 and 6 < clarifier.waits[0] <= 7
    assert "low tokens" in wrappers.CLARIFIER_STATS["calls"][-1]["pause_reasons"][0]


def test_clarifier_sdk_retries_are_off(clarifier):
    assert wrappers._REAL["clarifier"].client.max_retries == 0


def test_clarifier_import_guard(monkeypatch, tmp_path):
    fake = SimpleNamespace(__file__=str(tmp_path / "agent.py"))
    monkeypatch.setitem(sys.modules, "agent", fake)
    monkeypatch.setattr(wrappers, "_REAL", dict(wrappers._REAL))
    with pytest.raises(ImportError, match="wrong agent loaded"):
        wrappers._load_clarifier()
    assert "clarifier" not in wrappers._REAL


def test_clarifier_modules_come_from_new_folder(clarifier):
    for name in wrappers.CLARIFIER_MODULES:
        assert Path(sys.modules[name].__file__).resolve().parent == config.CLARIFIER_DIR.resolve()
    assert "clinicl-text-processor1" not in str(sys.path)


def test_clarifier_commit_unknown_never_crashes(monkeypatch, capsys):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(wrappers.subprocess, "run", no_git)
    assert wrappers._clarifier_commit() == "unknown"
    assert "commit unknown" in capsys.readouterr().out


def test_clarifier_other_commit_only_warns(monkeypatch, capsys):
    monkeypatch.setattr(wrappers.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout="abc1234\n"))
    assert wrappers._clarifier_commit() == "abc1234"
    assert "expected 51dacaf" in capsys.readouterr().out


def test_run_pipeline_several_cases_load_once(monkeypatch, calls, clarifier, capsys):
    from cdss_pipeline import run_pipeline
    set_c1(monkeypatch, calls, 0.9)
    inits = []
    monkeypatch.setattr(wrappers, "init_agents", lambda **k: inits.append(1) or {})
    monkeypatch.setattr("sys.argv", ["run_pipeline", "--verbose", "--text", "p1", "--text", "p2", "--text", "p3"])
    run_pipeline.main()
    out = capsys.readouterr().out
    assert inits == [1] and len(clarifier.requests) == 3  # one Groq call per case
    assert calls["knowledge_graph"] == 3
    assert "# CASE text3" in out and "=== Summary" in out and "=== Clarifier details" in out
    assert config.FAKE_WARNING not in out


def test_run_pipeline_json_has_clarifier_details_next_to_final_output(monkeypatch, calls, clarifier, capsys):
    from cdss_pipeline import run_pipeline
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr(wrappers, "init_agents", lambda **k: {})
    monkeypatch.setattr("sys.argv", ["run_pipeline", "--json", "--text", "chest pain"])
    run_pipeline.main()
    item = _json.loads(capsys.readouterr().out)
    assert item["final_output"]["analysis_status"] == ANALYSED
    assert item["clarifier_details"]["entities"][0]["text"] == "fake symptom"


@pytest.fixture
def fake_mode(monkeypatch, tmp_path):
    """CDSS_CLARIFIER_FAKE=1: his real code with the built-in fake Groq server."""
    monkeypatch.setattr(config, "CLARIFIER_FAKE", True)
    monkeypatch.setattr(config, "GROQ_LIMITS_FILE", tmp_path / "groq_limits.json")
    monkeypatch.setattr(wrappers, "_REAL", dict(wrappers._REAL))
    monkeypatch.setattr(wrappers, "_clarifier_commit", lambda: "51dacaf")
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "clarifier": "real"})


def test_fake_mode_warning_in_output(monkeypatch, calls, fake_mode, capsys):
    from cdss_pipeline import run_pipeline
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr("sys.argv", ["run_pipeline", "--verbose", "--text", "fake run text"])
    run_pipeline.main()
    out = capsys.readouterr().out
    assert out.count(config.FAKE_WARNING) >= 3  # top banner, final_output notes, bottom banner
    assert out.lstrip().startswith("!")
    assert "analysed" in out and "fake symptom" in out


def test_fake_mode_fail_trigger_only_on_special_string(monkeypatch, calls, fake_mode):
    set_c1(monkeypatch, calls, 0.9)
    wrappers.init_agents()
    state, _ = run("history of heart failure, FAIL")  # ordinary words never trigger it
    assert state["final_output"]["analysis_status"] == ANALYSED
    state, _ = run(f"some text {config.CLARIFIER_FAKE_FAIL}")
    assert state["final_output"]["analysis_status"] == CLARIFIER_ERROR
    assert config.FAKE_WARNING in state["final_output"]["notes"]


def test_replay_clarifier_is_analysed(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    state, _ = run()
    assert state["final_output"]["analysis_status"] == ANALYSED
    assert state["clarifier_details"] == {"mode": "replay"}


# ---------------------------------------------------------------- Vector RAG: Qdrant check (fake client, no server)

import json as _json
from types import SimpleNamespace as _NS

GOOD_QDRANT = {"symptom_chunks": (2690, 768, "Cosine", "green"),
               "literature_chunks": (201134, 768, "Cosine", "green")}


class _FakeQdrant:
    """Read-only fake of the QdrantClient methods check_qdrant may use (no create/delete methods at all)."""

    def __init__(self, collections=None, reachable=True):
        self.collections = dict(GOOD_QDRANT if collections is None else collections)
        self.reachable = reachable

    def get_collections(self):
        if not self.reachable:
            raise ConnectionRefusedError("connection refused")
        return _NS(collections=[_NS(name=n) for n in self.collections])

    def get_collection(self, name):
        _, size, distance, status = self.collections[name]
        return _NS(status=status, config=_NS(params=_NS(vectors=_NS(size=size, distance=distance))))

    def count(self, name, exact):
        assert exact is True  # always EXACT counts
        return _NS(count=self.collections[name][0])


def test_qdrant_check_ok():
    report = wrappers.check_qdrant(_FakeQdrant())
    assert report == {"symptom_chunks": {"points": 2690, "status": "green"},
                      "literature_chunks": {"points": 201134, "status": "green"}}


def test_qdrant_check_unreachable():
    with pytest.raises(wrappers.RagDatabaseError, match="Qdrant not reachable at .*start_qdrant.ps1"):
        wrappers.check_qdrant(_FakeQdrant(reachable=False))


def test_qdrant_check_missing_collection():
    with pytest.raises(wrappers.RagDatabaseError, match="'literature_chunks' not found"):
        wrappers.check_qdrant(_FakeQdrant({"symptom_chunks": GOOD_QDRANT["symptom_chunks"]}))


@pytest.mark.parametrize("change, message", [
    ((201133, 768, "Cosine", "green"), "201133 points, expected exactly 201134"),
    ((201134, 384, "Cosine", "green"), "vector size 384, expected 768"),
    ((201134, 768, "Dot", "green"), "distance Dot, expected Cosine"),
    ((201134, 768, "Cosine", "red"), "status red"),
])
def test_qdrant_check_wrong_collection(change, message):
    with pytest.raises(wrappers.RagDatabaseError, match=message):
        wrappers.check_qdrant(_FakeQdrant({**GOOD_QDRANT, "literature_chunks": change}))


def test_qdrant_check_failure_stops_whole_run(monkeypatch):
    from cdss_pipeline import run_pipeline
    monkeypatch.setattr(wrappers, "_qdrant_client", lambda *a, **k: _FakeQdrant(reachable=False))
    monkeypatch.setattr(config, "AGENT_MODES", {a: "replay" for a in config.AGENTS} | {"vector_rag": "real"})
    started = []
    monkeypatch.setattr(wrappers, "init_agents", lambda **k: started.append("init") or {})
    with pytest.raises(SystemExit, match="(?s)STOPPED: Vector RAG database check failed.*Qdrant not reachable"):
        run_pipeline.startup()
    assert started == []  # nothing loaded before the check


@pytest.mark.parametrize("env, expected", [
    ({"RAG_ANSWER_MODEL": "a/b", "OPENROUTER_MODEL": "c/d"}, ("a/b", "RAG_ANSWER_MODEL")),
    ({"RAG_ANSWER_MODEL": "  ", "OPENROUTER_MODEL": "c/d"}, ("c/d", "OPENROUTER_MODEL")),
    ({}, ("", "his default (DEFAULT_MODEL in answerer.py)")),
])
def test_rag_answer_model_source(env, expected):
    assert config.resolve_rag_answer_model(env) == expected


def test_frozen_rag_package_comes_from_frozen_repo():
    """The venv .pth must point to Symptom_RAG_Frozen\\src (same package name as the old repo)."""
    import symptom_rag_analyzer
    assert Path(symptom_rag_analyzer.__file__).resolve().parent == (
        config.RAG_DIR / "src" / "symptom_rag_analyzer").resolve()


# ---------------------------------------------------------------- Vector RAG: his real orchestrator/adapter, fake retriever/answerer

from symptom_rag_analyzer.reasoning.answerer import DEFAULT_MODEL as _HIS_DEFAULT_MODEL
from symptom_rag_analyzer.reasoning.answerer import DiagnosticCandidate as _Cand
from symptom_rag_analyzer.reasoning.answerer import RAGAnswer as _RAGAnswer
from symptom_rag_analyzer.retrieval.search import RetrievedEvidence as _Ev
from symptom_rag_analyzer.retrieval.search import VectorRetrievalResult as _VRR


class _FakeStore:
    def search(self, vector, top_k=5):
        return []


class _FakeRetriever:
    """Stands in for his BiomedicalRetriever (no BioBERT, no Qdrant)."""
    created = []

    def __init__(self, qdrant_client=None):
        self.qdrant_client, self.top_ks = qdrant_client, []
        self.embedding_model = _NS(embed_text=lambda text: [0.0] * 768)
        self.symptom_store, self.literature_store = _FakeStore(), _FakeStore()
        _FakeRetriever.created.append(self)

    def retrieve(self, query, top_k=5):
        self.top_ks.append(top_k)
        sym = [_Ev(text=f"symptom {i}", score=0.9 - i / 100, filename="symptoms.csv", page_number=None,
                   chunk_index=i, source_type="symptom-disease-dataset") for i in range(top_k)]
        lit = [_Ev(text=f"paper {i}", score=0.8 - i / 100, filename=f"PMC{i}.xml", page_number=None,
                   chunk_index=i, source_type="pubmed-pmc-literature", metadata={"pmcid": f"PMC{i}"})
               for i in range(top_k)]
        return _VRR(symptom_evidence=sym, literature_evidence=lit)


class _FakeOpenAIClient:
    max_retries = 2  # the openai SDK default

    def with_options(self, max_retries):
        client = _FakeOpenAIClient()
        client.max_retries = max_retries
        return client


class _FakeAnswerer:
    """Stands in for his BiomedicalAnswerer (no OpenRouter). Script: list of exceptions / None."""
    created = []

    def __init__(self, model=_HIS_DEFAULT_MODEL):
        self.model, self.client, self.calls, self.script = model, _FakeOpenAIClient(), [], []
        _FakeAnswerer.created.append(self)

    def answer(self, clinical_context, evidence_context):
        self.calls.append(clinical_context)
        step = self.script.pop(0) if self.script else None
        if isinstance(step, Exception):
            raise step
        return _RAGAnswer(diagnostic_candidates=[
            _Cand("Myocardial Infarction (Heart Attack)", 1, "chest pain and sweating", [1, 6]),
            _Cand("Angina", 2, "exertional chest pain", [2]),
            _Cand("Pulmonary Embolism", 3, "shortness of breath", [7]),
            _Cand("Aortic Dissection", 4, "severe chest pain", [8])], limitations="fake limitations")


@pytest.fixture
def frozen_rag(monkeypatch):
    """Load the real-mode Vector RAG through OUR _load_vector_rag with his modules' heavy parts faked."""
    import symptom_rag_analyzer.reasoning.answerer as answerer_mod
    import symptom_rag_analyzer.retrieval.search as search_mod
    _FakeRetriever.created, _FakeAnswerer.created = [], []
    order = []
    monkeypatch.setattr(search_mod, "BiomedicalRetriever", _FakeRetriever)
    monkeypatch.setattr(answerer_mod, "BiomedicalAnswerer", _FakeAnswerer)
    monkeypatch.setattr(wrappers, "check_qdrant", lambda client=None: order.append("check") or {"ok": 1})
    monkeypatch.setattr(wrappers, "_qdrant_client", lambda *a, **k: order.append("client") or "OUR_CLIENT")
    monkeypatch.setattr(wrappers, "_git_commit", lambda folder, expected, label: "7a80dfc")
    monkeypatch.setattr(wrappers, "_REAL", dict(wrappers._REAL))
    monkeypatch.setattr(wrappers, "RAG_STATS", {"openrouter_calls": 0, "retries": 0, "failures": 0, "calls": []})
    waits = []
    monkeypatch.setattr(wrappers, "_sleep", waits.append)
    monkeypatch.setattr(config, "RAG_ANSWER_MODEL", "test/model:free")
    monkeypatch.setattr(config, "RAG_ANSWER_MODEL_SOURCE", "OPENROUTER_MODEL")
    info = wrappers._load_vector_rag()
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "vector_rag": "real"})
    return _NS(info=info, order=order, waits=waits)


def test_load_vector_rag_report_model_and_no_hidden_retries(frozen_rag):
    info = frozen_rag.info
    assert frozen_rag.order[0] == "check"  # Qdrant checked before his code connects
    assert info["rag_answer_model"] == "test/model:free" and info["rag_answer_model_source"] == "OPENROUTER_MODEL"
    assert info["commit"] == "7a80dfc" and info["collections"] == {"ok": 1}
    assert len(_FakeRetriever.created) == 1 and _FakeRetriever.created[0].qdrant_client == "OUR_CLIENT"
    answerer = _FakeAnswerer.created[0]
    assert answerer.model == "test/model:free"
    assert answerer.client.max_retries == 0  # hidden SDK retries OFF


def test_load_vector_rag_uses_his_default_model_when_unset(monkeypatch, frozen_rag):
    monkeypatch.setattr(config, "RAG_ANSWER_MODEL", "")
    info = wrappers._load_vector_rag()
    assert info["rag_answer_model"] == _HIS_DEFAULT_MODEL == "openai/gpt-5.6-luna"


def test_real_openai_client_with_options_turns_retries_off():
    from openai import OpenAI
    client = OpenAI(api_key="offline-test", base_url="https://openrouter.ai/api/v1")
    assert client.max_retries == 2 and client.with_options(max_retries=0).max_retries == 0


def test_rag_import_guard(monkeypatch, frozen_rag):
    monkeypatch.setitem(sys.modules, "symptom_rag_analyzer", _NS(__file__=str(Path("C:/elsewhere/__init__.py"))))
    with pytest.raises(ImportError, match="wrong symptom_rag_analyzer loaded"):
        wrappers._load_vector_rag()


def test_rag_orchestrator_per_top_k_sharing_one_retriever_and_answerer(monkeypatch, calls, frozen_rag):
    set_c1(monkeypatch, calls, 0.3)  # 2 loops: top_k 5 -> 10 -> 15
    state, _ = run()
    rag = wrappers._REAL["vector_rag"]
    assert sorted(rag["orchestrators"]) == [5, 10, 15]
    for top_k, orch in rag["orchestrators"].items():
        assert orch.top_k == top_k and orch.retriever is rag["retriever"] and orch.answerer is rag["answerer"]
    assert len(_FakeRetriever.created) == 1 and len(_FakeAnswerer.created) == 1  # BioBERT loaded once
    assert rag["retriever"].top_ks == [5, 10, 15]
    assert [c["top_k"] for c in wrappers.RAG_STATS["calls"]] == [5, 10, 15]
    assert state["rag_output"]["metadata"]["requested_top_k"] == 15
    assert not state["errors"]


def test_rag_output_mapping_and_downstream_fields(monkeypatch, calls, frozen_rag):
    seen = {}

    def fusion(medico_state):
        seen["medico"] = medico_state
        calls["fusion"] += 1
        out = wrappers._sample("fusion_case1.json")
        out["confidence"]["confidence_1"]["score"] = 0.9
        out["fallback_used"] = False
        return out
    monkeypatch.setattr(wrappers, "run_fusion", fusion)
    state, _ = run()
    rag = state["rag_output"]
    assert rag["status"] == "success" and rag["agent"] == "symptom_rag"
    assert [(c["condition"], c["rank"]) for c in rag["diagnostic_candidates"]][:2] == [
        ("Myocardial Infarction (Heart Attack)", 1), ("Angina", 2)]
    assert rag["diagnostic_candidates"][0]["supporting_evidence"] == [1, 6]
    assert len(rag["evidence"]) == 10  # top_k 5 from EACH collection
    assert {e["source_type"] for e in rag["evidence"]} == {"symptom-disease-dataset", "pubmed-pmc-literature"}
    assert all(isinstance(e["relevance_score"], float) for e in rag["evidence"])
    assert rag["metadata"]["total_evidence_count"] == 10 and rag["metadata"]["requested_top_k"] == 5
    # downstream: Evidence Scanner = top 3 RAG conditions (cleaned) + KG; final output; Fusion gets it unchanged
    assert calls["evidence"][:3] == ["myocardial infarction", "angina", "pulmonary embolism"]
    assert state["final_output"]["most_likely_diagnosis"]["condition"] == "Myocardial Infarction (Heart Attack)"
    assert seen["medico"]["raw_vector_evidence"] == rag
    assert _FakeAnswerer.created[0].calls[0] == state["clarifier_output"]["clinical_text"]


def test_rag_openrouter_failure_recorded_no_loop(monkeypatch, calls, frozen_rag):
    set_c1(monkeypatch, calls, 0.3)
    _FakeAnswerer.created[0].script = [ValueError("Invalid OpenRouter response: response was not valid JSON.")]
    state, _ = run()
    assert any(e["node"] == "vector_rag" and "not valid JSON" in e["message"] for e in state["errors"])
    assert state["loop_count"] == 0
    assert "Vector RAG unavailable" in state["final_output"]["notes"]
    assert wrappers.RAG_STATS["failures"] == 1 and frozen_rag.waits == []


# ---------------------------------------------------------------- OpenRouter retry for the RAG Answerer (no network, no waiting)

def _openrouter_fail(code):
    """What his answerer raises: RuntimeError("OpenRouter request failed.") from the openai APIStatusError."""
    import openai
    response = httpx.Response(code, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))
    try:
        raise openai.APIStatusError(f"HTTP {code}", response=response, body=None)
    except Exception as cause:
        try:
            raise RuntimeError("OpenRouter request failed.") from cause
        except RuntimeError as err:
            return err


class _ScriptedAnswerer:
    """answer() raises/returns according to a script, e.g. [503, 503, "ok"]."""

    def __init__(self, script):
        self.script, self.attempts = list(script), 0

    def answer(self, clinical_context, evidence_context):
        self.attempts += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, int):
            raise _openrouter_fail(step)
        return step


@pytest.fixture
def retry_env(monkeypatch):
    waits = []
    monkeypatch.setattr(wrappers, "_sleep", waits.append)
    monkeypatch.setattr(wrappers, "RAG_STATS", {"openrouter_calls": 0, "retries": 0, "failures": 0, "calls": [{}]})
    return waits


def _retrying(script):
    ans = _ScriptedAnswerer(script)
    wrappers._with_openrouter_retry(ans)
    return ans


@pytest.mark.parametrize("code", [429, 502, 503])
def test_openrouter_retry_then_success(retry_env, code):
    ans = _retrying([code, "ok"])
    assert ans.answer("c", "e") == "ok"
    assert ans.attempts == 2
    assert wrappers.RAG_STATS["retries"] == 1 and wrappers.RAG_STATS["openrouter_calls"] == 2
    assert wrappers.RAG_STATS["calls"][-1]["retry_codes"] == [code]


def test_openrouter_retry_waits_are_5_then_15(retry_env):
    assert _retrying([429, 503, "ok"]).answer("c", "e") == "ok"
    assert retry_env == [5, 15] and wrappers.RAG_STATS["calls"][-1]["retry_wait_seconds"] == 20


def test_openrouter_retry_gives_up_after_two(retry_env):
    ans = _retrying([503, 503, 503, "never"])
    with pytest.raises(RuntimeError, match="OpenRouter request failed"):
        ans.answer("c", "e")
    assert ans.attempts == 3 and wrappers.RAG_STATS["retries"] == 2 and wrappers.RAG_STATS["failures"] == 1


@pytest.mark.parametrize("code", [400, 401, 404, 500])
def test_openrouter_no_retry_for_other_codes(retry_env, code):
    ans = _retrying([code, "ok"])
    with pytest.raises(RuntimeError):
        ans.answer("c", "e")
    assert ans.attempts == 1 and retry_env == [] and wrappers.RAG_STATS["retries"] == 0
    assert wrappers.RAG_STATS["calls"][-1]["error_code"] == code


def test_openrouter_no_retry_for_non_api_error(retry_env):
    ans = _retrying([ValueError("Invalid OpenRouter response: response was not valid JSON."), "ok"])
    with pytest.raises(ValueError, match="not valid JSON"):
        ans.answer("c", "e")
    assert ans.attempts == 1 and retry_env == []


def test_openrouter_retry_exhausted_recorded_in_pipeline(monkeypatch, calls, frozen_rag):
    set_c1(monkeypatch, calls, 0.3)
    _FakeAnswerer.created[0].script = [_openrouter_fail(503)] * 3
    state, _ = run()
    msg = [e["message"] for e in state["errors"] if e["node"] == "vector_rag"][0]
    assert "OpenRouter request failed" in msg and "503" in msg
    assert state["loop_count"] == 0
    assert wrappers.RAG_STATS["retries"] == 2 and frozen_rag.waits == [5, 15]


# ---------------------------------------------------------------- leakage helper (fake scroll, no Qdrant)

def test_export_literature_ids(tmp_path):
    from cdss_pipeline import export_literature_ids as exp

    class Scroller:
        pages = [[_NS(payload={"pmcid": "PMC2", "pmid": 22}), _NS(payload={"pmcid": "PMC1", "pmid": "11"})],
                 [_NS(payload={"pmcid": "PMC1", "pmid": 11}), _NS(payload={}), _NS(payload={"pmid": "33"})]]

        def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
            assert collection_name == "literature_chunks" and with_vectors is False
            assert with_payload == ["pmid", "pmcid"]  # only the two id fields
            index = offset or 0
            return self.pages[index], (index + 1 if index + 1 < len(self.pages) else None)

    out = tmp_path / "ids.txt"
    summary = exp.write_ids(*exp.collect_ids(Scroller()), out)
    assert summary == {"points": 5, "without_ids": 1, "articles": 3, "pmcids": 2, "pmids": 3, "file": str(out)}
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[2:] == ["PMCID\tPMID", "\t33", "PMC1\t11", "PMC2\t22"]


# ---------------------------------------------------------------- Evidence Scanner real mode (no network)

class _FakeScanner:
    """Stands in for her compiled graph: records order, can fail for one disease."""

    def __init__(self, fail_for=()):
        self.order, self.fail_for, self.running = [], set(fail_for), 0

    def invoke(self, request):
        assert self.running == 0, "two diseases scanned at the same time"
        self.running += 1
        try:
            disease = request["request"]
            assert request["top_n"] == config.EVIDENCE_TOP_N
            self.order.append(disease)
            if disease in self.fail_for:
                raise RuntimeError(f"network down for {disease}")
            return {"hypothesis": disease, "broadened": False,
                    "evidence": [{"tier_label": "RCT", "score": 80.0, "title": f"T {disease}",
                                  "summary": "s", "url": None, "source": "PubMed"}]}
        finally:
            self.running -= 1


@pytest.fixture
def real_evidence(monkeypatch):
    monkeypatch.setattr(wrappers, "EVIDENCE_STATS", {"calls": []})
    monkeypatch.setattr(config, "AGENT_MODES", {a: "replay" for a in config.AGENTS} | {"evidence_scanner": "real"})

    def use(scanner):
        monkeypatch.setitem(wrappers._REAL, "evidence_scanner", scanner)
        return scanner
    return use


def test_real_evidence_stored_unchanged_under_clean_name(monkeypatch, calls, real_evidence):
    set_c1(monkeypatch, calls, 0.9)
    scanner = real_evidence(_FakeScanner())
    monkeypatch.setattr(wrappers, "run_evidence_scanner", _REAL_RUN_EVIDENCE)
    state, _ = run()
    assert "myocardial infarction" in state["evidence_output"]  # "(Heart Attack)" removed
    assert state["evidence_output"]["myocardial infarction"] == scanner.invoke(
        {"request": "myocardial infarction", "top_n": config.EVIDENCE_TOP_N})
    assert len(scanner.order) == 7  # 6 scanned in the pipeline + 1 in this assert


def test_real_evidence_one_failure_does_not_stop_others(monkeypatch, calls, real_evidence):
    set_c1(monkeypatch, calls, 0.9)
    scanner = real_evidence(_FakeScanner(fail_for={"heart arrhythmia"}))
    monkeypatch.setattr(wrappers, "run_evidence_scanner", _REAL_RUN_EVIDENCE)
    state, _ = run()
    assert len(scanner.order) == 6
    assert "heart arrhythmia" not in state["evidence_output"]
    assert len(state["evidence_output"]) == 5
    assert any(e["node"] == "evidence_scanner (heart arrhythmia)" and "network down" in e["message"]
               for e in state["errors"])
    failed = [r for r in wrappers.EVIDENCE_STATS["calls"] if r.get("error")]
    assert [r["disease"] for r in failed] == ["heart arrhythmia"]


def test_real_evidence_one_at_a_time_and_once_across_loops(monkeypatch, calls, real_evidence):
    set_c1(monkeypatch, calls, 0.3)  # 2 loops
    scanner = real_evidence(_FakeScanner())  # asserts nothing runs in parallel
    monkeypatch.setattr(wrappers, "run_evidence_scanner", _REAL_RUN_EVIDENCE)
    state, _ = run()
    assert state["loop_count"] == 2
    assert len(scanner.order) == len(set(scanner.order)) == 6
    assert [r["items"] for r in wrappers.EVIDENCE_STATS["calls"]] == [1] * 6


class _FakeResponse:
    def __init__(self, status):
        self.status_code = status


class _FakeRequestsModule:
    class HTTPError(Exception):
        pass

    def __init__(self, statuses):
        self.statuses = list(statuses)

    def get(self, *a, **k):
        status = self.statuses.pop(0)
        if status == "timeout":
            raise TimeoutError("slow")
        return _FakeResponse(status)

    post = get


def test_counting_requests_counts_and_passes_through(monkeypatch):
    monkeypatch.setattr(wrappers, "EVIDENCE_STATS", {"calls": [wrappers._new_evidence_record("x")]})
    real = _FakeRequestsModule([200, 429, 500, "timeout"])
    proxy = wrappers._CountingRequests(real, "pubmed")
    assert proxy.HTTPError is real.HTTPError  # attributes pass through
    first = proxy.get("u")
    assert isinstance(first, _FakeResponse) and first.status_code == 200  # same response object
    proxy.post("u")
    proxy.get("u")
    with pytest.raises(TimeoutError):
        proxy.get("u")
    rec = wrappers.EVIDENCE_STATS["calls"][-1]
    assert rec["requests"] == {"pubmed": 4}
    assert rec["rate_limited"] == {"pubmed": 1}
    assert rec["http_errors"] == {"pubmed": 1}
    assert rec["failed_requests"] == {"pubmed": 1}


# ---------------------------------------------------------------- "/" name parts

from cdss_pipeline.graph import _hypothesis_results


@pytest.mark.parametrize("name, parts", [
    ("Heart Attack / Acute Myocardial Infarction", ["heart attack", "acute myocardial infarction"]),
    ("Coronary Artery Disease/Angina", ["coronary artery disease", "angina"]),
    ("Myocardial Infarction (Heart Attack) / Angina", ["myocardial infarction", "angina"]),
    ("  Asthma   /  ", ["asthma"]),
    ("angina / Angina", ["angina"]),
    ("asthma", ["asthma"]),
    ("", []),
])
def test_name_parts(name, parts):
    assert wrappers.name_parts(name) == parts


def _rag_by_top_k(names_by_top_k):
    """Fake Vector RAG: candidate names depend on top_k (5, 10, 15)."""
    def rag(clarifier_output, top_k):
        return {"agent": "symptom_rag", "status": "success",
                "diagnostic_candidates": [{"condition": n, "rank": i + 1, "justification": "x"}
                                          for i, n in enumerate(names_by_top_k[top_k])]}
    return rag


def test_skip_if_any_part_scanned_in_earlier_loop(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    monkeypatch.setattr(wrappers, "run_vector_rag", _rag_by_top_k({
        5: ["Coronary Artery Disease / Angina"],
        10: ["Angina"],                         # shares "angina": skipped
        15: ["Angina / Pericarditis"],          # shares "angina": skipped, pericarditis not scanned
    }))
    run()
    rag_scans = [d for d in calls["evidence"] if d not in
                 ("thymic carcinoma", "hodgkins lymphoma", "monosodium glutamate sensitivity")]
    assert rag_scans == ["coronary artery disease / angina"]


def test_skip_if_any_part_scanned_in_same_pass(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr(wrappers, "run_vector_rag", _rag_by_top_k({
        5: ["Hodgkins Lymphoma / Lymphoma", "Asthma", "Asthma (Severe)"]}))
    run()
    # "hodgkins lymphoma" from RAG first; the KG's "hodgkins lymphoma" and the 2nd asthma are skipped
    assert calls["evidence"] == ["hodgkins lymphoma / lymphoma", "asthma",
                                 "thymic carcinoma", "monosodium glutamate sensitivity"]


def test_disease_without_shared_part_still_scanned(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    monkeypatch.setattr(wrappers, "run_vector_rag", _rag_by_top_k({
        5: ["Heart Attack / Acute Myocardial Infarction"],
        10: ["Myocardial Infarction (Heart Attack)"],   # brackets removed: no shared part
        15: ["Myocardial Infarction"]}))
    run()
    assert "heart attack / acute myocardial infarction" in calls["evidence"]
    assert calls["evidence"].count("myocardial infarction") == 1  # known limitation: scanned again once


def test_fusion_lookup_by_shared_part():
    evidence = {"hodgkins lymphoma": {"hypothesis": "hodgkins lymphoma"},
                "asthma": {"hypothesis": "asthma"}}
    out = _hypothesis_results(evidence, ["hodgkins lymphoma / lymphoma", "asthma", "unknown"])
    assert out == {"hypothesis_results": [{"hypothesis": "hodgkins lymphoma"}, {"hypothesis": "asthma"}]}


def test_fusion_gets_evidence_for_skipped_rag_disease(monkeypatch, calls):
    """RAG "Lymphoma / Hodgkins Lymphoma" is scanned first, so the KG's "hodgkins lymphoma" is
    skipped; the KG side of Fusion must still get that evidence through the shared part."""
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr(wrappers, "run_vector_rag", _rag_by_top_k({5: ["Lymphoma / Hodgkins Lymphoma"]}))
    seen = {}

    def fusion(medico_state):
        seen.update(medico_state)
        return wrappers._sample("fusion_case1.json")
    monkeypatch.setattr(wrappers, "run_fusion", fusion)
    run()
    graph_side = [h["hypothesis"] for h in seen["raw_graph_web_evidence"]["hypothesis_results"]]
    assert "lymphoma / hodgkins lymphoma" in graph_side


def test_supported_by_both_shared_part():
    rag = {"status": "success", "diagnostic_candidates": [
        {"condition": "Coronary Artery Disease / Angina", "rank": 1}]}
    kg = {"status": "success", "known_diseases": ["coronary artery disease"],
          "disease_prediction": {"candidate_diseases": [{"disease": "Angina"}]}}
    assert supported_by_both(rag, kg) == ["angina"]  # known disease not counted


def test_supported_by_both_no_partial_word_match():
    rag = {"status": "success", "diagnostic_candidates": [{"condition": "Infarction", "rank": 1}]}
    kg = {"status": "success", "disease_prediction": {"candidate_diseases": [{"disease": "myocardial infarction"}]}}
    assert supported_by_both(rag, kg) == []


def test_rag_matches_known_disease_shared_part():
    rag = {"status": "success", "diagnostic_candidates": [
        {"condition": "Heart Attack / Myocardial Infarction (MI)", "rank": 1}]}
    assert rag_matches_known_disease(rag, {"diseases": ["myocardial infarction"]}) == ["myocardial infarction"]


# ---------------------------------------------------------------- Groq model override (no network)

import types as _types


@pytest.fixture
def fake_groq_modules(monkeypatch):
    mods = {}
    for name in wrappers.GROQ_MODULES:
        mod = _types.ModuleType(name)
        mod.GROQ_MODEL = "llama-3.3-70b-versatile"
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    return mods


def test_groq_no_override_keeps_her_model(monkeypatch, fake_groq_modules):
    monkeypatch.setattr(config, "GROQ_MODEL_OVERRIDE", "")
    info = wrappers._apply_groq_model_override()
    assert info == {"groq_model": "llama-3.3-70b-versatile", "groq_model_source": "her code"}
    assert all(m.GROQ_MODEL == "llama-3.3-70b-versatile" for m in fake_groq_modules.values())


def test_groq_override_sets_both_modules(monkeypatch, fake_groq_modules, capsys):
    monkeypatch.setattr(config, "GROQ_MODEL_OVERRIDE", "openai/gpt-oss-120b")
    info = wrappers._apply_groq_model_override()
    assert info == {"groq_model": "openai/gpt-oss-120b",
                    "groq_model_source": "override (GROQ_MODEL_OVERRIDE)"}
    assert all(m.GROQ_MODEL == "openai/gpt-oss-120b" for m in fake_groq_modules.values())
    out = capsys.readouterr().out
    assert "OVERRIDDEN" in out and "llama-3.3-70b-versatile" in out


def test_groq_override_does_not_change_her_files(monkeypatch):
    """Load her real modules (imports only, no network) with the override on; files unchanged."""
    files = [config.EVIDENCE_DIR / "tools" / f for f in ("query_generator.py", "summarizer.py")]
    before = [f.read_bytes() for f in files]
    wrappers._put_first_on_path(config.EVIDENCE_DIR)
    import tools.query_generator  # noqa: F401
    import tools.summarizer  # noqa: F401
    original = {n: sys.modules[n].GROQ_MODEL for n in wrappers.GROQ_MODULES}
    monkeypatch.setattr(config, "GROQ_MODEL_OVERRIDE", "openai/gpt-oss-120b")
    try:
        wrappers._apply_groq_model_override()
        assert sys.modules["tools.summarizer"].GROQ_MODEL == "openai/gpt-oss-120b"
    finally:
        for n, v in original.items():
            sys.modules[n].GROQ_MODEL = v
    assert [f.read_bytes() for f in files] == before


# ---------------------------------------------------------------- Fusion real mode (no network)

class _FakeMedicoPipeline:
    """Returns her WHOLE state like build_pipeline().invoke(): inputs + normalized text + outputs."""

    def __init__(self, c1_values, fallback=False):
        self.c1_values, self.fallback, self.inputs = list(c1_values), fallback, []

    def invoke(self, state):
        self.inputs.append(state)
        c1 = self.c1_values.pop(0)
        if self.fallback:
            wrappers._current_pass().setdefault("failures", []).append("fusion: APIConnectionError: no network")
        return {**state, "vector_evidence": "text", "graph_evidence": "KG text without a score",
                "confidence": {"confidence_1": {"score": c1}, "confidence_2": {"score": 0.6, "graph_score": 0.9},
                               "fusion": {"final_fused_score": 0.5}, "final_score": 0.5,
                               "passes_threshold": c1 >= 0.65},
                "fusion_report": "REPORT", "optimizer_response": "OPT NOTE",
                "fallback_used": self.fallback, "attempt": state["attempt"]}


@pytest.fixture
def real_fusion(monkeypatch):
    monkeypatch.setattr(wrappers, "FUSION_STATS", {"openrouter_calls": 0, "openrouter_failures": 0,
                                                   "openrouter_retries": 0, "passes": []})
    monkeypatch.setattr(config, "AGENT_MODES", {a: "replay" for a in config.AGENTS} | {"fusion": "real"})

    def use(pipeline):
        monkeypatch.setitem(wrappers._REAL, "fusion", pipeline)
        return pipeline
    return use


_REAL_RUN_FUSION = wrappers.run_fusion


def test_real_fusion_output_keys_and_c1(monkeypatch, calls, real_fusion):
    pipe = real_fusion(_FakeMedicoPipeline([0.3, 0.5, 0.7]))
    monkeypatch.setattr(wrappers, "run_fusion", _REAL_RUN_FUSION)
    state, _ = run()
    assert set(state["fusion_output"]) == set(wrappers.FUSION_OUTPUT_KEYS)  # only her output keys
    assert state["fusion_output"]["fusion_report"] == "REPORT"
    assert state["rag_confidence"] == 0.7 and state["loop_count"] == 2
    assert [p["attempt"] for p in pipe.inputs] == [1, 2, 3]
    assert state["final_output"]["optimizer_note"] == "OPT NOTE"
    assert state["final_output"]["fusion_report"] == "REPORT"
    assert [p["C1"] for p in wrappers.FUSION_STATS["passes"]] == [0.3, 0.5, 0.7]
    assert all(p["graph_score_default_0_90"] for p in wrappers.FUSION_STATS["passes"])
    assert not state["errors"]


def test_real_fusion_input_keys_and_halves(monkeypatch, calls, real_fusion):
    pipe = real_fusion(_FakeMedicoPipeline([0.9]))
    monkeypatch.setattr(wrappers, "run_fusion", _REAL_RUN_FUSION)
    monkeypatch.setattr(wrappers, "run_vector_rag", _rag_by_top_k({5: ["Lymphoma / Hodgkins Lymphoma"]}))
    state, _ = run()
    sent = pipe.inputs[0]
    assert set(sent) == {"raw_vector_evidence", "raw_vector_web_evidence", "raw_graph_evidence",
                         "raw_graph_web_evidence", "kg_safety_messages", "attempt"}
    assert sent["raw_vector_evidence"] == state["rag_output"]
    assert sent["raw_graph_evidence"] == wrappers.medico_kg_view(state["kg_output"])  # cleaned view
    assert list(sent["raw_graph_evidence"])[0] == "graph_confidence"
    assert "graph_confidence_details" not in sent["raw_graph_evidence"]
    assert sent["kg_safety_messages"] == state["kg_safety_messages"]
    vec = [h["hypothesis"] for h in sent["raw_vector_web_evidence"]["hypothesis_results"]]
    kg = [h["hypothesis"] for h in sent["raw_graph_web_evidence"]["hypothesis_results"]]
    assert vec == ["lymphoma / hodgkins lymphoma"]
    assert "lymphoma / hodgkins lymphoma" in kg and "thymic carcinoma" in kg  # shared part on KG side


def test_real_fusion_fallback_note_and_reason(monkeypatch, calls, real_fusion):
    real_fusion(_FakeMedicoPipeline([0.9], fallback=True))
    monkeypatch.setattr(wrappers, "run_fusion", _REAL_RUN_FUSION)
    state, _ = run()
    assert "Fusion AI report unavailable, fallback used" in state["final_output"]["notes"]
    msg = [e["message"] for e in state["errors"] if e["node"] == "fusion"][0]
    assert msg.startswith("Fusion AI report unavailable, fallback used") and "no network" in msg


def test_count_openrouter_counts_and_reraises(monkeypatch):
    monkeypatch.setattr(wrappers, "FUSION_STATS", {"openrouter_calls": 0, "openrouter_failures": 0,
                                                   "openrouter_retries": 0, "passes": [{}]})
    mod = _types.ModuleType("fake_medico")
    mod.ok = lambda *a: "fine"

    def bad(*a):
        raise ConnectionError("down")
    mod.bad = bad
    wrappers._count_openrouter(mod, "ok", "fusion")
    wrappers._count_openrouter(mod, "bad", "optimizer")
    assert mod.ok() == "fine"
    with pytest.raises(ConnectionError):
        mod.bad()
    stats = wrappers.FUSION_STATS
    assert stats["openrouter_calls"] == 2 and stats["openrouter_failures"] == 1
    assert stats["passes"][0]["calls"] == {"fusion": 1, "optimizer": 1}
    assert "optimizer: ConnectionError: down" in stats["passes"][0]["failures"][0]


def test_retry_log_counter(monkeypatch):
    import logging
    monkeypatch.setattr(wrappers, "FUSION_STATS", {"openrouter_calls": 0, "openrouter_failures": 0,
                                                   "openrouter_retries": 0, "passes": [{}]})
    wrappers._attach_retry_counter()
    log = logging.getLogger("openai._base_client")
    log.info("Retrying request to %s in %f seconds", "/chat/completions", 0.5)
    log.info("Some other message")
    assert wrappers.FUSION_STATS["openrouter_retries"] == 1
    assert wrappers.FUSION_STATS["passes"][0]["retries"] == 1


def test_load_fusion_uses_unique_name(monkeypatch):
    """Imports her real files (no API call): loaded as medico_graph, ours untouched."""
    import cdss_pipeline.graph as ours
    ours_file = ours.__file__
    monkeypatch.delitem(wrappers._REAL, "fusion", raising=False)
    info = wrappers._load_fusion()
    try:
        assert Path(sys.modules["medico_graph"].__file__).resolve() == (config.MEDICO_DIR / "graph.py").resolve()
        assert Path(sys.modules["fusion_agents"].__file__).resolve().parent == config.MEDICO_DIR.resolve()
        assert sys.modules["cdss_pipeline.graph"].__file__ == ours_file
        assert hasattr(wrappers._REAL["fusion"], "invoke")
        assert info["openrouter_key"] in ("set", "missing")
        assert "key" not in str(info["openrouter_model"]).lower()
    finally:
        wrappers._REAL.pop("fusion", None)


# ---------------------------------------------------------------- user_id redaction

def test_redact_hides_user_ids():
    msg = ("Error code: 404 - {'error': {'code': 404}, 'user_id': 'user_3K6T43lYh9twbv5TiT3BQ8g4L4x'} "
           'and {"user_id": "user_ABCDEFGHIJKLMN"} and bare user_ZZZZZZZZZZZZ')
    out = config.redact(msg)
    assert "3K6T43" not in out and "ABCDEFGHIJ" not in out and "ZZZZZZZZZZ" not in out
    assert "'user_id': 'user_***'" in out and '"user_id": "user_***"' in out


def test_redact_keeps_normal_words():
    assert config.redact("user_input and user_id field and user_abc") == "user_input and user_id field and user_abc"


# ---------------------------------------------------------------- failed RAG loop keeps previous result

def test_failed_rag_loop_keeps_previous_result(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    good = _rag_by_top_k({5: ["Asthma"]})

    def rag(clarifier_output, top_k):
        if top_k > 5:
            raise RuntimeError("OpenRouter request failed.")
        return good(clarifier_output, top_k)
    monkeypatch.setattr(wrappers, "run_vector_rag", rag)
    state, _ = run()
    assert state["rag_output"]["diagnostic_candidates"][0]["condition"] == "Asthma"  # kept
    assert state["loop_count"] == 1  # failed loop -> no further loop
    notes = state["final_output"]["notes"]
    assert "Vector RAG loop failed, previous result kept" in notes
    assert "Vector RAG unavailable" not in notes
    assert state["final_output"]["most_likely_diagnosis"]["condition"] == "Asthma"
    assert [e["loop"] for e in state["errors"] if e["node"] == "vector_rag"] == [1]


def test_failed_rag_first_pass_unchanged(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    monkeypatch.setattr(wrappers, "run_vector_rag",
                        lambda c, k: (_ for _ in ()).throw(RuntimeError("OpenRouter request failed.")))
    state, _ = run()
    assert state["rag_output"] is None and state["loop_count"] == 0
    assert "Vector RAG unavailable" in state["final_output"]["notes"]
    assert "Vector RAG loop failed, previous result kept" not in state["final_output"]["notes"]


# ---------------------------------------------------------------- bracket content for the two matches

@pytest.mark.parametrize("name, parts", [
    ("Heart attack (Myocardial Infarction) / Coronary Artery Disease",
     ["heart attack", "coronary artery disease", "myocardial infarction"]),
    ("Asthma (Severe / Acute)", ["asthma", "severe", "acute"]),
    ("Disease ()", ["disease"]),
    ("A (B (C))", ["a", "c", "b"]),  # nested: innermost first
    ("Plain name", ["plain name"]),
])
def test_match_parts(name, parts):
    assert wrappers.match_parts(name) == parts


def test_case1_name_matches_known_disease_via_brackets():
    rag = {"status": "success", "diagnostic_candidates": [
        {"condition": "Heart attack (Myocardial Infarction) / Coronary Artery Disease", "rank": 1}]}
    assert rag_matches_known_disease(rag, {"diseases": ["myocardial infarction"]}) == ["myocardial infarction"]


def test_supported_by_both_via_brackets():
    rag = {"status": "success", "diagnostic_candidates": [
        {"condition": "Heart attack (Myocardial Infarction)", "rank": 1}]}
    kg = {"status": "success", "disease_prediction": {"candidate_diseases": [{"disease": "myocardial infarction"}]}}
    assert supported_by_both(rag, kg) == ["myocardial infarction"]


def test_evidence_skip_still_ignores_brackets(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    monkeypatch.setattr(wrappers, "run_vector_rag", _rag_by_top_k({
        5: ["Heart Attack (Myocardial Infarction)"],
        10: ["Myocardial Infarction"],   # shares only the BRACKET part: still scanned
        15: ["Myocardial Infarction"]}))
    run()
    assert "heart attack" in calls["evidence"]
    assert calls["evidence"].count("myocardial infarction") == 1


def test_counting_requests_stores_groq_limits(monkeypatch):
    monkeypatch.setattr(wrappers, "EVIDENCE_STATS", {"calls": [wrappers._new_evidence_record("x")]})

    class Resp:
        status_code = 429
        headers = {"X-RateLimit-Limit-Tokens": "8000", "x-ratelimit-remaining-tokens": "12",
                   "retry-after": "7", "Authorization": "should-not-be-kept", "content-type": "json"}
    real = type("R", (), {"post": lambda self, *a, **k: Resp()})()
    wrappers._CountingRequests(real, "groq").post("u")
    rec = wrappers.EVIDENCE_STATS["calls"][-1]
    assert rec["groq_limits"] == {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "12",
                                  "retry-after": "7", "status": 429}
    assert rec["retry_after"] == ["7"]


# ---------------------------------------------------------------- Groq pacing (no network, no real waiting)

@pytest.fixture(autouse=True)
def _isolated_groq_limits(monkeypatch, tmp_path):
    """Every test: empty GROQ_LIMITS and a temp limits file (never the real .groq_limits.json)."""
    monkeypatch.setattr(wrappers, "GROQ_LIMITS", {})
    monkeypatch.setattr(config, "GROQ_LIMITS_FILE", tmp_path / ".groq_limits.json")


@pytest.mark.parametrize("text, seconds", [
    ("26.407s", 26.407), ("50m24s", 3024), ("1h2m3s", 3723), ("120ms", 0.12), ("7", 7.0), ("", None),
])
def test_parse_duration(text, seconds):
    result = wrappers._parse_duration(text)
    assert result == seconds if seconds is None else abs(result - seconds) < 1e-9


def test_no_pause_without_info():
    assert wrappers._groq_wait_seconds() == (0.0, None)


def test_no_pause_with_enough_tokens():
    wrappers._update_groq_limits({"x-ratelimit-remaining-tokens": "4479", "x-ratelimit-reset-tokens": "26.4s"})
    assert wrappers._groq_wait_seconds() == (0.0, None)


def test_pause_on_low_tokens_minus_elapsed(monkeypatch):
    wrappers._update_groq_limits({"x-ratelimit-remaining-tokens": "1200", "x-ratelimit-reset-tokens": "26s"})
    wrappers.GROQ_LIMITS["tokens_seen_at"] -= 10  # 10 s already passed
    wait, reason = wrappers._groq_wait_seconds()
    assert 15.5 <= wait <= 16.1 and reason == "low tokens: 1200 left"
    wrappers.GROQ_LIMITS["tokens_seen_at"] -= 100  # reset already over
    assert wrappers._groq_wait_seconds() == (0.0, None)


def test_pause_capped_at_60():
    wrappers._update_groq_limits({"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "5m"})
    assert wrappers._groq_wait_seconds()[0] == 60


def test_retry_after_used_once_and_larger_wins():
    wrappers._update_groq_limits({"retry-after": "7", "x-ratelimit-remaining-tokens": "100",
                                  "x-ratelimit-reset-tokens": "20s"})
    wait, reason = wrappers._groq_wait_seconds()
    assert 19.5 <= wait <= 20 and reason.startswith("low tokens")  # larger of 7 and ~20
    wrappers.GROQ_LIMITS["remaining_tokens"] = 7000
    wrappers._update_groq_limits({"retry-after": "7"})
    assert wrappers._groq_wait_seconds() == (7.0, "retry-after 7 s")
    assert wrappers._groq_wait_seconds() == (0.0, None)  # retry-after cleared after use


def test_pause_recorded_and_top_n_3(monkeypatch, real_evidence):
    waits = []
    monkeypatch.setattr(wrappers, "_sleep", waits.append)
    scanner = real_evidence(_FakeScanner())
    wrappers._update_groq_limits({"x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "12s"})
    _REAL_RUN_EVIDENCE("asthma")
    rec = wrappers.EVIDENCE_STATS["calls"][-1]
    assert len(waits) == 1 and 11.5 <= waits[0] <= 12
    assert rec["paused_seconds"] == waits[0] and rec["pause_reason"] == "low tokens: 500 left"
    assert config.EVIDENCE_TOP_N == 3 and scanner.order == ["asthma"]  # _FakeScanner checks top_n


def test_limits_file_saved_and_startup_text():
    assert wrappers.groq_requests_left_text() == "unknown (no Groq call seen yet)"
    wrappers._update_groq_limits({"x-ratelimit-remaining-requests": "965", "x-ratelimit-limit-requests": "1000",
                                  "x-ratelimit-reset-requests": "50m24s", "x-ratelimit-remaining-tokens": "4479",
                                  "x-ratelimit-reset-tokens": "26s", "retry-after": "3"})
    saved = json.loads(Path(config.GROQ_LIMITS_FILE).read_text(encoding="utf-8"))
    assert set(saved) == {"remaining_tokens", "reset_tokens_s", "tokens_seen_at", "remaining_requests",
                          "limit_requests", "reset_requests_s", "requests_seen_at"}  # numbers only
    text = wrappers.groq_requests_left_text()
    assert text.startswith("965 of 1000 (as of ") and "resets 50m24s after that" in text
    assert "probably reset" not in text
    saved["requests_seen_at"] -= 4000
    Path(config.GROQ_LIMITS_FILE).write_text(json.dumps(saved), encoding="utf-8")
    assert "probably reset since" in wrappers.groq_requests_left_text()


def test_limits_file_write_failure_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GROQ_LIMITS_FILE", tmp_path / "no_such_dir" / "x.json")
    wrappers._update_groq_limits({"x-ratelimit-remaining-tokens": "100"})  # must not raise
    assert wrappers.GROQ_LIMITS["remaining_tokens"] == 100


# ---------------------------------------------------------------- Groq error visibility (fix 2: Clarifier)

ACCESS_DENIED = "Access denied. Please check your network settings."


def test_clarifier_groq_403_message_saved_and_printed(monkeypatch, calls, clarifier, capsys):
    from cdss_pipeline import run_pipeline
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(httpx.Response(403, json={"error": {"message": ACCESS_DENIED}}))
    state, _ = run()
    d = state["clarifier_details"]
    assert d["groq_error_message"] == ACCESS_DENIED
    assert d["attempts"] == 1  # 403 is never retried
    assert f"Groq permission denied (HTTP 403) - Groq says: {ACCESS_DENIED}" in state["errors"][0]["message"]
    assert state["final_output"]["analysis_status"] == CLARIFIER_ERROR
    run_pipeline.print_clarifier_details(d)
    assert f"groq error: {ACCESS_DENIED}" in capsys.readouterr().out


def test_clarifier_groq_error_message_hides_keys(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    body = {"error": {"message": "bad key gsk_abcdefghijklmnop1234 and test-groq-key-123"}}
    clarifier.replies.append(httpx.Response(401, json=body))
    state, _ = run()
    text = str(state["clarifier_details"]) + str(state["errors"])
    assert "gsk_abcdefghijklmnop1234" not in text and "test-groq-key-123" not in text
    assert "bad key" in state["clarifier_details"]["groq_error_message"]


def test_clarifier_groq_message_only_from_last_attempt(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(httpx.Response(429, json={"error": {"message": "slow down"}}))
    state, _ = run()  # retry succeeds
    d = state["clarifier_details"]
    assert "groq_error_message" not in d and "Groq says: slow down" in d["attempt_errors"][0]
    assert state["final_output"]["analysis_status"] == ANALYSED


def test_clarifier_connection_error_has_no_groq_message(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies += [httpx.ConnectError("down"), httpx.ConnectError("down")]
    state, _ = run()
    assert "groq_error_message" not in state["clarifier_details"]
    assert state["final_output"]["analysis_status"] == CLARIFIER_ERROR


@pytest.mark.parametrize("body, message", [
    ('{"error": {"message": "Access denied."}}', "Access denied."),
    ('{"error": "plain"}', "plain"),
    ("not json  at   all", "not json at all"),
    ("", None),
])
def test_groq_error_text(body, message):
    assert wrappers.groq_error_text(body) == message


# ---------------------------------------------------------------- Groq fallback visibility (fix 3: Evidence Scanner)

_FAKE_HER_MODULE = """
FAIL = False
def _call_llm(prompt):
    if FAIL:
        raise RuntimeError("HTTP 403")
    return "ok"
def {fallback}(*args):
    return "fallback"
def job(x):
    try:
        return _call_llm(x)
    except Exception:
        return {fallback}(x)
"""


@pytest.fixture
def her_groq_modules(monkeypatch):
    """Fake tools.query_generator / tools.summarizer with her _call_llm + fallback pattern (no network)."""
    mods = {}
    for name, fallback in wrappers.EVIDENCE_GROQ_FALLBACKS.items():
        mod = _types.ModuleType(name)
        exec(_FAKE_HER_MODULE.format(fallback=fallback), mod.__dict__)
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    monkeypatch.setattr(wrappers, "EVIDENCE_STATS", {"calls": [wrappers._new_evidence_record("x")]})
    wrappers._count_evidence_groq_jobs()
    wrappers._count_evidence_groq_jobs()  # second load must not wrap twice
    return mods


def test_evidence_groq_jobs_and_fallbacks_counted(her_groq_modules):
    from concurrent.futures import ThreadPoolExecutor
    qg, summ = her_groq_modules["tools.query_generator"], her_groq_modules["tools.summarizer"]
    assert qg.job("a") == "ok"
    summ.FAIL = True
    with ThreadPoolExecutor(max_workers=3) as pool:  # her summaries run in threads
        assert list(pool.map(summ.job, range(9))) == ["fallback"] * 9
    rec = wrappers.EVIDENCE_STATS["calls"][-1]
    assert rec["groq_llm_calls"] == 10 and rec["groq_fallbacks"] == 9


def test_evidence_groq_wrapped_only_once(her_groq_modules):
    for name, fallback in wrappers.EVIDENCE_GROQ_FALLBACKS.items():
        mod = her_groq_modules[name]
        for func in ("_call_llm", fallback):
            assert getattr(mod, func)._cdss_counts
            assert not hasattr(getattr(mod, func).__wrapped__, "_cdss_counts")


def test_counting_requests_keeps_groq_error_message(monkeypatch):
    monkeypatch.setattr(wrappers, "EVIDENCE_STATS", {"calls": [wrappers._new_evidence_record("x")]})
    response = SimpleNamespace(status_code=403, headers={},
                               text=_json.dumps({"error": {"message": ACCESS_DENIED}}))
    proxy = wrappers._CountingRequests(SimpleNamespace(post=lambda *a, **k: response), "groq")
    assert proxy.post("u") is response
    rec = wrappers.EVIDENCE_STATS["calls"][-1]
    assert rec["groq_error_message"] == ACCESS_DENIED and rec["http_errors"] == {"groq": 1}


@pytest.fixture
def evidence_groq_run(monkeypatch, calls):
    """Replay graph, but each scanned disease adds a real-style EVIDENCE_STATS record with Groq counts."""
    monkeypatch.setattr(wrappers, "EVIDENCE_STATS", {"calls": []})
    plan = {"fallback_for": set(), "message": None, "order": []}

    def scan(disease):
        calls["evidence"].append(disease)
        plan["order"].append(disease)
        rec = wrappers._new_evidence_record(disease)
        rec["groq_llm_calls"] = 4
        if disease in plan["fallback_for"] or "*" in plan["fallback_for"]:
            rec["groq_fallbacks"] = 4
            rec["groq_error_message"] = plan["message"]
        wrappers.EVIDENCE_STATS["calls"].append(rec)
        return {"hypothesis": disease, "broadened": False, "evidence": []}
    monkeypatch.setattr(wrappers, "run_evidence_scanner", scan)
    return plan


def test_evidence_fallback_flag_note_and_counts(monkeypatch, calls, evidence_groq_run, capsys):
    from cdss_pipeline import run_pipeline
    set_c1(monkeypatch, calls, 0.9)
    evidence_groq_run["fallback_for"] = {"myocardial infarction"}
    evidence_groq_run["message"] = ACCESS_DENIED
    state, path = run()
    fo = state["final_output"]
    n = len(evidence_groq_run["order"])
    assert fo["evidence_fallback_used"] is True
    assert fo["evidence_groq"] == {"groq_calls": 4 * n, "fallbacks": 4,
                                   "diseases_with_fallback": ["myocardial infarction"],
                                   "groq_error_message": ACCESS_DENIED}
    assert (f"Evidence Scanner: Groq unavailable, used fallback for 4 of {4 * n} calls - "
            f"evidence quality reduced (Groq says: {ACCESS_DENIED})") in fo["notes"]
    assert set(state["evidence_groq"]) == set(state["evidence_output"])
    run_pipeline.print_report(state, path)
    out = capsys.readouterr().out
    assert "=== Evidence Scanner Groq" in out and f"Groq says: {ACCESS_DENIED}" in out


def test_evidence_no_fallback_no_note(monkeypatch, calls, evidence_groq_run):
    set_c1(monkeypatch, calls, 0.9)
    state, _ = run()
    fo = state["final_output"]
    assert fo["evidence_fallback_used"] is False and fo["evidence_groq"]["fallbacks"] == 0
    assert fo["evidence_groq"]["groq_calls"] == 4 * len(evidence_groq_run["order"])
    assert not any(n.startswith("Evidence Scanner:") for n in fo["notes"])


def test_evidence_groq_counts_add_up_across_loops(monkeypatch, calls, evidence_groq_run):
    set_c1(monkeypatch, calls, 0.3)  # 2 loops; each disease still scanned once
    evidence_groq_run["fallback_for"] = {"*"}
    state, _ = run()
    n = len(evidence_groq_run["order"])
    assert state["loop_count"] == 2 and n == len(set(evidence_groq_run["order"]))
    assert state["final_output"]["evidence_groq"]["groq_calls"] == 4 * n
    assert state["final_output"]["evidence_groq"]["fallbacks"] == 4 * n


def test_evidence_fallback_false_on_skipped_case(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(httpx.Response(403, json={"error": {"message": ACCESS_DENIED}}))
    state, _ = run()  # skipped after the Clarifier
    assert state["final_output"]["evidence_fallback_used"] is False
    assert state["final_output"]["evidence_groq"]["groq_calls"] == 0


def test_evidence_fallback_false_in_replay(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    state, _ = run()
    assert state["final_output"]["evidence_fallback_used"] is False
    assert state["evidence_groq"] == {}


# ---------------------------------------------------------------- Groq via Cloudflare AI Gateway (no network)

GW = "https://gateway.ai.cloudflare.com/v1/acc123/cdss/groq"


@pytest.mark.parametrize("env, expected", [
    ({"GROQ_VIA_GATEWAY": "true", "CF_ACCOUNT_ID": "acc123", "CF_AI_GATEWAY_ID": "cdss"}, (GW, "gateway cdss")),
    ({"GROQ_VIA_GATEWAY": "1", "CF_ACCOUNT_ID": " acc123 ", "CF_AI_GATEWAY_ID": "cdss"}, (GW, "gateway cdss")),
    ({"GROQ_VIA_GATEWAY": "false", "CF_ACCOUNT_ID": "acc123", "CF_AI_GATEWAY_ID": "cdss"}, ("", "direct")),
    ({"CF_ACCOUNT_ID": "acc123", "CF_AI_GATEWAY_ID": "cdss"}, ("", "direct")),
    ({"GROQ_VIA_GATEWAY": "true", "CF_AI_GATEWAY_ID": "cdss"},
     ("", "direct (GROQ_VIA_GATEWAY=true but CF_ACCOUNT_ID / CF_AI_GATEWAY_ID missing)")),
])
def test_groq_route_choice(env, expected):
    assert config.resolve_groq_gateway(env) == expected


@pytest.fixture
def gateway_clarifier(monkeypatch, tmp_path):
    """Real Clarifier code, gateway route, fake gateway server that passes Groq's rate-limit headers on."""
    monkeypatch.setattr(config, "GROQ_GATEWAY_BASE", GW)
    monkeypatch.setattr(config, "GROQ_ROUTE", "gateway cdss")
    monkeypatch.setattr(config, "GROQ_LIMITS_FILE", tmp_path / "groq_limits.json")
    monkeypatch.setattr(wrappers, "GROQ_LIMITS", {})
    monkeypatch.setattr(wrappers, "CLARIFIER_STATS", {"groq_requests": 0, "calls": []})
    monkeypatch.setattr(wrappers, "_REAL", dict(wrappers._REAL))
    monkeypatch.setattr(wrappers, "_clarifier_commit", lambda: "51dacaf")
    monkeypatch.setattr(wrappers, "_sleep", lambda s: None)
    monkeypatch.setattr(config, "CLARIFIER_FAKE", False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-123")
    seen = []

    def gateway(request):
        seen.append(request)
        text = _json.loads(request.content)["messages"][-1]["content"]
        return _first_word_reply(text).__class__(
            200, headers={"x-ratelimit-remaining-tokens": "7922", "x-ratelimit-reset-tokens": "585ms",
                          "x-ratelimit-remaining-requests": "999", "x-ratelimit-limit-requests": "1000",
                          "cf-aig-cache-status": "MISS"},
            content=_first_word_reply(text).content)
    info = wrappers._load_clarifier(transport=httpx.MockTransport(gateway))
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "clarifier": "real"})
    return _NS(info=info, seen=seen)


def test_clarifier_uses_gateway_url_skip_cache_and_keeps_pacing(gateway_clarifier):
    out = wrappers.run_clarifier("chest pain since morning")
    assert out["symptoms"] == ["fake symptom"]
    request = gateway_clarifier.seen[0]
    assert str(request.url) == f"{GW}/openai/v1/chat/completions"  # the gateway accepts the SDK path as is
    assert request.headers["cf-aig-skip-cache"] == "true"
    assert request.headers["authorization"] == "Bearer test-groq-key-123"
    assert wrappers.GROQ_LIMITS["remaining_tokens"] == 7922 and wrappers.GROQ_LIMITS["remaining_requests"] == 999
    assert wrappers._REAL["clarifier"].client.max_retries == 0
    assert gateway_clarifier.info["groq_route"] == "gateway cdss"


def test_clarifier_direct_route_unchanged(clarifier):
    wrappers.run_clarifier("chest pain")
    assert "api.groq.com" in str(wrappers._REAL["clarifier"].client.base_url)
    assert "cf-aig-skip-cache" not in wrappers._REAL["clarifier"].client.default_headers


@pytest.fixture
def her_groq_base(monkeypatch):
    """Fake tools.query_generator / tools.summarizer with her GROQ_BASE (no network)."""
    mods = {}
    for name in wrappers.GROQ_MODULES:
        mod = _types.ModuleType(name)
        mod.GROQ_BASE = config.GROQ_DIRECT_BASE
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    return mods


def test_evidence_groq_base_replaced_in_memory_for_gateway(monkeypatch, her_groq_base):
    monkeypatch.setattr(config, "GROQ_GATEWAY_BASE", GW)
    monkeypatch.setattr(config, "GROQ_ROUTE", "gateway cdss")
    assert wrappers._apply_groq_route() == {"groq_route": "gateway cdss"}
    assert all(m.GROQ_BASE == f"{GW}/chat/completions" for m in her_groq_base.values())


def test_evidence_groq_base_untouched_when_direct(monkeypatch, her_groq_base):
    monkeypatch.setattr(config, "GROQ_GATEWAY_BASE", "")
    monkeypatch.setattr(config, "GROQ_ROUTE", "direct")
    assert wrappers._apply_groq_route() == {"groq_route": "direct"}
    assert all(m.GROQ_BASE == config.GROQ_DIRECT_BASE for m in her_groq_base.values())


def test_her_groq_files_on_disk_unchanged():
    for name in ("query_generator.py", "summarizer.py"):
        text = (config.EVIDENCE_DIR / "tools" / name).read_text(encoding="utf-8")
        assert f'GROQ_BASE = "{config.GROQ_DIRECT_BASE}"' in text


@pytest.mark.parametrize("gateway, header_sent", [(GW, True), ("", False)])
def test_evidence_groq_skip_cache_header_and_limits_through_gateway(monkeypatch, tmp_path, gateway, header_sent):
    monkeypatch.setattr(config, "GROQ_GATEWAY_BASE", gateway)
    monkeypatch.setattr(config, "GROQ_LIMITS_FILE", tmp_path / "groq_limits.json")
    monkeypatch.setattr(wrappers, "GROQ_LIMITS", {})
    monkeypatch.setattr(wrappers, "EVIDENCE_STATS", {"calls": [wrappers._new_evidence_record("x")]})
    sent = {}
    response = _NS(status_code=200, text="{}", headers={"x-ratelimit-remaining-tokens": "1234",
                                                         "x-ratelimit-reset-tokens": "2s"})

    def post(url, **kwargs):
        sent.update(kwargs)
        return response
    proxy = wrappers._CountingRequests(_NS(post=post), "groq")
    proxy.post("u", headers={"Authorization": "Bearer k"}, json={})
    assert sent["headers"]["Authorization"] == "Bearer k"  # her header kept
    assert ("cf-aig-skip-cache" in sent["headers"]) is header_sent
    assert wrappers.GROQ_LIMITS["remaining_tokens"] == 1234  # pacing still sees the limits


def test_non_groq_sources_never_get_gateway_header(monkeypatch):
    monkeypatch.setattr(config, "GROQ_GATEWAY_BASE", GW)
    monkeypatch.setattr(wrappers, "EVIDENCE_STATS", {"calls": [wrappers._new_evidence_record("x")]})
    sent = {}
    proxy = wrappers._CountingRequests(_NS(get=lambda url, **k: sent.update(k) or _NS(status_code=200)), "pubmed")
    proxy.get("u", params={"q": 1})
    assert "headers" not in sent


# ---------------------------------------------------------------- agent on/off switches (ablation; replay, no network)

from cdss_pipeline import graph as _graph_mod


def run_off(disabled, text="test patient"):
    from cdss_pipeline import run_pipeline
    return run_pipeline.run(text, disabled)


def _first_pass(path):
    return path[:path.index("optimizer") + 1] if "optimizer" in path else path


@pytest.mark.parametrize("text, expected", [
    ("", []), ("rag", ["rag"]), (" Optimizer , evidence ", ["evidence", "optimizer"]),
    (["kg", "kg"], ["kg"]), ("rag,evidence,optimizer", ["evidence", "optimizer", "rag"]),
])
def test_parse_disabled(text, expected):
    assert config.parse_disabled(text) == expected


def test_parse_disabled_unknown_name():
    with pytest.raises(config.SwitchError, match="Unknown agent.*fusion"):
        config.parse_disabled("rag,fusion")


def test_rag_and_kg_disabled_stops_before_running(monkeypatch, capsys):
    from cdss_pipeline import run_pipeline
    started = []
    monkeypatch.setattr(run_pipeline, "startup", lambda *a, **k: started.append(1) or {})
    monkeypatch.setattr("sys.argv", ["run_pipeline", "--disable", "kg,rag", "--text", "x"])
    with pytest.raises(SystemExit, match="At least one of rag or kg must be enabled - nothing would suggest diagnoses."):
        run_pipeline.main()
    monkeypatch.setattr(config, "DISABLED_FROM_ENV", "rag, kg")  # same rule for CDSS_DISABLE
    monkeypatch.setattr("sys.argv", ["run_pipeline", "--text", "x"])
    with pytest.raises(SystemExit, match="At least one of rag or kg must be enabled"):
        run_pipeline.main()
    assert started == []  # nothing loaded, nothing run
    with pytest.raises(config.SwitchError, match="At least one of rag or kg"):
        run_off(["rag", "kg"])


def test_full_system_records_no_disabled_agents(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    state, _ = run_off([])
    assert state["final_output"]["disabled_agents"] == []
    assert not any(n.startswith("Ablation run") for n in state["final_output"]["notes"])


def test_rag_disabled(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)  # low C1 would loop in the full system
    state, path = run_off(["rag"])
    fo = state["final_output"]
    assert "vector_rag" not in path and calls["vector_rag_top_k"] == []
    assert calls["knowledge_graph"] == 1 and calls["fusion"] == 1
    assert state["loop_count"] == 0 and state["rag_confidence"] is None  # no loop without RAG
    assert calls["evidence"] == _graph_mod._kg_disease_names(state)  # KG candidates only
    assert fo["disabled_agents"] == ["rag"] and "Ablation run: disabled rag - not the full system" in fo["notes"]
    assert "Vector RAG disabled: C1 and fused confidence not available" in fo["notes"]
    assert "Vector RAG unavailable" not in fo["notes"]
    conf = fo["confidence"]
    assert conf["C1"] is None and conf["fused"] is None and conf["passes_threshold"] is None
    assert conf["C2"] is not None and "C1 (Vector RAG disabled)" in conf["not_available"]
    assert state["fusion_output"]["confidence"]["confidence_1"]["score"] == 0.3  # her raw number kept, not used
    assert fo["most_likely_diagnosis"] is None and fo["supported_by_both"] == []


def test_rag_disabled_skips_qdrant_check_and_loader(monkeypatch):
    from cdss_pipeline import run_pipeline
    monkeypatch.setattr(config, "AGENT_MODES", {a: "replay" for a in config.AGENTS} | {"vector_rag": "real"})
    monkeypatch.setattr(wrappers, "_qdrant_client", lambda *a, **k: _FakeQdrant(reachable=False))
    seen = []
    monkeypatch.setattr(wrappers, "init_agents", lambda **k: seen.append(k) or {})
    report = run_pipeline.startup(["rag"])  # Qdrant is "down", but RAG is off: no stop
    assert seen == [{"disabled": ["rag"]}] and report["disabled_agents"] == ["rag"]


def test_init_agents_skips_disabled_agents(monkeypatch):
    monkeypatch.setattr(config, "AGENT_MODES", {a: "replay" for a in config.AGENTS}
                        | {"vector_rag": "real", "knowledge_graph": "real"})
    monkeypatch.setattr(wrappers, "_REAL", {})
    monkeypatch.setitem(wrappers._LOADERS, "vector_rag", lambda: (_ for _ in ()).throw(AssertionError("loaded")))
    monkeypatch.setitem(wrappers._LOADERS, "knowledge_graph", lambda: {"ok": True})
    report = wrappers.init_agents(disabled=["rag"])
    assert "vector_rag" not in report and report["knowledge_graph"]["ok"] is True


def test_kg_disabled(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    state, path = run_off(["kg"])
    fo = state["final_output"]
    assert "knowledge_graph" not in path and calls["knowledge_graph"] == 0
    assert state["kg_safety_messages"] == [] and fo["rare_serious_conditions"] == []
    assert calls["evidence"] == _graph_mod._rag_disease_names(state)  # RAG candidates only
    assert fo["most_likely_diagnosis"] is not None
    conf = fo["confidence"]
    assert conf["C2"] is None and conf["fused"] is None and conf["C1"] == 0.9
    assert any(n.startswith("Knowledge Graph disabled") for n in fo["notes"])
    assert "Knowledge Graph unavailable" not in fo["notes"] and fo["disabled_agents"] == ["kg"]


def test_kg_disabled_still_loops_on_rag(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    state, _ = run_off(["kg"])
    assert calls["vector_rag_top_k"] == [5, 10, 15] and state["loop_count"] == 2


def test_evidence_disabled(monkeypatch, calls):
    seen = []

    def fusion(medico_state):
        seen.append(medico_state)
        calls["fusion"] += 1
        out = wrappers._sample("fusion_case1.json")
        out["confidence"]["confidence_1"]["score"] = 0.9
        out["fallback_used"] = False
        return out
    monkeypatch.setattr(wrappers, "run_fusion", fusion)
    state, path = run_off(["evidence"])
    fo = state["final_output"]
    assert "evidence_scanner" not in path and calls["evidence"] == [] and state["evidence_output"] == {}
    assert calls["fusion"] == 1 and calls["knowledge_graph"] == 1 and calls["vector_rag_top_k"] == [5]
    assert seen[0]["raw_vector_web_evidence"] == {"hypothesis_results": []}  # no web evidence at all
    assert seen[0]["raw_graph_web_evidence"] == {"hypothesis_results": []}
    assert fo["confidence"]["C1"] == 0.9 and fo["confidence"]["C2"] is not None  # kept, with a note
    assert ("Evidence Scanner disabled: C1/C2 computed without web evidence (web term 0, default agreement)"
            in fo["notes"])
    assert fo["evidence_fallback_used"] is False


def test_evidence_disabled_fusion_runs_once_per_pass_with_loops(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    state, path = run_off(["evidence"])
    assert calls["vector_rag_top_k"] == [5, 10, 15] and calls["fusion"] == 3
    assert calls["knowledge_graph"] == 1 and path.count("fusion") == 3


def test_optimizer_disabled(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    state, path = run_off(["optimizer"])
    assert state["loop_count"] == 0 and calls["vector_rag_top_k"] == [5] and calls["fusion"] == 1
    assert "Optimizer loop disabled: single pass" in state["final_output"]["notes"]
    assert state["final_output"]["confidence"]["C1"] == 0.3  # C1 still reported


@pytest.mark.parametrize("disabled, first_pass", [
    (["rag", "evidence"], ["clarifier", "knowledge_graph", "fusion", "optimizer"]),
    (["kg", "evidence"], ["clarifier", "vector_rag", "fusion", "optimizer"]),
    (["rag", "optimizer"], ["clarifier", "knowledge_graph", "evidence_scanner", "fusion", "optimizer"]),
    (["kg", "optimizer"], ["clarifier", "vector_rag", "evidence_scanner", "fusion", "optimizer"]),
    (["evidence", "optimizer"], ["clarifier", "vector_rag", "knowledge_graph", "fusion", "optimizer"]),
    (["rag", "evidence", "optimizer"], ["clarifier", "knowledge_graph", "fusion", "optimizer"]),
    (["kg", "evidence", "optimizer"], ["clarifier", "vector_rag", "fusion", "optimizer"]),
])
def test_switch_combinations(monkeypatch, calls, disabled, first_pass):
    set_c1(monkeypatch, calls, 0.3)
    state, path = run_off(disabled)
    assert sorted(_first_pass(path)) == sorted(first_pass)
    assert path[-1] == "final_output" and calls["fusion"] >= 1
    fo = state["final_output"]
    assert fo["disabled_agents"] == sorted(disabled)
    assert f"Ablation run: disabled {', '.join(sorted(disabled))} - not the full system" in fo["notes"]
    if "optimizer" in disabled or "rag" in disabled:
        assert state["loop_count"] == 0
    if "rag" in disabled:
        assert fo["confidence"]["C1"] is None and calls["vector_rag_top_k"] == []
    if "kg" in disabled:
        assert fo["confidence"]["C2"] is None and calls["knowledge_graph"] == 0
    if "evidence" in disabled:
        assert calls["evidence"] == []


def test_disabled_agents_recorded_on_clarifier_skip(monkeypatch, calls, clarifier):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(_groq_error(400))
    state, path = run_off(["rag"])
    assert path == ["clarifier", "final_output"]
    assert state["final_output"]["disabled_agents"] == ["rag"]
    assert state["final_output"]["analysis_status"] == CLARIFIER_ERROR


def test_run_pipeline_disable_option_end_to_end(monkeypatch, calls, capsys):
    from cdss_pipeline import run_pipeline
    set_c1(monkeypatch, calls, 0.3)
    seen = []
    monkeypatch.setattr(wrappers, "init_agents", lambda **k: seen.append(k) or {})
    monkeypatch.setattr("sys.argv", ["run_pipeline", "--disable", "rag", "--json", "--text", "chest pain"])
    run_pipeline.main()
    item = _json.loads(capsys.readouterr().out)
    assert seen == [{"disabled": ["rag"]}]
    assert item["final_output"]["disabled_agents"] == ["rag"] and item["final_output"]["confidence"]["C1"] is None


def test_confidence_hides_only_disabled_parts():
    out = {"confidence": {"confidence_1": {"score": 0.4}, "confidence_2": {"score": 0.08},
                          "final_score": 0.2, "passes_threshold": False}}
    full = _graph_mod._confidence(out)
    assert (full["C1"], full["C2"], full["fused"]) == (0.4, 0.08, 0.2) and "not_available" not in full
    kg_off = _graph_mod._confidence(out, {"kg"})
    assert (kg_off["C1"], kg_off["C2"], kg_off["fused"]) == (0.4, None, None)
    ev_off = _graph_mod._confidence(out, {"evidence", "optimizer"})
    assert (ev_off["C1"], ev_off["C2"], ev_off["fused"]) == (0.4, 0.08, 0.2)


# ---------------------------------------------------------------- KG symptom matching (fake mini graph, his code untouched)

import networkx as _nx

_REL = "disease_phenotype_positive"


def _mini_graph():
    G = _nx.Graph()
    phen = ["Knee pain", "Joint swelling", "Joint stiffness", "Stiff knee", "Pain", "Arthralgia",
            "Generalized morning stiffness"]
    for p in phen:
        G.add_node(p, type="effect/phenotype")
    G.add_node("rheumatoid arthritis", type="disease")
    G.add_node("Osgood-Schlatter disease", type="disease")
    for p in ["Knee pain", "Joint swelling", "Joint stiffness", "Pain", "Arthralgia", "Generalized morning stiffness"]:
        G.add_edge(p, "rheumatoid arthritis", relation=_REL)  # "Stiff knee" has NO disease
    G.add_edge("Knee pain", "Osgood-Schlatter disease", relation=_REL)
    return G


def _diseases_for(G, node, relation):
    return [n for n in G.neighbors(node) if G.nodes[n].get("type") == "disease"]


_SYN = {"knee pain": "Knee pain", "joint swelling": "Joint swelling", "joint stiffness": "Joint stiffness",
        "knee stiffness": "Stiff knee", "joint pain": "Arthralgia", "pain": "Pain"}


@pytest.fixture
def lookup():
    return wrappers.make_symptom_lookup(_mini_graph(), _SYN, _diseases_for, _REL)


def _ent(text, site=None, status="present"):
    return {"text": text, "type": "symptom", "status": status, "evidence_text": text, "body_site": site}


def _by_symptom(records):
    return {r["symptom"]: (r["graph_term"], r["rule"]) for r in records}


def test_lookup_only_linked_phenotypes(lookup):
    assert lookup("knee pain") == "Knee pain" and lookup("JOINT PAIN") == "Arthralgia"
    assert lookup("knee stiffness") is None  # "Stiff knee" exists but has 0 diseases
    assert lookup("rheumatoid arthritis") is None  # disease node, not a phenotype
    assert lookup("knee swelling") is None  # not in the graph: never invented


def test_match_knee_case_own_body_site(lookup):
    symptoms = ["knee pain", "knee swelling", "morning stiffness"]
    entities = [_ent(s, "knees") for s in symptoms]
    records = wrappers.match_symptoms(symptoms, entities, lookup)
    assert _by_symptom(records) == {"knee pain": ("Knee pain", "exact"),
                                    "knee swelling": ("Joint swelling", "joint_general"),
                                    "morning stiffness": ("Joint stiffness", "joint_general")}
    stiff = records[2]
    assert stiff["tried"][:2] == ["morning stiffness", "knee stiffness"]  # "Stiff knee" skipped (0 diseases)
    assert stiff["body_site"] == "knees" and stiff["case_body_site"] is None


def test_match_case_body_site_rule(lookup):
    symptoms = ["knee pain", "knee swelling", "morning stiffness"]
    entities = [_ent("knee pain", "both knees"), _ent("knee swelling", "knees"), _ent("morning stiffness")]
    records = wrappers.match_symptoms(symptoms, entities, lookup)
    assert _by_symptom(records)["morning stiffness"] == ("Joint stiffness", "case_body_site")
    assert records[2]["case_body_site"] == "both knees" and records[2]["body_site"] is None


def test_case_body_site_needs_a_joint_site(lookup):
    symptoms = ["chest pain", "morning stiffness"]
    entities = [_ent("chest pain", "chest"), _ent("morning stiffness")]
    records = wrappers.match_symptoms(symptoms, entities, lookup)
    assert _by_symptom(records) == {"chest pain": ("Pain", "plain_word"),  # non-joint site: no joint form
                                    "morning stiffness": (None, "unmatched")}
    assert "joint stiffness" not in records[1]["tried"]


def test_absent_entities_do_not_give_a_case_site(lookup):
    records = wrappers.match_symptoms(["morning stiffness"], [_ent("knee pain", "knee", status="absent"),
                                                              _ent("morning stiffness")], lookup)
    assert records[0]["rule"] == "unmatched"


def test_matched_terms_always_exist_in_graph(lookup):
    G = _mini_graph()
    symptoms = ["knee pain", "knee swelling", "joint pain", "wrist swelling", "fever", "pain"]
    entities = [_ent("wrist swelling", "right wrist")]
    for r in wrappers.match_symptoms(symptoms, entities, lookup):
        assert r["rule"] in wrappers.MATCH_RULES + ("unmatched",)
        assert r["graph_term"] is None or (r["graph_term"] in G and _diseases_for(G, r["graph_term"], _REL))


def test_run_knowledge_graph_uses_matched_terms(monkeypatch, lookup):
    seen = []

    def fake_kg(clarifier_output, top_k=5):
        seen.append(clarifier_output)
        return {"status": "success", "disease_prediction": {
            "candidate_diseases": [{"disease": "rheumatoid arthritis", "matched_symptom_count": 3,
                                    "coverage_fraction": 0.25, "matched_symptoms": clarifier_output["symptoms"]}],
            "metadata": {"symptoms_matched_count": 3}}, "medication_safety": {"warnings": []}}
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "knowledge_graph": "real"})
    monkeypatch.setattr(wrappers, "_REAL", {"knowledge_graph": fake_kg, "kg_lookup": lookup})
    clar = {"clinical_text": "t", "diseases": [], "symptoms": ["knee pain", "knee swelling", "joint swelling",
                                                               "morning stiffness", "fever"],
            "medications": ["aspirin"], "tests": []}
    out = wrappers.run_knowledge_graph(clar, [_ent("knee swelling", "knee")])
    # knee swelling has site "knee" -> case-level joint site for morning stiffness; fever stays unmatched
    assert seen[0]["symptoms"] == ["Knee pain", "Joint swelling", "Joint stiffness"]  # deduped graph names only
    assert seen[0]["medications"] == ["aspirin"]
    assert out["unmatched_symptoms"] == ["fever"]
    assert [m["rule"] for m in out["symptom_matching"]] == ["exact", "joint_general", "exact", "case_body_site",
                                                            "unmatched"]
    assert out["graph_confidence"] == 0.4  # e = 3/3, c = 0.25 -> 2*0.25/1.25
    assert out["graph_confidence_details"]["disease"] == "rheumatoid arthritis"


def test_knowledge_graph_node_passes_clarifier_entities(monkeypatch):
    seen = []
    monkeypatch.setattr(wrappers, "run_knowledge_graph", lambda c, e=None: seen.append(e) or {"status": "error"})
    _graph_mod.knowledge_graph_node({"clarifier_output": {"symptoms": ["x"]},
                                     "clarifier_details": {"entities": [_ent("x", "knee")]}})
    assert seen == [[_ent("x", "knee")]]


# ---------------------------------------------------------------- graph_confidence

@pytest.mark.parametrize("matched, recognised, coverage, expected", [
    (2, 3, 0.5, 0.571), (3, 3, 0.071, 0.133), (1, 1, 1.0, 1.0), (1, 4, 0.0, 0.0),
])
def test_graph_confidence_formula(matched, recognised, coverage, expected):
    kg = {"disease_prediction": {"candidate_diseases": [{"disease": "d", "matched_symptom_count": matched,
                                                         "coverage_fraction": coverage}],
                                 "metadata": {"symptoms_matched_count": recognised}}}
    value, details = wrappers.graph_confidence(kg)
    assert value == expected and details["recognised"] == recognised


@pytest.mark.parametrize("kg", [
    {"disease_prediction": {"candidate_diseases": [], "metadata": {"symptoms_matched_count": 0}}},
    {"disease_prediction": {"candidate_diseases": [], "metadata": {"symptoms_matched_count": 2}}},
    {"disease_prediction": None}, {},
])
def test_graph_confidence_none_not_zero(kg):
    assert wrappers.graph_confidence(kg)[0] is None


def test_replay_kg_gets_graph_confidence_and_final_output(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    state, _ = run()
    assert state["kg_output"]["graph_confidence"] is not None
    assert state["final_output"]["graph_confidence"] == state["kg_output"]["graph_confidence"]
    state, _ = run_off(["kg"])
    assert state["final_output"]["graph_confidence"] is None


# ---------------------------------------------------------------- Medico can read ONLY our graph_confidence (her real code)

@pytest.fixture
def medico_scoring(monkeypatch):
    monkeypatch.syspath_prepend(str(config.MEDICO_DIR))
    import confidence_scoring
    import evidence_adapter
    return _NS(render=evidence_adapter.to_plain_text, score=confidence_scoring.extract_graph_score)


def _her_regex_hits(text):
    import re as _re
    return _re.findall(r"(?:confidence|score|fidelity)[:\s]+([01](?:\.\d+)?)", text, _re.IGNORECASE)


def _tricky_kg(graph_conf):
    kg = copy.deepcopy(wrappers._load_json(str(config.KG_SAMPLE)))
    kg["graph_confidence"], kg["graph_confidence_details"] = graph_conf, {"e": 1.0, "coverage": 0.071}
    kg["match_score"] = 0.9                                   # a key Medico could read
    kg["extra"] = {"confidence": 1, "note": "model score: 0.77 and fidelity 1"}  # nested key + text
    return kg


def _medico_text(kg, messages, render):
    """What her graph.py builds: rendered KG evidence + the safety messages JSON appended."""
    view = wrappers.medico_kg_view(kg)
    safety = _json.dumps(wrappers._scrub_scores(messages), ensure_ascii=False, indent=2)
    return f"{render(view)}\n\nKG SAFETY MESSAGES:\n{safety}"


_TRICKY_MESSAGES = [{"type": "contraindication", "drug": "x", "target": "y", "severity": "major",
                     "message": "x is contraindicated (confidence: 0.99)"}]


def test_medico_reads_only_our_graph_confidence(medico_scoring):
    text = _medico_text(_tricky_kg(0.133), _TRICKY_MESSAGES, medico_scoring.render)
    assert _her_regex_hits(text) == ["0.133"]  # exactly one readable value: ours
    assert medico_scoring.score(text) == 0.133


def test_medico_reads_nothing_when_graph_confidence_none(medico_scoring):
    text = _medico_text(_tricky_kg(None), _TRICKY_MESSAGES, medico_scoring.render)
    assert _her_regex_hits(text) == []
    assert medico_scoring.score(text) == 0.0  # her structural part stays 0.0


def test_medico_view_keeps_kg_output_unchanged():
    kg = _tricky_kg(0.5)
    before = copy.deepcopy(kg)
    view = wrappers.medico_kg_view(kg)
    assert kg == before and list(view)[0] == "graph_confidence" and "match_score_value" in view
    assert wrappers.medico_kg_view(None) == {}


# ---------------------------------------------------------------- labels when RAG is off

def test_rag_off_labels_fusion_report_and_optimizer_note(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.3)
    state, _ = run_off(["rag"])
    fo, raw = state["final_output"], state["fusion_output"]
    for key, raw_key in (("fusion_report", "fusion_report"), ("optimizer_note", "optimizer_response")):
        assert fo[key].startswith(_graph_mod.RAG_OFF_LABEL) and fo[key].endswith(raw[raw_key])
        assert not raw[raw_key].startswith(_graph_mod.RAG_OFF_LABEL)  # her output itself unchanged
    state, _ = run_off([])
    assert not state["final_output"]["fusion_report"].startswith(_graph_mod.RAG_OFF_LABEL)


# ---------------------------------------------------------------- short (default) and verbose views

def _main(monkeypatch, capsys, *args):
    from cdss_pipeline import run_pipeline
    monkeypatch.setattr(wrappers, "init_agents", lambda **k: {})
    monkeypatch.setattr("sys.argv", ["run_pipeline", *args, "--text", "chest pain since morning"])
    run_pipeline.main()
    return capsys.readouterr().out


def test_short_view_by_default(monkeypatch, calls, capsys):
    set_c1(monkeypatch, calls, 0.9)
    out = _main(monkeypatch, capsys)
    for part in ("Case: chest pain since morning", "Analysis status: analysed", "Most likely diagnosis (Vector RAG):",
                 "Rare / serious conditions (KG):", "Medication safety:", "Supported by both:",
                 "Confidence: C1 0.9 | C2", "graph_confidence (KG)", "Notes:", "Errors: none"):
        assert part in out, part
    for verbose_only in ("=== Clarifier details", "=== Path", "=== Fusion report", "=== Startup loading",
                         "=== Evidence Scanner Groq", "=== KG symptom matching"):
        assert verbose_only not in out, verbose_only
    assert "Disabled agents" not in out


def test_verbose_view_prints_everything(monkeypatch, calls, capsys):
    set_c1(monkeypatch, calls, 0.9)
    out = _main(monkeypatch, capsys, "--verbose")
    for part in ("=== Startup loading", "=== Clarifier details", "=== Path", "=== KG symptom matching",
                 "=== Evidence Scanner Groq", "=== Fusion report", "=== Optimizer note", "graph_confidence (KG)"):
        assert part in out, part


def test_short_view_rag_off_says_not_available(monkeypatch, calls, capsys):
    set_c1(monkeypatch, calls, 0.3)
    out = _main(monkeypatch, capsys, "--disable", "rag")
    assert "Disabled agents (ablation): rag" in out
    assert "Confidence: C1 not available | C2" in out and "fused not available" in out


def test_short_view_warning_lines(monkeypatch, calls, evidence_groq_run, capsys):
    set_c1(monkeypatch, calls, 0.9)
    evidence_groq_run["fallback_for"] = {"*"}
    real_kg = wrappers.run_knowledge_graph

    def kg(c, e=None):
        out = real_kg(c, e)
        out["unmatched_symptoms"] = ["morning stiffness"]
        return out
    monkeypatch.setattr(wrappers, "run_knowledge_graph", kg)
    monkeypatch.setattr(wrappers, "run_vector_rag", lambda c, k: (_ for _ in ()).throw(RuntimeError("rag down")))
    out = _main(monkeypatch, capsys)
    assert "! Evidence Scanner used Groq fallbacks" in out
    assert "! Symptoms not found in the KG: morning stiffness" in out
    assert "! 1 agent error(s) - see Errors" in out and "vector_rag: RuntimeError: rag down" in out


def test_short_view_clarifier_error_warning(monkeypatch, calls, clarifier, capsys):
    set_c1(monkeypatch, calls, 0.9)
    clarifier.replies.append(_groq_error(401))
    out = _main(monkeypatch, capsys)
    assert "! Clarifier failed - case not analysed (Groq authentication failed (HTTP 401)" in out
    assert "Analysis status: clarifier_error" in out


def test_short_view_startup_load_error_warning(monkeypatch, calls, capsys):
    from cdss_pipeline import run_pipeline
    set_c1(monkeypatch, calls, 0.9)
    monkeypatch.setattr(wrappers, "init_agents", lambda **k: {"fusion": {"error": "ImportError: x"}})
    monkeypatch.setattr("sys.argv", ["run_pipeline", "--text", "t"])
    run_pipeline.main()
    assert "! fusion failed to load: ImportError: x" in capsys.readouterr().out


# ---------------------------------------------------------------- KG ranking_top10 (scoring only)

def test_ranking_top10_real_mode_from_his_functions_not_given_to_medico(monkeypatch, lookup):
    asked = []

    def fake_rank(symptoms, top_k=10):
        asked.append((list(symptoms), top_k))
        return {"candidate_diseases": [{"disease": f"d{i}", "matched_symptom_count": 1, "coverage_fraction": 0.1,
                                        "matched_symptoms": ["x"]} for i in range(12)]}

    def fake_kg(clarifier_output, top_k=5):
        return {"status": "success", "disease_prediction": {
            "candidate_diseases": [{"disease": "d0", "matched_symptom_count": 1, "coverage_fraction": 0.1}],
            "metadata": {"symptoms_matched_count": 1}}, "medication_safety": {"warnings": []}}
    monkeypatch.setattr(config, "AGENT_MODES", {**config.AGENT_MODES, "knowledge_graph": "real"})
    monkeypatch.setattr(wrappers, "_REAL", {"knowledge_graph": fake_kg, "kg_lookup": lookup, "kg_rank": fake_rank})
    out = wrappers.run_knowledge_graph({"symptoms": ["knee pain", "fever"]}, [])
    assert asked == [(["Knee pain"], 10)]  # same matched graph names as the KG got
    assert [r["disease"] for r in out["ranking_top10"]] == [f"d{i}" for i in range(10)]
    assert set(out["ranking_top10"][0]) == {"disease", "matched_symptom_count", "coverage_fraction"}
    assert "ranking_top10" not in wrappers.medico_kg_view(out)
    assert len(out["disease_prediction"]["candidate_diseases"]) == 1  # his top list used downstream unchanged


def test_ranking_top10_in_final_output_and_rag_candidates(monkeypatch, calls):
    set_c1(monkeypatch, calls, 0.9)
    state, _ = run()
    fo = state["final_output"]
    kg_top = [c["disease"] for c in state["kg_output"]["disease_prediction"]["candidate_diseases"]]
    assert fo["kg_ranking_top10"] == kg_top[:10]  # replay: his list as given
    assert fo["rag_candidates"][0] == fo["most_likely_diagnosis"]["condition"]
    assert calls["evidence"][:3] != [] and len(fo["rag_candidates"]) >= 1
    state, _ = run_off(["kg"])
    assert state["final_output"]["kg_ranking_top10"] == []
