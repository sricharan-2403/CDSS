"""Offline tests for the batch runner and the scoring script (fake pipeline results, no network)."""

import csv
import json
from pathlib import Path

import pytest

from cdss_pipeline import config, run_batch, run_pipeline, score, wrappers


# ---------------------------------------------------------------- helpers

def _case(case_id, set_="medqa", gold="Myocardial infarction", synonyms=(), flags=None, entities=None, text=None):
    return {"case_id": case_id, "set": set_, "input_text": text or f"text of {case_id}", "gold_diagnosis": gold,
            "gold_synonyms": list(synonyms), "gold_differential": [], "gold_med_flags": flags or [],
            "gold_entities": entities or {"symptoms": [], "absent": [], "medications": [], "diseases": []},
            "notes": ""}


def _fo(rag=(), kg=(), status="analysed", fallback=False, disabled=(), warnings=(), report="", both=(),
        conf=None, graph_conf=None):
    rag = list(rag)
    return {"analysis_status": status, "disabled_agents": list(disabled),
            "most_likely_diagnosis": {"condition": rag[0], "justification": "j"} if rag else None,
            "rag_candidates": rag, "kg_ranking_top10": list(kg), "supported_by_both": list(both),
            "medication_safety": {"warnings": list(warnings), "treatment_options": []},
            "confidence": conf or {"C1": 0.5, "C2": 0.4, "fused": 0.45}, "graph_confidence": graph_conf,
            "evidence_fallback_used": fallback, "fusion_report": report, "notes": []}


def _item(case, fo, loops=0, clarifier_output=None, entities=(), seconds=10.0, groq=4, openrouter=3):
    return {"id": case["case_id"], "final_output": fo, "case": case, "loop_count": loops,
            "clarifier_output": clarifier_output if clarifier_output is not None else {
                "clinical_text": case["input_text"], "symptoms": ["x"], "medications": [], "diseases": [],
                "tests": []},
            "clarifier_details": {"entities": list(entities)},
            "batch": {"seconds": seconds, "groq_calls": groq, "openrouter_calls": openrouter, "errors": []}}


def _write_jsonl(path, cases):
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- input checks

def test_example_cases_file_is_valid():
    cases = run_batch.load_cases(config.PACKAGE_DIR / "cases" / "example_cases.jsonl")
    assert [c["set"] for c in cases] == ["medqa", "medsafety", "clarifier"]


@pytest.mark.parametrize("lines, message", [
    (['{"case_id": "a", "set": "medqa", "input_text": "t"}', "not json"], "line 2: not valid JSON"),
    (['{"case_id": "a", "set": "medqa", "input_text": "t"}', '{"case_id": "a", "set": "medqa", "input_text": "u"}'],
     "duplicate case_id 'a'"),
    (['{"case_id": "a", "set": "medqa", "input_text": " "}'], "input_text is missing"),
    (['{"case_id": "a", "set": "pubmed", "input_text": "t"}'], "set must be one of"),
    (['{"set": "medqa", "input_text": "t"}'], "case_id is missing"),
    (['{"case_id": "a/b", "set": "medqa", "input_text": "t"}', '{"case_id": "a_b", "set": "medqa", "input_text": "t"}'],
     "same file name"),
    ([""], "no cases"),
])
def test_input_errors(tmp_path, lines, message):
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(run_batch.InputError, match=message):
        run_batch.load_cases(path)


# ---------------------------------------------------------------- batch runner (fake pipeline)

IDENTITY = {"disabled_agents": [], "agent_modes": {}, "clarifier_fake": False, "groq_route": "gateway cdss",
            "models": {"fusion": {"model": "m", "source": "s"}}, "commits": {"clarifier": "51dacaf"},
            "kg_top_k": 5, "kg_rank_top": 10, "threshold": 0.65, "max_loops": 2}


@pytest.fixture
def fake_pipeline(monkeypatch):
    """run_pipeline.run / startup replaced; each run adds 2 Groq + 3 OpenRouter calls to the counters."""
    log = {"runs": [], "startups": [], "crash_on": set(), "identity": dict(IDENTITY)}
    counters = {"groq": 0, "openrouter": 0}

    def run(text, disabled=()):
        log["runs"].append(text)
        if any(c in text for c in log["crash_on"]):
            raise RuntimeError("boom")
        counters["groq"] += 2
        counters["openrouter"] += 3
        return ({"final_output": _fo(rag=["Asthma"], kg=["asthma", "copd"]), "clarifier_output": {"symptoms": []},
                 "clarifier_details": {"entities": []}, "loop_count": 1, "errors": []},
                ["clarifier", "vector_rag", "final_output"])
    monkeypatch.setattr(run_pipeline, "run", run)
    monkeypatch.setattr(run_pipeline, "startup",
                        lambda disabled=(): log["startups"].append(list(disabled)) or
                        {"vector_rag": {"collections": {"symptom_chunks": {"points": 2690}}}})
    monkeypatch.setattr(wrappers, "llm_counters", lambda: dict(counters))
    monkeypatch.setattr(run_batch, "run_identity", lambda disabled: {**log["identity"], "disabled_agents": list(disabled)})
    return log


def _batch(tmp_path, cases, *extra):
    inp = _write_jsonl(tmp_path / "cases.jsonl", cases)
    run_batch.main(["--input", str(inp), "--out", str(tmp_path / "run"), *extra])
    return tmp_path / "run"


def test_batch_saves_cases_manifest_and_summary(tmp_path, fake_pipeline, capsys):
    out = _batch(tmp_path, [_case("c1"), _case("c2", "rare")])
    files = sorted(p.name for p in (out / "cases").glob("*.json"))
    assert files == ["c1.json", "c2.json"] and not list((out / "cases").glob("*.tmp"))
    item = json.loads((out / "cases" / "c1.json").read_text(encoding="utf-8"))
    assert set(item) >= {"id", "final_output", "clarifier_details", "clarifier_output", "loop_count", "case", "batch"}
    assert item["case"]["gold_diagnosis"] == "Myocardial infarction"
    assert item["batch"]["groq_calls"] == 2 and item["batch"]["openrouter_calls"] == 3
    assert item["batch"]["seconds"] >= 0 and item["batch"]["errors"] == []
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["groq_route"] == "gateway cdss" and manifest["commits"] == {"clarifier": "51dacaf"}
    assert manifest["qdrant"] == {"symptom_chunks": {"points": 2690}} and "resumed_at" not in manifest
    rows = list(csv.DictReader(open(out / "summary.csv", encoding="utf-8")))
    assert [r["case_id"] for r in rows] == ["c1", "c2"] and rows[0]["rag_top1"] == "Asthma"
    assert rows[0]["kg_top3"] == "asthma; copd" and rows[0]["loops"] == "1"
    printed = capsys.readouterr().out
    assert "[1/2] c1  analysed" in printed and "groq 2  openrouter 3  dx: Asthma | kg: asthma" in printed


def test_batch_resume_skips_saved_cases(tmp_path, fake_pipeline, capsys):
    _batch(tmp_path, [_case("c1"), _case("c2")])
    fake_pipeline["runs"].clear()
    fake_pipeline["startups"].clear()
    out = _batch(tmp_path, [_case("c1"), _case("c2"), _case("c3")])
    assert fake_pipeline["runs"] == ["text of c3"] and len(fake_pipeline["startups"]) == 1
    assert "3 case(s): 2 already saved (skipped), 1 to run" in capsys.readouterr().out
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["resumed_at"]) == 1
    fake_pipeline["startups"].clear()
    _batch(tmp_path, [_case("c1"), _case("c2"), _case("c3")])
    assert fake_pipeline["startups"] == []  # nothing pending: agents are not even loaded


def test_batch_broken_saved_file_is_run_again(tmp_path, fake_pipeline):
    out = _batch(tmp_path, [_case("c1")])
    (out / "cases" / "c1.json").write_text("{half written", encoding="utf-8")
    fake_pipeline["runs"].clear()
    _batch(tmp_path, [_case("c1")])
    assert fake_pipeline["runs"] == ["text of c1"]


def test_batch_never_mixes_settings(tmp_path, fake_pipeline):
    _batch(tmp_path, [_case("c1")])
    fake_pipeline["identity"]["commits"] = {"clarifier": "abc1234"}
    fake_pipeline["startups"].clear()
    with pytest.raises(SystemExit, match="different settings \\(commits\\).*new --out"):
        _batch(tmp_path, [_case("c1"), _case("c2")])
    assert fake_pipeline["startups"] == []
    with pytest.raises(SystemExit, match="different settings \\(disabled_agents\\)"):
        fake_pipeline["identity"]["commits"] = {"clarifier": "51dacaf"}
        _batch(tmp_path, [_case("c1"), _case("c2")], "--disable", "rag")


def test_batch_crashed_case_not_saved_and_retried(tmp_path, fake_pipeline, capsys):
    fake_pipeline["crash_on"] = {"c2"}
    out = _batch(tmp_path, [_case("c1"), _case("c2")])
    assert not (out / "cases" / "c2.json").exists()
    assert "c2\tRuntimeError: boom" in (out / "failed_cases.txt").read_text(encoding="utf-8")
    assert "c2  CRASHED (not saved, will be retried)" in capsys.readouterr().out
    fake_pipeline["crash_on"], fake_pipeline["runs"] = set(), []
    _batch(tmp_path, [_case("c1"), _case("c2")])
    assert fake_pipeline["runs"] == ["text of c2"] and (out / "cases" / "c2.json").exists()


def test_batch_limit_and_disable(tmp_path, fake_pipeline):
    out = _batch(tmp_path, [_case("c1"), _case("c2"), _case("c3")], "--limit", "2", "--disable", "rag")
    assert len(list((out / "cases").glob("*.json"))) == 2 and fake_pipeline["startups"] == [["rag"]]
    assert json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["qdrant"] == "rag disabled"


def test_batch_bad_switch_or_input_stops(tmp_path, fake_pipeline):
    with pytest.raises(SystemExit, match="At least one of rag or kg"):
        _batch(tmp_path, [_case("c1")], "--disable", "rag,kg")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"case_id": "a"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="input_text is missing"):
        run_batch.main(["--input", str(bad), "--out", str(tmp_path / "r2")])
    assert fake_pipeline["startups"] == []


def test_run_info_has_no_key_values():
    info = json.dumps(wrappers.run_info(["rag"]))
    for name in config.API_KEY_NAMES:
        value = __import__("os").getenv(name)
        if value:
            assert value not in info
    assert json.loads(info)["disabled_agents"] == ["rag"]


# ---------------------------------------------------------------- name matching and statistics

@pytest.mark.parametrize("predicted, gold, synonyms, rule", [
    ("Myocardial Infarctions", "myocardial infarction", (), "exact"),
    ("Myocardial-infarction.", "Myocardial infarction", (), "exact"),
    ("Heart attack (Myocardial infarction)", "myocardial infarction", (), "part"),
    ("Coronary Artery Disease / Angina", "angina", (), "part"),
    ("Heart attack", "Myocardial infarction", ("heart attack",), "synonym"),
    ("Acute myocardial infarction", "Myocardial infarction", (), None),
    ("", "x", (), None),
])
def test_match_rule(predicted, gold, synonyms, rule):
    assert score.match_rule(predicted, gold, synonyms) == rule


def test_wilson_known_values():
    assert score.wilson(8, 10) == {"k": 8, "n": 10, "p": 0.8, "low": 0.4902, "high": 0.9433}
    assert score.wilson(0, 0)["p"] is None
    w = score.wilson(0, 20)
    assert w["low"] == 0.0 and 0.16 < w["high"] < 0.17


@pytest.mark.parametrize("b, c, p", [(0, 0, 1.0), (1, 9, 0.021484), (0, 6, 0.03125), (5, 5, 1.0), (3, 0, 0.25)])
def test_mcnemar_exact(b, c, p):
    assert score.mcnemar_exact(b, c) == p


# ---------------------------------------------------------------- tests 1-4

def test_test1_rules_exclusions_and_review():
    review = []
    items = [
        _item(_case("ok"), _fo(rag=["Heart attack (Myocardial infarction)"])),
        _item(_case("wrong"), _fo(rag=["Asthma"], kg=["copd", "myocardial infarction"], report="consider MI")),
        _item(_case("fail"), _fo(rag=["x"], status="clarifier_error")),
        _item(_case("fb"), _fo(rag=["Myocardial infarction"], fallback=True)),
        _item(_case("norag"), _fo(kg=["myocardial infarction"], disabled=["rag"])),
        _item(_case("nogold", gold=""), _fo(rag=["x"])),
        _item(_case("text"), _fo(rag=["Asthma"], report="Most likely acute myocardial infarction.")),
    ]
    t1, per_case = score.test1(items, review)
    assert t1["rag_top1_accuracy"]["k"] == 1 and t1["rag_top1_accuracy"]["n"] == 3  # ok, wrong, text
    assert t1["correct_anywhere_structured"]["k"] == 3 and t1["correct_anywhere_structured"]["n"] == 4
    assert t1["secondary_mentioned_in_fusion_report"]["k"] == 1
    assert t1["excluded"] == {"not_analysed": 1, "evidence_fallback": 1, "no_gold_diagnosis": 1,
                              "rag_disabled_top1": 1}
    assert per_case["ok"]["top1_rule"] == "part" and per_case["wrong"]["anywhere_rule"] == "exact"
    assert ("test1_rag_top1", items[1]["case"], 1, "Asthma") in review


def test_test2_hits_mrr_and_fallback_kept():
    gold = "Rheumatoid arthritis"
    items = [_item(_case(f"c{i}", gold=gold), _fo(kg=["x"] * (rank - 1) + [gold] if rank else ["x", "y"],
                                                   fallback=(i == 0)))
             for i, rank in enumerate([1, 3, None, 10])]
    items.append(_item(_case("off", gold=gold), _fo(kg=[gold], disabled=["kg"])))
    review = []
    t2, per_case = score.test2(items, review)
    assert [t2[f"hit@{k}"]["k"] for k in (1, 3, 5, 10)] == [1, 2, 2, 3] and t2["n"] == 4
    assert t2["mrr"] == round((1 + 1 / 3 + 0 + 0.1) / 4, 4)
    assert t2["excluded"]["kg_disabled"] == 1 and per_case["c0"] == 1  # fallback case kept
    assert [r[2] for r in review if r[1]["case_id"] == "c2"] == [1, 2]  # whole ranking to review


def test_test3_medication_safety_definitions():
    flags = [{"type": "contraindication", "drug": "Sildenafil", "target": "myocardial infarction", "severity": ""},
             {"type": "drug_clash", "drug": "warfarin", "target": "aspirin", "severity": "moderate"},
             {"type": "drug_clash", "drug": "simvastatin", "target": "clarithromycin", "severity": "major"}]
    w = lambda t, d, tg, s: {"type": t, "drug": d, "target": tg, "severity": s, "message": "m"}
    items = [
        _item(_case("a", "medsafety", flags=flags), _fo(fallback=True, warnings=[
            w("contraindication", "sildenafil", "Myocardial Infarction", "unknown"),  # severity not scored
            w("drug_clash", "aspirin", "warfarin", "minor"),          # either order; wrong severity
            w("drug_clash", "Clarithromycin", "Simvastatin", "major"),  # right severity
            w("treatment_clash", "x", "y", "major"),                  # other types ignored
            w("contraindication", "x", "y", None)])),                  # false warning, unknown severity
        _item(_case("b", "medsafety"), _fo(warnings=[w("drug_clash", "a", "b", "moderate")])),  # clean, alarm
        _item(_case("c", "medsafety"), _fo()),                                                    # clean, quiet
        _item(_case("d", "medsafety"), _fo(status="no_clinical_items")),
        _item(_case("e", "medqa"), _fo(warnings=[w("drug_clash", "a", "b", "major")])),           # not medsafety
    ]
    review = []
    t3 = score.test3(items, review)
    assert t3["cases"] == 3 and t3["excluded"]["not_analysed"] == 1
    assert (t3["flag_catch_rate"]["k"], t3["flag_catch_rate"]["n"]) == (3, 3)
    assert (t3["case_false_alarm_rate"]["k"], t3["case_false_alarm_rate"]["n"]) == (1, 2)
    assert (t3["warning_precision"]["k"], t3["warning_precision"]["n"]) == (3, 5)
    sev = t3["severity_correct_among_caught_drug_clash"]
    assert (sev["k"], sev["n"]) == (1, 2)  # the contraindication flag is not in the denominator
    assert {s: v["k"] for s, v in t3["warnings_by_severity"].items()} == {"major": 1, "moderate": 1, "minor": 1,
                                                                          "unknown": 2}
    assert sum(1 for r in review if r[0] == "test3_unmatched_warning") == 2


def test_test4_clarifier():
    gold = {"symptoms": ["knee pain", "knee swelling"], "absent": ["fever"], "medications": ["ibuprofen"],
            "diseases": []}
    items = [
        _item(_case("k", "clarifier", gold="", entities=gold), _fo(fallback=True),
              clarifier_output={"symptoms": ["knee pains", "fever"], "medications": ["ibuprofen"],
                                "diseases": ["hypertension"]},
              entities=[{"text": "chills", "status": "absent"}]),
        _item(_case("err", "clarifier", gold="", entities=gold), _fo(status="clarifier_error"), clarifier_output={}),
        _item(_case("other", "medqa"), _fo()),
    ]
    t4 = score.test4(items)
    assert t4["cases"] == 1 and t4["excluded"]["clarifier_failed"] == 1
    assert (t4["symptoms"]["precision"]["k"], t4["symptoms"]["precision"]["n"]) == (1, 2)
    assert (t4["symptoms"]["recall"]["k"], t4["symptoms"]["recall"]["n"]) == (1, 2)
    assert t4["medications"]["recall"]["p"] == 1.0
    assert (t4["diseases"]["precision"]["k"], t4["diseases"]["precision"]["n"]) == (0, 1)  # clarifier set: all scored
    assert t4["absent_correct"]["k"] == 0 and t4["absent_leaked_as_present"]["k"] == 1


def test_loops_confidence_cost_and_exclusions():
    items = [_item(_case("a"), _fo(rag=["Myocardial infarction"], graph_conf=0.3), loops=0, seconds=10, groq=2),
             _item(_case("b"), _fo(rag=["Asthma"]), loops=2, seconds=20, groq=4),
             _item(_case("c"), _fo(rag=["Myocardial infarction"], fallback=True), loops=1),
             _item(_case("d"), _fo(status="no_clinical_items"))]
    scores, _, _ = score.score_items(items)
    assert set(scores["accuracy_by_loops"]) == {"0", "2"}  # fallback case c left out
    assert scores["accuracy_by_loops"]["0"]["p"] == 1.0 and scores["accuracy_by_loops"]["2"]["p"] == 0.0
    assert scores["confidence_analysis"]["top1_correct"]["graph_confidence"] == {"mean": 0.3, "n": 1}
    assert scores["cases"]["not_analysed_by_status"] == {"no_clinical_items": 1}
    assert scores["cases"]["evidence_fallback_cases"] == 1
    assert scores["cost"]["seconds_per_case"]["n"] == 4


def _save_run(path, items, manifest=None):
    (path / "cases").mkdir(parents=True)
    (path / "run_manifest.json").write_text(json.dumps(manifest or {"disabled_agents": []}), encoding="utf-8")
    for item in items:
        (path / "cases" / f"{item['case']['case_id']}.json").write_text(json.dumps(item), encoding="utf-8")
    return path


def test_score_main_files_and_compare(tmp_path, capsys):
    gold = "Myocardial infarction"
    full = [_item(_case(f"c{i}"), _fo(rag=[gold if i < 6 else "Asthma"], kg=[gold])) for i in range(8)]
    ablation = [_item(_case(f"c{i}"), _fo(kg=["x", gold] if i < 7 else ["x"], disabled=["rag"])) for i in range(8)]
    run_a = _save_run(tmp_path / "full", full)
    run_b = _save_run(tmp_path / "norag", ablation, {"disabled_agents": ["rag"]})
    scores = score.main(["--run", str(run_a), "--compare", str(run_b)])
    hit1 = scores["compare"]["metrics"]["test2_hit@1"]
    assert hit1["paired_cases"] == 8 and hit1["only_run_correct"] == 8 and hit1["only_other_correct"] == 0
    assert hit1["mcnemar_exact_p"] == score.mcnemar_exact(8, 0)
    assert scores["compare"]["metrics"]["test2_hit@3"]["only_run_correct"] == 1
    assert "test1_top1" not in scores["compare"]["metrics"] or \
        scores["compare"]["metrics"]["test1_top1"]["paired_cases"] == 0  # rag off in the other run
    text = (run_a / "scores.txt").read_text(encoding="utf-8")
    assert "RAG top-1 accuracy:               6/8 = 75.0% (95% CI" in text and "McNemar" in text
    assert json.loads((run_a / "scores.json").read_text(encoding="utf-8"))["test2_kg"]["n"] == 8
    rows = list(csv.DictReader(open(run_a / "unmatched_for_review.csv", encoding="utf-8")))
    assert {r["case_id"] for r in rows} == {"c6", "c7"} and rows[0]["reviewer_1"] == ""


def test_score_empty_run_stops(tmp_path):
    (tmp_path / "empty" / "cases").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no saved cases"):
        score.main(["--run", str(tmp_path / "empty")])
