"""
Knowledge Graph agent - single entry point.

Ties together the pieces built so far:
  symptoms -> predict_disease() + select_for_fusion()  (demo_pipeline.py)
  top diseases + current medications -> check_medication_safety()
  with DDInter severity                                 (kg_real_traversal.py)

The graph (relation_filtered_graph_v2.pkl) and the DDInter severity lookup
are loaded once per process by load_resources() and reused on every call.
File paths are resolved relative to this file, so the agent works from any
working directory. Does not modify any existing module.
"""

import logging
import time
from pathlib import Path

from build_full_graph_v2 import load_graph
from demo_pipeline import predict_disease, select_for_fusion
from kg_real_traversal import check_medication_safety, load_ddinter_severity

BASE_DIR = Path(__file__).resolve().parent
GRAPH_FILE = BASE_DIR / "relation_filtered_graph_v2.pkl"
DDINTER_FOLDER = BASE_DIR / "data_sources"

logger = logging.getLogger("kg_agent")

# Loaded once per process by load_resources(); each is set only after its
# own load succeeds, so a failed DDInter load doesn't force a graph reload.
_G = None
_NODE_LOOKUP = None  # lowercased node name -> node name, built with the graph
_SEVERITY_LOOKUP = None
_LOAD_STATS = {}


def load_resources():
    """
    Load the graph and the DDInter severity lookup once, then reuse them.

    Returns (G, severity_lookup, stats). stats["loaded_now"] is True only on
    the call that actually loaded something.
    """
    global _G, _NODE_LOOKUP, _SEVERITY_LOOKUP
    loaded_now = False

    if _G is None:
        t = time.perf_counter()
        G = load_graph(GRAPH_FILE)
        _LOAD_STATS["graph_load_seconds"] = round(time.perf_counter() - t, 2)
        _LOAD_STATS["graph_nodes"] = G.number_of_nodes()
        _LOAD_STATS["graph_edges"] = G.number_of_edges()
        _NODE_LOOKUP = {n.lower(): n for n in G.nodes()}
        _G = G
        loaded_now = True

    if _SEVERITY_LOOKUP is None:
        t = time.perf_counter()
        lookup, lookup_stats = load_ddinter_severity(_G, DDINTER_FOLDER)
        _LOAD_STATS["lookup_load_seconds"] = round(time.perf_counter() - t, 2)
        _LOAD_STATS["lookup_stats"] = lookup_stats
        _SEVERITY_LOOKUP = lookup
        loaded_now = True

    if loaded_now:
        logger.info(
            "Loaded KG resources: graph %d nodes / %d edges in %.2fs; "
            "DDInter %d pairs matched in %.2fs",
            _LOAD_STATS["graph_nodes"], _LOAD_STATS["graph_edges"],
            _LOAD_STATS["graph_load_seconds"],
            _LOAD_STATS["lookup_stats"]["pairs_matched"],
            _LOAD_STATS["lookup_load_seconds"],
        )

    return _G, _SEVERITY_LOOKUP, dict(_LOAD_STATS, loaded_now=loaded_now)


def run_kg_agent(clarifier_output, top_k=5):
    """
    Knowledge Graph agent entry point.

    INPUT:  Clarifier dict, e.g. {"symptoms": [...], "medications": [...],
            "diseases": [...]}. Other keys are ignored; a missing key means
            an empty list.
    OUTPUT: {"agent": "knowledge_graph", "status": "success" | "error",
             "known_diseases": the input "diseases" list, as given,
             "disease_prediction": select_for_fusion() result or None,
             "medication_safety": check_medication_safety() result or None,
             "error": None or "<stage>: <message>"}

    Medications are checked against the known diseases first, then the
    predicted ones. A disease in both lists is checked once, as "known".
    Each warning gets disease_source: "known", "predicted", or None
    (medication-vs-medication warnings have no disease).
    """
    output = {
        "agent": "knowledge_graph",
        "status": "success",
        "known_diseases": None,
        "disease_prediction": None,
        "medication_safety": None,
        "error": None,
    }

    def fail(stage, exc):
        output["status"] = "error"
        output["error"] = f"{stage}: {exc}"
        return output

    if not isinstance(clarifier_output, dict):
        return fail("input", f"clarifier_output must be a dict, got {type(clarifier_output).__name__}")
    medications = clarifier_output.get("medications") or []
    known = clarifier_output.get("diseases") or []
    output["known_diseases"] = known

    try:
        G, severity_lookup, _ = load_resources()
    except Exception as e:
        return fail("load_resources", e)

    try:
        prediction = select_for_fusion(predict_disease(clarifier_output, G=G), top_k)
    except Exception as e:
        return fail("disease_prediction", e)
    output["disease_prediction"] = prediction

    # Known diseases first, then predicted. check_medication_safety() keeps
    # only the first occurrence of each graph node, so a disease in both
    # lists is checked once, as known. With no diseases at all, Tier 2
    # still checks medications against each other.
    predicted = [c["disease"] for c in prediction["candidate_diseases"]]
    try:
        safety = check_medication_safety(G, known + predicted, medications,
                                         severity_lookup=severity_lookup)
    except Exception as e:
        return fail("medication_safety", e)

    # Same resolution rule as check_medication_safety(): case-insensitive
    # exact match to a disease node.
    known_nodes = set()
    for name in known:
        node = _NODE_LOOKUP.get(str(name).strip().lower())
        if node is not None and G.nodes[node].get("type") == "disease":
            known_nodes.add(node)

    labelled = []
    for w in safety["warnings"]:
        if w["disease"] is None:
            source = None
        else:
            source = "known" if w["disease"] in known_nodes else "predicted"
        new_w = {}
        for key, value in w.items():
            new_w[key] = value
            if key == "disease":
                new_w["disease_source"] = source
        labelled.append(new_w)
    safety["warnings"] = labelled
    for entry in safety.get("by_disease", []):
        entry["disease_source"] = "known" if entry["disease"] in known_nodes else "predicted"
    output["medication_safety"] = safety

    if safety.get("status") == "error":
        return fail("medication_safety", safety.get("error"))

    return output
