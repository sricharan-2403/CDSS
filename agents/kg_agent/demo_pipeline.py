"""
End-to-end demo tying together the graph-traversal pieces built so far:
symptom synonym normalization (symptom_synonyms_full.py) + the fixed,
pickled relation-filtered graph (build_full_graph_v2.py) -> candidate
diseases ranked by how much of their documented symptom profile the input
covers.

Reuses load_graph(), diseases_for(), phenotypes_for() from
build_full_graph_v2.py and SYMPTOM_SYNONYMS from symptom_synonyms_full.py.
Does not touch any existing file.
"""

import copy
import json
import math
import sys
from pathlib import Path

from build_full_graph_v2 import load_graph, diseases_for, phenotypes_for
from symptom_synonyms_full import SYMPTOM_SYNONYMS

RELATION = "disease_phenotype_positive"


def predict_disease(clarifier_output: dict, G=None) -> dict:
    """
    Input: clarifier_output dict with a "symptoms" field (list of strings),
    matching the real Clarifier output format. G is an optional
    already-loaded graph; if None, the graph is loaded from the pickle.

    Returns candidate diseases (union across all input symptoms) ranked by
    coverage_fraction = matched_symptom_count / total_documented_symptoms.
    """
    symptoms = clarifier_output.get("symptoms") or []

    if G is None:
        G = load_graph()
    node_lookup = {n.strip().lower(): n for n in G.nodes()}

    normalized_symptoms = {}
    symptom_to_node = {}
    symptoms_not_found = []

    for symptom in symptoms:
        normalized = SYMPTOM_SYNONYMS.get(symptom.strip().lower(), symptom)
        normalized_symptoms[symptom] = normalized

        node = node_lookup.get(normalized.strip().lower())
        if node is not None and G.nodes[node].get("type") != "effect/phenotype":
            node = None  # matched a node, but it's not a phenotype (e.g. disease-name collision)

        if node is None:
            symptoms_not_found.append(symptom)
            symptom_to_node[symptom] = None
        else:
            symptom_to_node[symptom] = node

    disease_matches = {}  # disease name -> set of original input symptoms that matched
    for symptom, node in symptom_to_node.items():
        if node is None:
            continue
        for disease in diseases_for(G, node, RELATION):
            disease_matches.setdefault(disease, set()).add(symptom)

    candidate_diseases = []
    for disease, matched in disease_matches.items():
        total = len(phenotypes_for(G, disease, RELATION))
        coverage = (len(matched) / total) if total else 0.0
        candidate_diseases.append({
            "disease": disease,
            "matched_symptom_count": len(matched),
            "total_documented_symptoms": total,
            "coverage_fraction": coverage,
            "matched_symptoms": sorted(matched),
        })

    candidate_diseases.sort(
        key=lambda c: (c["matched_symptom_count"], c["coverage_fraction"]),
        reverse=True,
    )

    return {
        "status": "success",
        "input_symptoms": symptoms,
        "normalized_symptoms": normalized_symptoms,
        "symptoms_not_found": symptoms_not_found,
        "candidate_diseases": candidate_diseases,
        "metadata": {
            "input_symptom_count": len(symptoms),
            "symptoms_matched_count": len(symptoms) - len(symptoms_not_found),
            "symptoms_not_found_count": len(symptoms_not_found),
            "candidate_disease_count": len(candidate_diseases),
        },
    }


def select_for_fusion(result: dict, top_k: int = 5) -> dict:
    """
    Input: the dict returned by predict_disease().

    Keeps diseases matching at least ceil(recognised / 2) of the recognised
    input symptoms, sorts them by matched count, then coverage (both highest
    first), then disease name (case-insensitive), and keeps the top_k.

    Returns a new dict with the same keys as the input, candidate_diseases
    replaced by the selected list, plus a "selection" block. The input dict
    is not modified.
    """
    recognised = result["metadata"]["symptoms_matched_count"]
    candidates = result["candidate_diseases"]

    if recognised == 0:
        min_matches = None
        kept = []
        note = "No input symptoms were recognised in the graph; nothing selected."
    else:
        min_matches = math.ceil(recognised / 2)
        kept = [c for c in candidates if c["matched_symptom_count"] >= min_matches]
        kept.sort(key=lambda c: (
            -c["matched_symptom_count"],
            -c["coverage_fraction"],
            c["disease"].lower(),
        ))
        note = None

    selected = kept[:top_k]
    tie_at_cutoff = (
        top_k > 0
        and len(kept) > top_k
        and kept[top_k]["matched_symptom_count"] == selected[-1]["matched_symptom_count"]
        and kept[top_k]["coverage_fraction"] == selected[-1]["coverage_fraction"]
    )

    out = copy.deepcopy({k: v for k, v in result.items() if k != "candidate_diseases"})
    out["candidate_diseases"] = copy.deepcopy(selected)
    out["selection"] = {
        "recognised_symptom_count": recognised,
        "min_matches_required": min_matches,
        "candidates_before_threshold": len(candidates),
        "candidates_after_threshold": len(kept),
        "top_k": top_k,
        "tie_at_cutoff": tie_at_cutoff,
    }
    if note is not None:
        out["selection"]["note"] = note
    return out


def _print_result(result, top_n=8):
    print("=" * 70)
    print("INPUT SYMPTOMS")
    print("=" * 70)
    for s in result["input_symptoms"]:
        print(f"  - {s}")

    print("\n" + "=" * 70)
    print("NORMALIZED MAPPING APPLIED")
    print("=" * 70)
    for orig, norm in result["normalized_symptoms"].items():
        marker = "(unchanged)" if orig == norm else ""
        print(f"  {orig!r:30} -> {norm!r}  {marker}")

    if result["symptoms_not_found"]:
        print("\nSYMPTOMS NOT FOUND IN GRAPH:")
        for s in result["symptoms_not_found"]:
            print(f"  - {s}")

    print("\n" + "=" * 70)
    print(f"TOP {top_n} CANDIDATE DISEASES (of {result['metadata']['candidate_disease_count']} total)")
    print("=" * 70)
    print(f"{'matched':>8}  {'coverage':>9}  {'total_symptoms':>15}  disease")
    for c in result["candidate_diseases"][:top_n]:
        print(f"{c['matched_symptom_count']:>8}  {c['coverage_fraction']*100:>8.1f}%  "
              f"{c['total_documented_symptoms']:>15}  {c['disease']}")


DEFAULT_CASE = {"symptoms": ["fever", "shortness of breath"]}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--symptoms":
        if len(sys.argv) < 3 or not sys.argv[2].strip():
            print("ERROR: --symptoms requires a comma-separated list, "
                  'e.g. --symptoms "joint pain,rash,fatigue"')
            sys.exit(1)
        symptom_list = [s.strip() for s in sys.argv[2].split(",") if s.strip()]
        clarifier_output = {"symptoms": symptom_list}
        print(f"SYMPTOMS FROM COMMAND LINE: {symptom_list}\n")
    elif len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if not input_path.exists():
            print(f"ERROR: input file not found: {input_path}")
            sys.exit(1)
        try:
            with open(input_path) as f:
                clarifier_output = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: {input_path} is not valid JSON: {e}")
            sys.exit(1)
        print(f"CLARIFIER INPUT: {input_path}\n")
    else:
        clarifier_output = DEFAULT_CASE
        print("No input given - running default case:")
        print(f"  {DEFAULT_CASE}")
        print("(usage: python demo_pipeline.py <path_to_json_file>")
        print('        python demo_pipeline.py --symptoms "joint pain,rash,fatigue")\n')

    result = predict_disease(clarifier_output)
    _print_result(result)
