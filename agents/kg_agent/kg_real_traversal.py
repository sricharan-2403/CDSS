"""
Knowledge Graph Traversal — REAL PrimeKG data
CDSS Project — Knowledge Graph module

This replaces the earlier hand-curated prototype with a graph built from
actual PrimeKG facts (primekg_subset.csv, filtered from the full dataset).

Architecture note: the graph does NOT re-derive the diagnosis from raw
symptoms (that's Vector RAG's job upstream, e.g. "chest pain -> MI, 0.82
similarity"). The graph's job starts once a candidate diagnosis exists:
it traces standard treatment -> drug interaction / direct contraindication
implications against the patient's current medications.
"""

import pandas as pd
import networkx as nx
import json
import functools
import sys
from itertools import combinations
from pathlib import Path

SUBSET_FILE = "primekg_subset.csv"

# Relation types relevant to clinical decision support.
# Excludes molecular-biology noise (bioprocess_bioprocess, molfunc_molfunc,
# pathway_protein, gene/protein edges) that got pulled in because a drug
# or disease name matched a keyword somewhere in an unrelated biology row.
RELEVANT_RELATIONS = {
    "indication",
    "contraindication",
    "off-label use",
    "drug_drug",
    "disease_disease",
    "disease_phenotype_positive",
    "phenotype_phenotype",
}


def build_graph(csv_path=SUBSET_FILE):
    df = pd.read_csv(csv_path)
    df = df[df["relation"].isin(RELEVANT_RELATIONS)]

    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        G.add_node(row["x_name"], type=row["x_type"])
        G.add_node(row["y_name"], type=row["y_type"])
        G.add_edge(
            row["x_name"], row["y_name"],
            relation=row["relation"],
            display_relation=row["display_relation"],
        )
    return G


def traverse(G, candidate_diagnosis, current_medications):
    """
    Given a candidate diagnosis (already identified by Vector RAG upstream)
    and the patient's current medications, find:
      1. Direct contraindications between a medication and the diagnosis
         itself
      2. Indirect risk: diagnosis's standard/off-label treatment, checked
         for drug-drug interaction against the patient's medications
    Returns the output contract handed to the Fusion agent.
    """
    result = {
        "candidate_diagnosis": candidate_diagnosis,
        "direct_flags": [],
        "indirect_flags": [],
    }

    # 1. Direct: medication <-> diagnosis contraindication
    for med in current_medications:
        if G.has_edge(med, candidate_diagnosis):
            for _, attrs in G.get_edge_data(med, candidate_diagnosis).items():
                if attrs["relation"] == "contraindication":
                    result["direct_flags"].append({
                        "path": [med, candidate_diagnosis],
                        "relation": attrs["display_relation"],
                        "note": "DrugBank-curated contraindication - genuine severity signal, unlike drug_drug edges below",
                    })

    # 2. Indirect: diagnosis -> (indication / off-label use) -> treatment
    #    -> drug_drug -> medication
    # NOTE: PrimeKG's drug_drug edges all carry the same generic label
    # ("synergistic interaction") regardless of actual clinical danger -
    # it records THAT two drugs interact, not HOW severe that interaction
    # is. We do not fabricate a severity here; that judgment needs a
    # separate signal (e.g. cross-checking the Evidence Scanner / PubMed).
    if G.has_node(candidate_diagnosis):
        for treatment, _, attrs in G.in_edges(candidate_diagnosis, data=True):
            if attrs["relation"] in ("indication", "off-label use"):
                for med in current_medications:
                    if G.has_edge(treatment, med):
                        for _, e_attrs in G.get_edge_data(treatment, med).items():
                            if e_attrs["relation"] == "drug_drug":
                                result["indirect_flags"].append({
                                    "path": [candidate_diagnosis, treatment, med],
                                    "treatment_relation": attrs["relation"],
                                    "interaction_type": e_attrs["display_relation"],
                                    "note": "PrimeKG does not grade interaction severity - flagged for downstream review, not auto-classified as dangerous",
                                })
    return result


@functools.lru_cache(maxsize=1)
def _cached_graph(csv_path):
    """Same CSV in -> same graph out, loaded once per process."""
    return build_graph(csv_path)


def graph_traversal_agent(clarifier_output, csv_path=SUBSET_FILE):
    """
    Callable entry point matching the multi-agent pipeline's shared
    input/output contract.

    INPUT  (Clarifier agent output): {"diseases": [...], "symptoms": [...],
            "medications": [...], "tests": [...]}
    OUTPUT: see safety_flags / evidence / metadata contract used by Fusion.

    Pure/deterministic: given the same clarifier_output and the same
    underlying CSV, always returns the same result. The only mutable state
    is an in-memory cache of the parsed graph (keyed by csv_path), which
    never changes the answer, only how fast repeat calls are.
    """
    diseases = clarifier_output.get("diseases") or []
    medications = clarifier_output.get("medications") or []

    try:
        G = _cached_graph(csv_path)
    except Exception as e:
        return {
            "agent": "graph_traversal",
            "status": "error",
            "query_context": {"diseases": diseases, "medications": medications},
            "safety_flags": {"direct_flags": [], "indirect_flags": []},
            "evidence": [],
            "metadata": {
                "entities_checked": 0,
                "entities_not_found": 0,
                "direct_flag_count": 0,
                "indirect_flag_count": 0,
            },
            "error": str(e),
        }

    # Node names in PrimeKG are case-sensitive strings (e.g. "myocardial
    # infarction", "Sildenafil"); the Clarifier's casing won't always match
    # exactly, so resolve case-insensitively but keep the original string
    # for everything we hand back downstream.
    node_lookup = {n.lower(): n for n in G.nodes()}

    def resolve(entity_name):
        return node_lookup.get(entity_name.strip().lower())

    resolved_diseases = [(d, resolve(d)) for d in diseases]
    resolved_meds = [(m, resolve(m)) for m in medications]
    not_found = [name for name, node in resolved_diseases + resolved_meds if node is None]

    direct_flags = []
    indirect_flags = []
    evidence = []

    # 1. Direct: (medication)-[contraindication]->(diagnosis), 1-hop.
    for med_orig, med_node in resolved_meds:
        if med_node is None:
            continue
        for dis_orig, dis_node in resolved_diseases:
            if dis_node is None:
                continue
            if not G.has_edge(med_node, dis_node):
                continue
            for _, attrs in G.get_edge_data(med_node, dis_node).items():
                if attrs["relation"] != "contraindication":
                    continue
                direct_flags.append({
                    "drug": med_orig,
                    "relation": "contraindication",
                    "target": dis_orig,
                    "status": "found",
                })
                evidence.append({
                    "content": f"{med_orig} has a documented contraindication with "
                               f"{dis_orig} ({attrs['display_relation']}).",
                    "source": "PrimeKG (DrugBank)",
                    "source_type": "graph_edge",
                })

    # 2. Indirect: (diagnosis)-[indication|off-label use]->(treatment)
    #    -[drug_drug]->(medication), the exact 2-hop chain shape only.
    for dis_orig, dis_node in resolved_diseases:
        if dis_node is None:
            continue
        for treatment_node, _, t_attrs in G.in_edges(dis_node, data=True):
            if t_attrs["relation"] not in ("indication", "off-label use"):
                continue
            for med_orig, med_node in resolved_meds:
                if med_node is None:
                    continue
                if not G.has_edge(treatment_node, med_node):
                    continue
                for _, e_attrs in G.get_edge_data(treatment_node, med_node).items():
                    if e_attrs["relation"] != "drug_drug":
                        continue
                    indirect_flags.append({
                        "drug": med_orig,
                        "path": [dis_orig, "treated_by", treatment_node, "drug_drug", med_orig],
                        "status": "found",
                    })
                    evidence.append({
                        "content": f"{treatment_node} is a standard treatment for {dis_orig} "
                                   f"and has a drug_drug interaction with {med_orig} "
                                   f"({e_attrs['display_relation']}).",
                        "source": "PrimeKG (DrugBank)",
                        "source_type": "graph_edge",
                    })

    return {
        "agent": "graph_traversal",
        "status": "success",
        "query_context": {"diseases": diseases, "medications": medications},
        "safety_flags": {
            "direct_flags": direct_flags,
            "indirect_flags": indirect_flags,
        },
        "evidence": evidence,
        "metadata": {
            "entities_checked": len(diseases) + len(medications),
            "entities_not_found": len(not_found),
            "direct_flag_count": len(direct_flags),
            "indirect_flag_count": len(indirect_flags),
            "not_found_entities": not_found,
        },
        "error": None,
    }


NOT_CHECKED_NOTE = (
    "These medications could not be matched to the graph and were not checked. "
    "No warnings does not mean safe for these medications."
)

DDINTER_FOLDER = "data_sources"
SEVERITY_RANK = {"major": 3, "moderate": 2, "minor": 1, "unknown": 0}


def ddinter_pair_key(a, b):
    """Unordered pair of graph node names -> lookup key."""
    return tuple(sorted((a, b)))


def load_ddinter_severity(G, folder=DDINTER_FOLDER):
    """
    Build a DDInter severity lookup for an already-loaded graph. Call once at
    startup (like load_graph()) and pass the lookup to
    check_medication_safety(); it is never loaded per check.

    Reads every ddinter_downloads_code_*.csv in folder. Drug_A / Drug_B are
    matched to graph drug nodes by case-insensitive exact name (bracketed
    route text such as "(topical)" is kept, so those rows mostly don't
    match). Rows where either drug doesn't match are skipped. If a pair ever
    appears with different levels, the most severe level is kept.

    Returns (lookup, stats): lookup maps ddinter_pair_key(node_a, node_b)
    -> "major" / "moderate" / "minor" / "unknown".
    """
    files = sorted(Path(folder).glob("ddinter_downloads_code_*.csv"))
    if not files:
        raise FileNotFoundError(f"No ddinter_downloads_code_*.csv files found in {folder}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    rows_read = len(df)
    df["Drug_A"] = df["Drug_A"].astype(str).str.strip()
    df["Drug_B"] = df["Drug_B"].astype(str).str.strip()
    df["Level"] = df["Level"].astype(str).str.strip().str.lower()

    drug_lookup = {n.lower(): n for n, attrs in G.nodes(data=True) if attrs.get("type") == "drug"}
    node_a = df["Drug_A"].str.lower().map(drug_lookup)
    node_b = df["Drug_B"].str.lower().map(drug_lookup)

    unique_pairs = {tuple(sorted((a.lower(), b.lower()))) for a, b in zip(df["Drug_A"], df["Drug_B"])}
    unique_names = set(df["Drug_A"].str.lower()) | set(df["Drug_B"].str.lower())

    lookup = {}
    for a, b, level in zip(node_a, node_b, df["Level"]):
        if not isinstance(a, str) or not isinstance(b, str) or a == b:
            continue
        if level not in SEVERITY_RANK:
            level = "unknown"
        key = ddinter_pair_key(a, b)
        if key not in lookup or SEVERITY_RANK[level] > SEVERITY_RANK[lookup[key]]:
            lookup[key] = level

    stats = {
        "files_read": len(files),
        "rows_read": rows_read,
        "unique_pairs": len(unique_pairs),
        "unique_names": len(unique_names),
        "names_matched": sum(1 for n in unique_names if n in drug_lookup),
        "pairs_matched": len(lookup),
    }
    return lookup, stats


def check_medication_safety(G, diseases, medications, severity_lookup=None):
    """
    Medication safety check against an already-loaded graph (e.g.
    relation_filtered_graph_v2.pkl via build_full_graph_v2.load_graph()).
    Does not load or build any graph itself.

    diseases: disease names in rank order. medications: current medications.
    Names resolve case-insensitively to exact node names (same rule as
    graph_traversal_agent), and must hit a node of the right type.

    Three warning tiers:
      1. contraindication: (medication)-[contraindication]-(disease)
      2. current_medication_interaction: (medication)-[drug_drug]-(medication)
      3. treatment_interaction_summary: (disease)-[indication|off-label use]-
         (treatment)-[drug_drug]-(medication), one summary per
         (disease, medication)

    PrimeKG drug_drug edges carry no severity. Without severity_lookup every
    warning has severity "unknown" and severity_source None. With a lookup
    from load_ddinter_severity(), Tier 2 and Tier 3 take their severity from
    DDInter where the pair is listed; Tier 1 stays "unknown".

    Besides the warnings list, the result has two grouped views of the same
    facts: "by_disease" (per checked disease: contraindicated current
    medications and the treatment drugs grouped from safest to riskiest by
    their worst clash) and "medication_pair_clashes" (the Tier 2 pairs).
    """
    diseases = list(diseases or [])
    medications = list(medications or [])

    result = {
        "agent": "knowledge_graph_safety",
        "status": "success",
        "diseases_checked": [],
        "diseases_not_found": [],
        "medications_input": medications,
        "medications_matched": {},
        "medications_not_checked": [],
        "warnings": [],
        "summary": {"tier_1": 0, "tier_2": 0, "tier_3": 0, "total": 0},
        "note": None,
        "error": None,
        "by_disease": [],
        "medication_pair_clashes": [],
    }

    try:
        if G is None:
            raise ValueError("No graph provided; pass an already-loaded graph as G.")

        node_lookup = {n.lower(): n for n in G.nodes()}

        def resolve(name, node_type):
            node = node_lookup.get(str(name).strip().lower())
            if node is None or G.nodes[node].get("type") != node_type:
                return None
            return node

        def relations_between(a, b):
            """Relations on edges between a and b, either direction."""
            rels = set()
            for u, v in ((a, b), (b, a)):
                for attrs in (G.get_edge_data(u, v) or {}).values():
                    rels.add(attrs.get("relation"))
            return rels

        # Resolve diseases, keeping rank order and dropping duplicate nodes.
        disease_nodes = []
        for name in diseases:
            node = resolve(name, "disease")
            if node is None:
                result["diseases_not_found"].append(name)
            elif node not in disease_nodes:
                disease_nodes.append(node)
        result["diseases_checked"] = disease_nodes
        disease_rank = {d: i for i, d in enumerate(disease_nodes)}

        # Resolve medications; two inputs hitting the same node are checked once.
        med_nodes = []
        for name in medications:
            node = resolve(name, "drug")
            if node is None:
                result["medications_not_checked"].append(name)
                continue
            result["medications_matched"][name] = node
            if node not in med_nodes:
                med_nodes.append(node)

        def pair_severity(a, b):
            """(level, source) for a drug pair; only called when a lookup is given."""
            level = severity_lookup.get(ddinter_pair_key(a, b))
            if level is None:
                return "unknown", None
            return level, "DDInter"

        warnings = []

        def add_warning(tier, wtype, disease, meds, path, message,
                        severity="unknown", severity_source=None, **extra):
            warning = {
                "tier": tier,
                "type": wtype,
                "disease": disease,
                "medications": meds,
                "path": path,
                "message": message,
            }
            warning.update(extra)
            warning["severity"] = severity
            warning["severity_source"] = severity_source
            warnings.append(warning)

        def clash_severity(a, b):
            """(level, source) for a drug pair, with or without a lookup."""
            if severity_lookup is None:
                return "unknown", None
            return pair_severity(a, b)

        # Per disease, for by_disease: contraindicated current medications.
        contraindicated = {d: [] for d in disease_nodes}

        # Tier 1: medication <-> disease contraindication.
        for disease in disease_nodes:
            for med in med_nodes:
                if "contraindication" in relations_between(med, disease):
                    contraindicated[disease].append(med)
                    add_warning(
                        1, "contraindication", disease, [med],
                        f"{med} -[contraindication]- {disease}",
                        f"{med} is recorded as contraindicated in {disease}.",
                    )

        # Tier 2: interaction between two current medications.
        for med_a, med_b in combinations(sorted(med_nodes, key=str.lower), 2):
            if "drug_drug" not in relations_between(med_a, med_b):
                continue
            message = f"{med_a} and {med_b} have a recorded interaction."
            severity, severity_source = "unknown", None
            if severity_lookup is not None:
                severity, severity_source = pair_severity(med_a, med_b)
                if severity_source is None:
                    qualifier = "severity not available"
                elif severity == "unknown":
                    qualifier = "listed in DDInter without a severity level"
                else:
                    qualifier = f"severity: {severity.capitalize()}, per DDInter"
                message = f"{med_a} and {med_b} have a recorded interaction ({qualifier})."
            add_warning(
                2, "current_medication_interaction", None, [med_a, med_b],
                f"{med_a} -[drug_drug]- {med_b}",
                message,
                severity=severity, severity_source=severity_source,
            )

        # Tier 3: disease's treatment drugs that interact with a current medication.
        by_disease = []
        for disease in disease_nodes:
            treatments = {}  # treatment drug -> link type ("indication" wins over off-label)
            for nbr in set(G.predecessors(disease)) | set(G.successors(disease)):
                if G.nodes[nbr].get("type") != "drug":
                    continue
                rels = relations_between(nbr, disease)
                if "indication" in rels:
                    treatments[nbr] = "indication"
                elif "off-label use" in rels:
                    treatments[nbr] = "off-label use"

            # Treatment drug -> current medications it has a drug_drug edge
            # with (never itself). Computed once, used by Tier 3 and by_disease.
            clash_meds = {
                t: [m for m in med_nodes if t != m and "drug_drug" in relations_between(t, m)]
                for t in treatments
            }

            by_disease.append(_disease_treatment_options(
                disease, treatments, clash_meds, med_nodes,
                contraindicated[disease], clash_severity,
            ))

            for med in med_nodes:
                interacting = sorted(
                    (t for t in treatments if med in clash_meds[t]),
                    key=str.lower,
                )
                if not interacting:
                    continue
                more = "" if len(interacting) <= 5 else ", ..."
                noun = "treatment drug" if len(treatments) == 1 else "treatment drugs"
                verb = "has" if len(interacting) == 1 else "have"
                path = f"{disease} -[indication/off-label use]- <treatment drug> -[drug_drug]- {med}"
                lead = (f"{len(interacting)} of {len(treatments)} {noun} for {disease} "
                        f"{verb} a recorded interaction with {med}")

                if severity_lookup is None:
                    examples = interacting[:5]
                    add_warning(
                        3, "treatment_interaction_summary", disease, [med], path,
                        f"{lead} (e.g. {', '.join(examples)}{more}).",
                        interacting_count=len(interacting),
                        treatment_drug_count=len(treatments),
                        examples=examples,
                        details=[{"drug": t, "link_type": treatments[t]} for t in interacting],
                    )
                    continue

                details = []
                for t in interacting:
                    level, source = pair_severity(t, med)
                    details.append({"drug": t, "link_type": treatments[t],
                                    "severity": level, "severity_source": source})
                counts = {level: 0 for level in ("major", "moderate", "minor", "unknown")}
                for d in details:
                    counts[d["severity"]] += 1
                worst = max((d["severity"] for d in details), key=SEVERITY_RANK.__getitem__)
                source = "DDInter" if any(d["severity_source"] for d in details) else None
                examples = [d["drug"] for d in sorted(
                    details, key=lambda d: (-SEVERITY_RANK[d["severity"]], d["drug"].lower()))][:5]
                counts_text = ", ".join(f"{level.capitalize()}: {n}" for level, n in counts.items() if n)
                add_warning(
                    3, "treatment_interaction_summary", disease, [med], path,
                    f"{lead} ({counts_text}); e.g. {', '.join(examples)}{more or '.'}",
                    severity=worst, severity_source=source,
                    interacting_count=len(interacting),
                    treatment_drug_count=len(treatments),
                    examples=examples,
                    details=details,
                    severity_counts=counts,
                )

        # Stable order: tier, then severity (most severe first), then disease
        # rank (Tier 2 has none), then names. Without a lookup every severity
        # is "unknown", so the severity key never changes the order.
        warnings.sort(key=lambda w: (
            w["tier"],
            -SEVERITY_RANK.get(w["severity"], 0),
            disease_rank.get(w["disease"], -1),
            [m.lower() for m in w["medications"]],
        ))

        result["warnings"] = warnings
        for w in warnings:
            result["summary"][f"tier_{w['tier']}"] += 1
        result["summary"]["total"] = len(warnings)
        if result["medications_not_checked"]:
            result["note"] = NOT_CHECKED_NOTE

        result["by_disease"] = by_disease
        result["medication_pair_clashes"] = [
            {"medications": w["medications"], "severity": w["severity"],
             "severity_source": w["severity_source"], "path": w["path"]}
            for w in warnings if w["tier"] == 2
        ]

    except Exception as e:
        result["status"] = "error"
        result["warnings"] = []
        result["summary"] = {"tier_1": 0, "tier_2": 0, "tier_3": 0, "total": 0}
        result["error"] = str(e)
        result["by_disease"] = []
        result["medication_pair_clashes"] = []

    return result


NO_TREATMENTS_NOTE = "The graph lists no treatment drugs for this disease."
TREATMENT_GROUPS = ("already_taken_by_patient", "no_known_clash",
                    "clash_minor_or_unknown", "clash_moderate", "clash_major")


def _disease_treatment_options(disease, treatments, clash_meds, med_nodes,
                               contraindicated_meds, clash_severity):
    """
    One by_disease entry: the disease's treatment drugs grouped from safest
    to riskiest by each drug's WORST clash with a current medication.
    disease_source is left None for the caller (e.g. kg_agent) to fill in.
    """
    groups = {g: [] for g in TREATMENT_GROUPS}
    for t in sorted(treatments, key=str.lower):
        clashes = []
        for m in sorted(clash_meds[t], key=str.lower):
            level, source = clash_severity(t, m)
            clashes.append({
                "medication": m,
                "severity": level,
                "severity_source": source,
                "path": f"{disease} -[{treatments[t]}]- {t} -[drug_drug]- {m}",
            })
        entry = {"drug": t, "link_type": treatments[t], "clashes_with": clashes}

        if t in med_nodes:
            group = "already_taken_by_patient"
        elif not clashes:
            group = "no_known_clash"
        else:
            worst = max((c["severity"] for c in clashes), key=SEVERITY_RANK.__getitem__)
            group = {"major": "clash_major", "moderate": "clash_moderate"}.get(
                worst, "clash_minor_or_unknown")
        groups[group].append(entry)

    return {
        "disease": disease,
        "disease_source": None,
        "current_medication_problems": [
            {"medication": m, "problem": "contraindicated in this disease",
             "path": f"{m} -[contraindication]- {disease}"}
            for m in contraindicated_meds
        ],
        "treatment_drug_count": len(treatments),
        "treatment_options": groups,
        "note": None if treatments else NO_TREATMENTS_NOTE,
    }


if __name__ == "__main__":
    G = build_graph()
    print("=" * 70)
    print("GRAPH STATS (real PrimeKG subset, filtered to clinical relations)")
    print("=" * 70)
    print(json.dumps({
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "relation_types_used": sorted(RELEVANT_RELATIONS),
    }, indent=2))

    DEFAULT_CASE = {
        "diseases": ["Myocardial Infarction"],
        "symptoms": ["chest pain", "sweating", "shortness of breath"],
        "medications": ["Sildenafil"],
        "tests": [],
    }

    if len(sys.argv) > 1:
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
        print("\n" + "=" * 70)
        print(f"CLARIFIER INPUT: {input_path}")
        print("=" * 70)
    else:
        clarifier_output = DEFAULT_CASE
        print("\n" + "=" * 70)
        print("No input file given - running default worked example:")
        print("55yo, chest pain/sweating/SOB/arm pain, on Sildenafil -> MI")
        print("(usage: python kg_real_traversal.py <path_to_json_file>)")
        print("=" * 70)

    output = graph_traversal_agent(clarifier_output)
    print(json.dumps(output, indent=2))
