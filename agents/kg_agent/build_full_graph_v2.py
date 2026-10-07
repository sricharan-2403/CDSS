"""
Rebuild the relation-filtered PrimeKG graph with the two data-quality fixes
found in data_quality_checks.py applied:

  1. x_id / y_id forced to a single consistent type (str) right after load.
  2. x_name / y_name casing normalized so case-variant duplicates (e.g.
     "Pneumonia" vs "pneumonia") collapse into one node. Rule used: keep the
     FIRST-encountered original casing (scanning x_name then y_name, in row
     order) as the canonical spelling for that lowercased+stripped identity.

Then: rebuild the graph, report size/timing, re-run the near-duplicate-name
check to confirm zero collisions, re-run the Fever/Dyspnea coverage_fraction
check and diff it against the pre-fix numbers, and finally pickle the graph
plus provide a small loader function.

Does not touch kg.csv, primekg_subset.csv, primekg_relation_filtered.csv,
kg_real_traversal.py, or any previously written script.
"""

import pickle
import time

import networkx as nx
import pandas as pd

INPUT_FILE = "primekg_relation_filtered.csv"
PICKLE_FILE = "relation_filtered_graph_v2.pkl"


def load_and_fix(csv_path=INPUT_FILE):
    df = pd.read_csv(csv_path, low_memory=False)

    # Fix 1: force x_id / y_id to a single consistent type (string).
    df["x_id"] = df["x_id"].astype(str).str.strip()
    df["y_id"] = df["y_id"].astype(str).str.strip()

    # Fix 2: canonicalize x_name / y_name casing. Build a lowercased+stripped
    # -> first-encountered-original-casing map by scanning x_name then y_name
    # in row order (interleaved via reshape, which flattens each row's
    # [x_name, y_name] pair in sequence).
    names_flat = pd.Series(df[["x_name", "y_name"]].to_numpy().reshape(-1)).astype(str).str.strip()
    keys_flat = names_flat.str.lower()
    first_mask = ~keys_flat.duplicated(keep="first")
    canon_map = dict(zip(keys_flat[first_mask], names_flat[first_mask]))

    df["x_name"] = df["x_name"].astype(str).str.strip().str.lower().map(canon_map)
    df["y_name"] = df["y_name"].astype(str).str.strip().str.lower().map(canon_map)

    return df


def build_graph_fixed(csv_path=INPUT_FILE):
    df = load_and_fix(csv_path)

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


def load_graph(pkl_path=PICKLE_FILE):
    """Loader: read the pre-built, fixed graph from disk instead of rebuilding from CSV."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def near_duplicate_check(node_names):
    groups = {}
    for name in node_names:
        key = name.strip().lower()
        groups.setdefault(key, set()).add(name)
    colliding = {k: v for k, v in groups.items() if len(v) > 1}
    return len(node_names), len(groups), colliding


def diseases_for(G, phenotype_name, relation):
    result = set()
    if not G.has_node(phenotype_name):
        return result
    for nbr in G.predecessors(phenotype_name):
        for _, attrs in G.get_edge_data(nbr, phenotype_name).items():
            if attrs["relation"] == relation and G.nodes[nbr].get("type") == "disease":
                result.add(nbr)
    for nbr in G.successors(phenotype_name):
        for _, attrs in G.get_edge_data(phenotype_name, nbr).items():
            if attrs["relation"] == relation and G.nodes[nbr].get("type") == "disease":
                result.add(nbr)
    return result


def phenotypes_for(G, disease_name, relation):
    result = set()
    for nbr in G.predecessors(disease_name):
        for _, attrs in G.get_edge_data(nbr, disease_name).items():
            if attrs["relation"] == relation and G.nodes[nbr].get("type") == "effect/phenotype":
                result.add(nbr)
    for nbr in G.successors(disease_name):
        for _, attrs in G.get_edge_data(disease_name, nbr).items():
            if attrs["relation"] == relation and G.nodes[nbr].get("type") == "effect/phenotype":
                result.add(nbr)
    return result


# Pre-fix results, from the earlier run against the unfixed graph (built from
# primekg_relation_filtered.csv with no casing/dtype normalization), used
# below to check whether the casing bug actually changed any numbers.
PREVIOUS_TOTALS = {
    "pulmonary blastoma": 8,
    "inhalational anthrax": 11,
    "wound botulism": 12,
    "Lujo hemorrhagic fever": 13,
    "letterer-Siwe disease": 14,
    "severe acute respiratory syndrome": 14,
    "Hughes-Stovin syndrome": 17,
    "idiopathic bronchiectasis": 18,
    "pulmonary non-tuberculous mycobacterial infection": 18,
    "acute lung injury": 19,
    "thrombocytopenic purpura": 19,
    "Hodgkins lymphoma": 20,
    "drug-induced lupus erythematosus": 20,
    "recurrent respiratory papillomatosis": 21,
    "amoebiasis due to Entamoeba histolytica": 22,
    "pneumocystosis": 22,
    "scrub typhus": 23,
    "Chagas disease": 24,
    "aregenerative anemia": 24,
    "tracheobronchopathia osteochondroplastica": 24,
    "autoimmune hemolytic anemia": 25,
    "cryptogenic organizing pneumonia": 25,
    "drug rash with eosinophilia and systemic symptoms": 25,
    "autoimmune pulmonary alveolar proteinosis": 26,
    "acute interstitial pneumonia": 30,
    "pauci-immune glomerulonephritis": 30,
    "Erdheim-Chester disease": 32,
    "Stevens-Johnson syndrome": 33,
    "paroxysmal nocturnal hemoglobinuria": 33,
    "connective tissue disease": 34,
    "diffuse alveolar hemorrhage (disease)": 34,
    "staphylococcal pneumonia": 34,
    "avian influenza": 35,
    "cryptococcosis": 36,
    "lymphangioleiomyomatosis": 36,
    "fructose-1,6-bisphosphatase deficiency": 39,
    "nocardiosis": 40,
    "sarcoidosis, susceptibility to": 41,
    "hepatocellular carcinoma": 42,
    "tropical endomyocardial fibrosis": 42,
    "Blau syndrome": 43,
    "kaposiform lymphangiomatosis": 44,
    "pediatric systemic lupus erythematosus": 46,
    "maternally-inherited Leigh syndrome": 47,
    "alveolar echinococcosis": 48,
    "aspergillosis": 52,
    "Wiskott-Aldrich syndrome": 64,
    "dermatomyositis": 67,
    "sarcoidosis": 67,
    "Fabry disease": 69,
    "idiopathic hypereosinophilic syndrome": 76,
    "Gaucher disease": 115,
    "glycogen storage disease": 156,
}
PREVIOUS_FEVER_TOTAL = 289
PREVIOUS_DYSPNEA_TOTAL = 179


def main():
    print("=" * 70)
    print("STEP 3: REBUILD WITH FIXES APPLIED")
    print("=" * 70)
    t0 = time.perf_counter()
    G = build_graph_fixed()
    elapsed = time.perf_counter() - t0

    print(f"Build time: {elapsed:.2f} sec")
    print(f"Node count: {G.number_of_nodes()}  (pre-fix was 36879)")
    print(f"Edge count: {G.number_of_edges()}")

    print("\n" + "=" * 70)
    print("STEP 4: NEAR-DUPLICATE NAME CHECK ON NEW GRAPH")
    print("=" * 70)
    exact_count, folded_count, colliding = near_duplicate_check(list(G.nodes()))
    print(f"Unique node names (exact match): {exact_count}")
    print(f"Unique node names (lowercased+stripped match): {folded_count}")
    print(f"Colliding groups remaining: {len(colliding)}")
    if colliding:
        print("UNEXPECTED - collisions remain:")
        for k, v in list(colliding.items())[:10]:
            print(f"  {k!r} -> {sorted(v)}")
    else:
        print("Confirmed: zero collisions remain.")

    print("\n" + "=" * 70)
    print("STEP 5: FEVER/DYSPNEA COVERAGE RE-CHECK ON FIXED GRAPH")
    print("=" * 70)
    fever_pos = diseases_for(G, "Fever", "disease_phenotype_positive")
    dyspnea_pos = diseases_for(G, "Dyspnea", "disease_phenotype_positive")
    intersection = sorted(fever_pos & dyspnea_pos)

    print(f"Fever total diseases: {len(fever_pos)}  (pre-fix was {PREVIOUS_FEVER_TOTAL})")
    print(f"Dyspnea total diseases: {len(dyspnea_pos)}  (pre-fix was {PREVIOUS_DYSPNEA_TOTAL})")
    print(f"Intersection size: {len(intersection)}  (pre-fix was {len(PREVIOUS_TOTALS)})")

    rows = []
    any_diff = False
    for d in intersection:
        total = len(phenotypes_for(G, d, "disease_phenotype_positive"))
        coverage = 2 / total
        prev_total = PREVIOUS_TOTALS.get(d)
        changed = prev_total is not None and prev_total != total
        if changed or prev_total is None:
            any_diff = True
        rows.append((d, total, coverage, prev_total, changed))

    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"\n{len(rows)} diseases, sorted by coverage_fraction (2/total) descending:")
    print(f"{'coverage':>9}  {'new_total':>9}  {'prev_total':>10}  {'changed?':>9}  disease")
    for d, total, coverage, prev_total, changed in rows:
        flag = "YES" if changed else ("NEW" if prev_total is None else "no")
        print(f"{coverage*100:>8.1f}%  {total:>9}  {str(prev_total):>10}  {flag:>9}  {d}")

    print("\n" + "-" * 70)
    if any_diff:
        print("DIFFERENCE FOUND: at least one disease's total/coverage changed after the fix.")
    else:
        print("NO DIFFERENCE: every disease's total connection count and coverage_fraction")
        print("is identical before and after the casing/dtype fix.")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("SAVING FIXED GRAPH TO PICKLE")
    print("=" * 70)
    t0 = time.perf_counter()
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_elapsed = time.perf_counter() - t0
    print(f"Saved {PICKLE_FILE} in {save_elapsed:.2f} sec")

    print("\n" + "=" * 70)
    print("VERIFYING LOADER FUNCTION (load_graph)")
    print("=" * 70)
    t0 = time.perf_counter()
    G_loaded = load_graph()
    load_elapsed = time.perf_counter() - t0
    print(f"load_graph() loaded in {load_elapsed:.2f} sec")
    print(f"Loaded graph nodes: {G_loaded.number_of_nodes()}, edges: {G_loaded.number_of_edges()}")
    print(f"Matches freshly-built graph: {G_loaded.number_of_nodes() == G.number_of_nodes() and G_loaded.number_of_edges() == G.number_of_edges()}")


if __name__ == "__main__":
    main()
