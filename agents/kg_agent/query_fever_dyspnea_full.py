"""
Re-run the fever / dyspnea disease-phenotype lookups against the full
relation-filtered graph (primekg_relation_filtered.csv via build_full_graph),
instead of the old primekg_subset.csv-based graph.

Does not touch kg.csv, primekg_subset.csv, or kg_real_traversal.py.
"""

from build_full_graph import build_graph

TARGET_RELATIONS = {"disease_phenotype_positive", "disease_phenotype_negative"}


def main():
    G = build_graph()

    # phenotype -> set of diseases, per relation type (positive vs negative)
    pheno_diseases = {}  # (phenotype_name, relation) -> set of disease names

    for u, v, data in G.edges(data=True):
        rel = data.get("relation")
        if rel not in TARGET_RELATIONS:
            continue
        u_type = G.nodes[u].get("type")
        v_type = G.nodes[v].get("type")
        if u_type == "effect/phenotype" and v_type == "disease":
            pheno, disease = u, v
        elif v_type == "effect/phenotype" and u_type == "disease":
            pheno, disease = v, u
        else:
            continue
        key = (pheno, rel)
        pheno_diseases.setdefault(key, set()).add(disease)

    all_phenos = sorted({p for (p, _rel) in pheno_diseases.keys()})
    fever_matches = [p for p in all_phenos if "fever" in p.lower()]
    dyspnea_matches = [
        p for p in all_phenos
        if "dyspnea" in p.lower() or "breath" in p.lower() or "respirat" in p.lower()
    ]

    print("=" * 70)
    print("FEVER-matching phenotype names:", fever_matches)
    print("DYSPNEA/BREATH-matching phenotype names:", dyspnea_matches)
    print("=" * 70)

    for p in fever_matches + dyspnea_matches:
        for rel in ("disease_phenotype_positive", "disease_phenotype_negative"):
            diseases = pheno_diseases.get((p, rel), set())
            print(f"\n{p} [{rel}] -> {len(diseases)} disease(s)")
            for d in sorted(diseases):
                print("  -", d)


if __name__ == "__main__":
    main()
