"""
Intersect diseases connected to "Fever" and "Dyspnea" via disease_phenotype_positive,
using the graph from build_full_graph.py (primekg_relation_filtered.csv).
Also flags any intersection disease that has a disease_phenotype_negative edge
to either symptom (contradictory signal).

Does not touch kg.csv, primekg_subset.csv, or kg_real_traversal.py.
"""

from build_full_graph import build_graph

SYMPTOMS = ["Fever", "Dyspnea"]


def diseases_for(G, phenotype_name, relation):
    result = set()
    if not G.has_node(phenotype_name):
        return result
    # phenotype -> disease direction is inconsistent in PrimeKG (x/y not fixed),
    # so check edges in both directions touching this node.
    for nbr in G.predecessors(phenotype_name):
        for _, attrs in G.get_edge_data(nbr, phenotype_name).items():
            if attrs["relation"] == relation and G.nodes[nbr].get("type") == "disease":
                result.add(nbr)
    for nbr in G.successors(phenotype_name):
        for _, attrs in G.get_edge_data(phenotype_name, nbr).items():
            if attrs["relation"] == relation and G.nodes[nbr].get("type") == "disease":
                result.add(nbr)
    return result


def main():
    G = build_graph()

    fever_pos = diseases_for(G, "Fever", "disease_phenotype_positive")
    dyspnea_pos = diseases_for(G, "Dyspnea", "disease_phenotype_positive")

    print("=" * 70)
    print(f"Fever [disease_phenotype_positive] -> {len(fever_pos)} diseases")
    print(f"Dyspnea [disease_phenotype_positive] -> {len(dyspnea_pos)} diseases")
    print("=" * 70)

    intersection = sorted(fever_pos & dyspnea_pos)
    print(f"\nINTERSECTION (both Fever AND Dyspnea positive): {len(intersection)} disease(s)")
    for d in intersection:
        print("  -", d)

    fever_neg = diseases_for(G, "Fever", "disease_phenotype_negative")
    dyspnea_neg = diseases_for(G, "Dyspnea", "disease_phenotype_negative")

    print("\n" + "=" * 70)
    print("CONTRADICTORY-SIGNAL CHECK (intersection diseases that also appear")
    print("as disease_phenotype_negative for Fever and/or Dyspnea)")
    print("=" * 70)

    contradictions = []
    for d in intersection:
        neg_fever = d in fever_neg
        neg_dyspnea = d in dyspnea_neg
        if neg_fever or neg_dyspnea:
            contradictions.append((d, neg_fever, neg_dyspnea))

    if not contradictions:
        print("None - no intersection disease has a disease_phenotype_negative edge for either symptom.")
    else:
        for d, neg_fever, neg_dyspnea in contradictions:
            flags = []
            if neg_fever:
                flags.append("Fever ruled out (disease_phenotype_negative)")
            if neg_dyspnea:
                flags.append("Dyspnea ruled out (disease_phenotype_negative)")
            print(f"  - {d}: {', '.join(flags)}")


if __name__ == "__main__":
    main()
