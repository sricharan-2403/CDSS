"""
For the Fever ∩ Dyspnea intersection (disease_phenotype_positive), compute
coverage_fraction = 2 / total_disease_phenotype_positive_connections per disease,
sorted descending (i.e. ascending by total connections - most "specific" disease
first). Reuses build_full_graph.build_graph.

Does not touch kg.csv, primekg_subset.csv, kg_real_traversal.py, or any
previously written script.
"""

from build_full_graph import build_graph

RELATION = "disease_phenotype_positive"


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


def main():
    G = build_graph()

    fever_pos = diseases_for(G, "Fever", RELATION)
    dyspnea_pos = diseases_for(G, "Dyspnea", RELATION)
    intersection = sorted(fever_pos & dyspnea_pos)

    rows = []
    for d in intersection:
        total = len(phenotypes_for(G, d, RELATION))
        coverage = 2 / total
        rows.append((d, total, coverage))

    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"{len(rows)} diseases, sorted by coverage_fraction (2/total) descending:\n")
    for d, total, coverage in rows:
        print(f"{d} — {coverage * 100:.1f}%  (2/{total})")


if __name__ == "__main__":
    main()
