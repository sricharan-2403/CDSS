"""
For the Fever/Dyspnea intersection (disease_phenotype_positive), compute the
specificity-weighted score (1/total_Fever_diseases + 1/total_Dyspnea_diseases,
constant across the intersection by construction) and, more usefully, each
disease's total phenotype-connection count via disease_phenotype_positive,
sorted ascending (fewest phenotypes = most "specific"/rare disease).

Reuses build_full_graph.build_graph. Does not touch kg.csv, primekg_subset.csv,
kg_real_traversal.py, or any previously written script.
"""

from build_full_graph import build_graph

SYMPTOMS = ["Fever", "Dyspnea"]
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

    total_fever = len(fever_pos)
    total_dyspnea = len(dyspnea_pos)
    score = (1 / total_fever) + (1 / total_dyspnea)

    print("=" * 70)
    print(f"Fever total diseases: {total_fever}  ->  weight 1/{total_fever} = {1/total_fever:.6f}")
    print(f"Dyspnea total diseases: {total_dyspnea}  ->  weight 1/{total_dyspnea} = {1/total_dyspnea:.6f}")
    print(f"Specificity-weighted score (same for all {len(intersection)} diseases): {score:.6f}")
    print("=" * 70)

    rows = []
    for d in intersection:
        phenos = phenotypes_for(G, d, RELATION)
        rows.append((d, len(phenos)))

    rows.sort(key=lambda r: r[1])

    print(f"\n{len(rows)} intersection diseases, sorted by total disease_phenotype_positive")
    print("connections ascending (fewest = most 'specific'/rare; most = most 'generalist'):\n")
    print(f"{'total_phenotypes':>16}   disease")
    for d, n in rows:
        print(f"{n:>16}   {d}")

    counts = [n for _, n in rows]
    print("\n" + "=" * 70)
    print(f"Min: {min(counts)}   Max: {max(counts)}   Median: {sorted(counts)[len(counts)//2]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
