"""
Load relation_filtered_graph_v2.pkl via load_graph() and verify node/edge
counts plus zero name collisions. Read-only check, no CSV/graph rebuilding.
"""

import time

from build_full_graph_v2 import load_graph


def main():
    t0 = time.perf_counter()
    G = load_graph()
    elapsed = time.perf_counter() - t0

    print(f"load_graph() time: {elapsed:.3f} sec")
    print(f"Node count: {G.number_of_nodes()}")
    print(f"Edge count: {G.number_of_edges()}")
    print(f"Edge count == 3,162,770: {G.number_of_edges() == 3162770}")
    print(f"Node count == 36,299: {G.number_of_nodes() == 36299}")

    groups = {}
    for name in G.nodes():
        key = name.strip().lower()
        groups.setdefault(key, set()).add(name)
    colliding = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"Unique node names (exact): {len(list(G.nodes()))}")
    print(f"Unique node names (lowercased+stripped): {len(groups)}")
    print(f"Colliding groups: {len(colliding)}")


if __name__ == "__main__":
    main()
