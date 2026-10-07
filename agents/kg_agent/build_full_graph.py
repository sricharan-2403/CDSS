"""
Build a NetworkX graph from primekg_relation_filtered.csv, using the same
graph type and node/edge structure as build_graph() in kg_real_traversal.py.
Times the build and reports size/memory.

Does not touch kg.csv, primekg_subset.csv, or kg_real_traversal.py.
"""

import time
import os
import pandas as pd
import networkx as nx

INPUT_FILE = "primekg_relation_filtered.csv"


def build_graph(csv_path=INPUT_FILE):
    df = pd.read_csv(csv_path)

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


def main():
    t0 = time.perf_counter()
    G = build_graph()
    elapsed = time.perf_counter() - t0

    print("=" * 70)
    print("FULL RELATION-FILTERED GRAPH BUILD")
    print("=" * 70)
    print(f"Build time: {elapsed:.2f} sec")
    print(f"Node count: {G.number_of_nodes()}")
    print(f"Edge count: {G.number_of_edges()}")

    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024 ** 2)
        print(f"Process RSS memory: {rss_mb:.1f} MB")
    except ImportError:
        print("psutil not installed - skipping process memory measurement.")

    return G


if __name__ == "__main__":
    main()
