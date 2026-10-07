"""
Data-quality diagnostics on primekg_relation_filtered.csv: dtype consistency
of x_id/y_id, exact duplicate rows, and near-duplicate node names that differ
only by casing/whitespace. Read-only - does not modify the CSV or build a graph.
"""

import pandas as pd

INPUT_FILE = "primekg_relation_filtered.csv"


def dtype_check(df):
    print("=" * 70)
    print("1. DTYPE CHECK: x_id / y_id")
    print("=" * 70)
    print(f"x_id pandas dtype: {df['x_id'].dtype}")
    print(f"y_id pandas dtype: {df['y_id'].dtype}")

    for col in ("x_id", "y_id"):
        type_counts = df[col].map(lambda v: type(v).__name__).value_counts()
        print(f"\n{col} - Python types actually present (with counts):")
        print(type_counts.to_string())

        if len(type_counts) > 1:
            print(f"\n{col} - MIXED: showing example rows per type:")
            seen_types = set()
            examples = []
            for idx, val in df[col].items():
                t = type(val).__name__
                if t not in seen_types:
                    seen_types.add(t)
                    examples.append((idx, val, t))
                if len(seen_types) == len(type_counts):
                    break
            for idx, val, t in examples:
                print(f"  row {idx}: {col} = {val!r}  (python type: {t})")
        else:
            print(f"{col} - consistently one type row-to-row.")


def duplicate_check(df):
    print("\n" + "=" * 70)
    print("2. DUPLICATE ROWS (same x_id, relation, y_id)")
    print("=" * 70)
    dup_mask = df.duplicated(subset=["x_id", "relation", "y_id"], keep=False)
    dup_rows = df[dup_mask]
    # count of *extra* duplicate rows (i.e. rows beyond the first occurrence)
    extra_dup_count = df.duplicated(subset=["x_id", "relation", "y_id"], keep="first").sum()
    print(f"Total rows involved in a duplicate group: {len(dup_rows)}")
    print(f"Extra duplicate rows (beyond first occurrence): {extra_dup_count}")

    if len(dup_rows) > 0:
        print("\nExample duplicate groups:")
        groups = dup_rows.groupby(["x_id", "relation", "y_id"])
        shown = 0
        for key, group in groups:
            print(f"\n  x_id={key[0]}, relation={key[1]}, y_id={key[2]}  ({len(group)} rows)")
            print(group[["x_name", "relation", "y_name", "x_id", "y_id"]].to_string(index=True))
            shown += 1
            if shown >= 3:
                break
    else:
        print("No exact duplicate rows found.")


def near_duplicate_node_names(df):
    print("\n" + "=" * 70)
    print("3. NEAR-DUPLICATE NODE NAMES (case/whitespace collisions)")
    print("=" * 70)

    all_names = pd.concat([df["x_name"], df["y_name"]], ignore_index=True).dropna()
    unique_exact = set(all_names.unique())

    groups = {}
    for name in unique_exact:
        key = name.strip().lower()
        groups.setdefault(key, set()).add(name)

    colliding = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"Unique node names (exact match): {len(unique_exact)}")
    print(f"Unique node names (lowercased+stripped match): {len(groups)}")
    print(f"Groups with more than one distinct original spelling: {len(colliding)}")

    if colliding:
        print("\nColliding groups:")
        for key, variants in sorted(colliding.items()):
            print(f"  {key!r} -> {sorted(variants)}")
    else:
        print("No case/whitespace collisions found.")

    return len(unique_exact), len(groups)


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}\n")

    dtype_check(df)
    duplicate_check(df)
    near_duplicate_node_names(df)


if __name__ == "__main__":
    main()
