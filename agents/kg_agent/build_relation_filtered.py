"""
Build primekg_relation_filtered.csv from the raw full PrimeKG kg.csv.

Unlike fetch_and_filter_primekg.py (which also applies a keyword filter for
case-study terms), this script keeps ALL rows for a fixed set of relation
types, with no keyword/entity filtering. Streams kg.csv in chunks to keep
memory manageable given the file is ~935MB / 8.1M rows.

Does not touch kg.csv, primekg_subset.csv, or fetch_and_filter_primekg.py.
"""

import pandas as pd

RAW_FILE = "kg.csv"
OUTPUT_FILE = "primekg_relation_filtered.csv"

KEEP_RELATIONS = {
    "contraindication",
    "disease_disease",
    "disease_phenotype_positive",
    "drug_drug",
    "indication",
    "off-label use",
    "phenotype_phenotype",
    "disease_phenotype_negative",
}

CHUNK_SIZE = 500_000


def main():
    first_chunk = True
    total_rows = 0
    counts = pd.Series(dtype="int64")

    for chunk in pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE):
        filtered = chunk[chunk["relation"].isin(KEEP_RELATIONS)]
        if filtered.empty:
            continue
        counts = counts.add(filtered["relation"].value_counts(), fill_value=0)
        total_rows += len(filtered)
        filtered.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

    counts = counts.astype("int64").sort_values(ascending=False)

    print("=" * 70)
    print(f"Wrote {OUTPUT_FILE}")
    print("=" * 70)
    print(f"TOTAL ROWS: {total_rows}")
    print()
    print("ROWS PER RELATION TYPE:")
    for rel, n in counts.items():
        print(f"{n:>10}  {rel}")


if __name__ == "__main__":
    main()
