"""
STEP 1 — run this on YOUR machine (not in the chat).
Requires: Python 3.8+, and the 'pandas' and 'requests' packages.
    pip install pandas requests

What this does:
  1. Downloads the real PrimeKG kg.csv file (~600MB-1GB) from Harvard Dataverse.
  2. Streams through it in chunks (so it never needs your whole RAM at once).
  3. Keeps only rows that mention our case-study terms (symptoms, diseases,
     drugs from the sildenafil/nitroglycerin worked example) - PLUS a few
     extra terms so we get a slightly wider net to explore.
  4. Saves the small filtered result as 'primekg_subset.csv' in the same
     folder you run this from.

Once it finishes, upload 'primekg_subset.csv' back in the chat and I'll
take it from there (build the graph, run traversal, visualize it).

NOTE: the download itself may take a few minutes depending on your
internet speed - it's a large file. The filtering step after that is fast.
"""

import requests
import pandas as pd
import io

DOWNLOAD_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
LOCAL_RAW_FILE = "kg.csv"
OUTPUT_FILE = "primekg_subset.csv"

# Terms relevant to our case study - lowercase, partial match.
# Feel free to add more terms here later (e.g. other symptoms/drugs
# your team wants to explore) and re-run just the filtering step.
KEYWORDS = [
    "chest pain",
    "diaphoresis",
    "sweating",
    "dyspnea",
    "shortness of breath",
    "myocardial infarction",
    "angina",
    "gastroesophageal reflux",
    "gerd",
    "hypertension",
    "nitroglycerin",
    "sildenafil",
    "aspirin",
]


def download_if_needed():
    import os
    if os.path.exists(LOCAL_RAW_FILE):
        print(f"{LOCAL_RAW_FILE} already exists locally, skipping download.")
        return
    print("Downloading PrimeKG kg.csv from Harvard Dataverse... this is a large file, please wait.")
    with requests.get(DOWNLOAD_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(LOCAL_RAW_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:5.1f}% downloaded", end="")
    print("\nDownload complete.")


def filter_in_chunks():
    print("Filtering kg.csv for case-study terms (this streams through the file, low memory)...")
    matched_chunks = []
    chunk_iter = pd.read_csv(LOCAL_RAW_FILE, chunksize=200_000)
    total_rows = 0
    for i, chunk in enumerate(chunk_iter):
        total_rows += len(chunk)
        # kg.csv columns are typically:
        # relation, display_relation, x_id, x_type, x_name, x_source,
        # y_id, y_type, y_name, y_source
        x_name = chunk["x_name"].astype(str).str.lower()
        y_name = chunk["y_name"].astype(str).str.lower()
        mask = pd.Series(False, index=chunk.index)
        for kw in KEYWORDS:
            mask |= x_name.str.contains(kw, na=False) | y_name.str.contains(kw, na=False)
        if mask.any():
            matched_chunks.append(chunk[mask])
        print(f"\r  processed {total_rows:,} rows so far...", end="")

    print("\nDone scanning.")
    if matched_chunks:
        result = pd.concat(matched_chunks, ignore_index=True)
        result.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(result):,} matching rows to {OUTPUT_FILE}")
        print("\nPreview:")
        print(result.head(20).to_string())
        print(f"\nUnique relation types found: {sorted(result['relation'].unique())}")
    else:
        print("No matches found - check the keyword list or column names.")


if __name__ == "__main__":
    download_if_needed()
    filter_in_chunks()
