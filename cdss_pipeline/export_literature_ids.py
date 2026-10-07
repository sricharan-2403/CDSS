r"""Export the unique PMCID / PMID list of Sri Charan's literature_chunks (for our leakage check).

READ-ONLY against Qdrant: it only scrolls payload fields pmid and pmcid (no vectors, no writes).
Qdrant must be running (start_qdrant.ps1). From the integration folder:
    .venv\Scripts\python.exe -B -m cdss_pipeline.export_literature_ids [--out leakage_check\literature_ids.txt]

Output: a short header (counts), then one line per unique article: PMCID<TAB>PMID
(empty field if the payload has no value).
"""

import argparse
from pathlib import Path

from cdss_pipeline import config

COLLECTION = "literature_chunks"
DEFAULT_OUT = config.INTEGRATION_DIR / "leakage_check" / "literature_ids.txt"


def _clean(value):
    return "" if value is None else str(value).strip()


def collect_ids(client, collection=COLLECTION, batch=2000):
    """-> (set of (pmcid, pmid) pairs, points scanned, points without any id). Read-only scroll."""
    pairs, scanned, missing, offset = set(), 0, 0, None
    while True:
        points, offset = client.scroll(collection_name=collection, limit=batch, offset=offset,
                                       with_payload=["pmid", "pmcid"], with_vectors=False)
        for point in points:
            payload = point.payload or {}
            pair = (_clean(payload.get("pmcid")), _clean(payload.get("pmid")))
            scanned += 1
            if pair == ("", ""):
                missing += 1
            else:
                pairs.add(pair)
        if offset is None:
            return pairs, scanned, missing


def write_ids(pairs, scanned, missing, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pmcids = {p for p, _ in pairs if p}
    pmids = {m for _, m in pairs if m}
    lines = [f"# {COLLECTION}: {scanned} points scanned, {missing} without pmid/pmcid",
             f"# unique articles (pmcid, pmid pairs): {len(pairs)}; unique PMCIDs: {len(pmcids)}; "
             f"unique PMIDs: {len(pmids)}",
             "PMCID\tPMID"]
    lines += [f"{p}\t{m}" for p, m in sorted(pairs)]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"points": scanned, "without_ids": missing, "articles": len(pairs),
            "pmcids": len(pmcids), "pmids": len(pmids), "file": str(out_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    from cdss_pipeline import wrappers
    client = wrappers._qdrant_client(timeout=120)
    wrappers.check_qdrant(client)  # stops with a clear message if Qdrant / the collections are wrong
    print(write_ids(*collect_ids(client), args.out))


if __name__ == "__main__":
    main()
