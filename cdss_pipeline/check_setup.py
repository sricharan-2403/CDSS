r"""Light setup check: agent folders and KG data files. Loads no model, no graph, no network.

From the integration (or cdss_release) folder:
    .venv\Scripts\python.exe -B -m cdss_pipeline.check_setup

KG data (graph pickle + DDInter CSVs) is not in git. Each file must be in the KG agent folder with the
SHA-256 listed in setup\kg_data\cdss_kg_data_SHA256.txt. Exit code 0 = all OK, 1 = something missing/wrong.
"""

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

from cdss_pipeline import config

EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"  # Vector RAG (his code)

AGENT_FOLDERS = {"Clarifier": "CLARIFIER_DIR", "Vector RAG": "RAG_DIR", "Knowledge Graph": "CODE_DIR",
                 "Evidence Scanner": "EVIDENCE_DIR", "Fusion (Medico)": "MEDICO_DIR"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sha_list(path):
    """-> {relative path: sha256} for the files INSIDE the zip (the zip's own line is skipped)."""
    expected = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and not line.startswith("#") and not parts[1].strip().endswith(".zip"):
            expected[parts[1].strip()] = parts[0].lower()
    return expected


def check_kg_data(kg_dir, sha_file):
    """-> list of problem lines ([] = OK). Prints one line per file."""
    if not Path(sha_file).is_file():
        return [f"SHA-256 list missing: {sha_file}"]
    problems = []
    for rel, want in read_sha_list(sha_file).items():
        path = Path(kg_dir) / rel
        if not path.is_file():
            problems.append(f"missing  {path}")
        elif sha256(path) != want:
            problems.append(f"WRONG SHA-256  {path} (damaged or a different version)")
        else:
            print(f"  OK  {rel}")
    return problems


def download_model():
    """One-time download of the embedding model into HF_HOME (.hf_cache). Runs are offline afterwards."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download
    print(f"Downloading {EMBEDDING_MODEL} into {os.environ['HF_HOME']} ...")
    snapshot_download(EMBEDDING_MODEL)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Light setup check (no model or graph is loaded).")
    parser.add_argument("--download-model", action="store_true",
                        help="download the Vector RAG embedding model once (needs internet)")
    if parser.parse_args().download_model:
        download_model()
    print(f"Layout: {'team repo (agents\\)' if config.REPO_LAYOUT else 'local (integration + ..\\code)'}")
    ok = True
    for label, attr in AGENT_FOLDERS.items():
        folder = getattr(config, attr)
        found = Path(folder).is_dir()
        ok &= found
        print(f"  {'OK ' if found else 'MISSING'}  {label}: {folder}")

    print("\nVector RAG import (.pth line) and embedding model")
    spec = importlib.util.find_spec("symptom_rag_analyzer")  # finds the folder only, imports nothing heavy
    want = Path(config.RAG_DIR) / "src" / "symptom_rag_analyzer"
    found_at = Path(spec.origin).parent if spec and spec.origin else None
    if found_at is not None and found_at.resolve() == want.resolve():
        print(f"  OK   symptom_rag_analyzer from {found_at}")
    else:
        ok = False
        print(f"  PROBLEM  symptom_rag_analyzer found at {found_at} - expected {want}")
        print(f"  Fix: create .venv\\Lib\\site-packages\\symptom_rag_analyzer_src.pth with the one line "
              f"{os.path.relpath(want.parent, Path(sys.prefix) / 'Lib' / 'site-packages')}")
    model = Path(os.environ["HF_HOME"]) / "hub" / f"models--{EMBEDDING_MODEL.replace('/', '--')}"
    if model.is_dir():
        print(f"  OK   embedding model in {model}")
    else:
        ok = False
        print(f"  MISSING  embedding model {EMBEDDING_MODEL} in {model}")
        print("  Fix (one download, needs internet): .venv\\Scripts\\python.exe -B -m cdss_pipeline.check_setup "
              "--download-model")

    print("\n.env (values never shown)")
    if not (config.INTEGRATION_DIR / ".env").is_file():
        ok = False
        print("  MISSING  .env - copy .env.example to .env and fill in your own keys")
    for name, state in config.key_status().items():
        if name in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "CLINICAL_TEXT_MODEL"):
            ok &= state == "set"
            print(f"  {'OK ' if state == 'set' else 'MISSING'}  {name}")

    print(f"\nKG data in {config.CODE_DIR}")
    problems = check_kg_data(config.CODE_DIR, config.KG_DATA_FILES_SHA256)
    for line in problems:
        print(f"  {line}")
    if problems:
        ok = False
        print("\n  How to fix: download cdss_kg_data.zip from", config.KG_DATA_URL)
        print(f"  and extract it INTO {config.CODE_DIR}")
        print("  (relation_filtered_graph_v2.pkl and the data_sources folder must sit next to kg_agent.py).")
        print("  Check the zip first: its SHA-256 is the cdss_kg_data.zip line in", config.KG_DATA_FILES_SHA256)

    print("\nSetup check:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
