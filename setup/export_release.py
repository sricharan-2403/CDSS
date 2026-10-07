r"""Build SDC_2\cdss_release (the team repo folder) from the local folders. Re-runnable.

From the integration folder:
    .venv\Scripts\python.exe -B setup\export_release.py            (builds / rebuilds cdss_release)
    .venv\Scripts\python.exe -B setup\export_release.py --dry-run  (only prints what it would do)

What it does:
  1. Each agent = `git archive` of its local clone at the PINNED commit (tracked files only: no .git, no .env,
     no data, no local changes) -> cdss_release\agents\<name>\. Writes agents\PINNED_COMMITS.txt.
  2. Copies OUR integration files from an allow-list (cdss_pipeline, setup, requirements, TESTING.md, ...).
  3. Writes .gitignore, .env.example (safe defaults, keys EMPTY) and README.md (from setup\release_README.md).
  4. Safety scan: stops with an error if any .env / .venv / .git / .hf_cache / big file got in.
Never touches the source folders (git archive is read-only). Never commits or pushes anything.
A rebuild deletes cdss_release only if it carries our marker file (so a wrong folder is never deleted).
"""

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

INTEGRATION = Path(__file__).resolve().parent.parent
SDC = INTEGRATION.parent
OUT = SDC / "cdss_release"
MARKER = ".cdss_release_marker"

# name in agents\ -> (local clone, pinned commit, GitHub URL)
AGENTS = {
    "kg_agent": (SDC / "code", "cb9d20a", "https://github.com/rakshitha-dev9/kg_agent"),
    "evidence_scanner": (INTEGRATION / "evidence_scanner", "d4fc935",
                         "https://github.com/VaishnaviKoosam/evidence_scanner"),
    "Symptom_RAG_Frozen": (INTEGRATION / "Symptom_RAG_Frozen", "7a80dfc",
                           "https://github.com/sricharan-2403/Symptom_RAG_Frozen"),
    "clinical_text_clarifier": (INTEGRATION / "clinical_text_clarifier", "51dacaf",
                                "https://github.com/varshith-3011/clinical_text_clarifier"),
    "Medico": (INTEGRATION / "Medico", "38a634f", "https://github.com/charanya-reddy/Medico"),
}

# Our files (allow-list). Everything else in integration stays out: .env, .venv, .hf_cache, backups,
# downloads, results, retired, old Clarifier / old Vector RAG clones, review / check reports, CLAUDE.md.
OUR_FILES = ["cdss_pipeline", "setup", "requirements_combined.txt", "requirements_lock.txt", "TESTING.md",
             "kg_case1_output.json"]
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".env", "results", "leakage_check"}
SKIP_SUFFIXES = {".pyc"}

# .env.example: names with SAFE defaults only (models, URLs). Keys and account ids stay empty.
ENV_EXAMPLE = """# Copy this file to .env and fill in YOUR OWN keys. Never commit .env.
GROQ_API_KEY=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
CLINICAL_TEXT_MODEL=openai/gpt-oss-120b
# Temporary: Evidence Scanner Groq model (her model name is retired on Groq)
GROQ_MODEL_OVERRIDE=openai/gpt-oss-120b
# Optional: Vector RAG answer model (default: OPENROUTER_MODEL)
RAG_ANSWER_MODEL=
QDRANT_URL=http://localhost:6333
# Only if Groq answers 403 "Access denied" on your network (ask Rakshitha for the two ids)
GROQ_VIA_GATEWAY=false
CF_ACCOUNT_ID=
CF_AI_GATEWAY_ID=
CDSS_KG_DATA_URL=https://drive.google.com/drive/folders/16ztB2D_6Uyk1rEKDugCZDqUOIoQV9Ycw?usp=drive_link
# Optional folder overrides (absolute, or relative to this folder); leave empty for the repo layout
CDSS_KG_DIR=
CDSS_EVIDENCE_DIR=
CDSS_RAG_DIR=
CDSS_MEDICO_DIR=
CDSS_CLARIFIER_DIR=
"""

GITIGNORE = """\
.env
.env.*
!.env.example
.venv/
.hf_cache/
__pycache__/
*.pyc
.pytest_cache/
.groq_limits.json
results/
leakage_check/
backups/
downloads/
.cdss_release_marker
# KG data comes from the team Google Drive zip (see README), never from git
*.pkl
agents/kg_agent/data_sources/
agents/kg_agent/*.csv
agents/kg_agent/hp.json
"""

README_SOURCE = INTEGRATION / "setup" / "release_README.md"  # becomes README.md


def git(folder, *args, binary=False):
    proc = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=not binary)
    if proc.returncode != 0:
        err = proc.stderr if not binary else proc.stderr.decode(errors="replace")
        raise SystemExit(f"git {' '.join(args)} failed in {folder}: {err.strip()}")
    return proc.stdout


def export_agent(name, folder, commit, dest):
    full = git(folder, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    data = git(folder, "archive", "--format=tar", full, binary=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest / name, filter="data")
    head = git(folder, "rev-parse", "--short", "HEAD").strip()
    note = "" if full.startswith(head) else f"  (NOTE: local clone HEAD is {head}, exported pinned {commit})"
    print(f"  agents\\{name}  <- {folder} @ {commit}{note}")


def copy_ours(dest):
    def ignore(_dir, names):
        return [n for n in names if n in SKIP_NAMES or Path(n).suffix in SKIP_SUFFIXES]

    for rel in OUR_FILES:
        src = INTEGRATION / rel
        if src.is_dir():
            shutil.copytree(src, dest / rel, ignore=ignore)
        else:
            shutil.copy2(src, dest / rel)
        print(f"  {rel}")


def safety_scan(dest):
    bad = []
    for path in dest.rglob("*"):
        name = path.name.lower()
        if name in {".env", ".venv", ".git", ".hf_cache", "backups"} or (name.startswith(".env.")
                                                                          and name != ".env.example"):
            bad.append(f"forbidden: {path}")
        elif path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
            bad.append(f"over 50 MB: {path}")
        elif path.is_file() and path.suffix == ".pkl":
            bad.append(f"pickle (KG data goes via Drive): {path}")
    # Our real key values must not appear in any exported file (values are never printed)
    from dotenv import dotenv_values
    secrets = [v for k, v in dotenv_values(INTEGRATION / ".env").items()
               if v and len(v) >= 8 and (k.endswith("_KEY") or k.startswith("CF_"))]
    for path in dest.rglob("*"):
        if path.is_file() and path.stat().st_size < 20 * 1024 * 1024:
            data = path.read_bytes()
            if any(s.encode() in data for s in secrets):
                bad.append(f"contains a key / account value from .env: {path}")
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Target: {OUT}")
    for name, (folder, commit, _url) in AGENTS.items():
        print(f"  agent {name}: {folder} @ {commit}")
    print(f"  our files: {', '.join(OUR_FILES)}")
    if args.dry_run:
        print("Dry run: nothing written.")
        return 0

    if OUT.exists():
        if not (OUT / MARKER).is_file():
            raise SystemExit(f"{OUT} exists but was not made by this script (no {MARKER}) - not deleting it.")
        shutil.rmtree(OUT)
    (OUT / "agents").mkdir(parents=True)
    (OUT / MARKER).write_text("made by setup\\export_release.py; safe to rebuild\n", encoding="utf-8")

    print("Agents (git archive at pinned commits):")
    pinned = ["# folder  commit  source  (read by the pipeline when the folder has no .git)"]
    for name, (folder, commit, url) in AGENTS.items():
        export_agent(name, folder, commit, OUT / "agents")
        pinned.append(f"{name}  {commit}  {url}")
    (OUT / "agents" / "PINNED_COMMITS.txt").write_text("\n".join(pinned) + "\n", encoding="utf-8")

    print("Our integration files:")
    copy_ours(OUT)
    (OUT / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (OUT / ".env.example").write_text(ENV_EXAMPLE, encoding="utf-8")
    shutil.copy2(README_SOURCE, OUT / "README.md")

    bad = safety_scan(OUT)
    if bad:
        print("SAFETY SCAN FAILED - do not publish this folder:")
        for line in bad:
            print("  " + line)
        return 1
    print("Safety scan OK. Done (nothing committed or pushed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
