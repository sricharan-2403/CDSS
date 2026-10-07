"""Paths, settings and .env loading for the CDSS pipeline.

Key VALUES are never printed. Use key_status() to see only "set" / "missing".
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
INTEGRATION_DIR = PACKAGE_DIR.parent
SDC_DIR = INTEGRATION_DIR.parent
SAMPLES_DIR = PACKAGE_DIR / "samples"
KG_SAMPLE = INTEGRATION_DIR / "kg_case1_output.json"

load_dotenv(INTEGRATION_DIR / ".env")

# Two folder layouts, chosen automatically (each folder can also be set in .env, see _agent_dir):
#   local (Rakshitha's PC): integration\<teammate clones>, my KG code in ..\code
#   team repo (cdss_release): agents\<5 agent folders> + agents\PINNED_COMMITS.txt (no .git there)
AGENTS_ROOT = INTEGRATION_DIR / "agents"
REPO_LAYOUT = AGENTS_ROOT.is_dir()
PINNED_COMMITS_FILE = AGENTS_ROOT / "PINNED_COMMITS.txt"  # used when git cannot tell the commit


def _agent_dir(env_name, local_default, repo_name):
    """.env value (absolute, or relative to INTEGRATION_DIR) > repo layout > local layout."""
    value = os.getenv(env_name, "").strip()
    if value:
        return (INTEGRATION_DIR / value).resolve()
    return AGENTS_ROOT / repo_name if REPO_LAYOUT else local_default


# Teammate repos and my KG code (used by real mode in Step 13)
CODE_DIR = _agent_dir("CDSS_KG_DIR", SDC_DIR / "code", "kg_agent")
EVIDENCE_DIR = _agent_dir("CDSS_EVIDENCE_DIR", INTEGRATION_DIR / "evidence_scanner", "evidence_scanner")
# Sri Charan's FROZEN Vector RAG (github.com/sricharan-2403/Symptom_RAG_Frozen), pinned commit.
# The venv .pth points to RAG_DIR\src (same package name as the old Symptom_RAG_Analyzer, never used now).
RAG_DIR = _agent_dir("CDSS_RAG_DIR", INTEGRATION_DIR / "Symptom_RAG_Frozen", "Symptom_RAG_Frozen")
RAG_EXPECTED_COMMIT = "7a80dfc"  # other commit -> warning only
MEDICO_DIR = _agent_dir("CDSS_MEDICO_DIR", INTEGRATION_DIR / "Medico", "Medico")

# Varshit's NEW Clarifier (one Groq call per text, runs in the main process).
# The old clinicl-text-processor1 is never imported any more (same module names: agent, preprocess).
CLARIFIER_DIR = _agent_dir("CDSS_CLARIFIER_DIR", INTEGRATION_DIR / "clinical_text_clarifier",
                           "clinical_text_clarifier")
CLARIFIER_EXPECTED_COMMIT = "51dacaf"  # "Handle missing entities response"; other commit -> warning only

# KG data (not in git; team Google Drive zip). His kg_agent.py reads them from its own folder (CODE_DIR).
KG_DATA_FILES_SHA256 = INTEGRATION_DIR / "setup" / "kg_data" / "cdss_kg_data_SHA256.txt"
TEAM_DRIVE_URL = "https://drive.google.com/drive/folders/16ztB2D_6Uyk1rEKDugCZDqUOIoQV9Ycw?usp=drive_link"
KG_DATA_URL = os.getenv("CDSS_KG_DATA_URL", "").strip() or TEAM_DRIVE_URL

# Hugging Face models (Vector RAG embeddings) are cached here; set before transformers is imported
os.environ.setdefault("HF_HOME", str(INTEGRATION_DIR / ".hf_cache"))
# Never download models at runtime: use only what is already in .hf_cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

API_KEY_NAMES = ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]
OTHER_ENV_NAMES = ["OPENROUTER_MODEL", "CLINICAL_TEXT_MODEL", "RAG_ANSWER_MODEL", "QDRANT_URL",
                   "GROQ_VIA_GATEWAY", "CF_ACCOUNT_ID", "CF_AI_GATEWAY_ID"]

# Loop settings (0.65 threshold from base paper Table 2)
THRESHOLD = 0.65
MAX_LOOPS = 2
TOP_K_START = 5
TOP_K_STEP = 5

# How many diseases from each side the Evidence Scanner checks
EVIDENCE_TOP_RAG = 3
EVIDENCE_TOP_KG = 3

# Vector RAG database: Qdrant SERVER (native qdrant.exe v1.19.2 + V2 storage handoff, see setup\qdrant). Checked read-only
# at startup; the run stops if it is not reachable or a collection / count / size / distance is wrong.
QDRANT_URL = os.getenv("QDRANT_URL", "").strip() or "http://localhost:6333"
RAG_COLLECTIONS = {"symptom_chunks": 2690, "literature_chunks": 201134}  # name -> exact point count
RAG_VECTOR_SIZE = 768
RAG_DISTANCE = "Cosine"

def resolve_rag_answer_model(environ=os.environ):
    """Answerer model: RAG_ANSWER_MODEL, else OPENROUTER_MODEL, else "" (= his DEFAULT_MODEL). -> (model, source)"""
    for name in ("RAG_ANSWER_MODEL", "OPENROUTER_MODEL"):
        value = (environ.get(name) or "").strip()
        if value:
            return value, name
    return "", "his default (DEFAULT_MODEL in answerer.py)"


RAG_ANSWER_MODEL, RAG_ANSWER_MODEL_SOURCE = resolve_rag_answer_model()

# OpenRouter retry for the RAG Answerer (our wrapper only; the SDK's hidden retries are OFF)
OPENROUTER_RETRY_WAITS = [5, 15]   # = up to 2 retries
OPENROUTER_RETRY_CODES = {429, 502, 503}

# Evidence Scanner: evidence items kept per disease (her top_n; 3 = 1 + 3 Groq calls per disease)
EVIDENCE_TOP_N = 3

# Groq pacing (free tier: 8,000 tokens/min, 1,000 requests/day). Before each Groq-using call
# (one Evidence Scanner disease, one Clarifier text), wait for the token budget to refill if fewer
# than GROQ_MIN_TOKENS are left; never more than GROQ_MAX_PAUSE. Clarifier and Evidence Scanner
# share ONE tracker (same key; today also the same model via GROQ_MODEL_OVERRIDE).
GROQ_MIN_TOKENS = 2500
GROQ_MAX_PAUSE = 60
GROQ_LIMITS_FILE = INTEGRATION_DIR / ".groq_limits.json"  # last seen limits (numbers only)

# TEMPORARY: if set in .env, our wrapper replaces her GROQ_MODEL in memory (her files unchanged).
# Remove from .env once Vaishnavi fixes her model name.
GROQ_MODEL_OVERRIDE = os.getenv("GROQ_MODEL_OVERRIDE", "").strip()

# Groq route. Groq blocks our networks (403 "Access denied" from its Cloudflare), so Groq calls can go
# through our Cloudflare AI Gateway (authentication disabled): GROQ_VIA_GATEWAY=true + CF_ACCOUNT_ID +
# CF_AI_GATEWAY_ID. Otherwise (false, or a CF value missing) Groq is called directly.
GROQ_DIRECT_BASE = "https://api.groq.com/openai/v1/chat/completions"  # Vaishnavi's GROQ_BASE
GATEWAY_SKIP_CACHE = {"cf-aig-skip-cache": "true"}  # never serve a stored old answer


def resolve_groq_gateway(environ=os.environ):
    """-> (gateway Groq base URL or "", route label for the startup report)."""
    wanted = (environ.get("GROQ_VIA_GATEWAY") or "").strip().lower() in ("1", "true", "yes", "on")
    account = (environ.get("CF_ACCOUNT_ID") or "").strip()
    gateway = (environ.get("CF_AI_GATEWAY_ID") or "").strip()
    if wanted and account and gateway:
        return f"https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/groq", f"gateway {gateway}"
    if wanted:
        return "", "direct (GROQ_VIA_GATEWAY=true but CF_ACCOUNT_ID / CF_AI_GATEWAY_ID missing)"
    return "", "direct"


GROQ_GATEWAY_BASE, GROQ_ROUTE = resolve_groq_gateway()

# KG top_k passed to run_kg_agent
KG_TOP_K = 5
# Extra read-only KG ranking saved for SCORING only (hit@10); not given to Medico or the Evidence Scanner
KG_RANK_TOP = 10

# Each agent can be switched separately: "replay" (saved sample) or "real".
# Change one in .env, for example CDSS_MODE_KNOWLEDGE_GRAPH=real
AGENTS = ["clarifier", "vector_rag", "knowledge_graph", "evidence_scanner", "fusion"]
DEFAULT_MODES = {
    "clarifier": "real",            # new Clarifier: 1 Groq call per case (main process)
    "vector_rag": "real",           # frozen RAG: Qdrant server + 1 OpenRouter call per pass
    "knowledge_graph": "real",      # Step 13 part A
    "evidence_scanner": "real",     # Step 13 part D (Groq model in her code is retired: fallbacks used)
    "fusion": "real",               # Step 13 part E (OpenRouter, 2 calls per pass)
}
AGENT_MODES = {
    name: os.getenv(f"CDSS_MODE_{name.upper()}", DEFAULT_MODES[name]).strip().lower()
    for name in AGENTS
}


# Agent on/off switches (ablation runs): run_pipeline --disable rag,kg,evidence,optimizer or CDSS_DISABLE in .env.
# Clarifier and Fusion are always on. Each switch -> the agent it removes (None = no agent to load).
SWITCHES = {"rag": "vector_rag", "kg": "knowledge_graph", "evidence": "evidence_scanner", "optimizer": None}
SWITCH_LABELS = {"rag": "Vector RAG", "kg": "Knowledge Graph", "evidence": "Evidence Scanner",
                 "optimizer": "Optimizer loop"}


class SwitchError(ValueError):
    """Invalid --disable / CDSS_DISABLE value: the run must not start."""


def parse_disabled(text):
    """"rag, kg" / ["rag"] -> sorted list of switch names. Raises SwitchError on unknown names or rag+kg."""
    items = text if isinstance(text, (list, tuple, set)) else str(text or "").split(",")
    names = sorted({str(n).strip().lower() for n in items if str(n).strip()})
    unknown = [n for n in names if n not in SWITCHES]
    if unknown:
        raise SwitchError(f"Unknown agent(s) to disable: {', '.join(unknown)} (use: {', '.join(SWITCHES)})")
    if "rag" in names and "kg" in names:
        raise SwitchError("At least one of rag or kg must be enabled - nothing would suggest diagnoses.")
    return names


DISABLED_FROM_ENV = os.getenv("CDSS_DISABLE", "")  # parsed by run_pipeline (so a bad value stops the run)

# TEST-ONLY: the real Clarifier code runs, but with a fake Groq client (no network).
# A text containing CLARIFIER_FAKE_FAIL makes the fake reply leave out "entities" (-> error).
CLARIFIER_FAKE = bool(os.getenv("CDSS_CLARIFIER_FAKE", "").strip())
FAKE_WARNING = "FAKE CLARIFIER MODE - not real results"
CLARIFIER_FAKE_FAIL = "__FAKE_FAIL__"

# Clarifier: our wrapper retries ONCE (rate limit, timeout, connection error only), waiting at least this
CLARIFIER_RETRY_WAIT = 5


def key_status():
    """Return {name: "set" | "missing"} without ever exposing values."""
    return {name: ("set" if os.getenv(name) else "missing")
            for name in API_KEY_NAMES + OTHER_ENV_NAMES}


_USER_ID_FIELD = re.compile(r"""(['"]user_id['"]\s*:\s*['"])[^'"]*(['"])""")
_USER_ID_VALUE = re.compile(r"\buser_[A-Za-z0-9]{10,}\b")


def redact(text):
    """Replace API key values with *** and account user ids (e.g. OpenRouter user_...) with user_***."""
    text = str(text)
    for name in API_KEY_NAMES:
        value = os.getenv(name)
        if value and len(value) >= 6:
            text = text.replace(value, "***")
    text = _USER_ID_FIELD.sub(r"\1user_***\2", text)
    return _USER_ID_VALUE.sub("user_***", text)
