"""One wrapper per agent. Each can run in REPLAY mode (saved sample) or REAL mode.

The mode for each agent comes from config.AGENT_MODES, so agents can be
switched to real one at a time in Step 13.

Startup: init_agents() loads every real agent once in this process. The Clarifier
(Varshit's new version) is only a Groq client, so it needs almost no RAM.
"""

import copy
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

from cdss_pipeline import config

# Real agents, loaded once by init_agents()
_REAL = {}


# ---------------------------------------------------------------- helpers

def clean_name(name):
    """Lowercase, remove anything in brackets, trim extra spaces.

    "Myocardial Infarction (Heart Attack)" -> "myocardial infarction"
    """
    text, _ = _strip_brackets(str(name or "").lower())
    return re.sub(r"\s+", " ", text).strip()


_INNER_BRACKETS = re.compile(r"\(([^()\[\]]*)\)|\[([^()\[\]]*)\]")


def _strip_brackets(text):
    """Remove brackets innermost-first (handles nesting). Returns (text, [bracket contents])."""
    contents = []
    while True:
        new = _INNER_BRACKETS.sub(lambda m: contents.append(m.group(1) or m.group(2) or "") or " ", text)
        if new == text:
            return re.sub(r"[()\[\]]", " ", text), contents  # unmatched stray brackets
        text = new


def name_parts(name):
    """clean_name, then split on "/" into separate names (empty parts and duplicates dropped).

    "Heart Attack / Acute Myocardial Infarction" -> ["heart attack", "acute myocardial infarction"]
    Bracket content is still removed (not used as an extra part).
    """
    parts = []
    for part in clean_name(name).split("/"):
        part = part.strip()
        if part and part not in parts:
            parts.append(part)
    return parts


def match_parts(name):
    """name_parts PLUS the bracket content as extra parts (cleaned, split on "/").

    Used ONLY for supported_by_both and rag_matches_known_disease, never for evidence skipping.
    "Heart attack (Myocardial Infarction) / Coronary Artery Disease"
        -> ["heart attack", "coronary artery disease", "myocardial infarction"]
    """
    parts = name_parts(name)
    _, contents = _strip_brackets(str(name or ""))
    for content in contents:
        for part in name_parts(content):
            if part not in parts:
                parts.append(part)
    return parts


@lru_cache(maxsize=None)
def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sample(name):
    """Fresh copy of a replay sample, so callers can change it safely."""
    return copy.deepcopy(_load_json(str(config.SAMPLES_DIR / name)))


def _mode(agent):
    mode = config.AGENT_MODES.get(agent, "replay")
    if mode not in ("replay", "real"):
        raise ValueError(f"Unknown mode '{mode}' for {agent} (use replay or real)")
    return mode


def _put_first_on_path(folder):
    folder = str(folder)
    if folder in sys.path:
        sys.path.remove(folder)
    sys.path.insert(0, folder)


# ---------------------------------------------------------------- startup loading

def _load_knowledge_graph():
    """Import my kg_agent from ..\\code and load the graph + DDInter lookup once."""
    _put_first_on_path(config.CODE_DIR)
    import kg_agent  # noqa: E402  (needs CODE_DIR on sys.path)

    # Name clash guard: evidence_scanner also has a demo_pipeline.py
    demo = sys.modules.get("demo_pipeline")
    expected = (config.CODE_DIR / "demo_pipeline.py").resolve()
    if demo is None or Path(demo.__file__).resolve() != expected:
        raise ImportError(f"wrong demo_pipeline loaded: {getattr(demo, '__file__', None)} "
                          f"(expected {expected})")

    G, _, stats = kg_agent.load_resources()
    _REAL["knowledge_graph"] = kg_agent.run_kg_agent
    # Read-only helpers for OUR symptom matching (his graph, synonym table and relation; his code unchanged)
    _REAL["kg_lookup"] = make_symptom_lookup(G, demo.SYMPTOM_SYNONYMS, demo.diseases_for, demo.RELATION)
    # Read-only top-10 ranking for SCORING only (hit@10); his top-5 used downstream is unchanged
    _REAL["kg_rank"] = lambda symptoms, top_k=10: demo.select_for_fusion(
        demo.predict_disease({"symptoms": symptoms}, G=G), top_k)
    return {"graph_load_seconds": stats.get("graph_load_seconds"),
            "lookup_load_seconds": stats.get("lookup_load_seconds"),
            "graph_nodes": stats.get("graph_nodes"), "graph_edges": stats.get("graph_edges")}


def _check_from(module_name, folder):
    mod = sys.modules.get(module_name)
    path = Path(getattr(mod, "__file__", "") or "").resolve()
    if mod is None or path.parent != Path(folder).resolve():
        raise ImportError(f"wrong {module_name} loaded: {path} (expected it in {folder})")


class RagDatabaseError(RuntimeError):
    """Vector RAG database (Qdrant) check failed: the whole run must stop."""


def _qdrant_client(url=None, timeout=10):
    from qdrant_client import QdrantClient
    return QdrantClient(url=url or config.QDRANT_URL, timeout=timeout)


def check_qdrant(client=None):
    """Read-only check of the Qdrant server: reachable, both collections, EXACT counts, 768, Cosine.

    Returns {collection: {"points", "status"}}. Raises RagDatabaseError with a clear message.
    Never creates, changes or deletes anything (his QdrantVectorStore WOULD create a missing
    collection, so this must run before his code connects).
    """
    url = config.QDRANT_URL
    client = client or _qdrant_client(url)
    start_hint = "start it with: powershell -ExecutionPolicy Bypass -File <qdrant folder>\\start_qdrant.ps1"
    try:
        names = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        raise RagDatabaseError(f"Qdrant not reachable at {url} ({type(exc).__name__}: {str(exc)[:200]}) - "
                               f"{start_hint}")
    report = {}
    for name, expected in config.RAG_COLLECTIONS.items():
        if name not in names:
            raise RagDatabaseError(f"Qdrant collection '{name}' not found at {url} "
                                   f"(found: {sorted(names) or 'none'}) - restore it with setup\\qdrant\\setup_qdrant.ps1")
        try:
            info = client.get_collection(name)
            count = client.count(name, exact=True).count
        except Exception as exc:
            raise RagDatabaseError(f"Qdrant collection '{name}' cannot be read: {type(exc).__name__}: {str(exc)[:200]}")
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        distance = getattr(getattr(vectors, "distance", None), "value", getattr(vectors, "distance", None))
        status = getattr(getattr(info, "status", None), "value", getattr(info, "status", None))
        problems = []
        if count != expected:
            problems.append(f"{count} points, expected exactly {expected}")
        if size != config.RAG_VECTOR_SIZE:
            problems.append(f"vector size {size}, expected {config.RAG_VECTOR_SIZE}")
        if str(distance) != config.RAG_DISTANCE:
            problems.append(f"distance {distance}, expected {config.RAG_DISTANCE}")
        if str(status) == "red":
            problems.append("status red")
        if problems:
            raise RagDatabaseError(f"Qdrant collection '{name}' is wrong: " + "; ".join(problems)
                                   + " - do not add or delete points; report it to Sri Charan")
        report[name] = {"points": count, "status": str(status)}
    return report


def _pinned_commit(folder):
    """Commit for an agent folder from PINNED_COMMITS.txt (team repo, no .git); "" if not listed."""
    try:
        lines = Path(config.PINNED_COMMITS_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("#") and parts[0] == Path(folder).name:
            return parts[1]
    return ""


def _has_own_git(folder):
    return (Path(folder) / ".git").exists()


def _git_commit(folder, expected, label):
    """Short commit of an agent folder via git (read-only), else from PINNED_COMMITS.txt.
    Never raises: "unknown" on failure."""
    commit = ""
    if _has_own_git(folder):  # only the folder's OWN repo (in cdss_release, git would see the parent)
        try:
            proc = subprocess.run(["git", "-C", str(folder), "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True, timeout=10)
            commit = proc.stdout.strip() if proc.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            commit = ""
    commit = commit or _pinned_commit(folder)
    if not commit:
        print(f"WARNING: {label} commit unknown (git not found or failed, not in PINNED_COMMITS.txt)", flush=True)
        return "unknown"
    if not (commit.startswith(expected) or expected.startswith(commit)):
        print(f"WARNING: {label} is at commit {commit}, expected {expected}", flush=True)
    return commit


# Timing / calls of the Vector RAG (measured in OUR wrapper; rag_output is not changed).
# openrouter_calls counts every attempt (retries included), as OpenRouter counts them.
RAG_STATS = {"openrouter_calls": 0, "retries": 0, "failures": 0, "calls": []}

_sleep = time.sleep  # replaced in tests so no real waiting happens


def _current_call():
    return RAG_STATS["calls"][-1] if RAG_STATS["calls"] else {}


def _timed(obj, attr, key):
    """Wrap obj.attr on this instance only, recording its time in the current call's stats."""
    original = getattr(obj, attr)

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _current_call()[key] = round(time.perf_counter() - start, 2)
    setattr(obj, attr, wrapper)


def _openrouter_error_code(exc):
    """HTTP status behind his "OpenRouter request failed." (openai APIStatusError in the cause chain), or None."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        code = getattr(exc, "status_code", None)
        if isinstance(code, int):
            return code
        exc = exc.__cause__ or exc.__context__
    return None


def _with_openrouter_retry(answerer):
    """Retry answerer.answer ONLY for OpenRouter 429/502/503, waiting config.OPENROUTER_RETRY_WAITS.

    Records openrouter_seconds (all attempts, waits not included), retries, retry_codes and
    retry_wait_seconds for the current call. Any other error is raised at once.
    """
    original = answerer.answer

    def answer(*args, **kwargs):
        stats = _current_call()
        stats.update({"openrouter_seconds": 0.0, "retries": 0, "retry_codes": [], "retry_wait_seconds": 0})
        waits = list(config.OPENROUTER_RETRY_WAITS)
        while True:
            RAG_STATS["openrouter_calls"] += 1
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                code = _openrouter_error_code(exc)
                if code not in config.OPENROUTER_RETRY_CODES or not waits:
                    RAG_STATS["failures"] += 1
                    stats["error_code"] = code
                    raise
                wait = waits.pop(0)
                stats["retries"] += 1
                stats["retry_codes"].append(code)
                stats["retry_wait_seconds"] += wait
                RAG_STATS["retries"] += 1
                _sleep(wait)
            finally:
                stats["openrouter_seconds"] = round(stats["openrouter_seconds"] + time.perf_counter() - start, 2)
    answerer.answer = answer


def _load_vector_rag():
    """Check Qdrant (read-only), load his retriever (BioBERT once) and answerer once.

    Orchestrators are made per top_k on first use (5, 10, 15), all sharing that retriever + answerer.
    """
    collections = check_qdrant()
    from symptom_rag_analyzer.orchestrator import ClinicalTextClarifierOutput, SymptomRAGOrchestrator
    from symptom_rag_analyzer.output_adapter import SymptomRAGOutputAdapter
    from symptom_rag_analyzer.reasoning.answerer import DEFAULT_MODEL, BiomedicalAnswerer
    from symptom_rag_analyzer.retrieval.search import BiomedicalRetriever
    _check_package("symptom_rag_analyzer", config.RAG_DIR / "src" / "symptom_rag_analyzer")
    commit = _git_commit(config.RAG_DIR, config.RAG_EXPECTED_COMMIT, "Vector RAG (Symptom_RAG_Frozen)")

    retriever = BiomedicalRetriever(qdrant_client=_qdrant_client(timeout=60))
    answerer = BiomedicalAnswerer(model=config.RAG_ANSWER_MODEL or DEFAULT_MODEL)
    answerer.client = answerer.client.with_options(max_retries=0)  # hidden SDK retries OFF; ours below
    _timed(retriever.embedding_model, "embed_text", "embed_seconds")
    _timed(retriever.symptom_store, "search", "symptom_search_seconds")
    _timed(retriever.literature_store, "search", "literature_search_seconds")
    _with_openrouter_retry(answerer)
    _REAL["vector_rag"] = {"retriever": retriever, "answerer": answerer, "adapter": SymptomRAGOutputAdapter(),
                           "orchestrator_class": SymptomRAGOrchestrator,
                           "context_class": ClinicalTextClarifierOutput, "orchestrators": {}}
    return {"qdrant_url": config.QDRANT_URL, "collections": collections, "commit": commit,
            "rag_answer_model": answerer.model, "rag_answer_model_source": config.RAG_ANSWER_MODEL_SOURCE,
            "openrouter_key": "set" if os.getenv("OPENROUTER_API_KEY") else "missing"}


def _rag_orchestrator(top_k):
    """One orchestrator per top_k (his top_k is set at creation), created once and cached."""
    rag = _real("vector_rag")
    if top_k not in rag["orchestrators"]:
        rag["orchestrators"][top_k] = rag["orchestrator_class"](
            retriever=rag["retriever"], answerer=rag["answerer"], top_k=top_k)
    return rag["orchestrators"][top_k]


def _check_package(name, folder):
    mod = sys.modules.get(name)
    path = Path(getattr(mod, "__file__", "") or "").resolve()
    if mod is None or path.parent != Path(folder).resolve():
        raise ImportError(f"wrong {name} loaded: {path} (expected it in {folder})")


# ---------------------------------------------------------------- Evidence Scanner

# Per-disease stats (measured in OUR wrapper; evidence_output is not changed)
EVIDENCE_STATS = {"calls": []}
_EVIDENCE_LOCK = threading.Lock()  # her code runs searches in threads inside one disease

# Which of her modules talk to which service
EVIDENCE_SOURCES = {
    "tools.pubmed_tool": "pubmed",
    "tools.europepmc_tool": "europepmc",
    "tools.semantic_scholar_tool": "semantic_scholar",
    "tools.query_generator": "groq",
    "tools.summarizer": "groq",
}


# Her Groq "jobs" (one query generation / one summary) and the fallback each one uses on failure.
# Counted in memory per disease (her files unchanged); her own retries inside _call_llm = one job.
EVIDENCE_GROQ_FALLBACKS = {"tools.query_generator": "_fallback_queries", "tools.summarizer": "_fallback_summary"}

_KEY_LIKE = re.compile(r"\b(gsk|sk|xg)[-_][A-Za-z0-9_-]{8,}")


def groq_error_text(body):
    """Groq's own error message from a response body (error.message, else the raw text), short, keys hidden."""
    text = str(body or "").strip()
    try:
        error = json.loads(text).get("error")
        text = str(error.get("message") if isinstance(error, dict) else error or text)
    except (ValueError, AttributeError):
        pass
    text = re.sub(r"\s+", " ", text)[:300]
    return _KEY_LIKE.sub("***", config.redact(text)) or None


def _new_evidence_record(disease):
    return {"disease": disease, "requests": {}, "rate_limited": {}, "http_errors": {},
            "failed_requests": {}, "groq_llm_calls": 0, "groq_fallbacks": 0}


def _bump_evidence_count(key):
    with _EVIDENCE_LOCK:
        if EVIDENCE_STATS["calls"]:
            rec = EVIDENCE_STATS["calls"][-1]
            rec[key] = rec.get(key, 0) + 1


def _wrap_counted(module, func_name, key):
    """Replace module.func_name IN MEMORY with a wrapper that counts calls (once; her files unchanged)."""
    original = getattr(module, func_name)
    if getattr(original, "_cdss_counts", None):
        return

    def counted(*args, **kwargs):
        _bump_evidence_count(key)
        return original(*args, **kwargs)
    counted._cdss_counts = key
    counted.__wrapped__ = original
    setattr(module, func_name, counted)


class _CountingRequests:
    """Stand-in for the `requests` module inside ONE of her modules.

    Passes every call through unchanged and counts it (per source, with 429s and other
    HTTP errors) in the current disease's EVIDENCE_STATS record.
    """

    def __init__(self, real, source):
        self._real, self._source = real, source

    def __getattr__(self, name):  # HTTPError, exceptions, etc. pass through
        return getattr(self._real, name)

    def get(self, *args, **kwargs):
        return self._call("get", *args, **kwargs)

    def post(self, *args, **kwargs):
        return self._call("post", *args, **kwargs)

    def _bump(self, key):
        with _EVIDENCE_LOCK:
            rec = EVIDENCE_STATS["calls"][-1] if EVIDENCE_STATS["calls"] else _new_evidence_record("?")
            rec[key][self._source] = rec[key].get(self._source, 0) + 1

    def _call(self, method, *args, **kwargs):
        if self._source == "groq" and config.GROQ_GATEWAY_BASE:  # never a stored old answer from the gateway
            kwargs["headers"] = {**(kwargs.get("headers") or {}), **config.GATEWAY_SKIP_CACHE}
        self._bump("requests")
        try:
            response = getattr(self._real, method)(*args, **kwargs)
        except Exception:
            self._bump("failed_requests")  # timeout, connection error, ...
            raise
        status = getattr(response, "status_code", 200)
        if status == 429:
            self._bump("rate_limited")
        elif status >= 400:
            self._bump("http_errors")
        if self._source == "groq":
            self._store_limits(response, status)
            if status >= 400:
                message = groq_error_text(getattr(response, "text", ""))
                if message:
                    with _EVIDENCE_LOCK:
                        if EVIDENCE_STATS["calls"]:
                            EVIDENCE_STATS["calls"][-1]["groq_error_message"] = message
        return response

    def _store_limits(self, response, status):
        """Keep Groq's rate-limit headers (numbers and times only, never the key)."""
        headers = getattr(response, "headers", None) or {}
        limits = {k.lower(): v for k, v in headers.items()
                  if k.lower().startswith("x-ratelimit-") or k.lower() == "retry-after"}
        if limits:
            with _EVIDENCE_LOCK:
                rec = EVIDENCE_STATS["calls"][-1] if EVIDENCE_STATS["calls"] else {}
                rec["groq_limits"] = {**limits, "status": status}
                if "retry-after" in limits:
                    rec.setdefault("retry_after", []).append(limits["retry-after"])
                _update_groq_limits(limits)


# ---------------------------------------------------------------- Groq pacing

# Latest Groq rate-limit values seen (numbers/times only). Saved to config.GROQ_LIMITS_FILE.
GROQ_LIMITS = {}


def _parse_duration(text):
    """Groq time strings -> seconds: "26.407s", "50m24s", "1h2m3s", "120ms". Unknown -> None."""
    text = str(text or "").strip()
    if not text:
        return None
    try:
        return float(text)  # plain seconds (retry-after)
    except ValueError:
        pass
    total, found = 0.0, False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", text):
        found = True
        total += float(value) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[unit]
    return total if found else None


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _update_groq_limits(headers):
    """Store the latest values and save them to the limits file (failures ignored)."""
    now = time.time()
    if "x-ratelimit-remaining-tokens" in headers:
        GROQ_LIMITS["remaining_tokens"] = _to_int(headers["x-ratelimit-remaining-tokens"])
        GROQ_LIMITS["reset_tokens_s"] = _parse_duration(headers.get("x-ratelimit-reset-tokens"))
        GROQ_LIMITS["tokens_seen_at"] = now
    if "x-ratelimit-remaining-requests" in headers:
        GROQ_LIMITS["remaining_requests"] = _to_int(headers["x-ratelimit-remaining-requests"])
        GROQ_LIMITS["limit_requests"] = _to_int(headers.get("x-ratelimit-limit-requests"))
        GROQ_LIMITS["reset_requests_s"] = _parse_duration(headers.get("x-ratelimit-reset-requests"))
        GROQ_LIMITS["requests_seen_at"] = now
    if "retry-after" in headers:
        GROQ_LIMITS["retry_after_s"] = _parse_duration(headers["retry-after"])
    try:
        saved = {k: v for k, v in GROQ_LIMITS.items() if k != "retry_after_s"}
        Path(config.GROQ_LIMITS_FILE).write_text(json.dumps(saved), encoding="utf-8")
    except Exception:
        pass  # never break a run because the small limits file could not be written


def _groq_wait_seconds():
    """Seconds to wait before the next disease, and why. Never pauses when the budget is enough."""
    wait, reason = 0.0, None
    retry_after = GROQ_LIMITS.pop("retry_after_s", None)  # used once
    if retry_after:
        wait, reason = retry_after, f"retry-after {retry_after:g} s"
    left, reset = GROQ_LIMITS.get("remaining_tokens"), GROQ_LIMITS.get("reset_tokens_s")
    if left is not None and left < config.GROQ_MIN_TOKENS and reset:
        remaining_reset = reset - (time.time() - GROQ_LIMITS.get("tokens_seen_at", time.time()))
        if remaining_reset > wait:
            wait, reason = remaining_reset, f"low tokens: {left} left"
    wait = min(max(wait, 0.0), config.GROQ_MAX_PAUSE)
    return (round(wait, 1), reason) if wait > 0 else (0.0, None)


def groq_requests_left_text():
    """Startup line for the remaining daily Groq requests, from the saved limits file."""
    try:
        saved = json.loads(Path(config.GROQ_LIMITS_FILE).read_text(encoding="utf-8"))
    except Exception:
        return "unknown (no Groq call seen yet)"
    left, limit = saved.get("remaining_requests"), saved.get("limit_requests")
    seen = saved.get("requests_seen_at")
    if left is None or seen is None:
        return "unknown (no Groq call seen yet)"
    text = f"{left} of {limit} (as of {time.strftime('%H:%M', time.localtime(seen))}"
    reset = saved.get("reset_requests_s")
    if reset is not None:
        text += f"; resets {int(reset // 60)}m{int(reset % 60):02d}s after that"
        if time.time() > seen + reset:
            text += "; probably reset since"
    return text + ")"


class _SemanticScholar429Once(logging.Filter):
    """Her aggregator logs one warning per failed source call; without a Semantic Scholar key EVERY call
    gets HTTP 429. Show that warning once per case (reset_case_warnings), hide the repeats (counted)."""

    def __init__(self):
        super().__init__()
        self.shown, self.hidden = False, 0

    def filter(self, record):
        message = record.getMessage()
        if "semantic_scholar_search" not in message or "HTTP 429" not in message:
            return True
        if self.shown:
            self.hidden += 1
            return False
        self.shown = True
        record.msg, record.args = (message + " (no Semantic Scholar API key; further 429s in this case "
                                   "are not shown)"), ()
        return True


S2_429_FILTER = _SemanticScholar429Once()


def reset_case_warnings():
    """Call at the start of every case (run_pipeline.run)."""
    S2_429_FILTER.shown, S2_429_FILTER.hidden = False, 0


def _load_evidence_scanner():
    """Import Vaishnavi's scanner, check module origins, build its graph ONCE."""
    aggregator_log = logging.getLogger("tools.evidence_aggregator")
    if S2_429_FILTER not in aggregator_log.filters:
        aggregator_log.addFilter(S2_429_FILTER)
    _put_first_on_path(config.EVIDENCE_DIR)
    import agents.evidence_scanner_agent as scanner_mod  # noqa: E402
    _check_package("agents", config.EVIDENCE_DIR / "agents")
    _check_package("tools", config.EVIDENCE_DIR / "tools")
    demo = sys.modules.get("demo_pipeline")
    if demo is not None and Path(demo.__file__).resolve().parent != config.CODE_DIR.resolve():
        raise ImportError(f"demo_pipeline clash: {demo.__file__} is loaded instead of ..\\code's")

    for module_name, source in EVIDENCE_SOURCES.items():
        mod = sys.modules[module_name]
        if not isinstance(mod.requests, _CountingRequests):
            mod.requests = _CountingRequests(mod.requests, source)
    _count_evidence_groq_jobs()

    groq_info = {**_apply_groq_model_override(), **_apply_groq_route()}
    _REAL["evidence_scanner"] = scanner_mod.build_evidence_scanner()
    return {"groq_key": "set" if sys.modules["tools.summarizer"].GROQ_API_KEY else "missing",
            **groq_info, "top_n": config.EVIDENCE_TOP_N,
            "groq_requests_left_today": groq_requests_left_text()}


def _count_evidence_groq_jobs():
    """Count her Groq jobs (_call_llm) and fallbacks (_fallback_queries / _fallback_summary) per disease."""
    for module_name, fallback in EVIDENCE_GROQ_FALLBACKS.items():
        mod = sys.modules[module_name]
        _wrap_counted(mod, "_call_llm", "groq_llm_calls")
        _wrap_counted(mod, fallback, "groq_fallbacks")


GROQ_MODULES = ("tools.query_generator", "tools.summarizer")


def _apply_groq_model_override():
    """If GROQ_MODEL_OVERRIDE is set, replace her GROQ_MODEL in memory (both modules).

    Her code reads GROQ_MODEL on every request, so this applies to all later calls.
    Her files on disk are never changed. Returns the model actually used and its source.
    """
    original = {name: sys.modules[name].GROQ_MODEL for name in GROQ_MODULES}
    if config.GROQ_MODEL_OVERRIDE:
        for name in GROQ_MODULES:
            sys.modules[name].GROQ_MODEL = config.GROQ_MODEL_OVERRIDE
        print(f"Groq model OVERRIDDEN by GROQ_MODEL_OVERRIDE: {config.GROQ_MODEL_OVERRIDE} "
              f"(her code says {', '.join(sorted(set(original.values())))})", flush=True)
    used = {name: sys.modules[name].GROQ_MODEL for name in GROQ_MODULES}
    values = sorted(set(used.values()))
    return {"groq_model": values[0] if len(values) == 1 else used,
            "groq_model_source": "override (GROQ_MODEL_OVERRIDE)" if config.GROQ_MODEL_OVERRIDE
            else "her code"}


def _apply_groq_route():
    """Gateway route: replace her GROQ_BASE in memory (both modules; read on every request).

    Her files on disk are never changed. Direct route: her GROQ_BASE is left as it is.
    """
    if config.GROQ_GATEWAY_BASE:
        for name in GROQ_MODULES:
            sys.modules[name].GROQ_BASE = f"{config.GROQ_GATEWAY_BASE}/chat/completions"
    return {"groq_route": config.GROQ_ROUTE}


# ---------------------------------------------------------------- Fusion (Charanya's Medico)

# Keys of her returned state that we keep in fusion_output (unchanged)
FUSION_OUTPUT_KEYS = ("confidence", "fusion_report", "optimizer_response", "fallback_used")

# Per-pass stats (measured in OUR wrapper; fusion_output is not changed)
FUSION_STATS = {"openrouter_calls": 0, "openrouter_failures": 0, "openrouter_retries": 0, "passes": []}

# Copy of her extract_graph_score pattern, only to REPORT whether C2's graph part used the 0.90 default
_GRAPH_SCORE_PATTERN = re.compile(r"(?:confidence|score|fidelity)[:\s]+([01](?:\.\d+)?)", re.IGNORECASE)


def _current_pass():
    return FUSION_STATS["passes"][-1] if FUSION_STATS["passes"] else {}


def _count_openrouter(module, func_name, label):
    """Wrap module.func_name IN MEMORY: count calls and failures, record the reason, re-raise."""
    original = getattr(module, func_name)

    def wrapper(*args, **kwargs):
        FUSION_STATS["openrouter_calls"] += 1
        rec = _current_pass()
        rec.setdefault("calls", {})[label] = rec.get("calls", {}).get(label, 0) + 1
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            FUSION_STATS["openrouter_failures"] += 1
            rec.setdefault("failures", []).append(
                f"{label}: " + config.redact(f"{type(exc).__name__}: {str(exc)[:300]}"))
            raise  # her fallback still runs exactly as before
    setattr(module, func_name, wrapper)


class _RetryLogCounter(logging.Handler):
    """Counts the OpenAI client's own "Retrying request" log lines (its built-in retries)."""

    def emit(self, record):
        if "Retrying request" in record.getMessage():
            FUSION_STATS["openrouter_retries"] += 1
            rec = _current_pass()
            rec["retries"] = rec.get("retries", 0) + 1


def _attach_retry_counter():
    logger = logging.getLogger("openai._base_client")
    if not any(isinstance(h, _RetryLogCounter) for h in logger.handlers):
        logger.addHandler(_RetryLogCounter())
        logger.setLevel(min(logger.level or logging.INFO, logging.INFO))


def _load_fusion():
    """Load her graph.py as "medico_graph" (name clash with ours), check modules, build pipeline ONCE."""
    _put_first_on_path(config.MEDICO_DIR)
    spec = importlib.util.spec_from_file_location("medico_graph", config.MEDICO_DIR / "graph.py")
    medico_graph = importlib.util.module_from_spec(spec)
    sys.modules["medico_graph"] = medico_graph
    spec.loader.exec_module(medico_graph)
    for name in ("fusion_agents", "evidence_adapter", "confidence_scoring"):
        _check_from(name, config.MEDICO_DIR)

    _count_openrouter(medico_graph, "run_fusion_agent", "fusion")
    _count_openrouter(medico_graph, "run_optimizer_agent", "optimizer")
    _attach_retry_counter()
    _REAL["fusion"] = medico_graph.build_pipeline()

    llm = sys.modules["fusion_agents"].llm
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    from_env = bool(os.getenv("OPENROUTER_MODEL"))
    info = {"openrouter_model": model,
            "openrouter_model_source": "OPENROUTER_MODEL" if from_env else "her default (gpt-4o-mini, PAID)",
            "openrouter_key": "set" if os.getenv("OPENROUTER_API_KEY") else "missing"}
    if not from_env:
        print("WARNING: OPENROUTER_MODEL is not set - Fusion uses her paid default gpt-4o-mini", flush=True)
    return info


# ---------------------------------------------------------------- Clarifier (Varshit's new version)

# His top-level modules; all must come from config.CLARIFIER_DIR (import guard)
CLARIFIER_MODULES = ("agent", "config", "schemas", "grounding", "preprocess")
# The ONLY keys passed on as clarifier_output (same 5 keys as the old Clarifier)
CLARIFIER_KEYS = ("clinical_text", "diseases", "symptoms", "medications", "tests")
CLARIFIER_LISTS = CLARIFIER_KEYS[1:]
# Error texts from his _safe_groq_error_message() that are worth ONE retry
_CLARIFIER_RETRY_TEXTS = ("rate limit", "timed out", "connection error")

# Measured in OUR wrapper: Groq requests seen by the response hook, one record per case
CLARIFIER_STATS = {"groq_requests": 0, "calls": []}
CLARIFIER_INFO = {}       # commit and model, set at load
_CLARIFIER_LAST = {}      # HTTP status of the latest Clarifier Groq response


class ClarifierError(RuntimeError):
    """Clarifier failed; .details goes to clarifier_details (attempts, error, ...)."""

    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def _clarifier_commit():
    """Short commit of the Clarifier folder via git (read-only). Never raises: "unknown" on failure."""
    return _git_commit(config.CLARIFIER_DIR, config.CLARIFIER_EXPECTED_COMMIT, "Clarifier")


def _clarifier_response_hook(response):
    """httpx response hook on OUR Groq client: count the request, feed the shared Groq pacing,
    and keep Groq's own error message (his code hides it behind a generic text)."""
    CLARIFIER_STATS["groq_requests"] += 1
    _CLARIFIER_LAST["status"] = response.status_code
    limits = {k.lower(): v for k, v in response.headers.items()
              if k.lower().startswith("x-ratelimit-") or k.lower() == "retry-after"}
    if limits:
        with _EVIDENCE_LOCK:
            _update_groq_limits(limits)
    if response.status_code >= 400:
        try:
            response.read()
            _CLARIFIER_LAST["groq_error_message"] = groq_error_text(response.text)
        except Exception:
            pass  # never break the call because the error body could not be read


def _make_clarifier_client(api_key, transport=None):
    """Our own Groq client: SDK retries OFF (our wrapper decides), hook sees every response.

    Routed through the Cloudflare AI Gateway when config.GROQ_GATEWAY_BASE is set (the gateway accepts
    the SDK's own .../groq/openai/v1/chat/completions path), with its cache skipped.
    """
    from groq import DefaultHttpxClient, Groq
    http_client = DefaultHttpxClient(transport=transport,
                                     event_hooks={"response": [_clarifier_response_hook]})
    if config.GROQ_GATEWAY_BASE:
        return Groq(api_key=api_key, max_retries=0, http_client=http_client,
                    base_url=config.GROQ_GATEWAY_BASE, default_headers=config.GATEWAY_SKIP_CACHE)
    return Groq(api_key=api_key, max_retries=0, http_client=http_client)


def _fake_groq_transport():
    """TEST-ONLY fake Groq server: one present symptom grounded on the first word of the text.

    A text containing config.CLARIFIER_FAKE_FAIL gets a reply without "entities" (his code -> error).
    """
    import httpx

    def handler(request):
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        if config.CLARIFIER_FAKE_FAIL in text:
            payload = {"demographics": {"age": None, "sex": None}}
        else:
            words = text.split()
            payload = {"entities": [{"text": "fake symptom", "type": "symptom", "status": "present",
                                     "evidence_text": words[0] if words else "-", "body_site": None,
                                     "duration": None, "severity": None, "value": None}],
                       "demographics": {"age": None, "sex": None}}
        return httpx.Response(200, json={
            "id": "fake", "object": "chat.completion", "created": 0, "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(payload)}}]})
    return httpx.MockTransport(handler)


def _load_clarifier(transport=None):
    """Import Varshit's new Clarifier, check where every module came from, create ONE agent."""
    _put_first_on_path(config.CLARIFIER_DIR)
    import agent as clarifier_agent  # noqa: E402  (generic name: checked below)
    for name in CLARIFIER_MODULES:
        _check_from(name, config.CLARIFIER_DIR)

    commit = _clarifier_commit()
    if config.CLARIFIER_FAKE:
        client = _make_clarifier_client("fake-key", transport or _fake_groq_transport())
    else:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing (needed by the Clarifier)")
        client = _make_clarifier_client(api_key, transport)
    clarifier = clarifier_agent.ClinicalTextProcessingAgent(client=client)
    _REAL["clarifier"] = clarifier
    CLARIFIER_INFO.update({"commit": commit, "model": clarifier.model})
    return {"model": clarifier.model,
            "model_source": "CLINICAL_TEXT_MODEL" if os.getenv("CLINICAL_TEXT_MODEL") else "his default",
            "commit": commit, "groq_key": "set" if os.getenv("GROQ_API_KEY") else "missing",
            "groq_route": config.GROQ_ROUTE, "fake": config.CLARIFIER_FAKE}


_LOADERS = {"clarifier": _load_clarifier, "knowledge_graph": _load_knowledge_graph,
            "vector_rag": _load_vector_rag, "evidence_scanner": _load_evidence_scanner,
            "fusion": _load_fusion}


def init_agents(disabled=()):
    """Load every agent that is in real mode and not disabled (ablation switch), once. Call at startup.

    Returns {agent: {"seconds": ..., **details}} or {"error": ...} per agent.
    A failed load does not stop the pipeline: that agent's node will record an error.
    """
    skip = {config.SWITCHES[name] for name in disabled if config.SWITCHES.get(name)}
    report = {}
    for agent, loader in _LOADERS.items():
        if _mode(agent) != "real" or agent in _REAL or agent in skip:
            continue
        start = time.perf_counter()
        try:
            details = loader()
            report[agent] = {"seconds": round(time.perf_counter() - start, 2), **details}
        except Exception as exc:
            report[agent] = {"error": config.redact(f"{type(exc).__name__}: {exc}")}
    return report


def _real(agent):
    if agent not in _REAL:
        raise RuntimeError(f"{agent}: real mode selected but not loaded (call init_agents() at startup)")
    return _REAL[agent]


# ---------------------------------------------------------------- agents

def _clarifier_retryable(result):
    """Retry ONLY rate limit, timeout or connection errors (never auth, bad request, model not found)."""
    if _CLARIFIER_LAST.get("status") == 429:
        return True
    message = str(result.get("error") or "").lower()
    return any(text in message for text in _CLARIFIER_RETRY_TEXTS)


def _check_usable(result):
    """Raise ValueError unless the result has the 5 keys with the right types."""
    if not isinstance(result.get("clinical_text"), str):
        raise ValueError("clinical_text is missing or not a string")
    for key in CLARIFIER_LISTS:
        items = result.get(key)
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
            raise ValueError(f"{key} is missing or not a list of strings")


def run_clarifier_full(patient_text):
    """-> (clarifier_output with EXACTLY the 5 keys, clarifier_details).

    Raises ClarifierError when his result has status "error" or is not usable,
    so clarifier_node records it in errors. Empty lists are never passed on silently.
    """
    if _mode("clarifier") == "replay":
        # No sample in Varshit's repo; built from ..\code\tests\case_01_mi_sildenafil.json
        return _sample("clarifier_case1.json"), {"mode": "replay"}
    clarifier = _real("clarifier")
    details = {"mode": "fake" if config.CLARIFIER_FAKE else "real",
               "commit": CLARIFIER_INFO.get("commit", "unknown"), "model": clarifier.model,
               "attempts": 0, "paused_seconds": 0.0}
    CLARIFIER_STATS["calls"].append(details)
    start = time.perf_counter()
    result = {}
    try:
        for attempt in (1, 2):
            wait, reason = _groq_wait_seconds()  # shared pacing (also uses retry-after once)
            if attempt == 2:
                wait = max(wait, config.CLARIFIER_RETRY_WAIT)
                reason = reason or f"retry wait {config.CLARIFIER_RETRY_WAIT} s"
            if wait:
                _sleep(wait)
                details["paused_seconds"] = round(details["paused_seconds"] + wait, 1)
                details.setdefault("pause_reasons", []).append(reason)
            _CLARIFIER_LAST.clear()
            details["attempts"] = attempt
            raw = clarifier.process(patient_text)
            result = raw.model_dump() if hasattr(raw, "model_dump") else raw
            if not isinstance(result, dict):
                raise ClarifierError(f"Clarifier returned an unusable result ({type(raw).__name__})", details)
            if result.get("status") != "error":
                break
            groq_message = _CLARIFIER_LAST.get("groq_error_message")
            details.setdefault("attempt_errors", []).append(
                config.redact(result.get("error")) + (f" - Groq says: {groq_message}" if groq_message else ""))
            if not _clarifier_retryable(result):
                break
    finally:
        details["seconds"] = round(time.perf_counter() - start, 2)

    details.update({"status": result.get("status"), "model": result.get("model") or clarifier.model,
                    "entities": result.get("entities") or [],
                    "demographics": result.get("demographics") or {},
                    "removed_by_grounding_check": result.get("removed_by_grounding_check", 0)})
    if result.get("status") == "error":
        details["error"] = config.redact(result.get("error") or "no error message")
        groq_message = _CLARIFIER_LAST.get("groq_error_message")  # from the LAST attempt only
        if groq_message:
            details["groq_error_message"] = groq_message
            details["error"] += f" - Groq says: {groq_message}"
        raise ClarifierError(f"Clarifier failed: {details['error']}", details)
    try:
        _check_usable(result)
    except ValueError as exc:
        details["error"] = f"unusable result: {exc}"
        raise ClarifierError(f"Clarifier returned an unusable result: {exc}", details)
    return {key: copy.deepcopy(result[key]) for key in CLARIFIER_KEYS}, details


def run_clarifier(patient_text):
    """-> {clinical_text, diseases, symptoms, medications, tests} (details dropped)"""
    return run_clarifier_full(patient_text)[0]


def run_vector_rag(clarifier_output, top_k):
    """-> output_adapter adapt() dict (diagnostic_candidates, evidence, ...)"""
    if _mode("vector_rag") == "replay":
        out = _sample("rag_case1.json")  # from Medico\example_input.json
        out.setdefault("metadata", {})["requested_top_k"] = top_k
        return out
    rag = _real("vector_rag")
    orchestrator = _rag_orchestrator(top_k)  # his top_k is fixed per orchestrator
    context = rag["context_class"](
        **{k: clarifier_output.get(k, "" if k == "clinical_text" else [])
           for k in ("clinical_text", "diseases", "symptoms", "medications", "tests")})
    stats = {"top_k": top_k}
    RAG_STATS["calls"].append(stats)
    start = time.perf_counter()
    try:
        out = rag["adapter"].adapt(context, orchestrator.analyze(context))
        # his contract is passed on unchanged; we only add the top_k we asked for (as in the replay sample)
        out.setdefault("metadata", {})["requested_top_k"] = top_k
        return out
    finally:
        stats["total_seconds"] = round(time.perf_counter() - start, 2)


def run_knowledge_graph(clarifier_output, entities=None):
    """-> run_kg_agent() output + OUR symptom_matching / unmatched_symptoms (real mode) + graph_confidence.

    entities = clarifier_details["entities"] (for body_site). Real mode: each Clarifier symptom is matched to
    an existing graph phenotype by OUR rules (match_symptoms) and his run_kg_agent gets those graph names.
    """
    if _mode("knowledge_graph") == "replay":
        out = copy.deepcopy(_load_json(str(config.KG_SAMPLE)))
    else:
        lookup, matching, kg_input = _REAL.get("kg_lookup"), None, clarifier_output
        if lookup is not None:
            matching = match_symptoms(clarifier_output.get("symptoms") or [], entities or [], lookup)
            terms = list(dict.fromkeys(m["graph_term"] for m in matching if m["graph_term"]))
            kg_input = {**clarifier_output, "symptoms": terms}
        out = _real("knowledge_graph")(kg_input, top_k=config.KG_TOP_K)
        if out.get("status") == "error":
            raise RuntimeError(f"run_kg_agent returned an error: {out.get('error')}")
        if matching is not None:
            out["symptom_matching"] = matching
            out["unmatched_symptoms"] = [m["symptom"] for m in matching if not m["graph_term"]]
        if _REAL.get("kg_rank") is not None:
            ranked = _REAL["kg_rank"](kg_input.get("symptoms") or [], config.KG_RANK_TOP)
            out["ranking_top10"] = _ranking(ranked.get("candidate_diseases"))
    if "ranking_top10" not in out:  # replay / no ranker: his top list as given
        out["ranking_top10"] = _ranking(((out.get("disease_prediction") or {}).get("candidate_diseases")))
    out["graph_confidence"], out["graph_confidence_details"] = graph_confidence(out)
    return out


def _ranking(candidates):
    """Scoring-only ranking: [{disease, matched_symptom_count, coverage_fraction}] (never given to other agents)."""
    return [{"disease": c.get("disease"), "matched_symptom_count": c.get("matched_symptom_count"),
             "coverage_fraction": c.get("coverage_fraction")} for c in (candidates or [])][:config.KG_RANK_TOP]


# ---------------------------------------------------------------- KG symptom matching (OUR wrapper, his graph)

# Body sites that are joints: a joint symptom may also match the general "Joint ..." phenotype
JOINT_SITES = {"knee", "hip", "ankle", "wrist", "elbow", "shoulder", "finger", "toe", "jaw", "spine", "joint",
               "knuckle"}
_SITE_FILLER = {"both", "left", "right", "bilateral", "the", "of", "and", "a", "an", "in", "on"}
MATCH_RULES = ("exact", "body_site", "joint_general", "case_body_site", "plain_word")


def make_symptom_lookup(G, synonyms, diseases_for, relation):
    """text -> graph phenotype name, using HIS synonym table + exact node name (like his predict_disease),
    but only phenotypes linked to at least 1 disease. Never invents a name."""
    nodes = {str(n).strip().lower(): n for n in G.nodes()}
    linked = {}

    def lookup(text):
        key = str(text).strip().lower()
        node = nodes.get(str(synonyms.get(key, text)).strip().lower())
        if node is None or G.nodes[node].get("type") != "effect/phenotype":
            return None
        if node not in linked:
            linked[node] = bool(diseases_for(G, node, relation))
        return node if linked[node] else None
    return lookup


def _singular(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _site_words(site):
    return [_singular(w) for w in re.findall(r"[a-z]+", str(site or "").lower()) if w not in _SITE_FILLER]


def _is_joint(words):
    return any(w in JOINT_SITES for w in words)


def _norm(text):
    return " ".join(str(text or "").lower().split())


def match_symptoms(symptoms, entities, lookup):
    """For each Clarifier symptom try, in order (first graph phenotype found wins):
    exact -> body_site combos (entity's own site) -> joint_general ("joint <word>", own site is a joint)
    -> case_body_site ("joint <word>", no own site but another present symptom has a joint site)
    -> plain_word (main word alone). Returns one record per symptom (graph_term None = unmatched)."""
    present = [e for e in entities if isinstance(e, dict) and e.get("type") == "symptom"
               and e.get("status") == "present"]
    site_of = {}
    for e in present:
        site_of.setdefault(_norm(e.get("text")), e.get("body_site"))
    case_site = next((s for s in site_of.values() if s and _is_joint(_site_words(s))), None)

    records = []
    for symptom in symptoms:
        own_site = site_of.get(_norm(symptom))
        site_words = _site_words(own_site)
        words = [w for w in re.findall(r"[a-z]+", _norm(symptom)) if _singular(w) not in site_words]
        word = words[-1] if words else (_norm(symptom).split() or [""])[-1]
        site = " ".join(site_words)
        steps = [("exact", [symptom])]
        if site:
            steps.append(("body_site", [f"{site} {word}", f"{word} of {site}", f"{word} of the {site}",
                                        f"{site} joint {word}"]))
            if _is_joint(site_words):
                steps.append(("joint_general", [f"joint {word}"]))
        elif case_site:
            steps.append(("case_body_site", [f"joint {word}"]))
        steps.append(("plain_word", [word]))

        record = {"symptom": symptom, "body_site": own_site, "case_body_site": None,
                  "graph_term": None, "rule": "unmatched", "tried": []}
        for rule, candidates in steps:
            for candidate in candidates:
                if not candidate.strip() or candidate in record["tried"]:
                    continue
                record["tried"].append(candidate)
                node = lookup(candidate)
                if node:
                    record.update({"graph_term": node, "rule": rule,
                                   "case_body_site": case_site if rule == "case_body_site" else None})
                    break
            if record["graph_term"]:
                break
        records.append(record)
    return records


def graph_confidence(kg_output):
    """Top predicted disease d: e = matched(d) / recognised symptoms, c = coverage(d),
    graph_confidence = 2*e*c / (e + c). -> (value or None, details). None (not 0) when nothing is recognised."""
    prediction = (kg_output or {}).get("disease_prediction") or {}
    candidates = prediction.get("candidate_diseases") or []
    recognised = (prediction.get("metadata") or {}).get("symptoms_matched_count") or 0
    if not recognised:
        return None, {"reason": "no patient symptom recognised in the graph"}
    if not candidates:
        return None, {"reason": "no candidate disease"}
    top = candidates[0]
    e = top.get("matched_symptom_count", 0) / recognised
    c = top.get("coverage_fraction") or 0.0
    value = 0.0 if e + c == 0 else 2 * e * c / (e + c)
    return round(value, 3), {"disease": top.get("disease"), "e": round(e, 3), "coverage": round(c, 3),
                             "matched": top.get("matched_symptom_count"), "recognised": recognised}


# Medico's extract_graph_score takes the FIRST "confidence|score|fidelity: <0/1 number>" in the KG text
# as the structural part of C2. Only OUR graph_confidence may provide it.
_SCORE_WORD = re.compile(r"confidence|score|fidelity", re.IGNORECASE)
_SCORE_TEXT = re.compile(r"(confidence|score|fidelity)(?=[:\s]+[01])", re.IGNORECASE)


def _scrub_scores(value):
    """Rename keys / break text that Medico's regex could read as a confidence (her code unchanged)."""
    if isinstance(value, dict):
        return {(f"{k}_value" if _SCORE_WORD.search(str(k)) else k): _scrub_scores(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_scores(v) for v in value]
    if isinstance(value, str):
        return _SCORE_TEXT.sub(r"\1_", value)
    return value


def medico_kg_view(kg_output):
    """What Medico receives as raw_graph_evidence: graph_confidence FIRST (the only readable confidence),
    everything else scrubbed; no confidence line at all when graph_confidence is None (her part stays 0.0)."""
    if not kg_output:
        return {}
    view = {}
    if kg_output.get("graph_confidence") is not None:
        view["graph_confidence"] = kg_output["graph_confidence"]
    rest = {k: v for k, v in kg_output.items()
            if k not in ("graph_confidence", "graph_confidence_details", "ranking_top10")}  # top 10 = scoring only
    view.update(_scrub_scores(rest))  # top-level keys are scrubbed too
    return view


def run_evidence_scanner(disease):
    """One disease per call -> {hypothesis, broadened, evidence: [...]}"""
    if _mode("evidence_scanner") == "replay":
        samples = _load_json(str(config.SAMPLES_DIR / "evidence_samples.json"))
        if disease in samples:
            return copy.deepcopy(samples[disease])
        return {"hypothesis": disease, "broadened": False, "evidence": [],
                "replay": "no sample"}
    scanner = _real("evidence_scanner")
    record = _new_evidence_record(disease)
    wait, reason = _groq_wait_seconds()
    if wait:
        _sleep(wait)
    record.update({"paused_seconds": wait, "pause_reason": reason})
    with _EVIDENCE_LOCK:
        EVIDENCE_STATS["calls"].append(record)
    start = time.perf_counter()
    try:
        result = scanner.invoke({"request": disease, "top_n": config.EVIDENCE_TOP_N})
        record["items"] = len(result.get("evidence") or [])
        record["broadened"] = result.get("broadened")
        return result
    except Exception as exc:
        record["error"] = config.redact(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = round(time.perf_counter() - start, 2)


def run_fusion(medico_state):
    """Medico build_pipeline().invoke(state) -> {confidence, fusion_report, optimizer_response, fallback_used}"""
    if _mode("fusion") == "replay":
        return _sample("fusion_case1.json")  # from Medico\example_output.json
    pipeline = _real("fusion")
    rec = {"attempt": medico_state.get("attempt")}
    FUSION_STATS["passes"].append(rec)
    start = time.perf_counter()
    try:
        state = pipeline.invoke(medico_state)
    finally:
        rec["seconds"] = round(time.perf_counter() - start, 2)
    out = {k: state[k] for k in FUSION_OUTPUT_KEYS if k in state}  # her output keys, unchanged
    conf = out.get("confidence") or {}
    rec.update({
        "C1": (conf.get("confidence_1") or {}).get("score"),
        "C2": (conf.get("confidence_2") or {}).get("score"),
        "fused": (conf.get("fusion") or {}).get("final_fused_score"),
        "graph_score": (conf.get("confidence_2") or {}).get("graph_score"),
        "graph_score_default_0_90": not _GRAPH_SCORE_PATTERN.search(str(state.get("graph_evidence", ""))),
        "fallback_used": out.get("fallback_used"),
    })
    return out


def fusion_failure_reason():
    """Reason(s) the last Fusion pass's AI calls failed, if our wrapper caught any."""
    return "; ".join(_current_pass().get("failures", [])) or None


# ---------------------------------------------------------------- KG safety list

def _treatment_clash_items(disease, options):
    items = []
    for bucket in ("clash_major", "clash_moderate"):
        for entry in options.get(bucket, []):
            for clash in entry.get("clashes_with", []):
                severity = clash.get("severity", "unknown")
                if severity not in ("moderate", "major"):
                    continue
                items.append({
                    "type": "treatment_clash",
                    "drug": entry["drug"],
                    "target": clash["medication"],
                    "severity": severity,
                    "message": (f"{entry['drug']} (a treatment option for {disease}) has a "
                                f"{severity} interaction with current medicine {clash['medication']}."),
                })
    return items


SUMMARY_WORDING = {
    "minor": "a minor interaction",
    "unknown": "an interaction of unknown severity (not rated in DDInter)",
}


def _treatment_summary_items(disease, options):
    """Up to two items per (disease, current medicine): one "minor", one "unknown".

    drug = "N treatment options", target = the current medicine, disease named in message.
    Any severity other than "minor" in this bucket counts as "unknown".
    """
    groups = {}  # (medication, severity) -> [treatment drug names]
    for entry in options.get("clash_minor_or_unknown", []):
        for clash in entry.get("clashes_with", []):
            severity = "minor" if clash.get("severity") == "minor" else "unknown"
            groups.setdefault((clash["medication"], severity), []).append(entry["drug"])

    items = []
    for (med, severity) in sorted(groups, key=lambda k: (k[0].lower(), k[1] != "minor")):
        drugs = groups[(med, severity)]
        noun = "treatment option" if len(drugs) == 1 else "treatment options"
        verb = "has" if len(drugs) == 1 else "have"
        items.append({
            "type": "treatment_clash_summary",
            "drug": f"{len(drugs)} {noun}",
            "target": med,
            "severity": severity,
            "message": (f"{len(drugs)} {noun} for {disease} {verb} {SUMMARY_WORDING[severity]} "
                        f"with {med}: {', '.join(drugs)}."),
        })
    return items


def make_kg_safety_messages(kg_output):
    """Flatten KG medication_safety into [{type, drug, target, severity, message}]."""
    safety = (kg_output or {}).get("medication_safety") or {}
    items = []

    for w in safety.get("warnings", []):
        if w.get("type") == "contraindication":
            for drug in w.get("medications", []):
                message = w.get("message") or ""
                if "contraindicated" not in message:
                    message = f"{drug} is contraindicated in {w['disease']}."
                items.append({"type": "contraindication", "drug": drug, "target": w["disease"],
                              "severity": w.get("severity", "unknown"), "message": message})
        elif w.get("type") == "current_medication_interaction":
            meds = w.get("medications", [])
            if len(meds) == 2:
                items.append({"type": "drug_clash", "drug": meds[0], "target": meds[1],
                              "severity": w.get("severity", "unknown"),
                              "message": w.get("message", "")})

    for entry in safety.get("by_disease", []):
        options = entry.get("treatment_options") or {}
        items.extend(_treatment_clash_items(entry["disease"], options))
        items.extend(_treatment_summary_items(entry["disease"], options))
    return items


# ---------------------------------------------------------------- batch runs: counters and run info

def llm_counters():
    """Running totals of LLM requests by provider (our counters; retries included). Diff before/after a case."""
    evidence_groq = sum((rec.get("requests") or {}).get("groq", 0) for rec in EVIDENCE_STATS["calls"])
    return {"groq": CLARIFIER_STATS["groq_requests"] + evidence_groq,
            "openrouter": RAG_STATS["openrouter_calls"] + FUSION_STATS["openrouter_calls"]}


AGENT_DIRS = {"clarifier": "CLARIFIER_DIR", "vector_rag": "RAG_DIR", "knowledge_graph": "CODE_DIR",
              "evidence_scanner": "EVIDENCE_DIR", "fusion": "MEDICO_DIR"}


def agent_commits():
    """Short git commit of every agent's folder (read-only; "unknown" if git fails)."""
    return {agent: _git_commit(getattr(config, attr), "", agent) for agent, attr in AGENT_DIRS.items()}


def run_info(disabled=()):
    """What produced a batch run: switches, modes, models (with source), commits, Groq route. No key values."""
    clarifier_model = os.getenv("CLINICAL_TEXT_MODEL") or "openai/gpt-oss-120b"
    return {
        "disabled_agents": sorted(disabled),
        "agent_modes": dict(config.AGENT_MODES),
        "clarifier_fake": config.CLARIFIER_FAKE,
        "groq_route": config.GROQ_ROUTE,
        "models": {
            "clarifier": {"model": clarifier_model,
                          "source": "CLINICAL_TEXT_MODEL" if os.getenv("CLINICAL_TEXT_MODEL") else "his default"},
            "evidence_scanner": {"model": config.GROQ_MODEL_OVERRIDE or "her code (GROQ_MODEL)",
                                 "source": "GROQ_MODEL_OVERRIDE" if config.GROQ_MODEL_OVERRIDE else "her code"},
            "vector_rag_answerer": {"model": config.RAG_ANSWER_MODEL or "openai/gpt-5.6-luna",
                                    "source": config.RAG_ANSWER_MODEL_SOURCE},
            "vector_rag_embedding": {"model": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                                     "source": "his code"},
            "fusion": {"model": os.getenv("OPENROUTER_MODEL") or "gpt-4o-mini (her paid default)",
                       "source": "OPENROUTER_MODEL" if os.getenv("OPENROUTER_MODEL") else "her default"},
        },
        "commits": agent_commits(),
        "kg_top_k": config.KG_TOP_K, "kg_rank_top": config.KG_RANK_TOP,
        "threshold": config.THRESHOLD, "max_loops": config.MAX_LOOPS,
    }
