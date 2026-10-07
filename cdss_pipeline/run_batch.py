r"""Batch runner: run many JSONL cases through the pipeline, one saved JSON per case (resumable).

From the integration folder (start Qdrant first unless --disable includes rag):
    .venv\Scripts\python.exe -B -m cdss_pipeline.run_batch --input cases.jsonl --out results\<run_name>
        [--disable rag,kg,evidence,optimizer] [--limit N]

Writes into --out:
    run_manifest.json   what produced the run (switches, models + sources, commits, Groq route, Qdrant counts)
    cases\<case_id>.json one per case = the --json result + "case" (input + gold) + "batch" (seconds, calls, ...)
    failed_cases.txt    cases that crashed (not saved, so the next run retries them)
    summary.csv         one row per saved case
RESUME: a case that already has a saved file is skipped (no quota spent again). A saved run made with other
switches, models or commits is never mixed: the run stops and asks for a new --out.
Input format: see TESTING.md.
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

from cdss_pipeline import config

SETS = ("medqa", "rare", "medsafety", "clarifier", "other")
MANIFEST = "run_manifest.json"
SUMMARY_FIELDS = ["case_id", "set", "analysis_status", "disabled_agents", "seconds", "groq_calls",
                  "openrouter_calls", "loops", "evidence_fallback_used", "errors", "rag_top1", "kg_top3",
                  "C1", "C2", "fused", "graph_confidence"]


class InputError(ValueError):
    """The JSONL input is not usable: the run must not start."""


def safe_name(case_id):
    return re.sub(r"[^A-Za-z0-9._-]", "_", case_id)


def load_cases(path):
    """Read and check the JSONL input. Returns a list of case dicts (all fields, gold included)."""
    cases, ids, files = [], set(), {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"line {number}: not valid JSON ({exc.msg})")
        if not isinstance(case, dict):
            raise InputError(f"line {number}: each line must be one JSON object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise InputError(f"line {number}: case_id is missing")
        if case_id in ids:
            raise InputError(f"line {number}: duplicate case_id {case_id!r}")
        if files.setdefault(safe_name(case_id), case_id) != case_id:
            raise InputError(f"line {number}: case_id {case_id!r} gives the same file name as "
                             f"{files[safe_name(case_id)]!r}")
        if not isinstance(case.get("input_text"), str) or not case["input_text"].strip():
            raise InputError(f"line {number}: input_text is missing ({case_id})")
        if case.get("set") not in SETS:
            raise InputError(f"line {number}: set must be one of {', '.join(SETS)} ({case_id})")
        ids.add(case_id)
        cases.append(case)
    if not cases:
        raise InputError("no cases in the input file")
    return cases


def run_identity(disabled):
    """The parts of the manifest that must be identical to resume a run."""
    from cdss_pipeline import wrappers
    info = wrappers.run_info(disabled)
    return {k: info[k] for k in ("disabled_agents", "agent_modes", "clarifier_fake", "groq_route", "models",
                                 "commits", "kg_top_k", "kg_rank_top", "threshold", "max_loops")}


def check_manifest(out_dir, identity):
    """Existing run: identity must match exactly, else stop (no mixed runs). -> saved manifest or None."""
    path = Path(out_dir) / MANIFEST
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    diff = sorted(k for k in identity if saved.get(k) != identity[k])
    if diff:
        raise SystemExit(f"\nSTOPPED: {out_dir} was made with different settings ({', '.join(diff)}). "
                         "Use a new --out so results are never mixed.")
    return saved


def write_manifest(out_dir, identity, saved, startup_report):
    """Write a new manifest, or add this session (resumed_at + startup report) to the saved one."""
    now = datetime.now().isoformat(timespec="seconds")
    if saved is None:
        manifest = {"created_at": now, **identity}
    else:
        manifest = saved
        manifest.setdefault("resumed_at", []).append(now)
    manifest["qdrant"] = ("rag disabled" if "rag" in identity["disabled_agents"]
                          else (startup_report.get("vector_rag") or {}).get("collections", "not loaded"))
    manifest.setdefault("startup_reports", []).append(startup_report)
    (Path(out_dir) / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    return manifest


def saved_result(path):
    """A finished case file, or None (missing or broken -> run the case again)."""
    try:
        item = json.loads(Path(path).read_text(encoding="utf-8"))
        return item if isinstance(item, dict) and "final_output" in item and "batch" in item else None
    except (OSError, ValueError):
        return None


def summary_row(item):
    fo, batch, case = item.get("final_output") or {}, item.get("batch") or {}, item.get("case") or {}
    conf = fo.get("confidence") or {}
    diagnosis = fo.get("most_likely_diagnosis") or {}
    return {"case_id": case.get("case_id"), "set": case.get("set"),
            "analysis_status": fo.get("analysis_status"),
            "disabled_agents": ",".join(fo.get("disabled_agents") or []),
            "seconds": batch.get("seconds"), "groq_calls": batch.get("groq_calls"),
            "openrouter_calls": batch.get("openrouter_calls"), "loops": item.get("loop_count"),
            "evidence_fallback_used": fo.get("evidence_fallback_used"), "errors": len(batch.get("errors") or []),
            "rag_top1": diagnosis.get("condition"), "kg_top3": "; ".join((fo.get("kg_ranking_top10") or [])[:3]),
            "C1": conf.get("C1"), "C2": conf.get("C2"), "fused": conf.get("fused"),
            "graph_confidence": fo.get("graph_confidence")}


def write_summary(out_dir):
    rows = [summary_row(item) for item in (saved_result(p) for p in sorted((Path(out_dir) / "cases").glob("*.json")))
            if item]
    with open(Path(out_dir) / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _progress(index, total, case_id, item):
    fo, batch = item["final_output"] or {}, item["batch"]
    diagnosis = (fo.get("most_likely_diagnosis") or {}).get("condition")
    kg = (fo.get("kg_ranking_top10") or [None])[0]
    flags = (" FALLBACK" if fo.get("evidence_fallback_used") else "") + (
        f" errors {len(batch['errors'])}" if batch["errors"] else "")
    return (f"[{index}/{total}] {case_id}  {fo.get('analysis_status')}  {batch['seconds']} s  "
            f"groq {batch['groq_calls']}  openrouter {batch['openrouter_calls']}  "
            f"dx: {diagnosis or '-'} | kg: {kg or '-'}{flags}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file, one case per line (see TESTING.md)")
    parser.add_argument("--out", required=True, help=r"results folder, e.g. results\full_run_1")
    parser.add_argument("--disable", default=None, help="ablation: rag, kg, evidence, optimizer (comma-separated)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases of the input")
    args = parser.parse_args(argv)

    from cdss_pipeline import run_pipeline, wrappers
    try:
        disabled = config.parse_disabled(args.disable if args.disable is not None else config.DISABLED_FROM_ENV)
        cases = load_cases(args.input)
    except (config.SwitchError, InputError) as exc:
        raise SystemExit(f"\nSTOPPED: {exc}")
    if args.limit is not None:
        cases = cases[:max(args.limit, 0)]
    out_dir, cases_dir = Path(args.out), Path(args.out) / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    identity = run_identity(disabled)
    saved_manifest = check_manifest(out_dir, identity)  # stop BEFORE loading anything if settings differ
    pending = [c for c in cases if saved_result(cases_dir / f"{safe_name(c['case_id'])}.json") is None]
    print(f"{len(cases)} case(s): {len(cases) - len(pending)} already saved (skipped), {len(pending)} to run"
          + (f"; disabled: {', '.join(disabled)}" if disabled else ""), flush=True)
    if pending:
        write_manifest(out_dir, identity, saved_manifest, run_pipeline.startup(disabled))

    for index, case in enumerate(pending, start=1):
        before, start = wrappers.llm_counters(), time.perf_counter()
        try:
            state, path = run_pipeline.run(case["input_text"], disabled)
        except Exception as exc:  # unexpected crash: logged, not saved -> retried next time
            message = config.redact(f"{type(exc).__name__}: {exc}")
            with open(out_dir / "failed_cases.txt", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{case['case_id']}\t{message}\n")
            print(f"[{index}/{len(pending)}] {case['case_id']}  CRASHED (not saved, will be retried): {message}",
                  flush=True)
            continue
        after = wrappers.llm_counters()
        item = run_pipeline.json_result(case["case_id"], state)
        item["case"] = case
        item["batch"] = {"seconds": round(time.perf_counter() - start, 2),
                         "groq_calls": after["groq"] - before["groq"],
                         "openrouter_calls": after["openrouter"] - before["openrouter"],
                         "path": path, "errors": state.get("errors") or [],
                         "finished_at": datetime.now().isoformat(timespec="seconds")}
        target = cases_dir / f"{safe_name(case['case_id'])}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(item, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(target)  # a stopped run never leaves a half-written result
        print(_progress(index, len(pending), case["case_id"], item), flush=True)

    saved = write_summary(out_dir)
    print(f"\n{saved} case(s) saved in {cases_dir}; summary.csv written. Score with:\n"
          f"  .venv\\Scripts\\python.exe -B -m cdss_pipeline.score --run {out_dir}")


if __name__ == "__main__":
    main()
