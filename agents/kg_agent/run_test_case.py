"""
Interactive runner for the Knowledge Graph agent.

Lists the Clarifier test cases in tests/, lets you pick one by number, runs
run_kg_agent() on it and prints a readable report. The medication section is
built from medication_safety's "by_disease" and "medication_pair_clashes"
views; the returned data is not changed.

usage: python run_test_case.py
"""

import json
import logging
import textwrap
from pathlib import Path

from kg_agent import run_kg_agent

TESTS_DIR = Path(__file__).resolve().parent / "tests"

DISCLAIMER = (
    "Treatment drugs are options from the knowledge graph for a doctor to consider, "
    "not prescriptions. 'No known clash' means no interaction is recorded in our data, "
    "not that the drug is guaranteed safe."
)

# Safest to riskiest; (group key, heading, show which medicines it clashes with).
GROUP_HEADINGS = [
    ("already_taken_by_patient", "Already taken by the patient", False),
    ("no_known_clash", "No known clash with current medicines", False),
    ("clash_minor_or_unknown", "Clash, minor or unknown severity", False),
    ("clash_moderate", "Clash, moderate severity, use with care", True),
    ("clash_major", "Clash, major severity, avoid unless the doctor decides otherwise", True),
]

WRAP_WIDTH = 100


def _print_wrapped(heading, items, indent):
    """Print 'heading: a, b, c' wrapped at WRAP_WIDTH, continuation lines indented."""
    print(textwrap.fill(
        f"{heading}: {', '.join(items)}", width=WRAP_WIDTH,
        initial_indent=" " * indent, subsequent_indent=" " * (indent + 4),
        break_on_hyphens=False,
    ))


def _name(name):
    """Quote names that contain a comma (e.g. "Heparin, bovine") so lists stay unambiguous."""
    return f'"{name}"' if "," in name else name


def _drug_label(entry, show_clashes):
    if not show_clashes:
        return _name(entry["drug"])
    meds = ", ".join(_name(c["medication"]) for c in entry["clashes_with"])
    return f"{_name(entry['drug'])} (clashes with {meds})"


def _print_disease(entry):
    source = f" ({entry['disease_source']} disease)" if entry.get("disease_source") else ""
    print(f"\nDISEASE: {entry['disease']}{source}")

    print("  Problems with current medicines:")
    if not entry["current_medication_problems"]:
        print("    none")
    for p in entry["current_medication_problems"]:
        print(f"    - {_name(p['medication'])}: should NOT be used in this disease")

    print(f"  Treatment drugs for this disease ({entry['treatment_drug_count']} in the graph):")
    if entry["note"]:
        print(f"    {entry['note']}")
    for key, heading, show_clashes in GROUP_HEADINGS:
        drugs = entry["treatment_options"][key]
        if drugs:
            _print_wrapped(f"{heading} ({len(drugs)})",
                           [_drug_label(d, show_clashes) for d in drugs], indent=4)


def _severity_text(clash):
    source = clash["severity_source"] or "not in DDInter"
    return f"severity {clash['severity']} ({source})"


def _print_report(case_name, case, output):
    print("=" * 70)
    print(f"KNOWLEDGE GRAPH AGENT - {case_name} - status: {output['status']}")
    print("=" * 70)
    print(f"Symptoms:     {', '.join(case.get('symptoms') or []) or '(none)'}")
    print(f"Medications:  {', '.join(case.get('medications') or []) or '(none)'}")
    print(f"Known diseases (from Clarifier): {', '.join(case.get('diseases') or []) or '(none)'}")
    if output["error"]:
        print(f"\nERROR: {output['error']}")

    prediction = output["disease_prediction"]
    if prediction is not None:
        print("\nTOP DISEASES (predicted from symptoms)")
        if not prediction["candidate_diseases"]:
            print("  (none selected)")
        for rank, c in enumerate(prediction["candidate_diseases"], 1):
            print(f"  {rank}. {c['disease']}  "
                  f"(matched {c['matched_symptom_count']}/{c['total_documented_symptoms']} "
                  f"documented symptoms, coverage {c['coverage_fraction']:.3f})")
        if prediction["symptoms_not_found"]:
            print(f"  symptoms not found: {', '.join(prediction['symptoms_not_found'])}")

    safety = output["medication_safety"]
    if safety is not None:
        print("\n" + "=" * 70)
        print("MEDICATION SAFETY")
        print("=" * 70)
        print(textwrap.fill(DISCLAIMER, width=WRAP_WIDTH))
        if safety["diseases_not_found"]:
            print(f"\nDiseases not found in graph: {', '.join(safety['diseases_not_found'])}")

        # Diseases with no treatment drugs and no medicine problems have
        # nothing to show, so they are listed together on one line.
        no_treatments = []
        for entry in safety["by_disease"]:
            if entry["treatment_drug_count"] == 0 and not entry["current_medication_problems"]:
                no_treatments.append(entry["disease"])
            else:
                _print_disease(entry)
        if no_treatments:
            print()
            _print_wrapped("No treatment drugs in the graph for",
                           [_name(d) for d in no_treatments], indent=0)

        print("\nPATIENT'S MEDICINES CLASHING WITH EACH OTHER:")
        if not safety["medication_pair_clashes"]:
            print("  none")
        for clash in safety["medication_pair_clashes"]:
            print(f"  - {' + '.join(_name(m) for m in clash['medications'])}: {_severity_text(clash)}")

        if safety["note"]:
            print(f"\nNOTE: {safety['note']}")
            print(f"Not checked: {', '.join(safety['medications_not_checked'])}")


def choose_case(files):
    print("Test cases in tests/:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    while True:
        try:
            choice = input("Choose a test case number (or q to quit): ").strip()
        except EOFError:
            return None
        if choice.lower() in ("", "q"):
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print(f"  Please enter a number from 1 to {len(files)}, or q.")


def main():
    files = sorted(TESTS_DIR.glob("*.json"))
    if not files:
        print(f"No .json test cases found in {TESTS_DIR}")
        return

    path = choose_case(files)
    if path is None:
        return

    try:
        with open(path, encoding="utf-8") as f:
            case = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read {path.name}: {e}")
        return
    if not isinstance(case, dict):
        print(f"{path.name} must contain a JSON object, got {type(case).__name__}")
        return

    print(f"\nRunning {path.name} ...")
    _print_report(path.name, case, run_kg_agent(case))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
