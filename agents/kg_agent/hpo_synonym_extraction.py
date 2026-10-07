"""
Build a full symptom-synonym lookup dictionary by matching every phenotype
node in our graph (relation_filtered_graph_v2.pkl) against the official HPO
ontology (hp.json), via id reconstruction: PrimeKG phenotype id -> zero-padded
HP:XXXXXXX -> hp.json node -> its listed synonyms.

Reuses load_graph() and load_and_fix() from build_full_graph_v2.py (no
rebuild - just loads the pickle and re-reads the CSV for id/name pairs).
Does not touch primekg_relation_filtered.csv, kg_real_traversal.py,
relation_filtered_graph_v2.pkl, or any small manual symptom_synonyms file
that may exist elsewhere.
"""

import json

from build_full_graph_v2 import load_graph, load_and_fix

HP_FILE = "hp.json"
OUTPUT_FILE = "symptom_synonyms_full.py"


def load_hp_index(hp_path=HP_FILE):
    with open(hp_path, encoding="utf-8") as f:
        data = json.load(f)
    nodes = data["graphs"][0]["nodes"]
    by_id = {}
    for n in nodes:
        nid = n.get("id", "")
        if "HP_" in nid:
            short = nid.rsplit("/", 1)[-1].replace("_", ":")
            by_id[short] = n
    return by_id


def reconstruct_hp_id(primekg_id):
    try:
        num = int(str(primekg_id).strip())
    except ValueError:
        return None
    return f"HP:{num:07d}"


def main():
    print("Loading graph (from pickle)...")
    G = load_graph()
    phenotype_names = {n for n, attrs in G.nodes(data=True) if attrs.get("type") == "effect/phenotype"}
    print(f"Phenotype nodes in graph: {len(phenotype_names)}")

    print("Loading + canonicalizing CSV for id/name pairs (no graph rebuild)...")
    df = load_and_fix()

    name_to_id = {}
    inconsistent = []
    for _, row in df[df["x_type"] == "effect/phenotype"][["x_id", "x_name"]].drop_duplicates().iterrows():
        name, pid = row["x_name"], row["x_id"]
        if name in name_to_id and name_to_id[name] != pid:
            inconsistent.append((name, name_to_id[name], pid))
        name_to_id.setdefault(name, pid)
    for _, row in df[df["y_type"] == "effect/phenotype"][["y_id", "y_name"]].drop_duplicates().iterrows():
        name, pid = row["y_name"], row["y_id"]
        if name in name_to_id and name_to_id[name] != pid:
            inconsistent.append((name, name_to_id[name], pid))
        name_to_id.setdefault(name, pid)

    print(f"Distinct canonical phenotype names with an id: {len(name_to_id)}")
    if inconsistent:
        print(f"WARNING: {len(inconsistent)} names had inconsistent ids across rows. Examples:")
        for ex in inconsistent[:5]:
            print(" ", ex)

    print("Loading hp.json index...")
    hp_by_id = load_hp_index()
    print(f"HPO terms available: {len(hp_by_id)}")

    matched = 0
    unmatched = []
    synonym_dict = {}
    key_collisions = []

    for name in sorted(phenotype_names):
        pid = name_to_id.get(name)
        # self-mapping for every phenotype node, matched or not
        key = name.strip().lower()
        if key not in synonym_dict:
            synonym_dict[key] = name
        elif synonym_dict[key] != name:
            key_collisions.append((key, synonym_dict[key], name, "self-name"))

        if pid is None:
            unmatched.append((name, None, "no id found in CSV"))
            continue

        hp_id = reconstruct_hp_id(pid)
        if hp_id is None:
            unmatched.append((name, pid, "id not numeric"))
            continue

        node = hp_by_id.get(hp_id)
        if node is None:
            unmatched.append((name, hp_id, "HP id not found in hp.json"))
            continue

        matched += 1
        for syn in node.get("meta", {}).get("synonyms", []):
            val = syn.get("val")
            if not val:
                continue
            skey = val.strip().lower()
            if skey not in synonym_dict:
                synonym_dict[skey] = name
            elif synonym_dict[skey] != name:
                key_collisions.append((skey, synonym_dict[skey], name, "synonym"))

    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print(f"Total phenotype terms in graph: {len(phenotype_names)}")
    print(f"Matched to an HPO id: {matched}")
    print(f"Unmatched: {len(unmatched)}")
    print(f"Total synonym dictionary entries (keys): {len(synonym_dict)}")
    print(f"Key collisions (same lowercased key claimed by two different canonical names): {len(key_collisions)}")

    if unmatched:
        print(f"\nUnmatched examples (up to 15 of {len(unmatched)}):")
        for name, ref, reason in unmatched[:15]:
            print(f"  {name!r}  (id={ref}) -> {reason}")

    if key_collisions:
        print(f"\nKey collision examples (up to 10 of {len(key_collisions)}):")
        for skey, first_name, second_name, kind in key_collisions[:10]:
            print(f"  {skey!r} -> kept {first_name!r}, also claimed by {second_name!r} ({kind})")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write("Auto-generated symptom-synonym lookup: lowercased synonym/name -> canonical\n")
        f.write("phenotype node name as it appears in relation_filtered_graph_v2.pkl.\n\n")
        f.write("Built by hpo_synonym_extraction.py from primekg_relation_filtered.csv\n")
        f.write("phenotype ids reconstructed into HPO ids (HP:XXXXXXX) and looked up in hp.json.\n")
        f.write(f"Source phenotype terms: {len(phenotype_names)}, matched to HPO: {matched}, ")
        f.write(f"unmatched: {len(unmatched)}, total dictionary entries: {len(synonym_dict)}.\n")
        f.write('"""\n\n')
        f.write("SYMPTOM_SYNONYMS = ")
        f.write(json.dumps(synonym_dict, indent=2, sort_keys=True, ensure_ascii=False))
        f.write("\n")

    print(f"\nSaved {OUTPUT_FILE} ({len(synonym_dict)} entries)")

    print("\n" + "=" * 70)
    print("TEST LOOKUP")
    print("=" * 70)
    test_terms = ["fever", "shortness of breath", "breathlessness", "pyrexia", "headache", "nausea"]
    for term in test_terms:
        result = synonym_dict.get(term.lower())
        print(f"  {term!r:30} -> {result!r}")


if __name__ == "__main__":
    main()
