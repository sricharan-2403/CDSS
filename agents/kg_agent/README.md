# Knowledge Graph agent

A knowledge-graph agent for a clinical decision support system (CDSS). It uses
the PrimeKG medical knowledge graph to do two things:

1. **Disease prediction from symptoms.** Symptom names are matched to graph
   phenotypes (with synonyms from the Human Phenotype Ontology, so for example
   "high temperature" can still match "Fever"). Diseases are ranked by how many
   of the patient's symptoms they share and how much of each disease's known
   symptom profile is covered. The top 5 are kept.

2. **Medication safety.** The patient's current medicines are checked against
   the diseases the Clarifier already knows about (checked first) and the
   predicted diseases:
   - **Problems with current medicines:** a medicine recorded as contraindicated
     in one of the diseases.
   - **Medicines clashing with each other:** two of the patient's medicines with
     a recorded interaction.
   - **Treatment drugs for each disease**, grouped from safest to riskiest:
     already taken by the patient, no known clash, clash of minor or unknown
     severity, moderate clash, major clash.

   PrimeKG records *that* two drugs interact but not *how badly*, so severity
   (major / moderate / minor / unknown) comes from the DDInter database.

> **Not a prescribing tool.** Treatment drugs are options from the knowledge
> graph for a doctor to consider, not prescriptions. "No known clash" means no
> interaction is recorded in our data, not that a drug is guaranteed safe.

## Files

**The agent**

| File | What it does |
|---|---|
| `kg_agent.py` | Entry point. `run_kg_agent(clarifier_output)` takes the Clarifier's symptoms, medications and known diseases and returns the disease prediction plus the medication safety result. Loads the graph and the DDInter data once and reuses them. |
| `demo_pipeline.py` | Disease prediction: `predict_disease()` ranks diseases from symptoms, `select_for_fusion()` keeps the top candidates. |
| `kg_real_traversal.py` | Medication safety: `check_medication_safety()` and `load_ddinter_severity()`. Also contains the older `graph_traversal_agent()`, which used the small `primekg_subset.csv` graph. |
| `build_full_graph_v2.py` | Builds the graph from `primekg_relation_filtered.csv` (with ID-type and name-casing fixes), saves it as `relation_filtered_graph_v2.pkl`, and provides `load_graph()`. |
| `symptom_synonyms_full.py` | Generated symptom-synonym dictionary (about 34,000 entries) from HPO. Included in the repo so you don't need to rebuild it. |
| `run_test_case.py` | Lists the cases in `tests/`, lets you pick one by number, runs the agent and prints a readable report. |

**Data builders**

| File | What it does |
|---|---|
| `fetch_and_filter_primekg.py` | Downloads PrimeKG `kg.csv` if it isn't there yet, and makes the small keyword-filtered `primekg_subset.csv` used by the older agent. |
| `build_relation_filtered.py` | Makes `primekg_relation_filtered.csv` from `kg.csv`, keeping 8 clinical relation types. |
| `hpo_synonym_extraction.py` | Rebuilds `symptom_synonyms_full.py` from the graph and HPO's `hp.json`. |

**Earlier analysis and checks**

| File | What it does |
|---|---|
| `build_full_graph.py` | First version of the full-graph build (no data fixes); used by the fever/dyspnea scripts. |
| `data_quality_checks.py` | Checks the filtered CSV for ID-type problems, duplicate rows and names that differ only in casing. |
| `fever_dyspnea_intersection.py`, `fever_dyspnea_coverage.py`, `fever_dyspnea_specificity.py`, `query_fever_dyspnea_full.py` | Experiments on diseases linked to both fever and dyspnea, which led to the coverage-based ranking. |
| `verify_pickle_load.py` | Checks that the saved graph loads with the expected node/edge counts. |
| `visualize_real_graph.py`, `kg_real_visual.png` | Picture of the myocardial infarction + sildenafil example from the older agent. |

**Tests**

- `tests/`: Clarifier input cases (JSON) used by `run_test_case.py`.
- `test_outputs/`: saved outputs from earlier test runs.

## Getting the data

The data files are **not** in this repository (they are large, and DDInter's
licence doesn't allow redistributing it here). Put them in this folder as follows.

1. **PrimeKG `kg.csv`** (about 940 MB), from Harvard Dataverse:
   <https://dataverse.harvard.edu/api/access/datafile/6180620>
   (dataset page: <https://doi.org/10.7910/DVN/IXA7BM>).
   Save it here as `kg.csv`, or run `python fetch_and_filter_primekg.py`, which
   downloads it if it is missing and also makes `primekg_subset.csv`.

2. **Filtered CSV:** `python build_relation_filtered.py`
   → `primekg_relation_filtered.csv` (about 380 MB).

3. **Graph:** `python build_full_graph_v2.py`
   → `relation_filtered_graph_v2.pkl` (about 135 MB). This takes a while and
   needs plenty of RAM.

4. **DDInter severity data:** download the 8 files
   `ddinter_downloads_code_A.csv`, `_B`, `_D`, `_H`, `_L`, `_P`, `_R`, `_V` from
   <https://ddinter.scbdd.com> (Download page) into a `data_sources/` folder.
   DDInter is licensed **CC BY-NC-SA 4.0** (non-commercial, share-alike), so it
   is not included here.

5. *(Optional, only to rebuild the synonyms)* Download HPO's `hp.json` from
   <https://hpo.jax.org> into this folder and run `python hpo_synonym_extraction.py`.

Expected layout:

```
code/
  kg.csv                          (only needed for steps 2-3)
  primekg_relation_filtered.csv
  relation_filtered_graph_v2.pkl
  data_sources/ddinter_downloads_code_*.csv
```

## How to run

```
pip install -r requirements.txt
python run_test_case.py
```

Choose a test case by number. The first run loads the graph and DDInter data
(about 10 seconds and about 1.5 GB of RAM); later calls in the same process take
about 0.1 seconds.

To use the agent from your own code:

```python
from kg_agent import run_kg_agent

result = run_kg_agent({
    "symptoms": ["fever", "cough", "headache", "fatigue"],
    "medications": ["Warfarin", "Ibuprofen"],
    "diseases": [],          # known diseases from the Clarifier, optional
})
```

## Citations

- **PrimeKG:** Chandak P, Huang K, Zitnik M. Building a knowledge graph to
  enable precision medicine. *Scientific Data* 10, 67 (2023).
  <https://doi.org/10.1038/s41597-023-01960-3>
- **DDInter:** Xiong G, Yang Z, Yi J, et al. DDInter: an online drug-drug
  interaction database towards improving clinical decision-making and patient
  safety. *Nucleic Acids Research* 50(D1), D1200-D1207 (2022).
  <https://doi.org/10.1093/nar/gkab880>
- **Human Phenotype Ontology:** Köhler S, Gargano M, Matentzoglu N, et al.
  The Human Phenotype Ontology in 2021. *Nucleic Acids Research* 49(D1),
  D1207-D1217 (2021). <https://doi.org/10.1093/nar/gkaa1043>
