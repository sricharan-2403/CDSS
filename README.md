# CDSS - Clinical Decision Support System (5 agents, LangGraph)

## What this is
You type a short patient description (symptoms, history, medicines). Five agents work on it:

1. **Clarifier** (Varshit): turns the text into a clean list of symptoms, diseases, medicines and tests (Groq).
2. **Vector RAG** (Sri Charan): searches symptom and literature collections in Qdrant and suggests diagnoses (OpenRouter).
3. **Knowledge Graph** (Rakshitha): suggests diagnoses from a medical graph (PrimeKG) and checks medication safety
   (contraindications, drug clashes from DDInter).
4. **Evidence Scanner** (Vaishnavi): looks up web evidence (PubMed, Europe PMC) for the top diagnoses (Groq).
5. **Fusion + Optimizer** (Charanya): combines everything into one report with confidence scores; if the
   confidence is low, it asks Vector RAG to search again (at most 2 extra rounds).

The result is a research prototype for a college project. **It is not medical advice.**

The code of each agent is in `agents\` as a plain copy at a fixed commit (see `agents\PINNED_COMMITS.txt`).
The glue code is `cdss_pipeline\`.

## What your laptop needs
- Windows 10/11, internet connection.
- **Python 3.13.6** (same version as ours; other versions may install different packages).
- Free RAM: **about 4-4.5 GB** for the full system; **about 2.5 GB** with `--disable rag` (no Qdrant).
- About **10 GB** free disk (Python packages, Qdrant data, KG data, model).
- Your **own** Groq API key (free, console.groq.com) and OpenRouter API key (openrouter.ai).

## Downloads (team Google Drive, one folder)
<https://drive.google.com/drive/folders/16ztB2D_6Uyk1rEKDugCZDqUOIoQV9Ycw?usp=drive_link>

Everything below is in that one folder:
- `cdss_kg_data.zip` (about 26 MB): Knowledge Graph data (graph + DDInter tables). Team use only
  (DDInter licence: CC BY-NC-SA 4.0). Its SHA-256 is in `setup\kg_data\cdss_kg_data_SHA256.txt`.
- `Qdrant_Symptom_RAG_Handoff_v2.tar.gz` (about 717 MB) + `CHECKSUMS_POINT_MIGRATION.txt` +
  `POINT_LEVEL_MIGRATION_VALIDATION.txt`: the Vector RAG database.

## Setup, step by step
Open PowerShell in this repo folder.

1. **Virtual environment**
   ```
   py -3.13 -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install -r requirements_lock.txt
   ```
2. **Vector RAG import line.** Create the file `.venv\Lib\site-packages\symptom_rag_analyzer_src.pth`
   containing exactly this one line:
   ```
   ..\..\..\agents\Symptom_RAG_Frozen\src
   ```
3. **KG data.** Download `cdss_kg_data.zip`, check it
   (`Get-FileHash cdss_kg_data.zip` must equal the `cdss_kg_data.zip` line in
   `setup\kg_data\cdss_kg_data_SHA256.txt`), then extract it **into** `agents\kg_agent\`.
   Afterwards `agents\kg_agent\relation_filtered_graph_v2.pkl` and `agents\kg_agent\data_sources\` must exist.
4. **Embedding model** (one download, about 420 MB):
   ```
   .venv\Scripts\python.exe -B -m cdss_pipeline.check_setup --download-model
   ```
5. **Qdrant (Vector RAG database).** Download the 3 Qdrant files into one folder, e.g. `C:\Qdrant_handoff_v2`, then:
   ```
   powershell -ExecutionPolicy Bypass -File setup\qdrant\setup_qdrant.ps1 -HandoffDir "C:\Qdrant_handoff_v2"
   ```
   It installs Qdrant v1.19.2, checks the archive, extracts it and verifies 2,690 + 201,134 points.
   Details: `setup\qdrant\README.txt`. You can skip this step and use `--disable rag` (see below).
6. **Keys.** Copy `.env.example` to `.env` and fill in **your own** `GROQ_API_KEY` and `OPENROUTER_API_KEY`.
   Keep the model names as they are in `.env.example`.
7. **Check everything** (light, loads nothing big):
   ```
   .venv\Scripts\python.exe -B -m cdss_pipeline.check_setup
   ```
   It must end with `Setup check: OK`.

## Run one case
Start Qdrant first (keep its window open):
`powershell -ExecutionPolicy Bypass -File <qdrant folder>\start_qdrant.ps1`
```
.venv\Scripts\python.exe -B -m cdss_pipeline.run_pipeline
.venv\Scripts\python.exe -B -m cdss_pipeline.run_pipeline --text "56 year old man with chest pain ..."
```
- Default output is **short**: diagnosis, warnings, medication safety, confidence.
- `--verbose` prints everything; `--json` prints the raw result.
- Startup takes about 30 s (the graph and model load once). One case takes 1-3 minutes.

**Without Qdrant** (or with little RAM): add `--disable rag`. Then only the Knowledge Graph suggests diagnoses,
so Tests 2-4 still make sense but Test 1 (Vector RAG) does not.

## Batch testing and scoring
See [TESTING.md](TESTING.md). In short:
```
.venv\Scripts\python.exe -B -m cdss_pipeline.run_batch --input my_cases.jsonl --out results\<run_name>
.venv\Scripts\python.exe -B -m cdss_pipeline.score --run results\<run_name>
```
A batch can be stopped and started again with the same command; finished cases are skipped.

## What to send back to Rakshitha
The **whole** `results\<run_name>` folder (zip it). It contains every case, the run settings
(`run_manifest.json`) and the scores. Also say which laptop / network you used and anything odd you saw.

## Troubleshooting
- **Groq says 403 "Access denied"**: your network is blocked by Groq. Use the Cloudflare AI Gateway:
  in `.env` set `GROQ_VIA_GATEWAY=true`, `CF_ACCOUNT_ID` and `CF_AI_GATEWAY_ID` (ask Rakshitha), or try another
  network / no VPN.
- **"Qdrant is not reachable" / wrong counts**: start `start_qdrant.ps1` and keep the window open. Never delete
  or change the Qdrant storage folder; send the window text to Rakshitha.
- **Low RAM / laptop freezes**: close browsers and other apps, or use `--disable rag`.
- **Quota limits** (Groq: 1,000 requests/day; free OpenRouter models: daily limits; "429"): wait and run the same
  command again later - finished cases are not repeated.
- **`check_setup` shows a problem**: it prints how to fix it.

## Rules
- **Never commit `.env`** or share your keys.
- **Do not change the agent code** in `agents\`. Report problems to the agent's owner instead.
- Report results honestly: small samples (for example 20 cases) give wide confidence intervals; say so.
- Do not commit the KG data, Qdrant files or `results\` (they are in `.gitignore`).
