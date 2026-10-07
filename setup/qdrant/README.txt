Qdrant setup for the CDSS pipeline (Vector RAG = Sri Charan's frozen Symptom RAG Analyzer)

Method: V2 STORAGE handoff (since 2026-10-07). The old .snapshot files are damaged and are NOT used.
The V2 archive is a packed Qdrant storage folder, not a snapshot: never upload it to
/collections/<name>/snapshots/upload.

Needed once per laptop (Windows 10/11, no Docker, about 2.7 GB free disk for archive + storage):
  1. Download from the team Google Drive folder (same folder as cdss_kg_data.zip):
       Qdrant_Symptom_RAG_Handoff_v2.tar.gz   (751,579,030 bytes)
       CHECKSUMS_POINT_MIGRATION.txt, POINT_LEVEL_MIGRATION_VALIDATION.txt
     Put them in one folder, e.g. C:\Qdrant_handoff_v2. Do NOT commit them to git.
     Archive SHA-256: 4FB65B2EBC4D71D505956DC5EB7DB613B3C9BB617F4F92AC7CEAA4D433E5A5FD
  2. From the repo folder run:
       powershell -ExecutionPolicy Bypass -File setup\qdrant\setup_qdrant.ps1 -HandoffDir "C:\Qdrant_handoff_v2"
     Optional: -QdrantDir "<folder for qdrant.exe and its storage>"
     (default: a folder named qdrant next to the repo folder).
     It downloads qdrant.exe v1.19.2 (official GitHub release, SHA-256 checked), checks the archive
     SHA-256, extracts it into a NEW EMPTY folder storage_v2 (collections\ directly inside), writes
     config.yaml (127.0.0.1 only, telemetry off), starts Qdrant and verifies:
       symptom_chunks 2,690 points, literature_chunks 201,134 points, 768 dims, Cosine, green.
     Safe to run again. It never deletes storage_v2 and never changes the collections.
     Qdrant uses about 0.75 GB RAM.

Vector RAG code: in the team repo it is already in agents\Symptom_RAG_Frozen (pinned 7a80dfc).
Make it importable in the venv: create the file
    .venv\Lib\site-packages\symptom_rag_analyzer_src.pth
containing exactly this one (relative) line:
    team repo:              ..\..\..\agents\Symptom_RAG_Frozen\src
    (Rakshitha's local PC:  ..\..\..\Symptom_RAG_Frozen\src)
OPENROUTER_API_KEY must be in .env (the RAG Answerer uses OpenRouter).

Before every pipeline run:
  powershell -ExecutionPolicy Bypass -File "<QdrantDir>\start_qdrant.ps1"
  Keep that window open while the pipeline runs (closing it stops Qdrant).
  The pipeline checks Qdrant at QDRANT_URL (.env, default http://localhost:6333) and stops with a
  clear message if it is not running or the collections are wrong.

If something fails: do NOT delete or change storage_v2. Copy the text of the Qdrant window, the
command you ran, the qdrant.exe version and the error, and send them to Rakshitha.
