# Start the local Qdrant server for the CDSS pipeline (Vector RAG).
# setup_qdrant.ps1 copies this file next to qdrant.exe and config.yaml; run it from there:
#     powershell -ExecutionPolicy Bypass -File <qdrant folder>\start_qdrant.ps1
# Keep this window open while the pipeline runs (closing it stops Qdrant).
# Qdrant listens on 127.0.0.1 only (see config.yaml); storage paths are relative to this folder.

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$exe = Join-Path $here "qdrant.exe"
$config = Join-Path $here "config.yaml"
if (-not (Test-Path $exe)) { throw "qdrant.exe not found in $here - run setup_qdrant.ps1 first" }
if (-not (Test-Path $config)) { throw "config.yaml not found in $here - run setup_qdrant.ps1 first" }

Set-Location $here   # relative storage paths in config.yaml resolve from here
$Host.UI.RawUI.WindowTitle = "Qdrant (CDSS) - keep this window open"
Write-Host "Starting Qdrant from $here (http://127.0.0.1:6333). Keep this window open."
& $exe --config-path $config
