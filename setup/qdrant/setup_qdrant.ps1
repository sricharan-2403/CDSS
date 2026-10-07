<#
One-time Qdrant setup for the CDSS pipeline (Vector RAG, Sri Charan's frozen collections), V2 method.
Windows 10/11, PowerShell 5.1+, no Docker. Safe to run again: finished steps are skipped.

    powershell -ExecutionPolicy Bypass -File setup\qdrant\setup_qdrant.ps1 `
        -HandoffDir "<folder with Qdrant_Symptom_RAG_Handoff_v2.tar.gz>" [-QdrantDir "<where qdrant.exe + storage go>"]

The V2 handoff is a packed Qdrant STORAGE folder, NOT a snapshot. Never upload it to the snapshot endpoint.
Steps:
 1. qdrant.exe v1.19.2 (official GitHub release, SHA-256 checked); an older qdrant.exe is renamed, not deleted
 2. check the archive SHA-256 (stops on mismatch: download it again)
 3. extract it into a NEW EMPTY folder storage_v2 (collections\ directly inside); an existing storage_v2 is
    never deleted or overwritten
 4. write config.yaml (storage_v2, 127.0.0.1 only, telemetry off) and copy start_qdrant.ps1
 5. start Qdrant in its own window (start_qdrant.ps1) if it is not running yet
 6. verify version, names, exact point counts, 768 dims, Cosine, status green; print qdrant.exe RAM
Never changes collection settings, never deletes or re-ingests data.
#>
param(
    [Parameter(Mandatory = $true)][string]$HandoffDir,
    [string]$QdrantDir = "",   # default set below ($PSScriptRoot is empty in param defaults on PS 5.1)
    [string]$QdrantUrl = "http://127.0.0.1:6333"
)
if (-not $QdrantDir) { $QdrantDir = Join-Path $PSScriptRoot "..\..\..\qdrant" }

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Version = "1.19.2"
$ZipName = "qdrant-x86_64-pc-windows-msvc.zip"
$ZipUrl = "https://github.com/qdrant/qdrant/releases/download/v$Version/$ZipName"
$ZipSha256 = "7D86596F16C6E85D45A50312F5E16CCB51E059B62E3D7308110CB65B4C799A4D"  # digest on the GitHub release
$ArchiveName = "Qdrant_Symptom_RAG_Handoff_v2.tar.gz"
$ArchiveSha256 = "4FB65B2EBC4D71D505956DC5EB7DB613B3C9BB617F4F92AC7CEAA4D433E5A5FD"  # from Sri Charan's guide
$StorageName = "storage_v2"
$Expected = [ordered]@{ "symptom_chunks" = 2690; "literature_chunks" = 201134 }
$VectorSize = 768
$Distance = "Cosine"

function Step($text) { Write-Host ""; Write-Host "=== $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "STOPPED: $text" -ForegroundColor Red; exit 1 }

function Get-QdrantRam {
    $p = Get-Process -Name qdrant -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $p) { return "qdrant.exe not running" }
    $p.Refresh()
    return ("working set {0:N0} MB, private {1:N0} MB" -f ($p.WorkingSet64 / 1MB), ($p.PrivateMemorySize64 / 1MB))
}

function Test-Qdrant {
    try { return Invoke-RestMethod -Uri "$QdrantUrl/" -TimeoutSec 3 } catch { return $null }
}

function Get-Collection($name) {
    try { return (Invoke-RestMethod -Uri "$QdrantUrl/collections/$name" -TimeoutSec 30).result }
    catch { return $null }
}

function Get-ExactCount($name) {
    return (Invoke-RestMethod -Method Post -Uri "$QdrantUrl/collections/$name/points/count" `
        -Body '{"exact": true}' -ContentType "application/json" -TimeoutSec 300).result.count
}

$QdrantDir = [IO.Path]::GetFullPath($QdrantDir)
$HandoffDir = [IO.Path]::GetFullPath($HandoffDir)
$Storage = Join-Path $QdrantDir $StorageName
Write-Host "Qdrant folder:  $QdrantDir"
Write-Host "Handoff folder: $HandoffDir"
Write-Host "Qdrant URL:     $QdrantUrl"

# ---------------------------------------------------------------- 1. qdrant.exe
Step "1. qdrant.exe v$Version"
New-Item -ItemType Directory -Force $QdrantDir | Out-Null
$exe = Join-Path $QdrantDir "qdrant.exe"
if (Test-Path $exe) {
    $have = (& $exe --version) -replace '^qdrant\s+', ''
    if ($have.Trim() -ne $Version) {
        $old = Join-Path $QdrantDir "qdrant_v$($have.Trim()).exe"
        Write-Host "qdrant.exe is version $have - renaming it to $(Split-Path $old -Leaf)"
        Move-Item $exe $old
    }
}
if (Test-Path $exe) {
    Write-Host "qdrant.exe v$Version already present"
} else {
    $zip = Join-Path $QdrantDir $ZipName
    Write-Host "Downloading $ZipUrl"
    Invoke-WebRequest -Uri $ZipUrl -OutFile $zip -UseBasicParsing
    $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash
    if ($hash -ne $ZipSha256) { Remove-Item $zip; Fail "qdrant zip SHA-256 $hash does not match $ZipSha256 (zip deleted)" }
    Write-Host "zip SHA-256 OK"
    Expand-Archive -Path $zip -DestinationPath $QdrantDir -Force
    Remove-Item $zip
    if (-not (Test-Path $exe)) { Fail "qdrant.exe not found after unpacking $ZipName" }
    Write-Host "qdrant.exe $(& $exe --version) unpacked"
}

# ---------------------------------------------------------------- 2 + 3. archive check and extract
Step "2. V2 storage archive"
$symptomDir = Join-Path $Storage "collections\symptom_chunks"
if (Test-Path $symptomDir) {
    Write-Host "$StorageName already extracted ($symptomDir exists), skipped"
} else {
    $archive = Join-Path $HandoffDir $ArchiveName
    if (-not (Test-Path $archive)) { Fail "$ArchiveName not found in $HandoffDir" }
    Write-Host "hashing $ArchiveName (about 720 MB) ..."
    $got = (Get-FileHash -Algorithm SHA256 $archive).Hash
    if ($got -ne $ArchiveSha256) { Fail "archive SHA-256 MISMATCH (got $got, expected $ArchiveSha256) - download it again" }
    Write-Host "  SHA-256 matches"

    Step "3. extract into a new empty $StorageName"
    if ((Test-Path $Storage) -and (Get-ChildItem -Force $Storage | Select-Object -First 1)) {
        Fail "$Storage exists and is not empty but has no collections\symptom_chunks - not touching it. Move it away yourself."
    }
    New-Item -ItemType Directory -Force $Storage | Out-Null
    Write-Host "extracting (about 1.9 GB) ..."
    & tar.exe -xzf $archive -C $Storage
    if ($LASTEXITCODE -ne 0) { Fail "tar.exe could not extract $archive (storage left as is for diagnosis)" }
    foreach ($name in $Expected.Keys) {
        if (-not (Test-Path (Join-Path $Storage "collections\$name"))) {
            Fail "collections\$name is not directly inside $Storage after extracting (wrong layout?)"
        }
    }
    Write-Host "extracted: collections\ is directly inside $Storage"
}

# ---------------------------------------------------------------- 4. config + start script
Step "4. config.yaml and start_qdrant.ps1"
$config = @"
# Written by setup_qdrant.ps1 (V2 storage handoff). Paths are relative to this folder.
storage:
  storage_path: ./$StorageName
  snapshots_path: ./snapshots_v2
  temp_path: ./tmp
service:
  host: 127.0.0.1
  http_port: 6333
  grpc_port: 6334
telemetry_disabled: true
"@
$configPath = Join-Path $QdrantDir "config.yaml"
if ((Test-Path $configPath) -and -not (Select-String -Path $configPath -SimpleMatch "./$StorageName" -Quiet)) {
    Copy-Item $configPath (Join-Path $QdrantDir "config_old.yaml") -Force
    Write-Host "old config.yaml kept as config_old.yaml"
}
Set-Content -Path $configPath -Value $config -Encoding ASCII
Copy-Item (Join-Path $PSScriptRoot "start_qdrant.ps1") (Join-Path $QdrantDir "start_qdrant.ps1") -Force
Write-Host "config.yaml written, start_qdrant.ps1 copied"

# ---------------------------------------------------------------- 5. start Qdrant
Step "5. Qdrant server"
$info = Test-Qdrant
if ($null -eq $info) {
    Write-Host "starting Qdrant in its own window (keep it open)"
    Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File",
        "`"$(Join-Path $QdrantDir 'start_qdrant.ps1')`"")
    for ($i = 0; $i -lt 120 -and $null -eq $info; $i++) { Start-Sleep -Seconds 1; $info = Test-Qdrant }
    if ($null -eq $info) { Fail "Qdrant did not answer at $QdrantUrl within 120 s - read the Qdrant window; do not delete $StorageName" }
}
if ($info.version -ne $Version) {
    Fail "Qdrant at $QdrantUrl is version $($info.version), expected $Version (an old Qdrant still running? close its window)"
}
Write-Host "Qdrant $($info.version) is running at $QdrantUrl"

# ---------------------------------------------------------------- 6. verify
Step "6. verify"
$ok = $true
foreach ($name in $Expected.Keys) {
    $c = $null
    for ($i = 0; $i -lt 60 -and ($null -eq $c -or $c.status -ne "green"); $i++) {
        if ($i -gt 0) { Start-Sleep -Seconds 2 }
        $c = Get-Collection $name
    }
    if ($null -eq $c) { Write-Host "  [FAIL] $name missing"; $ok = $false; continue }
    $v = $c.config.params.vectors
    $count = Get-ExactCount $name
    $checks = [ordered]@{
        "exact count $count = $($Expected[$name])" = ($count -eq $Expected[$name])
        "size $($v.size) = $VectorSize" = ($v.size -eq $VectorSize)
        "distance $($v.distance) = $Distance" = ($v.distance -eq $Distance)
        "status $($c.status) = green" = ($c.status -eq "green")
    }
    foreach ($k in $checks.Keys) {
        $mark = "OK  "; if (-not $checks[$k]) { $mark = "FAIL"; $ok = $false }
        Write-Host "  [$mark] $name $k"
    }
}
Write-Host "qdrant.exe RAM: $(Get-QdrantRam)"
if (-not $ok) { Fail "verification failed (see FAIL lines above). Do not delete $StorageName; send the Qdrant window text to Rakshitha." }
Write-Host ""
Write-Host "Qdrant setup complete. Before each pipeline run start it with:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$(Join-Path $QdrantDir 'start_qdrant.ps1')`""
