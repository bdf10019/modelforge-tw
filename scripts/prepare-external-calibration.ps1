[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot "configs\external-calibration.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$rawDirectory = Join-Path $projectRoot "data\external\raw"
$zipPath = Join-Path $rawDirectory "ai4i-2020.zip"
$csvPath = Join-Path $rawDirectory $config.source.csv_file
$modelforge = Join-Path $projectRoot ".venv\Scripts\modelforge.exe"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (-not (Test-Path -LiteralPath $modelforge)) {
    throw "請先執行 scripts\bootstrap.ps1"
}

New-Item -ItemType Directory -Force -Path $rawDirectory | Out-Null
if (-not (Test-Path -LiteralPath $zipPath) -or
    (Get-Sha256 $zipPath) -ne $config.source.zip_sha256) {
    Invoke-WebRequest -Uri $config.source.download_url -OutFile $zipPath
}
if ((Get-Sha256 $zipPath) -ne $config.source.zip_sha256) {
    throw "AI4I ZIP SHA-256 不符"
}

if (-not (Test-Path -LiteralPath $csvPath) -or
    (Get-Sha256 $csvPath) -ne $config.source.csv_sha256) {
    Expand-Archive -LiteralPath $zipPath -DestinationPath $rawDirectory -Force
}
if ((Get-Sha256 $csvPath) -ne $config.source.csv_sha256) {
    throw "AI4I CSV SHA-256 不符"
}

& $modelforge prepare-external-calibration --config $configPath --source $csvPath
if ($LASTEXITCODE -ne 0) { throw "外部校準集產生失敗" }
