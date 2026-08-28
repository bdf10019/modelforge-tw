[CmdletBinding()]
param([switch]$Smoke)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$config = if ($Smoke) { "configs/training-smoke.json" } else { "configs/training.json" }
if (-not (Test-Path -LiteralPath $venvPython)) { throw "請先執行 scripts\bootstrap.ps1" }

Push-Location $projectRoot
try {
    & $venvPython -m modelforge_tw.cli train --config $config
    if ($LASTEXITCODE -ne 0) { throw "訓練失敗，exit code: $LASTEXITCODE" }
}
finally {
    Pop-Location
}
