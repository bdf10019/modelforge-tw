[CmdletBinding()]
param(
    [ValidateSet("core", "ml")]
    [string]$Profile = "ml",
    [switch]$DownloadModel
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "建立 .venv 失敗" }
    }
    & $venvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "更新 Python 建置工具失敗" }

    if ($Profile -eq "ml") {
        & $venvPython -m pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
        if ($LASTEXITCODE -ne 0) { throw "安裝 CUDA PyTorch 失敗" }
        & $venvPython -m pip install -e ".[dev,ml]"
    }
    else {
        & $venvPython -m pip install -e ".[dev]"
    }
    if ($LASTEXITCODE -ne 0) { throw "安裝專案套件失敗" }

    & $venvPython -m modelforge_tw.cli generate-data
    if ($LASTEXITCODE -ne 0) { throw "產生資料失敗" }
    if ($DownloadModel) {
        & $venvPython -m modelforge_tw.cli probe-model --model-id Qwen/Qwen3-1.7B --load-in-4bit
        if ($LASTEXITCODE -ne 0) { throw "模型下載或 4-bit 載入探針失敗" }
    }
}
finally {
    Pop-Location
}
