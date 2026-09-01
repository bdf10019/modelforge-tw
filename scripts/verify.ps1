[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$checks = @(
    @("-m", "pip", "check"),
    @("-m", "modelforge_tw.cli", "generate-data"),
    @("-m", "modelforge_tw.cli", "audit-data"),
    @("-m", "ruff", "format", "--check", "."),
    @("-m", "ruff", "check", "."),
    @("-m", "mypy", "src"),
    @("-m", "pytest", "-q")
)

Push-Location $projectRoot
try {
    foreach ($arguments in $checks) {
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "驗證失敗：python $($arguments -join ' ')" }
    }

    $parseErrors = @()
    Get-ChildItem -LiteralPath $PSScriptRoot -Filter "*.ps1" | ForEach-Object {
        $tokens = $null
        $errors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile(
            $_.FullName,
            [ref]$tokens,
            [ref]$errors
        )
        $parseErrors += $errors
    }
    if ($parseErrors.Count -gt 0) {
        throw "PowerShell 語法驗證失敗：$($parseErrors[0].Message)"
    }

    & $python -m pip wheel . --no-deps --wheel-dir build\verify-wheel
    if ($LASTEXITCODE -ne 0) { throw "wheel 建置失敗" }
    $wheel = Get-ChildItem -LiteralPath (Join-Path $projectRoot "build\verify-wheel") `
        -Filter "modelforge_tw-*.whl" |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $wheel) { throw "找不到驗證用 wheel" }
    & $python -c @'
import sys
import zipfile

required = {
    "modelforge_tw/web/index.html",
    "modelforge_tw/web/styles.css",
    "modelforge_tw/web/app.js",
    "modelforge_tw/web/favicon.svg",
}
with zipfile.ZipFile(sys.argv[1]) as archive:
    missing = sorted(required - set(archive.namelist()))
if missing:
    raise SystemExit(f"wheel 缺少 Web 資源：{missing}")
'@ $wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "wheel 內容驗證失敗" }
    $temporaryBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $temporaryInstall = Join-Path $temporaryBase ("modelforge-wheel-" + [guid]::NewGuid())
    [void](New-Item -ItemType Directory -Path $temporaryInstall)
    try {
        & $python -m pip install --no-deps --target $temporaryInstall $wheel.FullName
        if ($LASTEXITCODE -ne 0) { throw "wheel 暫存安裝失敗" }
        & $python -c @'
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import modelforge_tw
from modelforge_tw.service import create_app
from modelforge_tw.settings import Settings

module_path = Path(modelforge_tw.__file__).resolve()
if not module_path.is_relative_to(target):
    raise SystemExit(f"載入的不是暫存 wheel：{module_path}")
app = create_app(settings=Settings())
paths = {route.path for route in app.routes}
required_paths = {"/", "/healthz", "/readyz", "/v1/triage", "/v1/chat/completions", "/metrics"}
missing = sorted(required_paths - paths)
if missing:
    raise SystemExit(f"安裝後 app 缺少路由：{missing}")
'@ $temporaryInstall
        if ($LASTEXITCODE -ne 0) { throw "wheel 安裝後 smoke test 失敗" }
    }
    finally {
        $resolvedTemporaryInstall = [System.IO.Path]::GetFullPath($temporaryInstall)
        if (-not $resolvedTemporaryInstall.StartsWith($temporaryBase)) {
            throw "拒絕清理暫存根目錄以外的路徑"
        }
        Remove-Item -LiteralPath $resolvedTemporaryInstall -Recurse -Force
    }
    & (Join-Path $PSScriptRoot "privacy-check.ps1")
    if ($LASTEXITCODE -ne 0) { throw "隱私檢查失敗" }
    & (Join-Path $PSScriptRoot "verify-local-evidence.ps1")
    if ($LASTEXITCODE -ne 0) { throw "本機證據快照驗證失敗" }
}
finally {
    Pop-Location
}
