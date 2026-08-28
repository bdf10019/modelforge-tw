[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$checks = @(
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
    & (Join-Path $PSScriptRoot "privacy-check.ps1")
}
finally {
    Pop-Location
}
