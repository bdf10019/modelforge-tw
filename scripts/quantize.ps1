[CmdletBinding()]
param(
    [string]$MergedModel = "artifacts/qwen3-1.7b-triage-merged",
    [string]$OutputDirectory = "artifacts/gguf"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$converter = Join-Path $projectRoot "tools\llama.cpp-src\convert_hf_to_gguf.py"
$binaryRoot = Join-Path $projectRoot "tools\llama.cpp-bin"
$quantizer = Get-ChildItem -LiteralPath $binaryRoot -Filter "llama-quantize.exe" -Recurse | Select-Object -First 1
$mergedPath = Join-Path $projectRoot $MergedModel
$outputRoot = Join-Path $projectRoot $OutputDirectory
$f16 = Join-Path $outputRoot "modelforge-tw-f16.gguf"
$q8 = Join-Path $outputRoot "modelforge-tw-q8_0.gguf"
$q4 = Join-Path $outputRoot "modelforge-tw-q4_k_m.gguf"

if (-not (Test-Path -LiteralPath $converter)) { throw "請先執行 scripts\prepare-llama.ps1" }
if (-not $quantizer) { throw "找不到 llama-quantize.exe" }
if (-not (Test-Path -LiteralPath $mergedPath)) { throw "找不到 merged model：$mergedPath" }
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

& $python $converter $mergedPath --outfile $f16 --outtype f16
if ($LASTEXITCODE -ne 0) { throw "HF → GGUF 轉換失敗" }

# The official Windows quantizer currently decodes non-ASCII argv paths through
# the active code page. Keep the canonical artifacts in the project directory,
# but use an ASCII-only, same-volume hard-link bridge while invoking the binary.
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $projectRoot)
$bridgeRoot = Join-Path $workspaceRoot ("modelforge-gguf-" + [guid]::NewGuid().ToString("N"))
$bridgeF16 = Join-Path $bridgeRoot "model-f16.gguf"
$bridgeQ8 = Join-Path $bridgeRoot "model-q8_0.gguf"
$bridgeQ4 = Join-Path $bridgeRoot "model-q4_k_m.gguf"

try {
    New-Item -ItemType Directory -Path $bridgeRoot | Out-Null
    New-Item -ItemType HardLink -Path $bridgeF16 -Target $f16 | Out-Null

    & $quantizer.FullName $bridgeF16 $bridgeQ8 Q8_0
    if ($LASTEXITCODE -ne 0) { throw "Q8_0 量化失敗" }
    Move-Item -Force -LiteralPath $bridgeQ8 -Destination $q8

    & $quantizer.FullName $bridgeF16 $bridgeQ4 Q4_K_M
    if ($LASTEXITCODE -ne 0) { throw "Q4_K_M 量化失敗" }
    Move-Item -Force -LiteralPath $bridgeQ4 -Destination $q4
}
finally {
    foreach ($temporaryFile in @($bridgeF16, $bridgeQ8, $bridgeQ4)) {
        if (Test-Path -LiteralPath $temporaryFile) {
            Remove-Item -Force -LiteralPath $temporaryFile
        }
    }
    if (Test-Path -LiteralPath $bridgeRoot) {
        Remove-Item -Force -LiteralPath $bridgeRoot
    }
}

$files = @($f16, $q8, $q4) | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $_
    [ordered]@{ file = $item.Name; bytes = $item.Length; sha256 = $hash.Hash.ToLowerInvariant() }
}
$report = [ordered]@{
    schema_version = "1.0"
    evidence_kind = "measured"
    source = $MergedModel
    llama_cpp_tag = "b10665"
    files = $files
}
$reportPath = Join-Path $outputRoot "quantization-report.json"
$reportTemporary = Join-Path $outputRoot (".quantization-report." + [guid]::NewGuid().ToString("N") + ".tmp")
try {
    $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportTemporary -Encoding utf8
    Move-Item -Force -LiteralPath $reportTemporary -Destination $reportPath
}
finally {
    if (Test-Path -LiteralPath $reportTemporary) {
        Remove-Item -Force -LiteralPath $reportTemporary
    }
}
Write-Host "GGUF 轉換與量化完成：$outputRoot"
