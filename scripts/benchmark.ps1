[CmdletBinding()]
param(
    [ValidateSet("fixture-adapted", "transformers", "llama-http")]
    [string]$Engine = "fixture-adapted",
    [string]$Output = "outputs/benchmark-report.json",
    [string]$ModelId = "Qwen/Qwen3-1.7B",
    [string]$ModelRevision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    [string]$ModelVersion = "",
    [string]$PromptVersion = "optimized-v2",
    [string]$Adapter = "artifacts/qwen3-1.7b-triage-lora-v5/final-adapter",
    [string]$BaseUrl = "http://127.0.0.1:8081",
    [string]$Quantization = "",
    [int]$ContextWindowTokens = 2048,
    [ValidateSet("risk-precedence-v2", "risk-precedence-v3")]
    [string]$SafetyPolicyVersion = "risk-precedence-v2",
    [int[]]$Concurrency = @(1, 2, 4),
    [int]$Requests = 30
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$modelforge = Join-Path $projectRoot ".venv\Scripts\modelforge.exe"
if ($Engine -in @("transformers", "llama-http") -and [string]::IsNullOrWhiteSpace($ModelVersion)) {
    throw "實測 engine 必須用 -ModelVersion 指定不可混淆的部署版本"
}
if ($Engine -eq "llama-http" -and [string]::IsNullOrWhiteSpace($Quantization)) {
    throw "llama-http 必須用 -Quantization 指定實際量化格式"
}
$arguments = @("benchmark", "--engine", $Engine, "--concurrency")
$arguments += $Concurrency
$arguments += @(
    "--requests", $Requests,
    "--output", $Output,
    "--model-id", $ModelId,
    "--model-revision", $ModelRevision,
    "--prompt-version", $PromptVersion,
    "--max-new-tokens", "192",
    "--context-window-tokens", $ContextWindowTokens,
    "--safety-policy-version", $SafetyPolicyVersion
)
if (-not [string]::IsNullOrWhiteSpace($ModelVersion)) {
    $arguments += @("--model-version", $ModelVersion)
}
if ($Engine -eq "transformers") {
    $arguments += @("--adapter", $Adapter, "--load-in-4bit")
}
elseif ($Engine -eq "llama-http") {
    $arguments += @("--base-url", $BaseUrl, "--quantization", $Quantization)
}

Push-Location $projectRoot
try {
    & $modelforge @arguments
    if ($LASTEXITCODE -ne 0) { throw "benchmark 失敗，exit code: $LASTEXITCODE" }
}
finally {
    Pop-Location
}
