[CmdletBinding()]
param(
    [ValidateSet("fixture", "transformers", "llama-http")]
    [string]$Engine = "fixture",
    [int]$Port = 8000,
    [string]$ModelId = "Qwen/Qwen3-1.7B",
    [string]$ModelRevision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    [string]$ModelVersion = "qwen3-1.7b-qlora-v5",
    [string]$Quantization = "q8_0",
    [string]$PromptVersion = "optimized-v2",
    [string]$AdapterPath = "artifacts/qwen3-1.7b-triage-lora-v5/final-adapter",
    [string]$AdapterTrainingReport = "artifacts/qwen3-1.7b-triage-lora-v5/training-report.json",
    [string]$LlamaBaseUrl = "http://127.0.0.1:8081",
    [int]$ContextWindowTokens = 2048,
    [ValidateSet("risk-precedence-v2", "risk-precedence-v3")]
    [string]$SafetyPolicyVersion = "risk-precedence-v2"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) { throw "請先執行 scripts\bootstrap.ps1" }

$env:MODELFORGE_ENGINE = $Engine
$env:MODELFORGE_MODEL_ID = $ModelId
$env:MODELFORGE_MODEL_REVISION = $ModelRevision
$env:MODELFORGE_MODEL_VERSION = $ModelVersion
$env:MODELFORGE_QUANTIZATION = $Quantization
$env:MODELFORGE_PROMPT_VERSION = $PromptVersion
$env:MODELFORGE_CONTEXT_WINDOW_TOKENS = $ContextWindowTokens
$env:MODELFORGE_SAFETY_POLICY_VERSION = $SafetyPolicyVersion
if ($Engine -eq "transformers") {
    $env:MODELFORGE_ADAPTER_PATH = $AdapterPath
    $env:MODELFORGE_ADAPTER_TRAINING_REPORT = $AdapterTrainingReport
    $env:MODELFORGE_LOAD_IN_4BIT = "true"
}
elseif ($Engine -eq "llama-http") {
    $env:MODELFORGE_LLAMA_BASE_URL = $LlamaBaseUrl
}

Push-Location $projectRoot
try {
    & $venvPython -m modelforge_tw.cli serve --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
}
