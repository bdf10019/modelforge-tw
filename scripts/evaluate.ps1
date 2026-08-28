[CmdletBinding()]
param([switch]$Full)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$model = "Qwen/Qwen3-1.7B"
$revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
$adapter = "artifacts/qwen3-1.7b-triage-lora-v5/final-adapter"
$splits = if ($Full) { @("test", "challenge") } else { @("challenge") }
$modelforge = Join-Path $projectRoot ".venv\Scripts\modelforge.exe"
if (-not (Test-Path -LiteralPath $modelforge)) { throw "請先執行 scripts\bootstrap.ps1" }

function Invoke-Evaluation([string]$Label, [string[]]$ExtraArguments) {
    & $modelforge evaluate --model-id $model --model-revision $revision --load-in-4bit --max-new-tokens 192 --splits @splits --output "outputs/eval-$Label.json" @ExtraArguments
    return $LASTEXITCODE
}

Push-Location $projectRoot
try {
    $baseExit = Invoke-Evaluation "A-base" @("--engine", "transformers", "--prompt-version", "minimal-v1")
    $promptExit = Invoke-Evaluation "B-prompt-v2" @("--engine", "transformers", "--prompt-version", "optimized-v2")
    $peftExit = Invoke-Evaluation "C-peft-v5" @("--engine", "transformers", "--model-version", "qwen3-1.7b-qlora-v5", "--prompt-version", "optimized-v2", "--adapter", $adapter)
    if ($peftExit -ne 0) { throw "PEFT 候選未通過發布門檻；請檢查 outputs/eval-C-peft-v5.json" }
    Write-Host "A exit=$baseExit; B exit=$promptExit; C exit=$peftExit（A/B 預期可不通過）"
}
finally {
    Pop-Location
}
