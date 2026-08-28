[CmdletBinding()]
param(
    [ValidateSet("q8_0", "q4_k_m")]
    [string]$Quantization = "q8_0",
    [int]$Port = 8081,
    [int]$ContextSize = 4096,
    [int]$Parallel = 4
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$server = Get-ChildItem -LiteralPath (Join-Path $projectRoot "tools\llama.cpp-bin") -Filter "llama-server.exe" -Recurse | Select-Object -First 1
if (-not $server) { throw "找不到 llama-server.exe；請先執行 scripts\prepare-llama.ps1" }

$fileName = if ($Quantization -eq "q8_0") { "modelforge-tw-q8_0.gguf" } else { "modelforge-tw-q4_k_m.gguf" }
$model = Join-Path $projectRoot "artifacts\gguf\$fileName"
if (-not (Test-Path -LiteralPath $model)) { throw "找不到 GGUF：$model；請先執行 scripts\quantize.ps1" }

# The official Windows server currently decodes non-ASCII argv paths through
# the active code page. Keep the model in the project and expose a temporary
# same-volume hard link with an ASCII-only path for the lifetime of the server.
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $projectRoot)
$bridgeRoot = Join-Path $workspaceRoot ("modelforge-serve-" + [guid]::NewGuid().ToString("N"))
$bridgeModel = Join-Path $bridgeRoot $fileName

try {
    New-Item -ItemType Directory -Path $bridgeRoot | Out-Null
    New-Item -ItemType HardLink -Path $bridgeModel -Target $model | Out-Null

    & $server.FullName `
        --model $bridgeModel `
        --alias "modelforge-tw-$Quantization" `
        --host 127.0.0.1 `
        --port $Port `
        --ctx-size $ContextSize `
        --parallel $Parallel `
        --n-gpu-layers all `
        --flash-attn on
}
finally {
    if (Test-Path -LiteralPath $bridgeModel) {
        Remove-Item -Force -LiteralPath $bridgeModel
    }
    if (Test-Path -LiteralPath $bridgeRoot) {
        Remove-Item -Force -LiteralPath $bridgeRoot
    }
}
