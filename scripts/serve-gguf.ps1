[CmdletBinding()]
param(
    [ValidateSet("q8_0", "q4_k_m")]
    [string]$Quantization = "q8_0",
    [int]$Port = 8081,
    [int]$ContextSize = 8192,
    [int]$Parallel = 4,
    [string]$ExpectedServerSha256 = "20a83f9ed723c6307749863842657da86918e8b612a3eaa44fe635d147341550"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot

function Assert-NoReparsePointInPath {
    param([string]$LiteralPath)
    $current = [IO.Path]::GetFullPath($LiteralPath)
    while (-not [string]::IsNullOrEmpty($current)) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "llama.cpp serving 路徑不能包含 reparse point：$current"
            }
        }
        $parent = Split-Path -Parent $current
        if ([string]::IsNullOrEmpty($parent) -or $parent -eq $current) { break }
        $current = $parent
    }
}

if ($Parallel -lt 1) { throw "Parallel 必須至少為 1" }
$minimumSlotContext = 1536
$slotContext = [math]::Floor($ContextSize / $Parallel)
if ($slotContext -lt $minimumSlotContext) {
    throw "每個推論 slot 只有 $slotContext tokens，低於最低 $minimumSlotContext；請提高 ContextSize 或降低 Parallel"
}
$serverPath = Join-Path $projectRoot "tools\llama.cpp-bin\llama-server.exe"
if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) { throw "找不到 llama-server.exe；請先執行 scripts\prepare-llama.ps1" }
Assert-NoReparsePointInPath -LiteralPath $serverPath
$server = Get-Item -LiteralPath $serverPath
$serverHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $server.FullName).Hash.ToLowerInvariant()
if ($serverHash -ne $ExpectedServerSha256.ToLowerInvariant()) { throw "llama-server.exe SHA-256 不符" }

$fileName = if ($Quantization -eq "q8_0") { "modelforge-tw-q8_0.gguf" } else { "modelforge-tw-q4_k_m.gguf" }
$model = Join-Path $projectRoot "artifacts\gguf\$fileName"
$quantizationReportPath = Join-Path $projectRoot "artifacts\gguf\quantization-report.json"
if (-not (Test-Path -LiteralPath $model -PathType Leaf)) { throw "找不到 GGUF：$model；請先執行 scripts\quantize.ps1" }
if (-not (Test-Path -LiteralPath $quantizationReportPath -PathType Leaf)) { throw "找不到 quantization report" }
Assert-NoReparsePointInPath -LiteralPath $model
Assert-NoReparsePointInPath -LiteralPath $quantizationReportPath
$quantizationReport = Get-Content -LiteralPath $quantizationReportPath -Raw | ConvertFrom-Json
$modelAttestations = @($quantizationReport.files | Where-Object { $_.file -eq $fileName })
if ($modelAttestations.Count -ne 1) { throw "quantization report 缺少唯一 GGUF 證明" }
$modelItem = Get-Item -LiteralPath $model
$modelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $model).Hash.ToLowerInvariant()
if ($modelItem.Length -ne $modelAttestations[0].bytes -or $modelHash -ne $modelAttestations[0].sha256) {
    throw "GGUF 與 quantization report 不符"
}

# The official Windows server currently decodes non-ASCII argv paths through
# the active code page. Keep the model in the project and expose a temporary
# same-volume hard link with an ASCII-only path for the lifetime of the server.
$workspaceRoot = Split-Path -Parent $projectRoot
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
