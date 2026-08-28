[CmdletBinding()]
param(
    [string]$Tag = "b10665",
    [string]$ExpectedBinarySha256 = "5573a0f58c8cd315c001a68f0ca2c60866352130ba10945a2a403f61916d727e",
    [string]$ExpectedRuntimeSha256 = "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e",
    [string]$ExpectedSourceCommit = "ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolsRoot = Join-Path $projectRoot "tools"
$binaryRoot = Join-Path $toolsRoot "llama.cpp-bin"
$sourceRoot = Join-Path $toolsRoot "llama.cpp-src"
$downloadRoot = Join-Path $toolsRoot "downloads"
$binaryZip = Join-Path $downloadRoot "llama-$Tag-bin-win-cuda-13.3-x64.zip"
$runtimeZip = Join-Path $downloadRoot "cudart-llama-bin-win-cuda-13.3-x64.zip"
$releaseRoot = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag"

New-Item -ItemType Directory -Force -Path $binaryRoot, $downloadRoot | Out-Null
if (-not (Test-Path -LiteralPath $binaryZip)) {
    Invoke-WebRequest -Uri "$releaseRoot/llama-$Tag-bin-win-cuda-13.3-x64.zip" -OutFile $binaryZip
}
if (-not (Test-Path -LiteralPath $runtimeZip)) {
    Invoke-WebRequest -Uri "$releaseRoot/cudart-llama-bin-win-cuda-13.3-x64.zip" -OutFile $runtimeZip
}
$binaryHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $binaryZip).Hash.ToLowerInvariant()
$runtimeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeZip).Hash.ToLowerInvariant()
if ($binaryHash -ne $ExpectedBinarySha256.ToLowerInvariant()) {
    throw "llama.cpp binary SHA-256 不符；保留下載檔供檢查"
}
if ($runtimeHash -ne $ExpectedRuntimeSha256.ToLowerInvariant()) {
    throw "CUDA runtime SHA-256 不符；保留下載檔供檢查"
}
Expand-Archive -LiteralPath $binaryZip -DestinationPath $binaryRoot -Force
Expand-Archive -LiteralPath $runtimeZip -DestinationPath $binaryRoot -Force

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
    git clone --depth 1 --branch $Tag https://github.com/ggml-org/llama.cpp.git $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw "下載 llama.cpp source 失敗" }
}
$sourceCommit = (& git -C $sourceRoot rev-parse HEAD).Trim().ToLowerInvariant()
if ($LASTEXITCODE -ne 0 -or $sourceCommit -ne $ExpectedSourceCommit.ToLowerInvariant()) {
    throw "llama.cpp source commit 不符"
}

$quantizer = Get-ChildItem -LiteralPath $binaryRoot -Filter "llama-quantize.exe" -Recurse | Select-Object -First 1
$server = Get-ChildItem -LiteralPath $binaryRoot -Filter "llama-server.exe" -Recurse | Select-Object -First 1
if (-not $quantizer -or -not $server) { throw "llama.cpp binary 壓縮檔缺少預期工具" }
Write-Host "llama.cpp $Tag 準備完成：$binaryRoot"
