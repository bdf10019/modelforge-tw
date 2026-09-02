[CmdletBinding()]
param(
    [string]$MergedModel = "artifacts/qwen3-1.7b-triage-merged",
    [string]$OutputDirectory = "artifacts/gguf",
    [string]$ExpectedLlamaCommit = "ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc",
    [string]$ExpectedQuantizerSha256 = "0cff41994af0a3425b63cfcdc841542ec7b709ceeced6217bc5fc156815b2122"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$sourceRoot = Join-Path $projectRoot "tools\llama.cpp-src"
$converter = Join-Path $sourceRoot "convert_hf_to_gguf.py"
$binaryRoot = Join-Path $projectRoot "tools\llama.cpp-bin"
$quantizer = Get-ChildItem -LiteralPath $binaryRoot -Filter "llama-quantize.exe" -Recurse | Select-Object -First 1
$projectPrefix = $projectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$mergedPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $MergedModel))
$outputRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
foreach ($candidatePath in @($mergedPath, $outputRoot)) {
    if (-not $candidatePath.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "量化路徑必須位於專案目錄內：$candidatePath"
    }
}
$f16 = Join-Path $outputRoot "modelforge-tw-f16.gguf"
$q8 = Join-Path $outputRoot "modelforge-tw-q8_0.gguf"
$q4 = Join-Path $outputRoot "modelforge-tw-q4_k_m.gguf"
$mergeReportPath = Join-Path $mergedPath "merge-report.json"

function Assert-NoReparsePointInPath {
    param([string]$LiteralPath)
    $current = [IO.Path]::GetFullPath($LiteralPath)
    while (-not [string]::IsNullOrEmpty($current)) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "量化路徑不能包含 reparse point：$current"
            }
        }
        $parent = Split-Path -Parent $current
        if ([string]::IsNullOrEmpty($parent) -or $parent -eq $current) { break }
        $current = $parent
    }
}

if (-not (Test-Path -LiteralPath $converter)) { throw "請先執行 scripts\prepare-llama.ps1" }
if (-not $quantizer) { throw "找不到 llama-quantize.exe" }
if (-not (Test-Path -LiteralPath $mergedPath)) { throw "找不到 merged model：$mergedPath" }
if (-not (Test-Path -LiteralPath $mergeReportPath)) { throw "找不到 merge report：$mergeReportPath" }
Assert-NoReparsePointInPath -LiteralPath $mergedPath
Assert-NoReparsePointInPath -LiteralPath $outputRoot
if (Test-Path -LiteralPath $outputRoot) {
    $outputItem = Get-Item -LiteralPath $outputRoot
    if (-not $outputItem.PSIsContainer -or ($outputItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "量化 output 必須是 regular directory"
    }
    if (@(Get-ChildItem -LiteralPath $outputRoot -Force).Count -ne 0) {
        throw "量化 output directory 必須為空，不能覆寫既有 artifact"
    }
}
else {
    New-Item -ItemType Directory -Path $outputRoot | Out-Null
}

$mergeReport = Get-Content -LiteralPath $mergeReportPath -Raw | ConvertFrom-Json
$sourceFiles = @()
foreach ($property in $mergeReport.files.PSObject.Properties) {
    $fileName = $property.Name
    if ([System.IO.Path]::GetFileName($fileName) -ne $fileName) {
        throw "merge report 包含不安全檔名：$fileName"
    }
    $sourceFile = Join-Path $mergedPath $fileName
    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
        throw "merge report 來源檔不存在：$fileName"
    }
    $item = Get-Item -LiteralPath $sourceFile
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "merge report 來源檔不能是 reparse point：$fileName"
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceFile).Hash.ToLowerInvariant()
    if ($item.Length -ne $property.Value.bytes -or $actualHash -ne $property.Value.sha256) {
        throw "merge report 來源檔驗證失敗：$fileName"
    }
    $sourceFiles += [ordered]@{
        file = $fileName
        bytes = $item.Length
        sha256 = $actualHash
    }
}
if ($sourceFiles.Count -eq 0) { throw "merge report 沒有可驗證的來源檔" }
$mergeReportHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $mergeReportPath).Hash.ToLowerInvariant()
$sourceCommit = (& git -C $sourceRoot rev-parse HEAD).Trim().ToLowerInvariant()
if ($LASTEXITCODE -ne 0 -or $sourceCommit -ne $ExpectedLlamaCommit.ToLowerInvariant()) {
    throw "llama.cpp source commit 不符"
}
$sourceStatus = @(& git -C $sourceRoot status --porcelain --untracked-files=normal)
if ($LASTEXITCODE -ne 0 -or $sourceStatus.Count -ne 0) {
    throw "llama.cpp source tree 必須保持乾淨"
}
if ($quantizer.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw "llama-quantize.exe 不能是 reparse point"
}
$quantizerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $quantizer.FullName).Hash.ToLowerInvariant()
if ($quantizerHash -ne $ExpectedQuantizerSha256.ToLowerInvariant()) {
    throw "llama-quantize.exe SHA-256 不符"
}

& $python $converter $mergedPath --outfile $f16 --outtype f16
if ($LASTEXITCODE -ne 0) { throw "HF → GGUF 轉換失敗" }

# The official Windows quantizer currently decodes non-ASCII argv paths through
# the active code page. Keep the canonical artifacts in the project directory,
# but use an ASCII-only, same-volume hard-link bridge while invoking the binary.
$workspaceRoot = Split-Path -Parent $projectRoot
$bridgeRoot = Join-Path $workspaceRoot ("modelforge-gguf-" + [guid]::NewGuid().ToString("N"))
$bridgeF16 = Join-Path $bridgeRoot "model-f16.gguf"
$bridgeQ8 = Join-Path $bridgeRoot "model-q8_0.gguf"
$bridgeQ4 = Join-Path $bridgeRoot "model-q4_k_m.gguf"

try {
    New-Item -ItemType Directory -Path $bridgeRoot | Out-Null
    New-Item -ItemType HardLink -Path $bridgeF16 -Target $f16 | Out-Null

    & $quantizer.FullName $bridgeF16 $bridgeQ8 Q8_0
    if ($LASTEXITCODE -ne 0) { throw "Q8_0 量化失敗" }
    Move-Item -LiteralPath $bridgeQ8 -Destination $q8

    & $quantizer.FullName $bridgeF16 $bridgeQ4 Q4_K_M
    if ($LASTEXITCODE -ne 0) { throw "Q4_K_M 量化失敗" }
    Move-Item -LiteralPath $bridgeQ4 -Destination $q4
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
    schema_version = "1.1"
    evidence_kind = "measured"
    source = $MergedModel
    source_merge_report_sha256 = $mergeReportHash
    source_files = $sourceFiles
    llama_cpp_tag = "b10665"
    llama_cpp_commit = $sourceCommit
    quantizer_sha256 = $quantizerHash
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
