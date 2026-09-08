[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$patterns = @(
    '(?i)(api[_-]?key|secret|password|passwd)\s*[:=]\s*["''][^"'']{8,}',
    '(?i)bearer\s+[a-z0-9._-]{16,}',
    'sk-[A-Za-z0-9_-]{16,}',
    'C:\\Users\\[^\\\s]+'
)
$excludedPrefixes = @(
    ".venv/", "artifacts/", "outputs/", "data/generated/", "tools/", "build/", "dist/",
    ".mypy_cache/", ".pytest_cache/", ".ruff_cache/"
)
$blockedExtensions = @(".safetensors", ".gguf", ".pt", ".pth", ".onnx", ".bin", ".sqlite", ".db")
$largeFileLimitBytes = 5MB
$findings = @()

Push-Location $projectRoot
try {
    $candidateFiles = & git ls-files --cached --others --exclude-standard
    if ($LASTEXITCODE -ne 0) { throw "git 公開檔案清單失敗" }
    $publicFiles = @($candidateFiles | Where-Object {
        $relativePath = $_.Replace("\", "/")
        $isExcluded = $relativePath -eq "scripts/privacy-check.ps1" -or
            $relativePath -match '(^|/)__pycache__/' -or
            $relativePath -match '(^|/)[^/]+\.egg-info/'
        foreach ($prefix in $excludedPrefixes) {
            if ($relativePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                $isExcluded = $true
                break
            }
        }
        -not $isExcluded
    })
    foreach ($relativePath in $publicFiles) {
        $item = Get-Item -LiteralPath $relativePath
        if ($item.Length -gt $largeFileLimitBytes) {
            $findings += "$relativePath 超過 5 MiB 公開檔案門檻：$($item.Length) bytes"
        }
        if ($blockedExtensions -contains $item.Extension.ToLowerInvariant()) {
            $findings += "$relativePath 使用不得提交的模型／資料庫副檔名"
        }
        $matches = Select-String -LiteralPath $item.FullName -Pattern $patterns -AllMatches
        foreach ($match in $matches) {
            $findings += "$relativePath`:$($match.LineNumber) 疑似包含敏感內容"
        }
    }
    if ($findings.Count -gt 0) {
        $findings | ForEach-Object { Write-Error $_ }
        throw "隱私／秘密掃描發現疑似敏感內容"
    }
    Write-Host "隱私／秘密掃描通過。"
}
finally {
    Pop-Location
}
