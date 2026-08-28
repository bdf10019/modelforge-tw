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
$exclude = @(
    '-g', '!.venv/**', '-g', '!artifacts/**', '-g', '!outputs/**',
    '-g', '!data/generated/**', '-g', '!tools/**', '-g', '!build/**', '-g', '!dist/**',
    '-g', '!.mypy_cache/**', '-g', '!.pytest_cache/**', '-g', '!.ruff_cache/**',
    '-g', '!**/__pycache__/**', '-g', '!**/*.egg-info/**', '-g', '!scripts/privacy-check.ps1'
)
$blockedExtensions = @(".safetensors", ".gguf", ".pt", ".pth", ".onnx", ".bin", ".sqlite", ".db")
$largeFileLimitBytes = 5MB
$findings = @()

Push-Location $projectRoot
try {
    foreach ($pattern in $patterns) {
        $result = & rg -n --hidden @exclude --regexp $pattern . 2>$null
        if ($LASTEXITCODE -eq 0) { $findings += $result }
        elseif ($LASTEXITCODE -ne 1) { throw "rg 隱私掃描失敗" }
    }
    $publicFiles = & rg --files --hidden @exclude
    if ($LASTEXITCODE -ne 0) { throw "rg 公開檔案清單失敗" }
    foreach ($relativePath in $publicFiles) {
        $item = Get-Item -LiteralPath $relativePath
        if ($item.Length -gt $largeFileLimitBytes) {
            $findings += "$relativePath 超過 5 MiB 公開檔案門檻：$($item.Length) bytes"
        }
        if ($blockedExtensions -contains $item.Extension.ToLowerInvariant()) {
            $findings += "$relativePath 使用不得提交的模型／資料庫副檔名"
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
