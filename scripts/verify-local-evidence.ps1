[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$snapshotPath = Join-Path $projectRoot "docs\measured-evidence.json"
$snapshot = Get-Content -LiteralPath $snapshotPath -Raw | ConvertFrom-Json
$external = $snapshot.external_calibration
$checkedReports = 0

function Assert-Equal([string]$Label, $Expected, $Actual) {
    if ($Expected -ne $Actual) {
        throw "$Label 不一致：snapshot=$Expected actual=$Actual"
    }
}

function Assert-Metric([string]$Label, $Expected, $Actual) {
    $delta = [math]::Abs([double]$Expected - [double]$Actual)
    if ($delta -gt 0.0000005) {
        throw "$Label 不一致：snapshot=$Expected actual=$Actual"
    }
}

foreach ($entry in $external.evaluations.PSObject.Properties) {
    $expected = $entry.Value
    $reportPath = Join-Path $projectRoot $expected.source_report
    if (-not (Test-Path -LiteralPath $reportPath)) { continue }
    $actual = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    Assert-Equal "$($entry.Name).run_id" $expected.run_id $actual.run_id
    Assert-Equal "$($entry.Name).cases" $expected.cases $actual.case_count
    Assert-Equal "$($entry.Name).prompt" $expected.prompt_version $actual.prompt_version
    Assert-Equal `
        "$($entry.Name).context" `
        $expected.context_window_tokens `
        $actual.context_window_tokens
    Assert-Equal `
        "$($entry.Name).release_passed" `
        $expected.release_passed `
        $actual.release_passed
    foreach ($metric in $expected.metrics.PSObject.Properties) {
        Assert-Metric `
            "$($entry.Name).metrics.$($metric.Name)" `
            $metric.Value `
            $actual.metrics.($metric.Name)
    }
    if ($null -ne $expected.PSObject.Properties["validation_warning_counts"]) {
        foreach ($warning in $expected.validation_warning_counts.PSObject.Properties) {
            Assert-Equal `
                "$($entry.Name).validation_warning_counts.$($warning.Name)" `
                $warning.Value `
                $actual.validation_warning_counts.($warning.Name)
        }
    }
    $checkedReports += 1
}

$generatedManifestPath = Join-Path $projectRoot "data\external\generated\manifest.json"
$manifestChecked = $false
if (Test-Path -LiteralPath $generatedManifestPath) {
    $manifest = Get-Content -LiteralPath $generatedManifestPath -Raw | ConvertFrom-Json
    $manifestHash = (Get-FileHash -LiteralPath $generatedManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert-Equal "external.manifest_sha256" $external.manifest_sha256 $manifestHash
    Assert-Equal "external.transform_version" $external.transform_version $manifest.transform_version
    Assert-Equal "external.calibration.records" $external.calibration.records $manifest.output.records
    Assert-Equal `
        "external.calibration.jsonl_sha256" `
        $external.calibration.jsonl_sha256 `
        $manifest.output.sha256
    Assert-Equal "external.holdout.records" $external.holdout.records $manifest.holdout_output.records
    Assert-Equal `
        "external.holdout.jsonl_sha256" `
        $external.holdout.jsonl_sha256 `
        $manifest.holdout_output.sha256
    Assert-Equal "external.source_row_overlap" $external.audit.source_row_overlap $manifest.source_row_overlap
    $manifestChecked = $true
}

[pscustomobject]@{
    status = "passed"
    checked_reports = $checkedReports
    manifest_checked = $manifestChecked
} | ConvertTo-Json
