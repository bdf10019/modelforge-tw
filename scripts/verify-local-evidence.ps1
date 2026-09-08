[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$snapshotPath = Join-Path $projectRoot "docs\measured-evidence.json"
$snapshot = Get-Content -LiteralPath $snapshotPath -Raw | ConvertFrom-Json
$external = $snapshot.external_calibration
$checkedReports = 0
$mainReportsChecked = 0
$comparisonChecked = $false
$trainingChecked = $false
$artifactReportsChecked = 0
$benchmarksChecked = 0
$currentBenchmarksChecked = 0
$registryChecked = $false

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
    $contractFields = @(
        "model_id",
        "model_version",
        "quantization",
        "max_new_tokens",
        "context_window_tokens",
        "structured_output_contract",
        "safety_policy_version",
        "evaluation_contract_version",
        "dataset_manifest_sha256",
        "release_gate_sha256",
        "evidence_kind"
    )
    foreach ($field in $contractFields) {
        Assert-Equal "$($entry.Name).$field" $expected.$field $actual.$field
    }
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

foreach ($entry in $snapshot.evaluations.PSObject.Properties) {
    $expected = $entry.Value
    $reportPath = Join-Path $projectRoot $expected.source_report
    if (-not (Test-Path -LiteralPath $reportPath)) { continue }
    $actual = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    Assert-Equal "$($entry.Name).run_id" $expected.run_id $actual.run_id
    Assert-Equal "$($entry.Name).cases" $expected.cases $actual.case_count
    Assert-Equal "$($entry.Name).evidence_kind" $snapshot.evidence_kind $actual.evidence_kind
    $contractFields = @(
        "prompt_version",
        "max_new_tokens",
        "structured_output_contract",
        "safety_policy_version",
        "evaluation_contract_version",
        "release_gate_sha256",
        "release_passed"
    )
    foreach ($field in $contractFields) {
        Assert-Equal "$($entry.Name).$field" $expected.$field $actual.$field
    }
    foreach ($metric in $expected.metrics.PSObject.Properties) {
        Assert-Metric `
            "$($entry.Name).metrics.$($metric.Name)" `
            $metric.Value `
            $actual.metrics.($metric.Name)
    }
    foreach ($latency in $expected.latency_ms.PSObject.Properties) {
        Assert-Metric `
            "$($entry.Name).latency_ms.$($latency.Name)" `
            $latency.Value `
            $actual.latency_ms.($latency.Name)
    }
    $mainReportsChecked += 1
}

$comparisonPath = Join-Path $projectRoot $snapshot.q8_to_q4_comparison.source_report
if (Test-Path -LiteralPath $comparisonPath) {
    $comparison = Get-Content -LiteralPath $comparisonPath -Raw | ConvertFrom-Json
    foreach ($check in $snapshot.q8_to_q4_comparison.compatibility_checks.PSObject.Properties) {
        Assert-Equal `
            "q8_to_q4.compatibility.$($check.Name)" `
            $check.Value `
            $comparison.quantization_gate.compatibility_checks.($check.Name)
    }
    foreach (
        $metric in $snapshot.q8_to_q4_comparison.metric_deltas_candidate_minus_baseline.PSObject.Properties
    ) {
        Assert-Metric `
            "q8_to_q4.metric_deltas.$($metric.Name)" `
            $metric.Value `
            $comparison.metric_deltas.($metric.Name)
    }
    Assert-Metric `
        "q8_to_q4.event_macro_f1_max_regression" `
        $snapshot.q8_to_q4_comparison.event_macro_f1_max_regression `
        $comparison.quantization_gate.checks.event_type_macro_f1.max_regression
    Assert-Equal `
        "q8_to_q4.quantization_gate_passed" `
        $snapshot.q8_to_q4_comparison.quantization_gate_passed `
        $comparison.quantization_gate.passed
    $comparisonChecked = $true
}

$trainingPath = Join-Path $projectRoot $snapshot.training.source_report
if (Test-Path -LiteralPath $trainingPath) {
    $training = Get-Content -LiteralPath $trainingPath -Raw | ConvertFrom-Json
    Assert-Equal "training.evidence_kind" $snapshot.evidence_kind $training.evidence_kind
    Assert-Equal "training.base_model" $snapshot.training.base_model $training.model_id
    Assert-Equal `
        "training.base_revision" `
        $snapshot.training.base_revision `
        $training.resolved_model_revision
    Assert-Metric "training.epochs" $snapshot.training.epochs $training.config.epochs
    Assert-Metric `
        "training.duration_seconds" `
        $snapshot.training.duration_seconds `
        $training.duration_seconds
    Assert-Metric "training.train_loss" $snapshot.training.train_loss $training.train_metrics.train_loss
    Assert-Metric `
        "training.validation_loss" `
        $snapshot.training.validation_loss `
        $training.eval_metrics.eval_loss
    Assert-Equal `
        "training.trainable_parameters" `
        $snapshot.training.trainable_parameters `
        $training.trainable_parameters
    Assert-Equal "training.total_parameters" $snapshot.training.total_parameters $training.total_parameters
    Assert-Metric `
        "training.trainable_percent" `
        $snapshot.training.trainable_percent `
        $training.trainable_percent
    Assert-Equal `
        "training.peak_allocated_bytes" `
        $snapshot.training.peak_allocated_bytes `
        $training.peak_vram_allocated_bytes
    Assert-Equal `
        "training.training_config_sha256" `
        $snapshot.training.training_config_sha256 `
        $training.training_config_sha256
    $adapter = $training.final_adapter_files.PSObject.Properties["adapter_model.safetensors"].Value
    Assert-Equal "training.adapter_sha256" $snapshot.training.adapter_sha256 $adapter.sha256
    $trainingChecked = $true
}

$mergeExpected = $snapshot.artifacts.merged_safetensors
$mergePath = Join-Path $projectRoot $mergeExpected.source_report
if (Test-Path -LiteralPath $mergePath) {
    $merge = Get-Content -LiteralPath $mergePath -Raw | ConvertFrom-Json
    $mergedFile = $merge.files.PSObject.Properties["model.safetensors"].Value
    Assert-Equal "artifacts.merged.bytes" $mergeExpected.bytes $mergedFile.bytes
    Assert-Equal "artifacts.merged.sha256" $mergeExpected.sha256 $mergedFile.sha256
    $artifactReportsChecked += 1
}

$quantizationPath = Join-Path $projectRoot $snapshot.artifacts.gguf_f16.source_report
if (Test-Path -LiteralPath $quantizationPath) {
    $quantization = Get-Content -LiteralPath $quantizationPath -Raw | ConvertFrom-Json
    Assert-Equal "artifacts.quantization.evidence_kind" $snapshot.evidence_kind $quantization.evidence_kind
    $quantizedFiles = @{
        gguf_f16 = "modelforge-tw-f16.gguf"
        gguf_q8_0 = "modelforge-tw-q8_0.gguf"
        gguf_q4_k_m = "modelforge-tw-q4_k_m.gguf"
    }
    foreach ($artifactName in $quantizedFiles.Keys) {
        $expected = $snapshot.artifacts.$artifactName
        $actual = @($quantization.files | Where-Object { $_.file -eq $quantizedFiles[$artifactName] })
        if ($actual.Count -ne 1) { throw "artifacts.$artifactName 找不到唯一的量化報告項目" }
        Assert-Equal "artifacts.$artifactName.bytes" $expected.bytes $actual[0].bytes
        Assert-Equal "artifacts.$artifactName.sha256" $expected.sha256 $actual[0].sha256
    }
    $artifactReportsChecked += 1
}

foreach ($entry in $snapshot.concurrency_benchmarks.PSObject.Properties) {
    $expected = $entry.Value
    $benchmarkPath = Join-Path $projectRoot $expected.source_report
    if (-not (Test-Path -LiteralPath $benchmarkPath)) { continue }
    $benchmark = Get-Content -LiteralPath $benchmarkPath -Raw | ConvertFrom-Json
    Assert-Equal "benchmark.$($entry.Name).evidence_kind" $snapshot.evidence_kind $benchmark.evidence_kind
    Assert-Equal "benchmark.$($entry.Name).quantization" $entry.Name $benchmark.quantization
    $levels = @($benchmark.levels)
    $requestCount = ($levels | Measure-Object -Property requests -Sum).Sum
    $errorCount = ($levels | Measure-Object -Property errors -Sum).Sum
    Assert-Equal "benchmark.$($entry.Name).requests" $expected.requests $requestCount
    Assert-Equal "benchmark.$($entry.Name).errors" $expected.errors $errorCount
    foreach ($level in $levels) {
        Assert-Equal `
            "benchmark.$($entry.Name).concurrency_$($level.concurrency).completed" `
            $level.requests `
            $level.completed
    }
    if ($null -ne $benchmark.PSObject.Properties["passed"]) {
        Assert-Equal "benchmark.$($entry.Name).passed" $true $benchmark.passed
    }
    Assert-Equal `
        "benchmark.$($entry.Name).host_gpu_peak_mib" `
        $expected.host_gpu_peak_mib `
        $benchmark.environment.peak_gpu_memory_used_mib
    $concurrency4 = @($levels | Where-Object { $_.concurrency -eq 4 })
    if ($concurrency4.Count -ne 1) { throw "benchmark.$($entry.Name) 找不到唯一的 concurrency 4" }
    Assert-Metric `
        "benchmark.$($entry.Name).concurrency_4.requests_per_second" `
        $expected.concurrency_4.requests_per_second `
        $concurrency4[0].requests_per_second
    Assert-Metric `
        "benchmark.$($entry.Name).concurrency_4.output_tokens_per_second" `
        $expected.concurrency_4.output_tokens_per_second `
        $concurrency4[0].output_tokens_per_second
    Assert-Metric `
        "benchmark.$($entry.Name).concurrency_4.latency_p95_ms" `
        $expected.concurrency_4.latency_p95_ms `
        $concurrency4[0].latency_ms.p95
    Assert-Metric `
        "benchmark.$($entry.Name).concurrency_4.ttft_p50_ms" `
        $expected.concurrency_4.ttft_p50_ms `
        $concurrency4[0].ttft_ms.p50
    $benchmarksChecked += 1
}

$currentSnapshotPath = Join-Path $projectRoot "docs\current-context-benchmarks.json"
if (Test-Path -LiteralPath $currentSnapshotPath) {
    $currentSnapshot = Get-Content -LiteralPath $currentSnapshotPath -Raw | ConvertFrom-Json
    foreach ($entry in $currentSnapshot.measurements.PSObject.Properties) {
        $expected = $entry.Value
        $reportPath = Join-Path $projectRoot $expected.source_report
        if (-not (Test-Path -LiteralPath $reportPath)) { continue }
        $reportHash = (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLowerInvariant()
        Assert-Equal "current_benchmark.$($entry.Name).sha256" $expected.source_report_sha256 $reportHash
        $actual = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        Assert-Equal `
            "current_benchmark.$($entry.Name).report" `
            ($expected.report | ConvertTo-Json -Depth 30 -Compress) `
            ($actual | ConvertTo-Json -Depth 30 -Compress)
        $currentBenchmarksChecked += 1
    }
}

$registryPath = Join-Path $projectRoot $snapshot.registry_source_report
if (Test-Path -LiteralPath $registryPath) {
    $registry = Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json
    $expected = $snapshot.registry_expected_state
    Assert-Equal "registry.released" $expected.released $registry.released
    Assert-Equal "registry.previous_good" $expected.previous_good $registry.previous_good
    $released = $registry.manifests.PSObject.Properties[$expected.released].Value
    $previous = $registry.manifests.PSObject.Properties[$expected.previous_good].Value
    $blocked = $registry.manifests.PSObject.Properties[$expected.blocked].Value
    Assert-Equal "registry.released.status" "released" $released.status
    Assert-Equal "registry.previous_good.status" "retired" $previous.status
    Assert-Equal "registry.blocked.status" "blocked" $blocked.status
    Assert-Equal "registry.blocked.release_passed" $false $blocked.release_passed
    $registryChecked = $true
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
    external_reports_checked = $checkedReports
    main_reports_checked = $mainReportsChecked
    comparison_checked = $comparisonChecked
    training_checked = $trainingChecked
    artifact_reports_checked = $artifactReportsChecked
    benchmarks_checked = $benchmarksChecked
    current_context_benchmarks_checked = $currentBenchmarksChecked
    registry_checked = $registryChecked
    external_manifest_checked = $manifestChecked
} | ConvertTo-Json
