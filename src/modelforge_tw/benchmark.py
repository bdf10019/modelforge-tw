from __future__ import annotations

import concurrent.futures
import functools
import json
import platform
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from modelforge_tw.engines.base import InferenceEngine
from modelforge_tw.io_utils import atomic_write_text
from modelforge_tw.schemas import DatasetRecord, InferenceMetadata

BENCHMARK_IDENTITY_FIELDS = (
    "engine",
    "model_id",
    "model_version",
    "model_revision",
    "quantization",
    "prompt_version",
    "max_new_tokens",
    "context_window_tokens",
    "structured_output_contract",
    "safety_policy_version",
    "evidence_kind",
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _gpu_snapshot() -> dict[str, Any] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        name, used, total, utilization, driver = [part.strip() for part in output.split(",")]
        return {
            "name": name,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_percent": int(utilization),
            "driver_version": driver,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class _ResourceSampler:
    def __init__(self, process: psutil.Process, interval_seconds: float = 0.5) -> None:
        self.process = process
        self.interval_seconds = interval_seconds
        self.peak_process_rss_bytes = process.memory_info().rss
        initial_gpu = _gpu_snapshot()
        self.peak_gpu_memory_used_mib = (
            initial_gpu["memory_used_mib"] if initial_gpu is not None else None
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def _sample_once(self) -> None:
        self.peak_process_rss_bytes = max(
            self.peak_process_rss_bytes,
            self.process.memory_info().rss,
        )
        gpu = _gpu_snapshot()
        if gpu is not None:
            used = gpu["memory_used_mib"]
            if self.peak_gpu_memory_used_mib is None:
                self.peak_gpu_memory_used_mib = used
            else:
                self.peak_gpu_memory_used_mib = max(self.peak_gpu_memory_used_mib, used)

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._sample_once()
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds * 2)


def run_benchmark(
    engine: InferenceEngine,
    records: list[DatasetRecord],
    *,
    concurrency_levels: list[int],
    requests_per_level: int,
    warmup_requests: int = 2,
) -> dict[str, Any]:
    if not records:
        raise ValueError("benchmark requires at least one record")
    if not concurrency_levels:
        raise ValueError("benchmark requires at least one concurrency level")
    if requests_per_level < 1 or any(level < 1 for level in concurrency_levels):
        raise ValueError("concurrency and request counts must be positive")
    if len(concurrency_levels) != len(set(concurrency_levels)):
        raise ValueError("concurrency levels must be unique")
    if warmup_requests < 0:
        raise ValueError("warmup request count cannot be negative")

    for index in range(warmup_requests):
        engine.generate(records[index % len(records)].input)

    streaming_generate = getattr(engine, "generate_streaming", None)
    benchmark_generate = streaming_generate if callable(streaming_generate) else engine.generate

    process = psutil.Process()
    rss_before = process.memory_info().rss
    gpu_before = _gpu_snapshot()
    sampler = _ResourceSampler(process)
    sampler.start()
    levels: list[dict[str, Any]] = []
    metadata_samples: list[InferenceMetadata] = []
    metadata_lock = threading.Lock()

    try:
        for concurrency in concurrency_levels:
            latencies: list[float] = []
            output_tokens: list[int] = []
            ttfts: list[float] = []
            errors: list[str] = []
            started = time.perf_counter()

            def execute(
                index: int,
                latency_sink: list[float],
                token_sink: list[int],
                ttft_sink: list[float],
                error_sink: list[str],
            ) -> None:
                request_started = time.perf_counter()
                try:
                    result = benchmark_generate(records[index % len(records)].input)
                    with metadata_lock:
                        if not metadata_samples:
                            metadata_samples.append(result.metadata)
                        else:
                            identity_mismatches = [
                                field
                                for field in BENCHMARK_IDENTITY_FIELDS
                                if getattr(result.metadata, field)
                                != getattr(metadata_samples[0], field)
                            ]
                            if identity_mismatches:
                                raise ValueError(
                                    "inference metadata changed during benchmark: "
                                    + ", ".join(identity_mismatches)
                                )
                    latency_sink.append((time.perf_counter() - request_started) * 1000)
                    if result.metadata.output_tokens is not None:
                        token_sink.append(result.metadata.output_tokens)
                    if result.metadata.ttft_ms is not None:
                        ttft_sink.append(result.metadata.ttft_ms)
                except Exception as error:  # benchmark must retain failures in its report
                    error_sink.append(f"{type(error).__name__}: {str(error)[:160]}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                worker = functools.partial(
                    execute,
                    latency_sink=latencies,
                    token_sink=output_tokens,
                    ttft_sink=ttfts,
                    error_sink=errors,
                )
                list(executor.map(worker, range(requests_per_level)))
            wall_seconds = time.perf_counter() - started
            completed = requests_per_level - len(errors)
            levels.append(
                {
                    "concurrency": concurrency,
                    "requests": requests_per_level,
                    "completed": completed,
                    "errors": len(errors),
                    "error_samples": errors[:5],
                    "wall_seconds": round(wall_seconds, 6),
                    "requests_per_second": round(
                        completed / wall_seconds if wall_seconds else 0, 4
                    ),
                    "latency_ms": {
                        "p50": round(_percentile(latencies, 0.50), 4),
                        "p95": round(_percentile(latencies, 0.95), 4),
                        "p99": round(_percentile(latencies, 0.99), 4),
                        "mean": round(statistics.fmean(latencies), 4) if latencies else 0,
                    },
                    "output_tokens": sum(output_tokens) if output_tokens else None,
                    "output_token_sample_count": len(output_tokens),
                    "output_tokens_per_second": (
                        round(sum(output_tokens) / wall_seconds, 4) if output_tokens else None
                    ),
                    "ttft_ms": (
                        {
                            "p50": round(_percentile(ttfts, 0.50), 4),
                            "p95": round(_percentile(ttfts, 0.95), 4),
                            "mean": round(statistics.fmean(ttfts), 4),
                        }
                        if ttfts
                        else None
                    ),
                    "ttft_sample_count": len(ttfts),
                    "ttft_status": (
                        "measured_from_streaming_response"
                        if completed > 0 and len(ttfts) == completed
                        else (
                            "partially_available_from_streaming_response"
                            if ttfts
                            else "not_available_for_non_streaming_adapter"
                        )
                    ),
                }
            )

    finally:
        sampler.stop()
    gpu_after = _gpu_snapshot()
    first = metadata_samples[0] if metadata_samples else None
    passed = all(
        level["errors"] == 0 and level["completed"] == level["requests"] for level in levels
    )
    return {
        "schema_version": "1.2",
        "created_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "failure_reason": None if passed else "one_or_more_benchmark_requests_failed",
        "evidence_kind": first.evidence_kind if first else engine.evidence_kind,
        "engine": first.engine if first else engine.name,
        "model_id": first.model_id if first else engine.model_id,
        "model_version": first.model_version if first else engine.model_version,
        "model_revision": first.model_revision
        if first
        else getattr(engine, "model_revision", None),
        "quantization": first.quantization if first else engine.quantization,
        "prompt_version": first.prompt_version if first else engine.prompt_version,
        "max_new_tokens": first.max_new_tokens if first else engine.max_new_tokens,
        "context_window_tokens": (
            first.context_window_tokens if first else getattr(engine, "context_window_tokens", None)
        ),
        "structured_output_contract": (
            first.structured_output_contract
            if first
            else getattr(engine, "structured_output_contract", None)
        ),
        "safety_policy_version": (
            first.safety_policy_version if first else getattr(engine, "safety_policy_version", None)
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": psutil.cpu_count(),
            "ram_total_bytes": psutil.virtual_memory().total,
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": process.memory_info().rss,
            "peak_process_rss_bytes": sampler.peak_process_rss_bytes,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "peak_gpu_memory_used_mib": sampler.peak_gpu_memory_used_mib,
            "gpu_memory_scope": "host-wide nvidia-smi reading",
        },
        "warmup_requests": warmup_requests,
        "levels": levels,
        "notes": [
            "A report passes only when every measured request completes without error.",
            "TTFT requires a streaming adapter and is not represented as zero.",
            "Fixture evidence validates the benchmark pipeline only and is not model performance.",
            "GPU memory is sampled for the whole device, not attributed to one process.",
        ],
    }


def write_benchmark(report: dict[str, Any], path: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
