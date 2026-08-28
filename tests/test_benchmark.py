from modelforge_tw.benchmark import run_benchmark
from modelforge_tw.dataset import generate_records
from modelforge_tw.engines import FixtureEngine


class StreamingFixtureEngine(FixtureEngine):
    def generate_streaming(self, request):
        result = self.generate(request)
        result.metadata = result.metadata.model_copy(update={"ttft_ms": 2.5})
        return result


def test_benchmark_keeps_unavailable_ttft_explicit():
    records = generate_records(1, 1, 1)[:4]
    report = run_benchmark(
        FixtureEngine("adapted"),
        records,
        concurrency_levels=[1, 2],
        requests_per_level=4,
        warmup_requests=1,
    )
    assert report["evidence_kind"] == "fixture"
    assert report["levels"][0]["ttft_ms"] is None
    assert report["levels"][0]["ttft_status"] == "not_available_for_non_streaming_adapter"
    assert report["levels"][0]["errors"] == 0
    assert (
        report["environment"]["peak_process_rss_bytes"]
        >= report["environment"]["process_rss_before_bytes"]
    )
    assert report["environment"]["gpu_memory_scope"] == "host-wide nvidia-smi reading"


def test_benchmark_records_streaming_ttft_when_engine_supports_it():
    records = generate_records(1, 1, 1)[:2]
    report = run_benchmark(
        StreamingFixtureEngine("adapted"),
        records,
        concurrency_levels=[1],
        requests_per_level=2,
        warmup_requests=0,
    )
    assert report["levels"][0]["ttft_status"] == "measured_from_streaming_response"
    assert report["levels"][0]["ttft_ms"]["p50"] == 2.5
