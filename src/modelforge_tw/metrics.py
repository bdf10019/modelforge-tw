from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class ServiceMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "modelforge_requests_total",
            "Triage requests by outcome and engine.",
            ["outcome", "engine"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "modelforge_request_duration_seconds",
            "End-to-end triage request duration.",
            ["engine"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "modelforge_inflight_requests",
            "Requests currently executing inference.",
            registry=self.registry,
        )
        self.validation_warnings = Counter(
            "modelforge_validation_warnings_total",
            "Server-side validation warnings by code.",
            ["code"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
