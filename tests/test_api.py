import asyncio
import threading
import time
from contextlib import asynccontextmanager

import httpx
import pytest

from modelforge_tw.engines import FixtureEngine
from modelforge_tw.service import create_app
from modelforge_tw.settings import Settings


def test_settings_reject_context_capacity_below_measured_minimum():
    with pytest.raises(ValueError):
        Settings(context_window_tokens=1024)


def test_service_defaults_to_attested_safety_policy_v2():
    assert Settings().safety_policy_version == "risk-precedence-v2"


@asynccontextmanager
async def app_client(app):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_triage_health_metrics_and_no_input_echo():
    app = create_app(engine_factory=lambda: FixtureEngine("adapted"))
    async with app_client(app) as client:
        assert (await client.get("/healthz")).status_code == 200
        ready = (await client.get("/readyz")).json()
        assert ready["evidence_kind"] == "fixture"
        response = await client.post(
            "/v1/triage",
            headers={"X-Request-ID": "test-request-1"},
            json={"incident_text": "泵浦軸承持續異音，需要人員確認。", "signals": []},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["request_id"] == "test-request-1"
        assert body["output"]["review_required"] is True
        assert "incident_text" not in body
        metrics = await client.get("/metrics")
        assert "modelforge_requests_total" in metrics.text


async def test_invalid_request_id_and_extra_input_are_rejected():
    app = create_app(engine_factory=lambda: FixtureEngine("adapted"))
    async with app_client(app) as client:
        assert (
            await client.post(
                "/v1/triage",
                headers={"X-Request-ID": "bad id with spaces"},
                json={"incident_text": "泵浦持續出現異音"},
            )
        ).status_code == 400
        assert (
            await client.post(
                "/v1/triage",
                json={
                    "incident_text": "泵浦持續出現異音",
                    "model_path": "C:/private/model",
                },
            )
        ).status_code == 422


async def test_openai_compatible_non_streaming_and_streaming():
    app = create_app(engine_factory=lambda: FixtureEngine("adapted"))
    async with app_client(app) as client:
        payload = {
            "model": "released",
            "messages": [{"role": "user", "content": "馬達溫度持續升高"}],
        }
        normal = await client.post("/v1/chat/completions", json=payload)
        assert normal.status_code == 200
        assert normal.json()["object"] == "chat.completion"
        assert "usage" not in normal.json()
        streamed = await client.post("/v1/chat/completions", json={**payload, "stream": True})
        assert streamed.status_code == 200
        assert "data: [DONE]" in streamed.text
        invalid_id = await client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "bad id with spaces"},
            json=payload,
        )
        assert invalid_id.status_code == 400


async def test_openai_compatible_rejects_unsupported_or_misleading_parameters():
    app = create_app(engine_factory=lambda: FixtureEngine("adapted"))
    async with app_client(app) as client:
        messages = [{"role": "user", "content": "馬達溫度持續升高"}]
        assert (
            await client.post(
                "/v1/chat/completions",
                json={"model": "not-deployed", "messages": messages},
            )
        ).status_code == 404
        assert (
            await client.post(
                "/v1/chat/completions",
                json={"model": "released", "messages": messages, "temperature": 0.2},
            )
        ).status_code == 422
        assert (
            await client.post(
                "/v1/chat/completions",
                json={"model": "released", "messages": messages, "top_p": 0.9},
            )
        ).status_code == 422
        assert (
            await client.post(
                "/v1/chat/completions",
                json={"model": "released", "messages": messages, "max_tokens": 64},
            )
        ).status_code == 400
        assert (
            await client.post(
                "/v1/chat/completions",
                json={"model": "released", "messages": [{"role": "user", "content": "太短"}]},
            )
        ).status_code == 422


class SlowFailingEngine(FixtureEngine):
    def generate(self, request):
        time.sleep(0.05)
        raise RuntimeError("do not expose this internal error")


class BlockingEngine(FixtureEngine):
    def __post_init__(self):
        super().__post_init__()
        self.calls = 0
        self.release = threading.Event()

    def generate(self, request):
        self.calls += 1
        self.release.wait(timeout=1)
        return super().generate(request)


async def test_failure_returns_conservative_fallback_without_error_detail():
    settings = Settings(request_timeout_seconds=1)
    app = create_app(settings=settings, engine_factory=lambda: SlowFailingEngine("adapted"))
    async with app_client(app) as client:
        response = await client.post("/v1/triage", json={"incident_text": "設備狀態無法確認"})
        assert response.status_code == 200
        body = response.json()
        assert body["output"]["event_type"] == "unknown"
        assert body["output"]["review_required"] is True
        assert "internal error" not in response.text


async def test_timed_out_worker_keeps_concurrency_slot_until_it_finishes():
    engine = BlockingEngine("adapted")
    settings = Settings(request_timeout_seconds=0.01, max_concurrency=1)
    app = create_app(settings=settings, engine_factory=lambda: engine)
    async with app_client(app) as client:
        first = await client.post(
            "/v1/triage", json={"incident_text": "設備持續異音但背景工作尚未結束"}
        )
        assert "TIMEOUT_FALLBACK" in first.json()["validation_warnings"]
        metrics_while_timed_out = (await client.get("/metrics")).text
        assert 'modelforge_active_engine_jobs{engine="fixture-adapted"} 1.0' in (
            metrics_while_timed_out
        )
        assert 'modelforge_validation_warnings_total{code="TIMEOUT_FALLBACK"} 1.0' in (
            metrics_while_timed_out
        )

        second_task = asyncio.create_task(
            client.post("/v1/triage", json={"incident_text": "第二筆設備事件等待容量釋放"})
        )
        await asyncio.sleep(0.03)
        assert engine.calls == 1
        assert second_task.done() is False

        engine.release.set()
        second = await asyncio.wait_for(second_task, timeout=1)
        assert second.status_code == 200
        for _ in range(20):
            if engine.calls == 2:
                break
            await asyncio.sleep(0.01)
        assert engine.calls == 2
