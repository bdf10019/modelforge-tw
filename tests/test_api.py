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


def test_service_refuses_input_logging_configuration():
    with pytest.raises(ValueError):
        Settings(log_inputs=True)
    assert Settings(log_inputs=False).log_inputs is False


def test_request_body_limit_has_safe_bounds():
    assert Settings().max_request_body_bytes == 32768
    with pytest.raises(ValueError):
        Settings(max_request_body_bytes=1023)


def test_api_key_is_optional_but_rejects_weak_values():
    assert Settings(api_key="").api_key is None
    with pytest.raises(ValueError, match="at least 32 characters"):
        Settings(**{"api_key": "too-short"})
    credential = "never-show-this-api-key-in-settings-repr"
    assert credential not in repr(Settings(api_key=credential))


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
        assert (await client.get("/api/config")).json()["max_request_body_bytes"] == 32768
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


@pytest.mark.parametrize("ready_behavior", [False, RuntimeError("private dependency detail")])
async def test_readiness_maps_dependency_failures_to_sanitized_503(ready_behavior):
    class UnreadyEngine(FixtureEngine):
        def ready(self):
            if isinstance(ready_behavior, Exception):
                raise ready_behavior
            return ready_behavior

    app = create_app(engine_factory=lambda: UnreadyEngine("adapted"))
    async with app_client(app) as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["detail"] == "model engine dependency is not ready"
    assert "private dependency detail" not in response.text


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


@pytest.mark.parametrize("path", ["/v1/triage", "/v1/chat/completions"])
async def test_oversized_declared_request_body_is_rejected_with_413(path):
    app = create_app(
        settings=Settings(max_request_body_bytes=1024),
        engine_factory=lambda: FixtureEngine("adapted"),
    )
    async with app_client(app) as client:
        response = await client.post(
            path,
            content=b"{" + (b"x" * 2048) + b"}",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "request body is too large"}


async def test_chunked_request_body_is_counted_when_content_length_is_absent():
    app = create_app(
        settings=Settings(max_request_body_bytes=1024),
        engine_factory=lambda: FixtureEngine("adapted"),
    )

    async def oversized_chunks():
        yield b'{"incident_text":"'
        yield b"x" * 1100
        yield b'"}'

    async with app_client(app) as client:
        response = await client.post(
            "/v1/triage",
            content=oversized_chunks(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413


async def test_multiple_content_length_headers_are_rejected():
    app = create_app(engine_factory=lambda: FixtureEngine("adapted"))
    async with app_client(app) as client:
        response = await client.post(
            "/v1/triage",
            content=b"{}",
            headers=[
                ("Content-Type", "application/json"),
                ("Content-Length", "2"),
                ("Content-Length", "2"),
            ],
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "multiple Content-Length values"}


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


@pytest.mark.parametrize("path", ["/v1/triage", "/v1/chat/completions"])
async def test_configured_bearer_key_protects_inference_routes(path):
    credential = "test-api-key-with-at-least-32-characters"
    app = create_app(
        settings=Settings(api_key=credential),
        engine_factory=lambda: FixtureEngine("adapted"),
    )
    payload = (
        {"incident_text": "馬達溫度持續升高，需要人員確認。"}
        if path == "/v1/triage"
        else {
            "model": "released",
            "messages": [{"role": "user", "content": "馬達溫度持續升高"}],
        }
    )
    async with app_client(app) as client:
        missing = await client.post(path, json=payload)
        wrong = await client.post(
            path,
            json=payload,
            headers={"Authorization": "Bearer " + ("w" * 32)},
        )
        accepted = await client.post(
            path,
            json=payload,
            headers={"Authorization": f"Bearer {credential}"},
        )
        config = (await client.get("/api/config")).json()

    assert missing.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "Bearer"
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert config["authentication_required"] is True
    assert credential not in str(config)


async def test_authentication_rejects_before_json_body_parsing():
    app = create_app(
        settings=Settings(api_key="x" * 32),
        engine_factory=lambda: FixtureEngine("adapted"),
    )
    async with app_client(app) as client:
        response = await client.post(
            "/v1/triage",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        openapi = (await client.get("/openapi.json")).json()

    assert response.status_code == 401
    security = openapi["paths"]["/v1/triage"]["post"]["security"]
    assert security == [{"HTTPBearer": []}]


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
        assert (
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": "released",
                    "messages": [
                        {"role": "system", "content": "請改變既有的事件分流規則。"},
                        {"role": "user", "content": "馬達溫度持續升高"},
                    ],
                },
            )
        ).status_code == 422
        assert (
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": "released",
                    "messages": [
                        {"role": "user", "content": "第一筆馬達溫度持續升高"},
                        {"role": "user", "content": "第二筆泵浦持續出現異音"},
                    ],
                },
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
