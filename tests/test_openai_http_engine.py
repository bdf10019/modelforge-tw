import json

import httpx
import pytest

from modelforge_tw.engines import openai_http
from modelforge_tw.engines.openai_http import OpenAIHTTPInferenceEngine
from modelforge_tw.schemas import TriageRequest

ORIGINAL_HTTPX_CLIENT = httpx.Client


class ChunkStream(httpx.SyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks


def make_engine(
    base_url: str = "http://model.test/",
    *,
    max_response_bytes: int = 262144,
) -> OpenAIHTTPInferenceEngine:
    return OpenAIHTTPInferenceEngine(
        base_url=base_url,
        model_id="model-q8",
        model_version="v5",
        quantization="q8_0",
        prompt_version="optimized-v2",
        timeout_seconds=5,
        max_tokens=192,
        context_window_tokens=2048,
        max_response_bytes=max_response_bytes,
    )


def valid_output_json() -> str:
    return json.dumps(
        {
            "event_type": "mechanical",
            "severity": "high",
            "summary_zh_tw": "泵浦持續異音，需要人工複核。",
            "evidence_spans": ["持續異音"],
            "missing_information": [],
            "safe_actions": ["isolate_and_escalate"],
            "review_required": True,
            "reason_codes": ["HIGH_SEVERITY"],
        },
        ensure_ascii=False,
    )


def install_transport(monkeypatch, handler) -> None:
    transport = httpx.MockTransport(handler)

    def direct_client(**kwargs):
        assert kwargs["trust_env"] is False
        return ORIGINAL_HTTPX_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(
        openai_http.httpx,
        "Client",
        direct_client,
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "file:///models/server",
        "http://user:password@model.test",
        "http://model.test?target=other",
        "http://model.test/#fragment",
        "http://model.test:invalid-port",
    ],
)
def test_model_server_base_url_rejects_ambiguous_or_credentialed_values(base_url):
    with pytest.raises(ValueError, match="base URL"):
        make_engine(base_url)


def test_model_response_limit_rejects_unsafe_configuration():
    with pytest.raises(ValueError, match="between 4096 and 4194304"):
        make_engine(max_response_bytes=4194305)


def test_model_server_defaults_to_attested_generation_budget():
    engine = OpenAIHTTPInferenceEngine(
        base_url="http://model.test",
        model_id="model-q8",
        model_version="v5",
        quantization="q8_0",
        prompt_version="optimized-v2",
    )

    assert engine.max_tokens == 192


def test_ready_is_fail_closed_for_status_and_connection_errors(monkeypatch):
    engine = make_engine()

    install_transport(monkeypatch, lambda request: httpx.Response(503, request=request))
    assert engine.ready() is False

    def disconnected(request):
        raise httpx.ConnectError("dependency unavailable", request=request)

    install_transport(monkeypatch, disconnected)
    assert engine.ready() is False


def test_non_streaming_generation_sends_strict_contract_and_records_usage(monkeypatch):
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": valid_output_json()}}],
                "usage": {"prompt_tokens": 87, "completion_tokens": 41},
            },
        )

    install_transport(monkeypatch, handler)
    result = make_engine().generate(TriageRequest(incident_text="泵浦持續異音且趨勢尚未停止。"))

    assert captured["model"] == "model-q8"
    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 192
    assert captured["stream"] is False
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert result.output.event_type == "mechanical"
    assert result.output.severity == "high"
    assert result.metadata.prompt_tokens == 87
    assert result.metadata.output_tokens == 41
    assert result.metadata.context_window_tokens == 2048
    assert result.metadata.safety_policy_version == "risk-precedence-v2"
    assert result.metadata.evidence_kind == "measured"
    assert result.metadata.generation_tokens_per_second is not None


def test_streaming_generation_uses_first_content_and_final_usage(monkeypatch):
    raw = valid_output_json()
    split_at = len(raw) // 2
    events = [
        {"choices": [{"delta": {"content": raw[:split_at]}}]},
        {"choices": [{"delta": {"content": raw[split_at:]}}]},
        {"choices": [], "usage": {"prompt_tokens": 90, "completion_tokens": 39}},
    ]
    stream_body = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)
    stream_body += "data: [DONE]\n\n"

    def handler(request):
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return httpx.Response(200, request=request, text=stream_body)

    install_transport(monkeypatch, handler)
    result = make_engine().generate_streaming(
        TriageRequest(incident_text="泵浦持續異音且趨勢尚未停止。")
    )

    assert result.raw_text == raw
    assert result.metadata.prompt_tokens == 90
    assert result.metadata.output_tokens == 39
    assert result.metadata.ttft_ms is not None
    assert result.metadata.ttft_ms <= result.metadata.elapsed_ms


def test_non_streaming_response_is_rejected_before_oversized_body_is_loaded(monkeypatch):
    engine = make_engine()
    engine.max_response_bytes = 64

    install_transport(
        monkeypatch,
        lambda request: httpx.Response(200, request=request, content=b"x" * 65),
    )

    with pytest.raises(ValueError, match="exceeds byte limit"):
        engine.generate(TriageRequest(incident_text="泵浦持續異音且趨勢尚未停止。"))


def test_chunked_non_streaming_response_stops_at_actual_byte_limit(monkeypatch):
    engine = make_engine()
    engine.max_response_bytes = 64

    install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            request=request,
            stream=ChunkStream(b"x" * 32, b"x" * 33),
        ),
    )

    with pytest.raises(ValueError, match="exceeds byte limit"):
        engine.generate(TriageRequest(incident_text="泵浦持續異音且趨勢尚未停止。"))


def test_model_response_rejects_invalid_declared_length(monkeypatch):
    engine = make_engine()
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            request=request,
            headers={"Content-Length": "invalid"},
            stream=ChunkStream(b"{}"),
        ),
    )

    with pytest.raises(ValueError, match="invalid Content-Length"):
        engine.generate(TriageRequest(incident_text="泵浦持續異音且趨勢尚未停止。"))


def test_streaming_response_rejects_oversized_unterminated_event(monkeypatch):
    engine = make_engine()
    engine.max_response_bytes = 64

    install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            request=request,
            stream=ChunkStream(b"data: ", b"x" * 59),
        ),
    )

    with pytest.raises(ValueError, match="exceeds byte limit"):
        engine.generate_streaming(TriageRequest(incident_text="泵浦持續異音且趨勢尚未停止。"))


def test_streaming_response_accepts_valid_final_event_without_newline(monkeypatch):
    raw = valid_output_json()
    event = json.dumps(
        {"choices": [{"delta": {"content": raw}}]},
        ensure_ascii=False,
    ).encode()
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            request=request,
            stream=ChunkStream(b"data: " + event),
        ),
    )

    result = make_engine().generate_streaming(
        TriageRequest(incident_text="泵浦持續異音且趨勢尚未停止。")
    )

    assert result.raw_text == raw


def test_streaming_response_rejects_invalid_utf8(monkeypatch):
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            request=request,
            stream=ChunkStream(b"data: \xff\n"),
        ),
    )

    with pytest.raises(ValueError, match="not valid UTF-8"):
        make_engine().generate_streaming(
            TriageRequest(incident_text="泵浦持續異音且趨勢尚未停止。")
        )


def test_invalid_model_output_returns_auditable_safe_fallback():
    result = make_engine()._build_result(
        request=TriageRequest(incident_text="泵浦狀態不明且沒有可驗證的量測資料。"),
        raw='{"event_type":',
        elapsed_ms=10,
        usage={},
    )

    assert result.output.event_type == "unknown"
    assert result.output.review_required is True
    assert result.output.safe_actions == ["collect_evidence"]
    assert "MODEL_OUTPUT_INVALID" in result.validation_warnings
    assert "SAFE_FALLBACK_APPLIED" in result.validation_warnings
