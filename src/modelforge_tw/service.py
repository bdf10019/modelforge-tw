from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from modelforge_tw.engines import FixtureEngine, InferenceEngine, OpenAIHTTPInferenceEngine
from modelforge_tw.metrics import ServiceMetrics
from modelforge_tw.safety import conservative_fallback
from modelforge_tw.schemas import InferenceMetadata, TriageRequest, TriageResponse
from modelforge_tw.settings import Settings

REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,96}$")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str = Field(min_length=1, max_length=4000)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    temperature: float = 0
    max_tokens: int = Field(default=256, ge=1, le=512)
    stream: bool = False


def build_engine(settings: Settings) -> InferenceEngine:
    if settings.engine in {"fixture", "fixture-adapted"}:
        return FixtureEngine("adapted")
    if settings.engine == "fixture-base":
        return FixtureEngine("base")
    if settings.engine == "fixture-prompt":
        return FixtureEngine("prompt")
    if settings.engine == "llama-http":
        return OpenAIHTTPInferenceEngine(
            base_url=settings.llama_base_url,
            model_id=settings.model_id,
            model_version=settings.model_version,
            quantization=settings.quantization,
            prompt_version=settings.prompt_version,
            timeout_seconds=settings.request_timeout_seconds,
            max_tokens=settings.max_new_tokens,
        )
    if settings.engine == "transformers":
        from modelforge_tw.engines.transformers_local import TransformersInferenceEngine

        return TransformersInferenceEngine(
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            adapter_path=settings.adapter_path,
            prompt_version=settings.prompt_version,
            model_version=settings.model_version,
            load_in_4bit=settings.load_in_4bit,
            max_new_tokens=settings.max_new_tokens,
        )
    raise ValueError(f"unsupported engine: {settings.engine}")


def create_app(
    *,
    settings: Settings | None = None,
    engine_factory: Callable[[], InferenceEngine] | None = None,
) -> FastAPI:
    selected_settings = settings or Settings()
    metrics = ServiceMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = engine_factory() if engine_factory else build_engine(selected_settings)
        app.state.semaphore = asyncio.Semaphore(selected_settings.max_concurrency)
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(
        title="ModelForge-TW",
        version="0.1.0",
        description="Validated Traditional Chinese industrial-event model serving",
        lifespan=lifespan,
    )
    web_root = files("modelforge_tw").joinpath("web")
    app.mount("/static", StaticFiles(directory=str(web_root)), name="static")

    def resolve_request_id(value: str | None) -> str:
        if value is not None and not REQUEST_ID.fullmatch(value):
            raise HTTPException(status_code=400, detail="invalid X-Request-ID")
        return value or str(uuid.uuid4())

    async def infer(request: TriageRequest, request_id: str) -> TriageResponse:
        engine: InferenceEngine = app.state.engine
        started = time.perf_counter()
        metrics.inflight.inc()
        try:
            async with app.state.semaphore:
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(engine.generate, request),
                        timeout=selected_settings.request_timeout_seconds,
                    )
                    outcome = "success"
                    for warning in result.validation_warnings:
                        metrics.validation_warnings.labels(code=warning).inc()
                    return TriageResponse(
                        request_id=request_id,
                        output=result.output,
                        metadata=result.metadata,
                        validation_warnings=result.validation_warnings,
                    )
                except TimeoutError:
                    outcome = "timeout_fallback"
                    fallback = conservative_fallback("MODEL_TIMEOUT")
                except Exception:
                    outcome = "error_fallback"
                    fallback = conservative_fallback("MODEL_INFERENCE_ERROR")
                return TriageResponse(
                    request_id=request_id,
                    output=fallback,
                    metadata=InferenceMetadata(
                        engine=engine.name,
                        model_id=engine.model_id,
                        model_version=engine.model_version,
                        quantization=engine.quantization,
                        prompt_version=engine.prompt_version,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        max_new_tokens=engine.max_new_tokens,
                        structured_output_contract=engine.structured_output_contract,
                        evidence_kind=engine.evidence_kind,
                    ),
                    validation_warnings=[outcome.upper()],
                )
        finally:
            metrics.inflight.dec()
            elapsed = time.perf_counter() - started
            metrics.requests.labels(
                outcome=locals().get("outcome", "unknown"), engine=engine.name
            ).inc()
            metrics.latency.labels(engine=engine.name).observe(elapsed)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(web_root.joinpath("index.html").read_text(encoding="utf-8"))

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready() -> dict[str, Any]:
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="model engine is not ready")
        engine: InferenceEngine = app.state.engine
        if not await asyncio.to_thread(engine.ready):
            raise HTTPException(status_code=503, detail="model engine dependency is not ready")
        return {
            "status": "ready",
            "engine": engine.name,
            "model_id": engine.model_id,
            "evidence_kind": engine.evidence_kind,
        }

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        engine: InferenceEngine = app.state.engine
        return {
            "engine": engine.name,
            "model_id": engine.model_id,
            "model_version": engine.model_version,
            "quantization": engine.quantization,
            "prompt_version": engine.prompt_version,
            "evidence_kind": engine.evidence_kind,
            "max_concurrency": selected_settings.max_concurrency,
            "input_logging": selected_settings.log_inputs,
        }

    @app.post("/v1/triage", response_model=TriageResponse)
    async def triage(
        payload: TriageRequest,
        x_request_id: str | None = Header(default=None),
    ) -> TriageResponse:
        return await infer(payload, resolve_request_id(x_request_id))

    @app.post("/v1/chat/completions")
    async def chat(payload: ChatCompletionRequest, request: Request) -> Response:
        user_messages = [message.content for message in payload.messages if message.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=422, detail="at least one user message is required")
        triage_request = TriageRequest(incident_text=user_messages[-1])
        request_id = resolve_request_id(request.headers.get("X-Request-ID"))
        result = await infer(triage_request, request_id)
        content = json.dumps(result.output.model_dump(mode="json"), ensure_ascii=False)
        body: dict[str, Any] = {
            "id": f"chatcmpl-{result.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.metadata.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        if result.metadata.prompt_tokens is not None and result.metadata.output_tokens is not None:
            body["usage"] = {
                "prompt_tokens": result.metadata.prompt_tokens or 0,
                "completion_tokens": result.metadata.output_tokens or 0,
                "total_tokens": (result.metadata.prompt_tokens or 0)
                + (result.metadata.output_tokens or 0),
            }
        if not payload.stream:
            return Response(
                content=json.dumps(body, ensure_ascii=False), media_type="application/json"
            )

        async def events() -> AsyncIterator[str]:
            chunk = {
                "id": body["id"],
                "object": "chat.completion.chunk",
                "created": body["created"],
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            done = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    return app


app = create_app()
