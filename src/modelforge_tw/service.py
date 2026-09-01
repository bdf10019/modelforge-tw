from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from modelforge_tw.engines import FixtureEngine, InferenceEngine, OpenAIHTTPInferenceEngine
from modelforge_tw.metrics import ServiceMetrics
from modelforge_tw.registry import FileRegistry
from modelforge_tw.safety import conservative_fallback
from modelforge_tw.schemas import InferenceMetadata, TriageRequest, TriageResponse
from modelforge_tw.settings import Settings

REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,96}$")
HTTP_TOKEN = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$")
BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
PROTECTED_API_PATHS = frozenset({"/v1/triage", "/v1/chat/completions"})
NO_STORE_PATHS = PROTECTED_API_PATHS | {"/", "/api/config"}
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
    "style-src 'self'; script-src 'self'; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'"
)


def _set_response_header(
    headers: list[tuple[bytes, bytes]], name: bytes, value: bytes
) -> list[tuple[bytes, bytes]]:
    return [(key, item) for key, item in headers if key.lower() != name] + [(name, value)]


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers = _set_response_header(headers, b"x-content-type-options", b"nosniff")
                headers = _set_response_header(headers, b"x-frame-options", b"DENY")
                headers = _set_response_header(headers, b"referrer-policy", b"no-referrer")
                headers = _set_response_header(
                    headers,
                    b"permissions-policy",
                    b"camera=(), microphone=(), geolocation=()",
                )
                if scope["path"] in NO_STORE_PATHS:
                    headers = _set_response_header(headers, b"cache-control", b"no-store")
                if scope["path"] == "/":
                    headers = _set_response_header(
                        headers,
                        b"content-security-policy",
                        CONTENT_SECURITY_POLICY.encode(),
                    )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        values = [value for name, value in scope["headers"] if name.lower() == b"x-request-id"]
        request_id: str | None = None
        if len(values) == 1:
            try:
                candidate = values[0].decode("ascii")
            except UnicodeDecodeError:
                candidate = ""
            if REQUEST_ID.fullmatch(candidate):
                request_id = candidate
        if len(values) > 1 or (values and request_id is None):
            rejection_id = str(uuid.uuid4())
            await _send_json_error(
                send,
                400,
                "invalid X-Request-ID",
                extra_headers=[(b"x-request-id", rejection_id.encode())],
            )
            return

        request_id = request_id or str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = _set_response_header(
                    list(message.get("headers", [])),
                    b"x-request-id",
                    request_id.encode(),
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_request_id)


async def _send_json_error(
    send: Send,
    status: int,
    detail: str,
    *,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    content = json.dumps({"detail": detail}, separators=(",", ":")).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(content)).encode()),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": content})


def _valid_bearer(authorization: str | None, expected: str) -> bool:
    scheme, separator, provided = (authorization or "").partition(" ")
    return (
        bool(separator)
        and scheme.lower() == "bearer"
        and secrets.compare_digest(provided, expected)
    )


class BearerAuthMiddleware:
    def __init__(self, app: ASGIApp, *, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            self.api_key is None
            or scope["type"] != "http"
            or scope["path"] not in PROTECTED_API_PATHS
        ):
            await self.app(scope, receive, send)
            return
        values = [value for name, value in scope["headers"] if name.lower() == b"authorization"]
        authorization: str | None = None
        if len(values) == 1:
            try:
                authorization = values[0].decode("ascii")
            except UnicodeDecodeError:
                pass
        if not _valid_bearer(authorization, self.api_key):
            await _send_json_error(
                send,
                401,
                "valid bearer authentication is required",
                extra_headers=[(b"www-authenticate", b"Bearer")],
            )
            return
        await self.app(scope, receive, send)


def _is_json_content_type(value: bytes) -> bool:
    try:
        content_type = value.decode("ascii")
    except UnicodeDecodeError:
        return False
    if any(
        (ord(character) < 32 and character != "\t") or ord(character) == 127
        for character in content_type
    ):
        return False

    media_type, *raw_parameters = content_type.split(";")
    media_type = media_type.strip().lower()
    if media_type != "application/json":
        category, separator, subtype = media_type.partition("/")
        if (
            not separator
            or category != "application"
            or not subtype.endswith("+json")
            or not subtype.removesuffix("+json")
            or not HTTP_TOKEN.fullmatch(subtype)
        ):
            return False

    parameter_names: set[str] = set()
    for raw_parameter in raw_parameters:
        name, separator, raw_value = raw_parameter.strip().partition("=")
        name = name.strip().lower()
        raw_value = raw_value.strip()
        if (
            not separator
            or not HTTP_TOKEN.fullmatch(name)
            or name in parameter_names
            or not raw_value
        ):
            return False
        parameter_names.add(name)
        if raw_value.startswith('"') or raw_value.endswith('"'):
            if len(raw_value) < 2 or not (raw_value.startswith('"') and raw_value.endswith('"')):
                return False
            value_without_quotes = raw_value[1:-1]
            if '"' in value_without_quotes or "\\" in value_without_quotes:
                return False
        else:
            if not HTTP_TOKEN.fullmatch(raw_value):
                return False
            value_without_quotes = raw_value
        if not value_without_quotes or any(
            ord(character) < 32 or ord(character) == 127 for character in value_without_quotes
        ):
            return False
        if name == "charset" and value_without_quotes.lower() not in {"utf-8", "utf8"}:
            return False
    return True


class JSONContentTypeMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] not in BODY_METHODS
            or scope["path"] not in PROTECTED_API_PATHS
        ):
            await self.app(scope, receive, send)
            return

        values = [value for name, value in scope["headers"] if name.lower() == b"content-type"]
        if len(values) > 1:
            await _send_json_error(send, 400, "multiple Content-Type values")
            return
        if not values or not _is_json_content_type(values[0]):
            await _send_json_error(send, 415, "Content-Type must be application/json")
            return

        content_encodings = [
            value for name, value in scope["headers"] if name.lower() == b"content-encoding"
        ]
        if len(content_encodings) > 1:
            await _send_json_error(send, 400, "multiple Content-Encoding values")
            return
        if content_encodings:
            try:
                content_encoding = content_encodings[0].decode("ascii").strip().lower()
            except UnicodeDecodeError:
                content_encoding = ""
            if content_encoding != "identity":
                await _send_json_error(send, 415, "request Content-Encoding is not supported")
                return
        await self.app(scope, receive, send)


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in BODY_METHODS:
            await self.app(scope, receive, send)
            return

        declared_lengths: list[int] = []
        has_transfer_encoding = False
        for name, value in scope["headers"]:
            if name.lower() == b"transfer-encoding":
                has_transfer_encoding = True
            if name.lower() != b"content-length":
                continue
            if not value.isdigit():
                await _send_json_error(send, 400, "invalid Content-Length")
                return
            declared_lengths.append(int(value))
        if len(declared_lengths) > 1:
            await _send_json_error(send, 400, "multiple Content-Length values")
            return
        if declared_lengths and has_transfer_encoding:
            await _send_json_error(send, 400, "ambiguous request body framing")
            return
        if declared_lengths and declared_lengths[0] > self.max_body_bytes:
            await _send_json_error(send, 413, "request body is too large")
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await _send_json_error(send, 413, "request body is too large")
                return
            if not message.get("more_body", False):
                break

        if declared_lengths and len(body) != declared_lengths[0]:
            await _send_json_error(send, 400, "Content-Length does not match the request body")
            return

        delivered = False

        async def replay_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=4000)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    temperature: Literal[0] = 0
    max_tokens: int | None = Field(default=None, ge=1, le=512)
    stream: bool = False

    @model_validator(mode="after")
    def validate_triage_user_message(self) -> ChatCompletionRequest:
        if len(self.messages) != 1 or self.messages[0].role != "user":
            raise ValueError("exactly one user message is supported")
        if not 8 <= len(self.messages[0].content.strip()) <= 2000:
            raise ValueError("the user message must contain 8 to 2000 characters")
        return self


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
            context_window_tokens=settings.context_window_tokens,
            max_response_bytes=settings.max_model_response_bytes,
            safety_policy_version=settings.safety_policy_version,
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
            safety_policy_version=settings.safety_policy_version,
        )
    raise ValueError(f"unsupported engine: {settings.engine}")


def validate_registry_deployment(
    settings: Settings,
    engine: InferenceEngine,
) -> str | None:
    role = settings.registry_deployment_role
    if role is None:
        return None
    manifest = FileRegistry(settings.registry_path).resolve_deployment(role)
    checks = {
        "evidence_kind": (
            engine.evidence_kind == "measured"
            and manifest.evaluation_evidence_kind == engine.evidence_kind
        ),
        "engine": manifest.evaluation_engine == engine.name,
        "model_id": manifest.evaluation_model_id == engine.model_id,
        "model_version": manifest.evaluation_model_version == engine.model_version,
        "quantization": manifest.quantization == engine.quantization,
        "prompt_version": manifest.prompt_version == engine.prompt_version,
        "max_new_tokens": manifest.max_new_tokens == engine.max_new_tokens,
        "structured_output_contract": (
            manifest.structured_output_contract == engine.structured_output_contract
        ),
        "safety_policy_version": manifest.safety_policy_version == engine.safety_policy_version,
    }
    mismatches = sorted(name for name, matched in checks.items() if not matched)
    if mismatches:
        raise ValueError("runtime does not match registry deployment: " + ", ".join(mismatches))
    return manifest.manifest_id


def create_app(
    *,
    settings: Settings | None = None,
    engine_factory: Callable[[], InferenceEngine] | None = None,
) -> FastAPI:
    selected_settings = settings or Settings()
    metrics = ServiceMetrics()
    engine_tasks: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        verified_manifest_id: str | None = None
        if (
            selected_settings.registry_deployment_role is not None
            and selected_settings.registry_artifact_root is not None
        ):
            verified_manifest_id = (
                FileRegistry(
                    selected_settings.registry_path,
                    artifact_root=selected_settings.registry_artifact_root,
                )
                .resolve_deployment(selected_settings.registry_deployment_role)
                .manifest_id
            )
        engine = engine_factory() if engine_factory else build_engine(selected_settings)
        registry_manifest_id = validate_registry_deployment(selected_settings, engine)
        if verified_manifest_id is not None and verified_manifest_id != registry_manifest_id:
            raise ValueError("registry deployment changed during startup verification")
        app.state.registry_manifest_id = registry_manifest_id
        app.state.engine = engine
        app.state.semaphore = asyncio.Semaphore(selected_settings.max_concurrency)
        app.state.queued_requests = 0
        engine_tasks.clear()
        app.state.engine_tasks = engine_tasks
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            unfinished = tuple(task for task in engine_tasks if not task.done())
            if unfinished:
                await asyncio.wait(
                    unfinished,
                    timeout=selected_settings.shutdown_grace_seconds,
                )

    app = FastAPI(
        title="ModelForge-TW",
        version="0.1.0",
        description="Validated Traditional Chinese industrial-event model serving",
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=selected_settings.max_request_body_bytes,
    )
    app.add_middleware(JSONContentTypeMiddleware)
    configured_api_key = (
        selected_settings.api_key.get_secret_value() if selected_settings.api_key else None
    )
    app.add_middleware(BearerAuthMiddleware, api_key=configured_api_key)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    web_root = files("modelforge_tw").joinpath("web")
    app.mount("/static", StaticFiles(directory=str(web_root)), name="static")

    bearer_scheme = HTTPBearer(auto_error=False)

    def require_api_key(
        credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    ) -> None:
        if configured_api_key is None:
            return
        authorization = (
            f"{credentials.scheme} {credentials.credentials}" if credentials is not None else None
        )
        if not _valid_bearer(authorization, configured_api_key):
            raise HTTPException(
                status_code=401,
                detail="valid bearer authentication is required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def infer(request: TriageRequest, request_id: str) -> TriageResponse:
        engine: InferenceEngine = app.state.engine
        started = time.perf_counter()
        metrics.inflight.inc()
        slot_acquired = False
        engine_task: asyncio.Task[Any] | None = None
        engine_job_started = False
        release_deferred = False
        queue_reserved = False
        engine_job_gauge = metrics.engine_jobs.labels(engine=engine.name)

        def release_after_background_completion(task: asyncio.Task[Any]) -> None:
            app.state.semaphore.release()
            engine_job_gauge.dec()
            if not task.cancelled():
                task.exception()

        try:
            active_jobs = sum(not task.done() for task in engine_tasks)
            admitted_limit = selected_settings.max_concurrency + selected_settings.max_queue_depth
            if active_jobs + app.state.queued_requests >= admitted_limit:
                outcome = "capacity_rejected"
                raise HTTPException(
                    status_code=503,
                    detail="inference capacity is temporarily unavailable",
                    headers={"Retry-After": "1"},
                )
            app.state.queued_requests += 1
            metrics.queued.inc()
            queue_reserved = True
            try:
                await asyncio.wait_for(
                    app.state.semaphore.acquire(),
                    timeout=selected_settings.queue_timeout_seconds,
                )
            except TimeoutError:
                outcome = "capacity_rejected"
                raise HTTPException(
                    status_code=503,
                    detail="inference capacity is temporarily unavailable",
                    headers={"Retry-After": "1"},
                ) from None
            slot_acquired = True
            engine_task = asyncio.create_task(asyncio.to_thread(engine.generate, request))
            engine_tasks.add(engine_task)
            engine_task.add_done_callback(engine_tasks.discard)
            app.state.queued_requests -= 1
            metrics.queued.dec()
            queue_reserved = False
            engine_job_gauge.inc()
            engine_job_started = True
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(engine_task),
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
                release_deferred = True
                engine_task.add_done_callback(release_after_background_completion)
            except Exception:
                outcome = "error_fallback"
                fallback = conservative_fallback("MODEL_INFERENCE_ERROR")
            metrics.validation_warnings.labels(code=outcome.upper()).inc()
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
                    safety_policy_version=engine.safety_policy_version,
                    evidence_kind=engine.evidence_kind,
                ),
                validation_warnings=[outcome.upper()],
            )
        except asyncio.CancelledError:
            outcome = "client_cancelled"
            if engine_task is not None and not release_deferred:
                release_deferred = True
                engine_task.add_done_callback(release_after_background_completion)
            raise
        finally:
            if queue_reserved:
                app.state.queued_requests -= 1
                metrics.queued.dec()
            if slot_acquired and not release_deferred:
                app.state.semaphore.release()
            if engine_job_started and not release_deferred:
                engine_job_gauge.dec()
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
        if selected_settings.registry_deployment_role is not None:
            try:
                current_manifest_id = validate_registry_deployment(selected_settings, engine)
            except Exception:
                raise HTTPException(
                    status_code=503,
                    detail="registry deployment binding is not ready",
                ) from None
            if current_manifest_id != app.state.registry_manifest_id:
                raise HTTPException(
                    status_code=503,
                    detail="registry deployment binding is not ready",
                )
        try:
            dependency_ready = await asyncio.to_thread(engine.ready)
        except Exception:
            raise HTTPException(
                status_code=503, detail="model engine dependency is not ready"
            ) from None
        if not dependency_ready:
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
            "structured_output_contract": engine.structured_output_contract,
            "safety_policy_version": engine.safety_policy_version,
            "evidence_kind": engine.evidence_kind,
            "max_concurrency": selected_settings.max_concurrency,
            "queue_timeout_seconds": selected_settings.queue_timeout_seconds,
            "max_queue_depth": selected_settings.max_queue_depth,
            "shutdown_grace_seconds": selected_settings.shutdown_grace_seconds,
            "max_request_body_bytes": selected_settings.max_request_body_bytes,
            "max_model_response_bytes": selected_settings.max_model_response_bytes,
            "authentication_required": selected_settings.api_key is not None,
            "registry_deployment_role": selected_settings.registry_deployment_role,
            "registry_manifest_id": app.state.registry_manifest_id,
            "registry_file_verification": (
                selected_settings.registry_deployment_role is not None
                and selected_settings.registry_artifact_root is not None
            ),
            "input_logging": selected_settings.log_inputs,
        }

    @app.post("/v1/triage", response_model=TriageResponse)
    async def triage(
        payload: TriageRequest,
        request: Request,
        _authentication: None = Depends(require_api_key),
    ) -> TriageResponse:
        return await infer(payload, request.state.request_id)

    @app.post("/v1/chat/completions")
    async def chat(
        payload: ChatCompletionRequest,
        request: Request,
        _authentication: None = Depends(require_api_key),
    ) -> Response:
        engine: InferenceEngine = app.state.engine
        if payload.model not in {None, "released", engine.model_id}:
            raise HTTPException(status_code=404, detail="requested model is not deployed")
        deployed_max_tokens = engine.max_new_tokens or selected_settings.max_new_tokens
        if payload.max_tokens is not None and payload.max_tokens < deployed_max_tokens:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be at least {deployed_max_tokens} for this schema",
            )
        triage_request = TriageRequest(incident_text=payload.messages[0].content)
        request_id = request.state.request_id
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
