import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.client import MiniMaxClient, UpstreamHTTPError
from app.config import load_settings
from app.translator import (
    ResponsesEventTranslator,
    UnsupportedFeatureError,
    build_response_context,
    format_sse,
    translate_anthropic_response,
    translate_responses_request,
)


def _extract_api_key(request: Request):
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "")


def _error_response(status_code, message, error_type="invalid_request_error"):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


def _write_last_request(body, request_log_path):
    if not request_log_path:
        return
    path = Path(request_log_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_app(settings_override=None, upstream_client=None):
    settings = load_settings(settings_override)
    client = upstream_client or MiniMaxClient(settings)
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/responses")
    async def responses(request: Request):
        if not settings.proxy_api_key:
            return _error_response(500, "Proxy API key is not configured", "server_error")

        if _extract_api_key(request) != settings.proxy_api_key:
            return _error_response(401, "Unauthorized", "authentication_error")

        body = await request.json()
        _write_last_request(body, settings.request_log_path)
        try:
            translated = translate_responses_request(body)
            response_context = build_response_context(body, model=translated.get("model"))
        except UnsupportedFeatureError as exc:
            return _error_response(400, str(exc), "unsupported_feature")

        stream = translated.get("stream", True)

        if not stream:
            try:
                upstream_body = await client.create_message(translated)
            except UpstreamHTTPError as exc:
                return _error_response(exc.status_code, str(exc.message), "upstream_error")
            try:
                translated_response = translate_anthropic_response(
                    upstream_body,
                    translated.get("model"),
                    response_context=response_context,
                )
            except UnsupportedFeatureError as exc:
                return _error_response(400, str(exc), "unsupported_feature")
            return JSONResponse(translated_response)

        async def event_stream():
            translator = ResponsesEventTranslator(
                model=translated.get("model"),
                response_context=response_context,
            )
            try:
                async for upstream_event in client.stream_messages(translated):
                    for event in translator.feed(upstream_event):
                        yield format_sse(event["event"], event["data"])
                for event in translator.finish():
                    yield format_sse(event["event"], event["data"])
                yield "data: [DONE]\n\n"
            except UpstreamHTTPError as exc:
                yield format_sse(
                    "error",
                    {"error": {"message": str(exc.message), "type": "upstream_error"}},
                )
                yield "data: [DONE]\n\n"
            except UnsupportedFeatureError as exc:
                yield format_sse(
                    "error",
                    {"error": {"message": str(exc), "type": "unsupported_feature"}},
                )
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type=None,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return app


app = create_app()
