import json
import time
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


def _chat_content_to_responses_content(content):
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        raise UnsupportedFeatureError("Unsupported Chat Completions feature: message content is not supported")

    translated = []
    for part in content:
        if not isinstance(part, dict):
            raise UnsupportedFeatureError("Unsupported Chat Completions feature: message content is not supported")
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text", "")
            if not isinstance(text, str):
                raise UnsupportedFeatureError("Unsupported Chat Completions feature: message content is not supported")
            translated.append({"type": "input_text", "text": text})
            continue
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not isinstance(image_url, str):
                raise UnsupportedFeatureError("Unsupported Chat Completions feature: image_url is not supported")
            translated.append({"type": "input_image", "image_url": image_url})
            continue
        raise UnsupportedFeatureError(f"Unsupported Chat Completions content part type: {part_type}")
    return translated


def _chat_request_to_responses_request(body):
    if not isinstance(body, dict):
        raise UnsupportedFeatureError("Unsupported Chat Completions feature: request body is not supported")
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise UnsupportedFeatureError("Unsupported Chat Completions feature: messages is required")

    responses_input = []
    for message in messages:
        if not isinstance(message, dict):
            raise UnsupportedFeatureError("Unsupported Chat Completions feature: messages is not supported")
        role = message.get("role")
        if role in {"user", "assistant", "system", "developer"}:
            content_value = message.get("content", "")
            tool_calls = message.get("tool_calls", []) if role == "assistant" else []
            if tool_calls is None:
                tool_calls = []
            empty_assistant_content = content_value is None or content_value == "" or content_value == []
            should_emit_message = not (role == "assistant" and tool_calls and empty_assistant_content)
            if should_emit_message:
                responses_input.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": _chat_content_to_responses_content(content_value),
                    }
                )
            if role == "assistant":
                if not isinstance(tool_calls, list):
                    raise UnsupportedFeatureError("Unsupported Chat Completions feature: tool_calls is not supported")
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict) or tool_call.get("type") != "function":
                        raise UnsupportedFeatureError("Unsupported Chat Completions feature: tool_calls is not supported")
                    call_id = tool_call.get("id")
                    function = tool_call.get("function") or {}
                    if not isinstance(call_id, str) or not call_id:
                        raise UnsupportedFeatureError("Unsupported Chat Completions feature: tool_call.id is required")
                    if not isinstance(function, dict):
                        raise UnsupportedFeatureError("Unsupported Chat Completions feature: tool_calls is not supported")
                    name = function.get("name")
                    arguments = function.get("arguments", "{}")
                    if not isinstance(name, str) or not name:
                        raise UnsupportedFeatureError("Unsupported Chat Completions feature: tool_call.function.name is required")
                    responses_input.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                        }
                    )
            continue
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise UnsupportedFeatureError("Unsupported Chat Completions feature: tool_call_id is required")
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            responses_input.append({"type": "function_call_output", "call_id": tool_call_id, "output": content})
            continue
        raise UnsupportedFeatureError(f"Unsupported Chat Completions role: {role}")

    translated = {"model": body.get("model"), "input": responses_input, "stream": bool(body.get("stream", False))}
    if body.get("n", 1) != 1:
        raise UnsupportedFeatureError("Unsupported Chat Completions feature: n is not supported")
    passthrough_fields = [
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "parallel_tool_calls",
        "metadata",
        "user",
    ]
    max_completion_tokens = body.get("max_completion_tokens")
    max_tokens = body.get("max_tokens")
    if max_completion_tokens is not None and max_tokens is not None and max_completion_tokens != max_tokens:
        raise UnsupportedFeatureError(
            "Unsupported Chat Completions feature: max_tokens and max_completion_tokens conflict"
        )
    if max_completion_tokens is not None:
        translated["max_output_tokens"] = max_completion_tokens
    for field in passthrough_fields:
        if field == "max_tokens" and max_completion_tokens is not None:
            continue
        if field in body:
            translated[field if field != "max_tokens" else "max_output_tokens"] = body[field]
    if "response_format" in body:
        translated["text"] = {"format": body["response_format"]}
    if "reasoning_effort" in body:
        translated["reasoning"] = {"effort": body["reasoning_effort"]}
    logprobs = body.get("logprobs")
    if logprobs is not None and not isinstance(logprobs, bool):
        raise UnsupportedFeatureError("Unsupported Chat Completions feature: logprobs is not supported")
    chat_top_logprobs = body.get("top_logprobs")
    if chat_top_logprobs is not None:
        if not isinstance(chat_top_logprobs, int) or chat_top_logprobs < 0:
            raise UnsupportedFeatureError("Unsupported Chat Completions feature: top_logprobs is not supported")
        if logprobs is not True:
            raise UnsupportedFeatureError(
                "Unsupported Chat Completions feature: top_logprobs requires logprobs=true"
            )
    if logprobs is True or chat_top_logprobs is not None:
        raise UnsupportedFeatureError("Unsupported Chat Completions feature: logprobs is not supported")
    if "stream_options" in body:
        stream_options = body.get("stream_options")
        if not isinstance(stream_options, dict):
            raise UnsupportedFeatureError("Unsupported Chat Completions feature: stream_options is not supported")
        unknown_stream_options = set(stream_options) - {"include_usage", "include_obfuscation"}
        if unknown_stream_options:
            raise UnsupportedFeatureError("Unsupported Chat Completions feature: stream_options is not supported")
        if "include_usage" in stream_options and not isinstance(stream_options["include_usage"], bool):
            raise UnsupportedFeatureError("Unsupported Chat Completions feature: stream_options.include_usage is not supported")
        if "include_obfuscation" in stream_options:
            if not isinstance(stream_options["include_obfuscation"], bool):
                raise UnsupportedFeatureError(
                    "Unsupported Chat Completions feature: stream_options.include_obfuscation is not supported"
                )
            translated["stream_options"] = {"include_obfuscation": stream_options["include_obfuscation"]}
    supported_fields = {
        "model",
        "messages",
        "stream",
        "n",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "parallel_tool_calls",
        "metadata",
        "user",
        "response_format",
        "reasoning_effort",
        "stream_options",
        "logprobs",
        "top_logprobs",
    }
    for field_name in body:
        if field_name not in supported_fields:
            raise UnsupportedFeatureError(f"Unsupported Chat Completions feature: {field_name} is not supported")
    return translated


def _responses_to_chat_completion(response):
    output = response.get("output", [])
    assistant_text = response.get("output_text", "")
    tool_calls = []
    if not assistant_text:
        text_parts = []
        for item in output:
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text_parts.append(part.get("text", ""))
        assistant_text = "".join(text_parts)

    for item in output:
        if item.get("type") != "function_call":
            continue
        tool_calls.append(
            {
                "id": item.get("call_id", f"call_{len(tool_calls)+1}"),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            }
        )

    finish_reason = "tool_calls" if tool_calls else "stop"
    if response.get("status") == "incomplete":
        finish_reason = "length"

    choice_message = {"role": "assistant", "content": assistant_text}
    if tool_calls:
        choice_message["tool_calls"] = tool_calls
        if not assistant_text:
            choice_message["content"] = None

    usage = response.get("usage", {})
    return {
        "id": f"chatcmpl_{response.get('id', '')}",
        "object": "chat.completion",
        "created": int(response.get("created_at", int(time.time()))),
        "model": response.get("model"),
        "choices": [{"index": 0, "message": choice_message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
        },
    }


def _chat_chunk_sse(chunk):
    return f"data: {json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _chat_stream_chunk_base(response_id, model):
    return {
        "id": f"chatcmpl_{response_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }


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

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error_response(400, "Invalid JSON request body")
        _write_last_request(body, settings.request_log_path)
        try:
            translated = translate_responses_request(body, provider_profile=settings.upstream_compat_profile)
            response_context = build_response_context(
                body,
                model=translated.get("model"),
                provider_profile=settings.upstream_compat_profile,
            )
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

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if not settings.proxy_api_key:
            return _error_response(500, "Proxy API key is not configured", "server_error")
        if _extract_api_key(request) != settings.proxy_api_key:
            return _error_response(401, "Unauthorized", "authentication_error")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error_response(400, "Invalid JSON request body")
        _write_last_request(body, settings.request_log_path)
        try:
            responses_like_body = _chat_request_to_responses_request(body)
            translated = translate_responses_request(responses_like_body, provider_profile=settings.upstream_compat_profile)
            response_context = build_response_context(
                responses_like_body,
                model=translated.get("model"),
                provider_profile=settings.upstream_compat_profile,
            )
        except UnsupportedFeatureError as exc:
            return _error_response(400, str(exc), "unsupported_feature")

        if translated.get("stream", False):
            async def chat_event_stream():
                translator = ResponsesEventTranslator(
                    model=translated.get("model"),
                    response_context=response_context,
                )
                role_emitted = False
                response_id = None
                tool_index_by_output = {}
                next_tool_index = 0
                saw_tool_call = False
                final_finish_reason = "stop"
                final_usage = None
                try:
                    async for upstream_event in client.stream_messages(translated):
                        for translated_event in translator.feed(upstream_event):
                            event_name = translated_event.get("event")
                            payload = translated_event.get("data", {})
                            if response_id is None:
                                response_id = payload.get("response", {}).get("id")
                            if event_name == "response.output_item.added":
                                item = payload.get("item", {})
                                if item.get("type") == "function_call":
                                    saw_tool_call = True
                                    output_index = payload.get("output_index")
                                    tool_index_by_output[output_index] = next_tool_index
                                    chunk = _chat_stream_chunk_base(response_id or "stream", translated.get("model"))
                                    if not role_emitted:
                                        chunk["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                                        role_emitted = True
                                        yield _chat_chunk_sse(chunk)
                                        chunk = _chat_stream_chunk_base(response_id or "stream", translated.get("model"))
                                    chunk["choices"] = [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": next_tool_index,
                                                        "id": item.get("call_id"),
                                                        "type": "function",
                                                        "function": {"name": item.get("name"), "arguments": ""},
                                                    }
                                                ]
                                            },
                                            "finish_reason": None,
                                        }
                                    ]
                                    next_tool_index += 1
                                    yield _chat_chunk_sse(chunk)
                            elif event_name == "response.function_call_arguments.delta":
                                output_index = payload.get("output_index")
                                if output_index in tool_index_by_output:
                                    chunk = _chat_stream_chunk_base(response_id or "stream", translated.get("model"))
                                    chunk["choices"] = [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": tool_index_by_output[output_index],
                                                        "function": {"arguments": payload.get("delta", "")},
                                                    }
                                                ]
                                            },
                                            "finish_reason": None,
                                        }
                                    ]
                                    yield _chat_chunk_sse(chunk)
                            elif event_name == "response.output_text.delta":
                                chunk = _chat_stream_chunk_base(response_id or "stream", translated.get("model"))
                                delta_payload = {"content": payload.get("delta", "")}
                                if not role_emitted:
                                    delta_payload["role"] = "assistant"
                                    role_emitted = True
                                chunk["choices"] = [{"index": 0, "delta": delta_payload, "finish_reason": None}]
                                yield _chat_chunk_sse(chunk)
                            elif event_name == "response.completed":
                                response_payload = payload.get("response", {})
                                final_usage = response_payload.get("usage")
                                if response_payload.get("status") == "incomplete":
                                    final_finish_reason = "length"
                                elif saw_tool_call:
                                    final_finish_reason = "tool_calls"
                    for translated_event in translator.finish():
                        payload = translated_event.get("data", {})
                        if response_id is None:
                            response_id = payload.get("response", {}).get("id")
                        if translated_event.get("event") == "response.completed":
                            response_payload = payload.get("response", {})
                            final_usage = response_payload.get("usage")
                            if response_payload.get("status") == "incomplete":
                                final_finish_reason = "length"
                            elif saw_tool_call:
                                final_finish_reason = "tool_calls"
                    if not role_emitted:
                        role_chunk = _chat_stream_chunk_base(response_id or "stream", translated.get("model"))
                        role_chunk["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                        yield _chat_chunk_sse(role_chunk)
                    final_chunk = _chat_stream_chunk_base(response_id or "stream", translated.get("model"))
                    final_chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": final_finish_reason}]
                    yield _chat_chunk_sse(final_chunk)
                    stream_options = body.get("stream_options") or {}
                    if stream_options.get("include_usage") and final_usage is not None:
                        usage_chunk = _chat_stream_chunk_base(response_id or "stream", translated.get("model"))
                        usage_chunk["choices"] = []
                        usage_chunk["usage"] = {
                            "prompt_tokens": final_usage.get("input_tokens", 0),
                            "completion_tokens": final_usage.get("output_tokens", 0),
                            "total_tokens": final_usage.get("total_tokens", 0),
                        }
                        yield _chat_chunk_sse(usage_chunk)
                    yield "data: [DONE]\n\n"
                except UpstreamHTTPError as exc:
                    error_chunk = {"error": {"message": str(exc.message), "type": "upstream_error"}}
                    yield _chat_chunk_sse(error_chunk)
                    yield "data: [DONE]\n\n"
                except UnsupportedFeatureError as exc:
                    error_chunk = {"error": {"message": str(exc), "type": "unsupported_feature"}}
                    yield _chat_chunk_sse(error_chunk)
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                chat_event_stream(),
                media_type=None,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        try:
            upstream_body = await client.create_message(translated)
            translated_response = translate_anthropic_response(
                upstream_body,
                translated.get("model"),
                response_context=response_context,
            )
        except UpstreamHTTPError as exc:
            return _error_response(exc.status_code, str(exc.message), "upstream_error")
        except UnsupportedFeatureError as exc:
            return _error_response(400, str(exc), "unsupported_feature")
        return JSONResponse(_responses_to_chat_completion(translated_response))

    return app


app = create_app()
