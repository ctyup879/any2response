import base64
import json
import mimetypes
import secrets
import time
import urllib.parse
import uuid


class UnsupportedFeatureError(ValueError):
    pass


UNSUPPORTED_REQUEST_FIELDS = {
    "conversation",
    "context_management",
    "prompt",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
}


RESPONSE_COMPLETED_STOP_REASONS = {None, "end_turn", "stop_sequence", "tool_use", "refusal"}
RESPONSE_INCOMPLETE_STOP_REASONS = {
    "max_tokens": {"reason": "max_output_tokens"},
    "model_context_window_exceeded": {"reason": "max_input_tokens"},
    "pause_turn": {"reason": "other"},
}
SUPPORTED_REASONING_SUMMARIES = {None, "auto", "concise", "detailed"}


def _stringify(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _parse_jsonish(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": value}
    if value is None:
        return {}
    return {"value": value}


def _parse_data_url(value):
    if not isinstance(value, str) or not value.startswith("data:"):
        return None
    header, _, data = value.partition(",")
    if not _:
        return None
    media_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    is_base64 = ";base64" in header
    return {"media_type": media_type, "data": data, "is_base64": is_base64}


def _guess_media_type(filename_or_url):
    media_type, _ = mimetypes.guess_type(filename_or_url or "")
    return media_type or "application/octet-stream"


def _is_textual_media_type(media_type):
    if not isinstance(media_type, str):
        return False
    if media_type.startswith("text/"):
        return True
    return media_type in {
        "application/json",
        "application/ld+json",
        "application/x-httpd-php",
        "application/x-javascript",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
    }


def _decode_data_url_text(parsed):
    data = parsed.get("data", "")
    if parsed.get("is_base64"):
        raw = base64.b64decode(data)
        return raw.decode("utf-8")
    return urllib.parse.unquote_to_bytes(data).decode("utf-8")


def _translate_image_block(item):
    image_url = item.get("image_url", "")
    if isinstance(image_url, dict):
        image_url = image_url.get("url", "")
    parsed = _parse_data_url(image_url)
    if parsed:
        if not parsed["media_type"].startswith("image/"):
            raise UnsupportedFeatureError(f"Unsupported image media type: {parsed['media_type']}")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": parsed["media_type"],
                "data": parsed["data"],
            },
        }
    if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": image_url}}
    raise UnsupportedFeatureError("Unsupported input_image source")


def _image_detail_instruction(detail):
    if detail in (None, "auto"):
        return None
    if detail == "low":
        return "Use low detail for the next image."
    if detail == "high":
        return "Use high detail for the next image."
    raise UnsupportedFeatureError("Unsupported Responses API feature: input_image.detail is not supported")


def _builtin_tool_input_schema(tool_type):
    if tool_type == "apply_patch":
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "object"},
            },
            "required": ["operation"],
            "additionalProperties": False,
        }
    if tool_type == "shell":
        return {
            "type": "object",
            "properties": {
                "commands": {"type": "array", "items": {"type": "string"}},
                "timeout_ms": {"type": "integer"},
                "max_output_length": {"type": "integer"},
                "environment": {"type": "object"},
            },
            "required": ["commands"],
            "additionalProperties": False,
        }
    raise UnsupportedFeatureError(f"Unsupported Responses API tool type: {tool_type}")


def _custom_tool_description(description, format_spec):
    description = description or ""
    format_type = format_spec.get("type", "text")
    if format_type == "text":
        return description
    if format_type == "grammar":
        syntax = format_spec.get("syntax")
        definition = format_spec.get("definition")
        if syntax not in {"lark", "regex"} or not isinstance(definition, str) or not definition.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: custom tool format is not supported")
        instruction = f"The tool input must be plain text matching this {syntax} grammar:\n{definition}"
        return f"{description}\n\n{instruction}" if description else instruction
    raise UnsupportedFeatureError("Unsupported Responses API feature: custom tool format is not supported")


def _translate_file_block(item):
    if item.get("file_id"):
        raise UnsupportedFeatureError("Unsupported Responses API feature: input_file.file_id is not supported")

    file_value = item.get("file_data") or item.get("file_url") or ""
    filename = item.get("filename") or ""
    parsed = _parse_data_url(file_value)

    if parsed:
        media_type = parsed["media_type"]
        source = {"type": "base64", "media_type": media_type, "data": parsed["data"]}
    elif isinstance(file_value, str) and file_value.startswith(("http://", "https://")):
        media_type = _guess_media_type(filename or file_value)
        source = {"type": "url", "url": file_value}
    else:
        raise UnsupportedFeatureError("Unsupported input_file source")

    if media_type.startswith("image/"):
        return {"type": "image", "source": source}
    if media_type == "application/pdf":
        return {"type": "document", "source": source}
    if parsed and _is_textual_media_type(media_type):
        return {"type": "text", "text": _decode_data_url_text(parsed)}
    raise UnsupportedFeatureError(f"Unsupported input_file media type: {media_type}")


def _translate_content_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        return [{"type": "text", "text": _stringify(content)}]

    blocks = []
    for item in content:
        if not isinstance(item, dict):
            blocks.append({"type": "text", "text": _stringify(item)})
            continue

        item_type = item.get("type")
        if item_type in {"input_text", "output_text", "text"}:
            blocks.append({"type": "text", "text": item.get("text", "")})
            continue
        if item_type in {"input_image", "image_url"}:
            instruction = _image_detail_instruction(item.get("detail"))
            if instruction:
                blocks.append({"type": "text", "text": instruction})
            blocks.append(_translate_image_block(item))
            continue
        if item_type in {"input_file", "file"}:
            blocks.append(_translate_file_block(item))
            continue
        if item_type in {"thinking", "redacted_thinking"}:
            block = {"type": item_type}
            if item.get("thinking"):
                block["thinking"] = item["thinking"]
            if item.get("signature"):
                block["signature"] = item["signature"]
            if item.get("data"):
                block["data"] = item["data"]
            blocks.append(block)
            continue
        if item_type == "tool_use":
            name = (item.get("name") or "").strip()
            if not name:
                continue
            blocks.append(
                {
                    "type": "tool_use",
                    "id": item.get("id") or item.get("call_id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": name,
                    "input": _parse_jsonish(item.get("input", item.get("arguments"))),
                }
            )
            continue
        if item_type == "tool_result":
            tool_use_id = item.get("tool_use_id") or item.get("call_id")
            if not tool_use_id:
                continue
            block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": item.get("content", item.get("output", "")),
            }
            if item.get("is_error") is True:
                block["is_error"] = True
            blocks.append(block)
            continue

        raise UnsupportedFeatureError(f"Unsupported content block type: {item_type}")

    return blocks


def _translate_tool_result_content(value):
    if isinstance(value, list):
        try:
            return _translate_content_blocks(value)
        except UnsupportedFeatureError:
            return _stringify(value)
    if isinstance(value, dict):
        if value.get("type"):
            return _translate_content_blocks([value])
        return _stringify(value)
    return _stringify(value)


def _translate_tools(tools):
    translated = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        tool_name = tool.get("name") or tool.get("function", {}).get("name", "")
        if tool_type in {"apply_patch", "shell"} and not tool_name:
            tool_name = tool_type
        if tool_type not in {None, "function", "custom", "apply_patch", "shell"}:
            if not str(tool_name).strip():
                continue
            raise UnsupportedFeatureError(
                f"Unsupported Responses API tool type: {tool_type}"
            )
        strict = tool.get("strict", tool.get("function", {}).get("strict"))
        if tool_type in {None, "function"} and strict is False:
            raise UnsupportedFeatureError(
                "Unsupported Responses API feature: function tool strict=false is not supported"
            )
        if tool_type == "custom":
            format_spec = tool.get("format") or {"type": "text"}
            if not isinstance(format_spec, dict):
                raise UnsupportedFeatureError("Unsupported Responses API feature: custom tool format is not supported")
            translated.append(
                {
                    "name": tool_name,
                    "description": _custom_tool_description(
                        tool.get("description")
                        or tool.get("function", {}).get("description", ""),
                        format_spec,
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                }
            )
            continue
        if tool_type in {"apply_patch", "shell"}:
            translated.append(
                {
                    "name": tool_name,
                    "description": tool.get("description")
                    or tool.get("function", {}).get("description", ""),
                    "input_schema": _builtin_tool_input_schema(tool_type),
                }
            )
            continue
        translated.append(
            {
                "name": tool_name,
                "description": tool.get("description")
                or tool.get("function", {}).get("description", ""),
                "input_schema": tool.get("parameters")
                or tool.get("function", {}).get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return translated


def _normalize_include(include):
    if include is None:
        return []
    if not isinstance(include, list):
        raise UnsupportedFeatureError("Unsupported Responses API include value")
    allowed = {"reasoning.encrypted_content"}
    normalized = []
    for value in include:
        if value not in allowed:
            raise UnsupportedFeatureError(f"Unsupported Responses API include value: {value}")
        normalized.append(value)
    return normalized


def _normalize_stream_options(stream_options, stream=True):
    if stream_options is None:
        return {"include_obfuscation": True} if stream else None
    if not stream:
        raise UnsupportedFeatureError("Unsupported Responses API feature: stream_options requires stream=true")
    if not isinstance(stream_options, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: stream_options is not supported")
    unknown_keys = set(stream_options) - {"include_obfuscation"}
    if unknown_keys:
        raise UnsupportedFeatureError("Unsupported Responses API feature: stream_options is not supported")
    include_obfuscation = stream_options.get("include_obfuscation", True)
    if not isinstance(include_obfuscation, bool):
        raise UnsupportedFeatureError("Unsupported Responses API feature: stream_options.include_obfuscation is not supported")
    return {"include_obfuscation": include_obfuscation}


def _validate_supported_request_fields(body):
    for field_name in UNSUPPORTED_REQUEST_FIELDS:
        if body.get(field_name) is not None:
            raise UnsupportedFeatureError(
                f"Unsupported Responses API feature: {field_name} is not supported"
            )


def _builtin_tool_type_for_name(name):
    if name == "apply_patch":
        return "apply_patch_call"
    if name == "shell":
        return "shell_call"
    return None


def _tool_type_lookup(response_context):
    lookup = {}
    if not isinstance(response_context, dict):
        return lookup
    for tool in response_context.get("tools", []):
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or tool.get("function", {}).get("name", "")
        if not isinstance(name, str) or not name.strip():
            continue
        tool_type = tool.get("type")
        if tool_type == "custom":
            lookup[name.strip()] = "custom_tool_call"
        elif tool_type == "apply_patch":
            lookup[name.strip()] = "apply_patch_call"
        elif tool_type == "shell":
            lookup[name.strip()] = "shell_call"
        else:
            lookup.setdefault(name.strip(), "function_call")
    return lookup


def _tool_payload_object(payload):
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_item_payload(call_id, name, payload, status, response_context=None):
    item_type = _tool_type_lookup(response_context).get(name)
    if item_type is None:
        item_type = _builtin_tool_type_for_name(name) or "function_call"
    item = {
        "id": f"fc_{call_id}",
        "type": item_type,
        "call_id": call_id,
        "status": status,
    }
    if item_type == "custom_tool_call":
        item["name"] = name
        item["input"] = _unwrap_custom_tool_payload(payload)
        return item
    if item_type == "apply_patch_call":
        payload_obj = _tool_payload_object(payload)
        operation = payload_obj.get("operation")
        if not isinstance(operation, dict):
            operation = payload_obj if isinstance(payload_obj, dict) and payload_obj else {}
        item["operation"] = operation
        return item
    if item_type == "shell_call":
        payload_obj = _tool_payload_object(payload)
        commands = payload_obj.get("commands")
        if isinstance(commands, str):
            commands = [commands]
        if isinstance(commands, list):
            item["commands"] = [str(command) for command in commands]
        else:
            fallback = _unwrap_custom_tool_payload(payload).strip()
            item["commands"] = [fallback] if fallback else []
        for field_name in ("max_output_length", "timeout_ms", "environment"):
            value = payload_obj.get(field_name)
            if value is not None:
                item[field_name] = value
        return item
    item["name"] = name
    item["arguments"] = payload
    return item


def _usage_payload(usage):
    usage = usage or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached_tokens = usage.get("cache_read_input_tokens", 0)
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _reasoning_encrypted_content(block, response_context=None):
    include = []
    if isinstance(response_context, dict):
        include = response_context.get("include", [])
    if "reasoning.encrypted_content" not in include:
        return None
    for field_name in ("data", "signature"):
        value = block.get(field_name)
        if value is not None:
            return _stringify(value)
    return None


def _reasoning_item_payload(item_id, text, encrypted_content=None):
    item = {
        "id": item_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
    }
    if text:
        item["content"] = [{"type": "reasoning_text", "text": text}]
    if encrypted_content is not None:
        item["encrypted_content"] = encrypted_content
    return item


def _normalize_response_tools(tools):
    normalized = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            normalized.append(tool)
            continue
        normalized_tool = dict(tool)
        tool_type = normalized_tool.get("type")
        if tool_type in {"apply_patch", "shell"} and not normalized_tool.get("name"):
            normalized_tool["name"] = tool_type
        if tool_type == "custom" and "format" not in normalized_tool:
            normalized_tool["format"] = {"type": "text"}
        if tool_type in {None, "function"} and "strict" not in normalized_tool:
            normalized_tool["strict"] = True
        normalized.append(normalized_tool)
    return normalized


def _custom_tool_input_value(value):
    if isinstance(value, dict) and set(value) == {"input"} and isinstance(value.get("input"), str):
        return value["input"]
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return _stringify(value)


def _custom_tool_payload(value):
    return {"input": _custom_tool_input_value(value)}


def _unwrap_custom_tool_payload(payload):
    if isinstance(payload, dict):
        if set(payload) == {"input"} and isinstance(payload.get("input"), str):
            return payload["input"]
        return _stringify(payload)
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return payload
        if isinstance(parsed, dict) and set(parsed) == {"input"} and isinstance(parsed.get("input"), str):
            return parsed["input"]
        return payload
    return _stringify(payload)


def _effective_response_tools(body):
    tools = _normalize_response_tools(body.get("tools", []))
    tool_choice = body.get("tool_choice")
    if not isinstance(tool_choice, dict):
        return tools

    choice_type = tool_choice.get("type")
    if choice_type == "custom":
        name = tool_choice.get("name")
        if name and all(tool.get("name") != name for tool in tools if isinstance(tool, dict)):
            tools.append({"type": "custom", "name": name, "format": {"type": "text"}})
        return tools
    if choice_type in {"apply_patch", "shell"}:
        name = choice_type
        if all(tool.get("name") != name for tool in tools if isinstance(tool, dict)):
            tools.append({"type": choice_type, "name": name})
    return tools


def _response_completion_from_stop_reason(stop_reason):
    if stop_reason in RESPONSE_COMPLETED_STOP_REASONS:
        return "completed", None
    return "incomplete", RESPONSE_INCOMPLETE_STOP_REASONS.get(stop_reason, {"reason": "other"})


def _tool_stream_event_spec(call, response_context=None):
    item_type = _tool_type_lookup(response_context).get(call.get("name", ""), "function_call")
    if item_type == "custom_tool_call":
        return {
            "delta_event": "response.custom_tool_call_input.delta",
            "done_event": "response.custom_tool_call_input.done",
            "field_name": "input",
        }
    return {
        "delta_event": "response.function_call_arguments.delta",
        "done_event": "response.function_call_arguments.done",
        "field_name": "arguments",
    }


def build_response_context(body, model=None):
    text_config = body.get("text")
    if not isinstance(text_config, dict):
        response_format = body.get("response_format")
        if isinstance(response_format, dict):
            text_config = {"format": response_format}
        else:
            text_config = {"format": {"type": "text"}}

    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        reasoning = {"effort": None, "summary": None}
    summary = reasoning.get("summary")
    if summary not in SUPPORTED_REASONING_SUMMARIES:
        raise UnsupportedFeatureError("Unsupported Responses API feature: reasoning.summary is not supported")
    reasoning = {
        "effort": reasoning.get("effort"),
        "summary": summary,
    }

    return {
        "model": model or body.get("model"),
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "metadata": body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
        "user": body.get("user"),
        "store": body.get("store", False),
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": _effective_response_tools(body),
        "text": text_config,
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0),
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "reasoning": reasoning,
        "previous_response_id": body.get("previous_response_id"),
        "truncation": body.get("truncation", "disabled"),
        "max_tool_calls": body.get("max_tool_calls"),
        "background": body.get("background", False),
        "include": _normalize_include(body.get("include")),
        "prompt_cache_key": body.get("prompt_cache_key"),
        "top_logprobs": body.get("top_logprobs"),
        "stream_options": _normalize_stream_options(body.get("stream_options"), body.get("stream", True)),
    }


def _translate_tool_choice(tool_choice):
    if tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "none":
        return {"type": "none"}
    if tool_choice == "required":
        return {"type": "any"}
    if isinstance(tool_choice, str):
        raise UnsupportedFeatureError(f"Unsupported Responses API feature: tool_choice value {tool_choice} is not supported")
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "function":
            name = tool_choice.get("name") or tool_choice.get("function", {}).get("name", "")
            if not name:
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool_choice.function requires a name")
            return {"type": "tool", "name": name}
        if choice_type == "tool":
            name = tool_choice.get("name", "")
            if not name:
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool_choice.tool requires a name")
            return {"type": "tool", "name": name}
        if choice_type == "custom":
            name = tool_choice.get("name", "")
            if not name:
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool_choice.custom requires a name")
            return {"type": "tool", "name": name}
        if choice_type == "apply_patch":
            return {"type": "tool", "name": "apply_patch"}
        if choice_type == "shell":
            return {"type": "tool", "name": "shell"}
        if choice_type == "none":
            return {"type": "none"}
        if choice_type == "required":
            return {"type": "any"}
        raise UnsupportedFeatureError(
            f"Unsupported Responses API feature: tool_choice type {choice_type} is not supported"
        )
    raise UnsupportedFeatureError("Unsupported Responses API feature: tool_choice is not supported")


def _allowed_tool_names(tool_choice):
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "allowed_tools":
        return None
    names = []
    for tool in tool_choice.get("tools", []):
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or tool.get("function", {}).get("name", "")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return set(names)


def _thinking_from_reasoning(body, max_tokens):
    if isinstance(body.get("thinking"), dict):
        return dict(body["thinking"])

    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return None

    if reasoning.get("summary") not in SUPPORTED_REASONING_SUMMARIES:
        raise UnsupportedFeatureError("Unsupported Responses API feature: reasoning.summary is not supported")

    effort = (reasoning.get("effort") or "").lower()
    if effort in {"", "none"}:
        return None

    ratios = {
        "minimal": 0.10,
        "low": 0.20,
        "medium": 0.50,
        "high": 0.80,
        "xhigh": 0.95,
    }
    ratio = ratios.get(effort)
    if ratio is None:
        raise UnsupportedFeatureError("Unsupported Responses API feature: reasoning.effort is not supported")

    if not isinstance(max_tokens, int) or max_tokens <= 1024:
        raise UnsupportedFeatureError(
            "Unsupported Responses API behavior: reasoning.effort requires max_output_tokens greater than 1024"
        )

    budget_tokens = int(max_tokens * ratio)
    budget_tokens = max(1024, budget_tokens)
    budget_tokens = min(budget_tokens, max_tokens - 1)
    if budget_tokens < 1024:
        return None
    return {"type": "enabled", "budget_tokens": budget_tokens}


def _ensure_tool_definitions(result):
    known_tools = {tool.get("name"): tool for tool in result.get("tools", []) if tool.get("name")}
    for message in result.get("messages", []):
        if message.get("role") != "assistant":
            continue
        for block in message.get("content", []):
            if block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if not name or name in known_tools:
                continue
            builtin_tool_type = name if name in {"apply_patch", "shell"} else None
            if builtin_tool_type:
                placeholder = {
                    "name": name,
                    "description": "",
                    "input_schema": _builtin_tool_input_schema(builtin_tool_type),
                }
            else:
                placeholder = {
                    "name": name,
                    "description": "",
                    "input_schema": {"type": "object", "properties": {}},
                }
            result.setdefault("tools", []).append(placeholder)
            known_tools[name] = placeholder


def _text_format_instruction(body):
    format_spec = None
    if isinstance(body.get("text"), dict):
        format_spec = body["text"].get("format")
    if format_spec is None:
        format_spec = body.get("response_format")
    if not isinstance(format_spec, dict):
        return None

    format_type = format_spec.get("type")
    if format_type not in {"text", "json_schema", "json_object"}:
        raise UnsupportedFeatureError("Unsupported Responses API feature: text.format is not supported")
    if format_type == "json_schema":
        schema = format_spec.get("schema") or format_spec.get("json_schema", {}).get("schema")
        if schema:
            schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
            strict = format_spec.get("strict", True)
            if strict not in (True, False, None):
                raise UnsupportedFeatureError("Unsupported Responses API feature: text.format.strict is not supported")
            if strict is False:
                return (
                    "You must respond with valid JSON that matches this JSON schema as closely as possible:\n"
                    f"```json\n{schema_json}\n```\n"
                    "Respond ONLY with the JSON object, no other text."
                )
            return (
                "You must respond with valid JSON that strictly follows this JSON schema:\n"
                f"```json\n{schema_json}\n```\n"
                "Respond ONLY with the JSON object, no other text."
            )
    if format_type == "json_object":
        return "You must respond with valid JSON. Respond ONLY with a JSON object, no other text."
    return None


def _text_verbosity_instruction(body):
    text_config = body.get("text")
    if text_config is None:
        return None
    if not isinstance(text_config, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: text is not supported")
    verbosity = text_config.get("verbosity")
    if verbosity is None:
        return None
    if verbosity == "low":
        return "Keep the response concise."
    if verbosity == "medium":
        return "Use a moderate level of detail."
    if verbosity == "high":
        return "Provide a detailed response."
    raise UnsupportedFeatureError("Unsupported Responses API feature: text.verbosity is not supported")


def _iter_input_items(raw_input):
    if raw_input is None:
        return []
    if isinstance(raw_input, str):
        return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": raw_input}]}]
    if isinstance(raw_input, dict):
        return [raw_input]
    if isinstance(raw_input, list):
        return raw_input
    return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": _stringify(raw_input)}]}]


def translate_responses_request(body):
    _validate_supported_request_fields(body)
    if body.get("background"):
        raise UnsupportedFeatureError("background mode is not supported")
    if body.get("previous_response_id"):
        raise UnsupportedFeatureError("Unsupported Responses API feature: previous_response_id is not supported")
    if body.get("store"):
        raise UnsupportedFeatureError("Unsupported Responses API feature: store is not supported")
    truncation = body.get("truncation")
    if truncation not in (None, "disabled"):
        raise UnsupportedFeatureError("Unsupported Responses API feature: truncation is not supported")
    if body.get("max_tool_calls") is not None:
        raise UnsupportedFeatureError("Unsupported Responses API feature: max_tool_calls is not supported")
    _normalize_include(body.get("include"))
    _normalize_stream_options(body.get("stream_options"), body.get("stream", True))
    top_logprobs = body.get("top_logprobs")
    if top_logprobs not in (None, 0):
        raise UnsupportedFeatureError("Unsupported Responses API feature: top_logprobs is not supported")

    result = {
        "model": body.get("model"),
        "messages": [],
        "stream": body.get("stream", True),
        "max_tokens": body.get("max_output_tokens") or 4096,
    }

    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        result["top_p"] = body["top_p"]
    if body.get("stop") is not None:
        result["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
    thinking = _thinking_from_reasoning(body, result["max_tokens"])
    if thinking:
        result["thinking"] = thinking

    system_segments = []
    if body.get("instructions"):
        system_segments.append(body["instructions"])
    text_format_instruction = _text_format_instruction(body)
    if text_format_instruction:
        system_segments.append(text_format_instruction)
    text_verbosity_instruction = _text_verbosity_instruction(body)
    if text_verbosity_instruction:
        system_segments.append(text_verbosity_instruction)

    allowed_tool_names = _allowed_tool_names(body.get("tool_choice"))
    tools = _translate_tools(body.get("tools", []))
    if allowed_tool_names is not None:
        tools = [tool for tool in tools if tool.get("name") in allowed_tool_names]

    tool_choice_input = body.get("tool_choice")
    if isinstance(tool_choice_input, dict) and tool_choice_input.get("type") == "allowed_tools":
        tool_choice_mode = tool_choice_input.get("mode")
        if tool_choice_mode == "required":
            tool_choice_input = "required"
        elif tool_choice_mode == "auto":
            tool_choice_input = "auto"
        else:
            raise UnsupportedFeatureError(
                "Unsupported Responses API feature: tool_choice.allowed_tools.mode is not supported"
            )

    tool_choice = _translate_tool_choice(tool_choice_input)
    if (
        isinstance(tool_choice_input, dict)
        and tool_choice_input.get("type") in {"custom", "apply_patch", "shell"}
        and tool_choice
        and tool_choice.get("type") == "tool"
        and all(tool.get("name") != tool_choice["name"] for tool in tools)
    ):
        choice_type = tool_choice_input.get("type")
        if choice_type == "custom":
            tools.append(
                {
                    "name": tool_choice["name"],
                    "description": "",
                    "input_schema": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                }
            )
        else:
            tools.append(
                {
                    "name": tool_choice["name"],
                    "description": "",
                    "input_schema": _builtin_tool_input_schema(choice_type),
                }
            )
    if tools:
        result["tools"] = tools
    if tool_choice:
        result["tool_choice"] = tool_choice

    current_assistant_blocks = []
    pending_tool_result_blocks = []

    def flush_assistant():
        if current_assistant_blocks:
            result["messages"].append({"role": "assistant", "content": list(current_assistant_blocks)})
            current_assistant_blocks.clear()

    def flush_tool_results():
        if pending_tool_result_blocks:
            result["messages"].append({"role": "user", "content": list(pending_tool_result_blocks)})
            pending_tool_result_blocks.clear()

    for item in _iter_input_items(body.get("input", [])):
        if not isinstance(item, dict):
            continue

        item_type = item.get("type") or ("message" if item.get("role") else None)
        if item_type == "message":
            flush_assistant()
            flush_tool_results()
            role = item.get("role", "user")
            if role not in {"user", "assistant", "developer", "system"}:
                raise UnsupportedFeatureError("Unsupported Responses API feature: message role is not supported")
            phase = item.get("phase")
            if phase not in {None, "commentary", "final_answer"}:
                raise UnsupportedFeatureError("Unsupported Responses API feature: message phase is not supported")
            if role in {"developer", "system"}:
                developer_blocks = _translate_content_blocks(item.get("content", []))
                if any(block.get("type") != "text" for block in developer_blocks):
                    raise UnsupportedFeatureError(
                        "Unsupported Responses API feature: developer messages only support text content"
                    )
                developer_text = "\n\n".join(block.get("text", "") for block in developer_blocks).strip()
                if developer_text:
                    system_segments.append(developer_text)
                continue
            result["messages"].append(
                {
                    "role": role,
                    "content": _translate_content_blocks(item.get("content", [])),
                }
            )
            continue

        if item_type in {"function_call", "custom_tool_call", "apply_patch_call", "shell_call"}:
            name = (item.get("name") or "").strip()
            if item_type == "apply_patch_call":
                name = "apply_patch"
            elif item_type == "shell_call":
                name = "shell"
            if not name:
                continue
            flush_tool_results()
            if item_type == "custom_tool_call":
                tool_input = _custom_tool_payload(item.get("input"))
            elif item_type == "apply_patch_call":
                tool_input = {"operation": _parse_jsonish(item.get("operation"))}
            elif item_type == "shell_call":
                tool_input = {}
                if item.get("commands") is not None:
                    commands = item.get("commands")
                    if isinstance(commands, str):
                        commands = [commands]
                    if not isinstance(commands, list):
                        raise UnsupportedFeatureError("Unsupported Responses API feature: shell_call.commands is not supported")
                    tool_input["commands"] = [str(command) for command in commands]
                for field_name in ("timeout_ms", "max_output_length", "environment"):
                    if item.get(field_name) is not None:
                        tool_input[field_name] = item[field_name]
            else:
                tool_input = _parse_jsonish(item.get("arguments", item.get("input")))
            current_assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": name,
                    "input": tool_input,
                }
            )
            continue

        if item_type in {
            "function_call_output",
            "custom_tool_call_output",
            "apply_patch_call_output",
            "shell_call_output",
        }:
            flush_assistant()
            call_id = item.get("call_id", "")
            if not call_id:
                continue
            pending_tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": _translate_tool_result_content(item.get("output", "")),
                }
            )
            continue

        if item_type == "reasoning":
            raise UnsupportedFeatureError("Unsupported Responses API feature: reasoning input items are not supported")

        if item_type == "item_reference":
            raise UnsupportedFeatureError("Unsupported Responses API feature: item_reference is not supported")

        raise UnsupportedFeatureError(f"Unsupported input item type: {item_type}")

    if system_segments:
        result["system"] = "\n\n".join(segment for segment in system_segments if segment)

    flush_assistant()
    flush_tool_results()

    all_tool_use_ids = set()
    for message in result["messages"]:
        if message.get("role") == "assistant":
            for block in message.get("content", []):
                if block.get("type") == "tool_use" and block.get("id"):
                    all_tool_use_ids.add(str(block["id"]))

    filtered_messages = []
    for message in result["messages"]:
        if message.get("role") == "user":
            filtered_blocks = []
            for block in message.get("content", []):
                if block.get("type") == "tool_result":
                    if block.get("tool_use_id") and str(block["tool_use_id"]) in all_tool_use_ids:
                        filtered_blocks.append(block)
                else:
                    filtered_blocks.append(block)
            if filtered_blocks:
                filtered_messages.append({"role": "user", "content": filtered_blocks})
            continue
        filtered_messages.append(message)

    result["messages"] = filtered_messages
    _ensure_tool_definitions(result)

    return result


def translate_anthropic_response(body, model, response_context=None):
    response_id = f"resp_{body.get('id', uuid.uuid4().hex)}"
    created_at = int(time.time())
    completed_at = int(time.time())
    output = []
    output_text_parts = []
    parallel_tool_calls = True if response_context is None else response_context.get("parallel_tool_calls", True)
    tool_call_seen = False
    response_status, incomplete_details = _response_completion_from_stop_reason(body.get("stop_reason"))
    message_status = "completed" if response_status == "completed" else "incomplete"

    for index, block in enumerate(body.get("content", [])):
        block_type = block.get("type")
        if block_type in {"thinking", "redacted_thinking"}:
            summary_text = block.get("thinking") or block.get("text") or ""
            output.append(
                _reasoning_item_payload(
                    f"rs_{response_id}_{index}",
                    summary_text,
                    encrypted_content=_reasoning_encrypted_content(block, response_context=response_context),
                )
            )
        elif block_type == "text":
            text = block.get("text", "")
            output_text_parts.append(text)
            output.append(
                {
                    "id": f"msg_{response_id}_{index}",
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "status": message_status,
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            )
        elif block_type == "tool_use":
            if not parallel_tool_calls and tool_call_seen:
                raise UnsupportedFeatureError(
                    "Unsupported Responses API behavior: parallel_tool_calls=false cannot accept multiple tool calls"
                )
            tool_call_seen = True
            output.append(
                _tool_item_payload(
                    block.get("id", ""),
                    block.get("name", ""),
                    json.dumps(block.get("input", {}), ensure_ascii=False),
                    "completed",
                    response_context=response_context,
                )
            )

    response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": completed_at,
        "status": response_status,
        "model": model,
        "error": None,
        "incomplete_details": incomplete_details,
        "output": output,
        "output_text": "".join(output_text_parts),
        "usage": _usage_payload(body.get("usage", {})),
    }
    if response_context:
        response.update(response_context)
    return response


class ResponsesEventTranslator:
    def __init__(self, model=None, response_context=None):
        self.model = model
        self.response_context = response_context or {}
        self.created_at = int(time.time())
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.sequence = 0
        self.started = False
        self.completed = False
        self.text_buffers = {}
        self.message_added = set()
        self.content_added = set()
        self.message_done = set()
        self.reasoning_buffers = {}
        self.reasoning_added = set()
        self.reasoning_done = set()
        self.reasoning_encrypted = {}
        self.tool_calls = {}
        self.tool_args = {}
        self.tool_seed_args = {}
        self.tool_done = set()
        self.ignored_tool_indexes = set()
        self.usage = {}
        self.stop_reason = None

    def _include_obfuscation(self):
        stream_options = self.response_context.get("stream_options") or {}
        return stream_options.get("include_obfuscation", True)

    def _delta_payload(self, payload):
        data = dict(payload)
        if self._include_obfuscation():
            data["obfuscation"] = secrets.token_hex(4)
        return data

    def _assistant_output_index(self):
        return 0

    def _final_response_status(self):
        return _response_completion_from_stop_reason(self.stop_reason)[0]

    def _final_incomplete_details(self):
        return _response_completion_from_stop_reason(self.stop_reason)[1]

    def _final_message_status(self):
        return "completed" if self._final_response_status() == "completed" else "incomplete"

    def _emit(self, event, data):
        self.sequence += 1
        payload = dict(data)
        payload["sequence_number"] = self.sequence
        return {"event": event, "data": payload}

    def _message_id(self, index):
        return f"msg_{self.response_id}_{index}"

    def _tool_item_id(self, call_id):
        return f"fc_{call_id}"

    def _build_output_items(self):
        items = []
        assistant_index = self._assistant_output_index()
        if assistant_index in self.reasoning_added:
            items.append(
                _reasoning_item_payload(
                    f"rs_{self.response_id}_{assistant_index}",
                    self.reasoning_buffers.get(assistant_index, ""),
                    encrypted_content=self.reasoning_encrypted.get(assistant_index),
                )
            )
        if assistant_index in self.message_added:
            text = self.text_buffers.get(assistant_index, "")
            items.append(
                {
                    "id": self._message_id(assistant_index),
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "status": self._final_message_status() if assistant_index in self.message_done else "in_progress",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            )
        for index in sorted(self.tool_calls):
            call = self.tool_calls[index]
            call_id = call["call_id"]
            args = self.tool_args.get(index, "") or self.tool_seed_args.get(index, "{}")
            items.append(
                _tool_item_payload(
                    call_id,
                    call["name"],
                    args,
                    "completed" if index in self.tool_done else "in_progress",
                    response_context=self.response_context,
                )
            )
        return items

    def _response_payload(self, status, include_output=False, include_usage=False, include_completed_at=False):
        payload = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "background": self.response_context.get("background", False),
            "error": None,
            "incomplete_details": self._final_incomplete_details() if status == "incomplete" else None,
            "model": self.model,
            "output": self._build_output_items() if include_output else [],
            "metadata": self.response_context.get("metadata", {}),
            "user": self.response_context.get("user"),
            "store": self.response_context.get("store", False),
            "tool_choice": self.response_context.get("tool_choice", "auto"),
            "tools": self.response_context.get("tools", []),
            "text": self.response_context.get("text", {"format": {"type": "text"}}),
            "temperature": self.response_context.get("temperature", 1.0),
            "top_p": self.response_context.get("top_p", 1.0),
            "parallel_tool_calls": self.response_context.get("parallel_tool_calls", True),
            "reasoning": self.response_context.get("reasoning", {"effort": None, "summary": None}),
            "previous_response_id": self.response_context.get("previous_response_id"),
            "truncation": self.response_context.get("truncation", "disabled"),
            "max_output_tokens": self.response_context.get("max_output_tokens"),
            "instructions": self.response_context.get("instructions"),
            "include": self.response_context.get("include", []),
            "prompt_cache_key": self.response_context.get("prompt_cache_key"),
            "top_logprobs": self.response_context.get("top_logprobs"),
        }
        max_tool_calls = self.response_context.get("max_tool_calls")
        if max_tool_calls is not None:
            payload["max_tool_calls"] = max_tool_calls
        if include_usage:
            if self.usage:
                payload["usage"] = _usage_payload(self.usage)
            else:
                payload["usage"] = None
        else:
            payload["usage"] = None
        if include_output:
            payload["output_text"] = "".join(
                self.text_buffers.get(idx, "") for idx in sorted(self.message_added)
            )
        if include_completed_at:
            payload["completed_at"] = int(time.time())
        return payload

    def _ensure_started(self):
        if self.started:
            return []
        self.started = True
        return [
            self._emit(
                "response.created",
                {
                    "type": "response.created",
                    "response": self._response_payload("in_progress"),
                },
            ),
            self._emit(
                "response.in_progress",
                {
                    "type": "response.in_progress",
                    "response": self._response_payload("in_progress"),
                },
            ),
        ]

    def _ensure_text_started(self, index):
        index = self._assistant_output_index()
        events = []
        msg_id = self._message_id(index)
        if index not in self.message_added:
            self.message_added.add(index)
            events.append(
                self._emit(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": index,
                        "item": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "phase": "final_answer",
                            "status": "in_progress",
                            "content": [],
                        },
                    },
                )
            )
        if index not in self.content_added:
            self.content_added.add(index)
            events.append(
                self._emit(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": msg_id,
                        "output_index": index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                    },
                )
            )
        return events

    def _close_text(self, index):
        index = self._assistant_output_index()
        if index in self.message_done:
            return []
        self.message_done.add(index)
        text = self.text_buffers.get(index, "")
        msg_id = self._message_id(index)
        return [
            self._emit(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": msg_id,
                    "output_index": index,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                },
            ),
            self._emit(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": msg_id,
                    "output_index": index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": [], "logprobs": []},
                },
            ),
            self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                        "item": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "phase": "final_answer",
                            "status": self._final_message_status(),
                            "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    },
                },
            ),
        ]

    def _ensure_reasoning_started(self, index):
        index = self._assistant_output_index()
        if index in self.reasoning_added:
            return []
        self.reasoning_added.add(index)
        item_id = f"rs_{self.response_id}_{index}"
        item = {"id": item_id, "type": "reasoning", "summary": []}
        encrypted_content = self.reasoning_encrypted.get(index)
        if encrypted_content is not None:
            item["encrypted_content"] = encrypted_content
        return [
            self._emit(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": item,
                },
            ),
            self._emit(
                "response.reasoning_summary_part.added",
                {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                },
            ),
        ]

    def _close_reasoning(self, index):
        index = self._assistant_output_index()
        if index not in self.reasoning_added or index in self.reasoning_done:
            return []
        self.reasoning_done.add(index)
        item_id = f"rs_{self.response_id}_{index}"
        text = self.reasoning_buffers.get(index, "")
        encrypted_content = self.reasoning_encrypted.get(index)
        return [
            self._emit(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index": 0,
                    "text": text,
                },
            ),
            self._emit(
                "response.reasoning_summary_part.done",
                {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": item_id,
                    "output_index": index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": text},
                },
            ),
            self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": _reasoning_item_payload(item_id, text, encrypted_content=encrypted_content),
                },
            ),
        ]

    def _close_tool(self, index):
        if index not in self.tool_calls or index in self.tool_done:
            return []
        self.tool_done.add(index)
        call = self.tool_calls[index]
        call_id = call["call_id"]
        args = self.tool_args.get(index, "") or self.tool_seed_args.get(index, "{}")
        event_spec = _tool_stream_event_spec(call, response_context=self.response_context)
        final_value = _unwrap_custom_tool_payload(args) if event_spec["field_name"] == "input" else args
        return [
            self._emit(
                event_spec["done_event"],
                {
                    "type": event_spec["done_event"],
                    "item_id": self._tool_item_id(call_id),
                    "output_index": index,
                    event_spec["field_name"]: final_value,
                },
            ),
            self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": _tool_item_payload(
                        call_id,
                        call["name"],
                        args,
                        "completed",
                        response_context=self.response_context,
                    ),
                },
            ),
        ]

    def _emit_completed(self):
        if self.completed:
            return []
        self.completed = True
        return [
            self._emit(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": self._response_payload(
                        self._final_response_status(),
                        include_output=True,
                        include_usage=True,
                        include_completed_at=True,
                    ),
                },
            )
        ]

    def feed(self, event):
        events = []
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message", {})
            if message.get("id"):
                self.response_id = f"resp_{message['id']}"
            return self._ensure_started()

        if not self.started:
            events.extend(self._ensure_started())

        if event_type == "content_block_start":
            index = event.get("index", 0)
            block = event.get("content_block", {})
            if block.get("type") == "text":
                events.extend(self._ensure_text_started(index))
            elif block.get("type") == "thinking":
                encrypted_content = _reasoning_encrypted_content(block, response_context=self.response_context)
                if encrypted_content is not None:
                    self.reasoning_encrypted[self._assistant_output_index()] = encrypted_content
                events.extend(self._ensure_reasoning_started(index))
            elif block.get("type") == "tool_use":
                if not self.response_context.get("parallel_tool_calls", True) and self.tool_calls:
                    raise UnsupportedFeatureError(
                        "Unsupported Responses API behavior: parallel_tool_calls=false cannot accept multiple tool calls"
                    )
                call_id = block.get("id", f"call_{uuid.uuid4().hex[:8]}")
                name = block.get("name", "")
                self.tool_calls[index] = {"call_id": call_id, "name": name}
                self.tool_args[index] = ""
                self.tool_seed_args[index] = json.dumps(block.get("input", {}), ensure_ascii=False)
                events.append(
                    self._emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": index,
                            "item": _tool_item_payload(
                                call_id,
                                name,
                                "",
                                "in_progress",
                                response_context=self.response_context,
                            ),
                        },
                    )
                )
            return events

        if event_type == "content_block_delta":
            index = event.get("index", 0)
            delta = event.get("delta", {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                index = self._assistant_output_index()
                self.text_buffers[index] = self.text_buffers.get(index, "") + text
                events.extend(self._ensure_text_started(index))
                events.append(
                    self._emit(
                        "response.output_text.delta",
                        self._delta_payload(
                            {
                            "type": "response.output_text.delta",
                            "item_id": self._message_id(index),
                            "output_index": index,
                            "content_index": 0,
                            "delta": text,
                            "logprobs": [],
                            }
                        ),
                    )
                )
                return events
            if delta_type == "thinking_delta":
                text = delta.get("thinking", "")
                index = self._assistant_output_index()
                self.reasoning_buffers[index] = self.reasoning_buffers.get(index, "") + text
                events.extend(self._ensure_reasoning_started(index))
                events.append(
                    self._emit(
                        "response.reasoning_summary_text.delta",
                        self._delta_payload(
                            {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": f"rs_{self.response_id}_{index}",
                            "output_index": index,
                            "summary_index": 0,
                            "delta": text,
                            }
                        ),
                    )
                )
                return events
            if delta_type == "input_json_delta":
                if index in self.ignored_tool_indexes or index not in self.tool_calls:
                    return events
                partial = delta.get("partial_json", "")
                self.tool_args[index] = self.tool_args.get(index, "") + partial
                call_id = self.tool_calls[index]["call_id"]
                event_spec = _tool_stream_event_spec(
                    self.tool_calls[index],
                    response_context=self.response_context,
                )
                events.append(
                    self._emit(
                        event_spec["delta_event"],
                        self._delta_payload(
                            {
                            "type": event_spec["delta_event"],
                            "item_id": self._tool_item_id(call_id),
                            "output_index": index,
                            "delta": partial,
                            }
                        ),
                    )
                )
                return events

        if event_type == "content_block_stop":
            index = event.get("index", 0)
            if index in self.ignored_tool_indexes:
                return events
            if index in self.tool_calls:
                events.extend(self._close_tool(index))
            return events

        if event_type == "message_delta":
            self.usage = event.get("usage", {}) or {}
            delta = event.get("delta", {}) or {}
            if delta.get("stop_reason") is not None:
                self.stop_reason = delta.get("stop_reason")
            return events

        if event_type == "message_stop":
            for index in sorted(self.message_added):
                events.extend(self._close_text(index))
            for index in sorted(self.reasoning_added):
                events.extend(self._close_reasoning(index))
            for index in sorted(self.tool_calls):
                events.extend(self._close_tool(index))
            events.extend(self._emit_completed())
            return events

        return events

    def finish(self):
        events = []
        for index in sorted(self.message_added):
            events.extend(self._close_text(index))
        for index in sorted(self.reasoning_added):
            events.extend(self._close_reasoning(index))
        for index in sorted(self.tool_calls):
            events.extend(self._close_tool(index))
        events.extend(self._emit_completed())
        return events


def format_sse(event_name, data):
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
