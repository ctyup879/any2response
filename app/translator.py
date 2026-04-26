import base64
import binascii
import json
import math
import mimetypes
import re
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

IGNORABLE_UNNAMED_HOSTED_TOOL_TYPES = {
    "web_search",
    "web_search_preview",
    "file_search",
    "computer_use",
    "computer_use_preview",
    "code_interpreter",
    "image_generation",
}

SUPPORTED_RESPONSES_TOOL_TYPES = {None, "function", "custom", "apply_patch", "shell", "mcp", "web_search"}
SUPPORTED_PROVIDER_PROFILES = {"minimax", "anthropic", "generic"}
SUPPORTED_INCLUDE_VALUES = {
    "reasoning.encrypted_content",
    "web_search_call.action.sources",
    "file_search_call.results",
    "computer_call_output.output.image_url",
    "message.output_text.logprobs",
    "message.input_image.image_url",
    "code_interpreter_call.outputs",
}
REASONING_BRIDGE_PREFIX = "a2r_reasoning_v1:"


def _has_supported_translatable_tool(tools, provider_profile="minimax"):
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "web_search" and not _provider_supports_web_search(provider_profile):
            continue
        if tool_type in SUPPORTED_RESPONSES_TOOL_TYPES:
            return True
    return False


RESPONSE_COMPLETED_STOP_REASONS = {None, "end_turn", "stop_sequence", "tool_use", "refusal"}
RESPONSE_INCOMPLETE_STOP_REASONS = {
    "max_tokens": {"reason": "max_output_tokens"},
}
SUPPORTED_REASONING_SUMMARIES = {None, "auto", "concise", "detailed"}


def _stringify(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _require_request_object(body):
    if not isinstance(body, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: request body is not supported")
    return body


def _normalize_model(value):
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedFeatureError("Unsupported Responses API feature: model is not supported")
    return value


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


def _normalize_provider_profile(provider_profile):
    if provider_profile is None:
        return "minimax"
    if not isinstance(provider_profile, str):
        raise UnsupportedFeatureError("Unsupported upstream capability profile")
    normalized = provider_profile.strip().lower()
    if normalized not in SUPPORTED_PROVIDER_PROFILES:
        raise UnsupportedFeatureError("Unsupported upstream capability profile")
    if normalized == "generic":
        return "anthropic"
    return normalized


def _provider_supports_message_media(provider_profile):
    return _normalize_provider_profile(provider_profile) != "minimax"


def _provider_supports_stop_sequences(provider_profile):
    return _normalize_provider_profile(provider_profile) != "minimax"


def _provider_supports_web_search(provider_profile):
    return _normalize_provider_profile(provider_profile) != "minimax"


def _should_ignore_unnamed_hosted_tool(tool_type, tool_name, provider_profile, has_supported_local_tool):
    if tool_name:
        return False
    if tool_type == "web_search" and _provider_supports_web_search(provider_profile):
        return False
    return tool_type in IGNORABLE_UNNAMED_HOSTED_TOOL_TYPES and has_supported_local_tool


def _provider_supports_files_api(provider_profile):
    return _normalize_provider_profile(provider_profile) == "anthropic"


def _provider_supports_mcp(provider_profile):
    return _normalize_provider_profile(provider_profile) == "anthropic"


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


def _validate_data_url_base64(parsed, error_message):
    if not parsed or not parsed.get("is_base64"):
        return
    try:
        base64.b64decode(parsed.get("data", ""), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UnsupportedFeatureError(error_message) from exc


def _decode_data_url_text(parsed):
    data = parsed.get("data", "")
    if parsed.get("is_base64"):
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise UnsupportedFeatureError("Unsupported input_file source") from exc
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedFeatureError("Unsupported input_file source") from exc
    try:
        return urllib.parse.unquote_to_bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedFeatureError("Unsupported input_file source") from exc


def _translate_image_block(item, provider_profile="minimax"):
    if item.get("file_id"):
        if not _provider_supports_files_api(provider_profile):
            raise UnsupportedFeatureError("Unsupported Responses API feature: input_image.file_id is not supported")
        file_id = item.get("file_id")
        if not isinstance(file_id, str) or not file_id.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: input_image.file_id is not supported")
        return {"type": "image", "source": {"type": "file", "file_id": file_id.strip()}}
    image_url = item.get("image_url", "")
    if isinstance(image_url, dict):
        image_url = image_url.get("url", "")
    parsed = _parse_data_url(image_url)
    if parsed:
        if not parsed["media_type"].startswith("image/"):
            raise UnsupportedFeatureError(f"Unsupported image media type: {parsed['media_type']}")
        _validate_data_url_base64(parsed, "Unsupported input_image source")
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
    raise UnsupportedFeatureError("Unsupported Responses API feature: input_image.detail is not supported")


def _normalize_max_output_tokens(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedFeatureError("Unsupported Responses API feature: max_output_tokens is not supported")
    return value


def _is_finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _normalize_temperature(value, provider_profile="minimax"):
    if value is None:
        return None
    if not _is_finite_number(value):
        raise UnsupportedFeatureError("Unsupported Responses API feature: temperature is not supported")
    normalized_profile = _normalize_provider_profile(provider_profile)
    if value < 0 or value > 1:
        raise UnsupportedFeatureError("Unsupported Responses API feature: temperature is not supported")
    if normalized_profile == "minimax" and value == 0:
        raise UnsupportedFeatureError("Unsupported Responses API feature: temperature is not supported")
    return value


def _normalize_top_p(value):
    if value is None:
        return None
    if not _is_finite_number(value):
        raise UnsupportedFeatureError("Unsupported Responses API feature: top_p is not supported")
    if value < 0 or value > 1:
        raise UnsupportedFeatureError("Unsupported Responses API feature: top_p is not supported")
    return value


def _normalize_stop(value, provider_profile="minimax"):
    if value is None:
        return None
    if not _provider_supports_stop_sequences(provider_profile):
        raise UnsupportedFeatureError("Unsupported Responses API feature: stop is not supported")
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        raise UnsupportedFeatureError("Unsupported Responses API feature: stop is not supported")
    if any(not isinstance(item, str) for item in value):
        raise UnsupportedFeatureError("Unsupported Responses API feature: stop is not supported")
    return value


def _normalize_parallel_tool_calls(value):
    if value is None:
        return True
    if not isinstance(value, bool):
        raise UnsupportedFeatureError("Unsupported Responses API feature: parallel_tool_calls is not supported")
    return value


def _normalize_metadata(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: metadata is not supported")
    if len(value) > 16:
        raise UnsupportedFeatureError("Unsupported Responses API feature: metadata is not supported")
    if any(not isinstance(key, str) for key in value):
        raise UnsupportedFeatureError("Unsupported Responses API feature: metadata is not supported")
    normalized = {}
    for key, item in value.items():
        if len(key) > 64:
            raise UnsupportedFeatureError("Unsupported Responses API feature: metadata is not supported")
        if isinstance(item, bool):
            normalized[key] = item
            continue
        if _is_finite_number(item):
            normalized[key] = item
            continue
        if isinstance(item, str) and len(item) <= 512:
            normalized[key] = item
            continue
        raise UnsupportedFeatureError("Unsupported Responses API feature: metadata is not supported")
    return normalized


def _normalize_user(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise UnsupportedFeatureError("Unsupported Responses API feature: user is not supported")
    return value


def _normalize_tool_description(description):
    if description is None:
        return ""
    if not isinstance(description, str):
        raise UnsupportedFeatureError("Unsupported Responses API feature: tool description is not supported")
    return description


def _normalize_stream(value):
    if value is None:
        return True
    if not isinstance(value, bool):
        raise UnsupportedFeatureError("Unsupported Responses API feature: stream is not supported")
    return value


def _normalize_instructions(value):
    if value is None:
        return None
    if not isinstance(value, str):
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise UnsupportedFeatureError("Unsupported Responses API feature: instructions is not supported")
    return value


def _validate_request_scalar_fields(body, provider_profile="minimax"):
    _normalize_max_output_tokens(body.get("max_output_tokens"))
    _normalize_temperature(body.get("temperature"), provider_profile=provider_profile)
    _normalize_top_p(body.get("top_p"))
    _normalize_stop(body.get("stop"), provider_profile=provider_profile)
    _normalize_parallel_tool_calls(body.get("parallel_tool_calls"))
    _normalize_metadata(body.get("metadata"))
    _normalize_user(body.get("user"))
    _normalize_stream(body.get("stream"))
    _normalize_instructions(body.get("instructions"))


def _builtin_tool_input_schema(tool_type, environment=None):
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
        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "object",
                    "properties": {
                        "commands": {"type": "array", "items": {"type": "string"}},
                        "timeout_ms": {"type": "integer"},
                        "max_output_length": {"type": "integer"},
                    },
                    "required": ["commands"],
                    "additionalProperties": False,
                },
                "environment": _shell_environment_schema(),
            },
            "required": ["action"],
            "additionalProperties": False,
        }
        environment = _normalize_shell_environment(environment)
        if environment is not None:
            schema["properties"]["environment"]["default"] = environment
        return schema
    raise UnsupportedFeatureError(f"Unsupported Responses API tool type: {tool_type}")


def _shell_environment_schema():
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "local"},
                    "skills": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "path": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["name", "path"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["type"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "container_reference"},
                    "container_id": {"type": "string"},
                },
                "required": ["type", "container_id"],
                "additionalProperties": False,
            },
        ]
    }


def _normalize_domain_list(value, feature_name):
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise UnsupportedFeatureError(f"Unsupported Responses API feature: {feature_name} is not supported")
    normalized = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise UnsupportedFeatureError(f"Unsupported Responses API feature: {feature_name} is not supported")
        normalized.append(item.strip())
    return normalized


def _normalize_web_search_user_location(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search.user_location is not supported")
    unknown_keys = set(value) - {"type", "city", "region", "country", "timezone"}
    if unknown_keys:
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search.user_location is not supported")
    if value.get("type") != "approximate":
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search.user_location is not supported")
    normalized = {"type": "approximate"}
    for key in ("city", "region", "country", "timezone"):
        field_value = value.get(key)
        if field_value is None:
            continue
        if not isinstance(field_value, str) or not field_value.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: web_search.user_location is not supported")
        normalized[key] = field_value.strip()
    return normalized


def _normalize_web_search_tool(tool, provider_profile="minimax"):
    if not _provider_supports_web_search(provider_profile):
        raise UnsupportedFeatureError("Unsupported Responses API tool type: web_search")
    normalized = {"type": "web_search", "name": "web_search"}
    name = tool.get("name")
    if name is not None:
        normalized["name"] = _normalize_tool_choice_name(name, "web_search.name")
    search_context_size = tool.get("search_context_size")
    if search_context_size is not None:
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search.search_context_size is not supported")
    filters = tool.get("filters")
    if filters is not None and not isinstance(filters, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search.filters is not supported")
    allowed_domains = tool.get("allowed_domains")
    blocked_domains = tool.get("blocked_domains")
    user_location = tool.get("user_location")
    if isinstance(filters, dict):
        if allowed_domains is None:
            allowed_domains = filters.get("allowed_domains")
        if blocked_domains is None:
            blocked_domains = filters.get("blocked_domains")
        if user_location is None:
            user_location = filters.get("user_location")
        if filters.get("search_context_size") is not None:
            raise UnsupportedFeatureError("Unsupported Responses API feature: web_search.search_context_size is not supported")
        unknown_filter_keys = set(filters) - {"allowed_domains", "blocked_domains", "user_location", "search_context_size"}
        if unknown_filter_keys:
            raise UnsupportedFeatureError("Unsupported Responses API feature: web_search.filters is not supported")
    allowed_domains = _normalize_domain_list(allowed_domains, "web_search.allowed_domains")
    blocked_domains = _normalize_domain_list(blocked_domains, "web_search.blocked_domains")
    user_location = _normalize_web_search_user_location(user_location)
    if allowed_domains is not None:
        normalized["allowed_domains"] = allowed_domains
    if blocked_domains is not None:
        normalized["blocked_domains"] = blocked_domains
    if user_location is not None:
        normalized["user_location"] = user_location
    return normalized


def _web_search_tool_payload(tool):
    payload = {
        "type": "web_search_20250305",
        "name": tool.get("name", "web_search"),
    }
    for field_name in ("allowed_domains", "blocked_domains", "user_location"):
        field_value = tool.get(field_name)
        if field_value is not None:
            payload[field_name] = field_value
    return payload


def _normalize_shell_skill(skill):
    if not isinstance(skill, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
    unknown_keys = set(skill) - {"name", "path", "description"}
    if unknown_keys:
        raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
    name = skill.get("name")
    path = skill.get("path")
    if not isinstance(name, str) or not name.strip():
        raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
    if not isinstance(path, str) or not path.strip():
        raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
    normalized = {"name": name, "path": path}
    description = skill.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
        normalized["description"] = description
    return normalized


def _custom_tool_description(description, format_spec):
    description = _normalize_tool_description(description)
    format_type = format_spec.get("type", "text")
    if format_type == "text":
        return description
    if format_type == "grammar":
        syntax = format_spec.get("syntax")
        definition = format_spec.get("definition")
        if syntax != "regex" or not isinstance(definition, str) or not definition.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: custom tool format is not supported")
        return description
    raise UnsupportedFeatureError("Unsupported Responses API feature: custom tool format is not supported")


def _custom_tool_input_schema(format_spec):
    format_type = format_spec.get("type", "text")
    if format_type == "text":
        input_property = {"type": "string"}
    elif format_type == "grammar":
        syntax = format_spec.get("syntax")
        definition = format_spec.get("definition")
        if syntax != "regex" or not isinstance(definition, str) or not definition.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: custom tool format is not supported")
        input_property = {"type": "string"}
        input_property["pattern"] = definition
    else:
        raise UnsupportedFeatureError("Unsupported Responses API feature: custom tool format is not supported")
    return {
        "type": "object",
        "properties": {"input": input_property},
        "required": ["input"],
        "additionalProperties": False,
    }


def _normalize_shell_environment(environment):
    if environment is None:
        return None
    if not isinstance(environment, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
    environment_type = environment.get("type")
    if environment_type == "local":
        unknown_keys = set(environment) - {"type", "skills"}
        if unknown_keys:
            raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
        normalized = {"type": "local"}
        skills = environment.get("skills")
        if skills is not None:
            if not isinstance(skills, list):
                raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
            normalized["skills"] = [_normalize_shell_skill(skill) for skill in skills]
        return normalized
    if environment_type == "container_reference":
        unknown_keys = set(environment) - {"type", "container_id"}
        if unknown_keys:
            raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
        container_id = environment.get("container_id")
        if not isinstance(container_id, str) or not container_id.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")
        return {"type": "container_reference", "container_id": container_id}
    raise UnsupportedFeatureError("Unsupported Responses API feature: shell environment is not supported")


def _normalize_mcp_server_label(value):
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedFeatureError("Unsupported Responses API feature: mcp.server_label is not supported")
    return value.strip()


def _normalize_mcp_server_url(value):
    if not isinstance(value, str) or not value.startswith(("https://", "http://")):
        raise UnsupportedFeatureError("Unsupported Responses API feature: mcp.server_url is not supported")
    return value


def _normalize_mcp_authorization(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedFeatureError("Unsupported Responses API feature: mcp.authorization is not supported")
    return value.strip()


def _normalize_mcp_allowed_tools(value):
    if value is None:
        return None
    raw_names = value
    if isinstance(value, dict):
        raw_names = value.get("tool_names")
    if not isinstance(raw_names, list) or not raw_names:
        raise UnsupportedFeatureError("Unsupported Responses API feature: mcp.allowed_tools is not supported")
    normalized = []
    for item in raw_names:
        if not isinstance(item, str) or not item.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: mcp.allowed_tools is not supported")
        name = item.strip()
        if name not in normalized:
            normalized.append(name)
    return normalized


def _normalize_mcp_require_approval(value):
    if value in (None, "never"):
        return value
    raise UnsupportedFeatureError("Unsupported Responses API feature: mcp.require_approval is not supported")


def _normalize_mcp_tool(tool, provider_profile="minimax"):
    if not _provider_supports_mcp(provider_profile):
        raise UnsupportedFeatureError("Unsupported Responses API tool type: mcp")
    connector_id = tool.get("connector_id")
    if connector_id is not None:
        raise UnsupportedFeatureError("Unsupported Responses API feature: mcp.connector_id is not supported")
    normalized = {
        "type": "mcp",
        "server_label": _normalize_mcp_server_label(tool.get("server_label")),
        "server_url": _normalize_mcp_server_url(tool.get("server_url")),
    }
    require_approval = _normalize_mcp_require_approval(tool.get("require_approval"))
    if require_approval is not None:
        normalized["require_approval"] = require_approval
    authorization = _normalize_mcp_authorization(tool.get("authorization"))
    if authorization is not None:
        normalized["authorization"] = authorization
    allowed_tools = _normalize_mcp_allowed_tools(tool.get("allowed_tools"))
    if allowed_tools is not None:
        normalized["allowed_tools"] = allowed_tools
    return normalized


def _translate_file_block(item, provider_profile="minimax"):
    if item.get("file_id"):
        if not _provider_supports_files_api(provider_profile):
            raise UnsupportedFeatureError("Unsupported Responses API feature: input_file.file_id is not supported")
        file_id = item.get("file_id")
        if not isinstance(file_id, str) or not file_id.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: input_file.file_id is not supported")
        media_type = item.get("mime_type")
        if media_type is None:
            media_type = _guess_media_type(item.get("filename") or "")
        if media_type.startswith("image/"):
            return {"type": "image", "source": {"type": "file", "file_id": file_id.strip()}}
        if media_type in {"application/pdf", "text/plain"}:
            return {"type": "document", "source": {"type": "file", "file_id": file_id.strip()}}
        raise UnsupportedFeatureError("Unsupported Responses API feature: input_file.file_id is not supported")

    file_value = item.get("file_data") or item.get("file_url") or ""
    filename = item.get("filename") or ""
    parsed = _parse_data_url(file_value)

    if parsed:
        _validate_data_url_base64(parsed, "Unsupported input_file source")
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


def _translate_content_blocks(content, allow_message_media=True, provider_profile="minimax"):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        raise UnsupportedFeatureError("Unsupported Responses API feature: message content is not supported")

    blocks = []
    for item in content:
        if not isinstance(item, dict):
            raise UnsupportedFeatureError("Unsupported Responses API feature: message content is not supported")

        item_type = item.get("type")
        if item_type in {"input_text", "output_text", "text"}:
            text = item.get("text")
            if not isinstance(text, str):
                raise UnsupportedFeatureError("Unsupported Responses API feature: text content is not supported")
            blocks.append({"type": "text", "text": text})
            continue
        if item_type in {"input_image", "image_url"}:
            if not allow_message_media:
                raise UnsupportedFeatureError(
                    "Unsupported Responses API feature: input_image is not supported by the upstream capability profile"
                )
            instruction = _image_detail_instruction(item.get("detail"))
            if instruction:
                blocks.append({"type": "text", "text": instruction})
            blocks.append(_translate_image_block(item, provider_profile=provider_profile))
            continue
        if item_type in {"input_file", "file"}:
            if not allow_message_media:
                raise UnsupportedFeatureError(
                    "Unsupported Responses API feature: input_file is not supported by the upstream capability profile"
                )
            blocks.append(_translate_file_block(item, provider_profile=provider_profile))
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
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool call name is required")
            call_id = item.get("id")
            if call_id is None:
                call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool call call_id is required")
            input_value = item["input"] if "input" in item else item.get("arguments")
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": _parse_tool_call_arguments(input_value),
                }
            )
            continue
        if item_type == "tool_result":
            tool_use_id = item.get("tool_use_id") if "tool_use_id" in item else item.get("call_id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool_result tool_use_id is required")
            content_value = item.get("content") if "content" in item else item.get("output", "")
            block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": _translate_tool_result_content(content_value),
            }
            if item.get("is_error") is True:
                block["is_error"] = True
            blocks.append(block)
            continue

        raise UnsupportedFeatureError(f"Unsupported content block type: {item_type}")

    return blocks


def _translate_tool_result_content(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _translate_content_blocks(value)
    if isinstance(value, dict) and value.get("type"):
        return _translate_content_blocks([value])
    raise UnsupportedFeatureError("Unsupported Responses API feature: tool_result content is not supported")


def _normalize_tool_call_output_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _translate_content_blocks(value)
    raise UnsupportedFeatureError("Unsupported Responses API feature: tool call output is not supported")


def _builtin_tool_description(tool_type, description, environment=None):
    return _normalize_tool_description(description)


def _require_named_tool(tool_type, tool_name):
    if tool_type in {None, "function", "custom"}:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: tool name is required")
        return tool_name.strip()
    if isinstance(tool_name, str):
        return tool_name.strip()
    return ""


def _normalize_function_parameters(tool):
    parameters = tool.get("parameters")
    if parameters is None and isinstance(tool.get("function"), dict):
        parameters = tool["function"].get("parameters")
    if parameters is None:
        return {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: tool parameters must be an object")
    if parameters.get("type") not in (None, "object"):
        raise UnsupportedFeatureError("Unsupported Responses API feature: tool parameters must be an object")
    return parameters


def _normalize_tool_choice_name(name, feature_name):
    if not isinstance(name, str) or not name.strip():
        raise UnsupportedFeatureError(f"Unsupported Responses API feature: {feature_name} requires a name")
    return name.strip()


def _parse_tool_call_arguments(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise UnsupportedFeatureError(
                "Unsupported Responses API feature: tool call arguments are not supported"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise UnsupportedFeatureError("Unsupported Responses API feature: tool call arguments are not supported")


def _parse_apply_patch_operation(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise UnsupportedFeatureError(
                "Unsupported Responses API feature: apply_patch_call.operation is not supported"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise UnsupportedFeatureError("Unsupported Responses API feature: apply_patch_call.operation is not supported")


def _mcp_toolset_payload(tool):
    payload = {
        "type": "mcp_toolset",
        "mcp_server_name": tool["server_label"],
    }
    allowed_tools = tool.get("allowed_tools")
    if allowed_tools:
        payload["default_config"] = {"enabled": False}
        payload["configs"] = {name: {"enabled": True} for name in allowed_tools}
    return payload


def _translate_tools(tools, provider_profile="minimax"):
    if tools is not None and not isinstance(tools, list):
        raise UnsupportedFeatureError("Unsupported Responses API feature: tools is not supported")
    has_supported_local_tool = _has_supported_translatable_tool(tools, provider_profile=provider_profile)
    translated = []
    mcp_servers = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            raise UnsupportedFeatureError("Unsupported Responses API feature: tools is not supported")
        tool_type = tool.get("type")
        tool_name = tool.get("name") or tool.get("function", {}).get("name", "")
        if _should_ignore_unnamed_hosted_tool(tool_type, tool_name, provider_profile, has_supported_local_tool):
            continue
        if tool_type not in SUPPORTED_RESPONSES_TOOL_TYPES:
            raise UnsupportedFeatureError(
                f"Unsupported Responses API tool type: {tool_type}"
            )
        if tool_type == "mcp":
            normalized_tool = _normalize_mcp_tool(tool, provider_profile=provider_profile)
            server = {
                "type": "url",
                "name": normalized_tool["server_label"],
                "url": normalized_tool["server_url"],
            }
            if normalized_tool.get("authorization"):
                server["authorization_token"] = normalized_tool["authorization"]
            translated.append(_mcp_toolset_payload(normalized_tool))
            mcp_servers.append(server)
            continue
        if tool_type == "web_search":
            translated.append(_web_search_tool_payload(_normalize_web_search_tool(tool, provider_profile=provider_profile)))
            continue
        if tool_type in {"apply_patch", "shell"} and not tool_name:
            tool_name = tool_type
        tool_name = _require_named_tool(tool_type, tool_name)
        if tool_type == "custom":
            format_spec = tool.get("format") or {"type": "text"}
            if not isinstance(format_spec, dict):
                raise UnsupportedFeatureError("Unsupported Responses API feature: custom tool format is not supported")
            translated.append(
                {
                    "name": tool_name,
                    "description": _custom_tool_description(
                        tool.get("description")
                        if "description" in tool
                        else tool.get("function", {}).get("description"),
                        format_spec,
                    ),
                    "input_schema": _custom_tool_input_schema(format_spec),
                }
            )
            continue
        if tool_type in {"apply_patch", "shell"}:
            environment = tool.get("environment")
            translated.append(
                {
                    "name": tool_name,
                    "description": _builtin_tool_description(
                        tool_type,
                        tool.get("description")
                        if "description" in tool
                        else tool.get("function", {}).get("description"),
                        environment,
                    ),
                    "input_schema": _builtin_tool_input_schema(tool_type, environment=environment),
                }
            )
            continue
        translated.append(
            {
                "name": tool_name,
                "description": _normalize_tool_description(
                    tool.get("description")
                    if "description" in tool
                    else tool.get("function", {}).get("description")
                ),
                "input_schema": _normalize_function_parameters(tool),
            }
        )
    return translated, mcp_servers


def _known_response_tool_names(tools):
    names = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        name = tool.get("name") or tool.get("function", {}).get("name", "")
        if tool_type == "mcp":
            for allowed_name in tool.get("allowed_tools") or []:
                if isinstance(allowed_name, str) and allowed_name.strip():
                    names.add(allowed_name.strip())
            continue
        if tool_type in {"apply_patch", "shell", "web_search"} and not name:
            name = tool_type
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _normalize_include(include):
    if include is None:
        return []
    if not isinstance(include, list):
        raise UnsupportedFeatureError("Unsupported Responses API include value")
    normalized = []
    for item in include:
        if item not in SUPPORTED_INCLUDE_VALUES:
            raise UnsupportedFeatureError("Unsupported Responses API feature: include is not supported")
        if item not in normalized:
            normalized.append(item)
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
    if name == "web_search":
        return "web_search_call"
    return None


def _tool_type_lookup(response_context):
    lookup = {}
    if not isinstance(response_context, dict):
        return lookup
    for tool in response_context.get("tools", []):
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "mcp":
            for name in tool.get("allowed_tools") or []:
                if isinstance(name, str) and name.strip():
                    lookup[name.strip()] = "mcp_call"
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
        elif tool_type == "web_search":
            lookup[name.strip()] = "web_search_call"
        else:
            lookup.setdefault(name.strip(), "function_call")
    return lookup


def _tool_definition_lookup(response_context):
    lookup = {}
    if not isinstance(response_context, dict):
        return lookup
    for tool in response_context.get("tools", []):
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or tool.get("function", {}).get("name", "")
        if not isinstance(name, str) or not name.strip():
            continue
        lookup[name.strip()] = tool
    return lookup


def _default_shell_environment(name, response_context=None):
    tool = _tool_definition_lookup(response_context).get(name)
    if not isinstance(tool, dict) or tool.get("type") != "shell":
        return None
    return _normalize_shell_environment(tool.get("environment"))


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


def _normalize_shell_action(action, payload=None):
    if action is None:
        action = payload if isinstance(payload, dict) else {}
    if not isinstance(action, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: shell_call.action is not supported")

    commands = action.get("commands")
    if isinstance(commands, str):
        commands = [commands]
    if commands is not None and not isinstance(commands, list):
        raise UnsupportedFeatureError("Unsupported Responses API feature: shell_call.action.commands is not supported")
    if isinstance(commands, list) and any(not isinstance(command, str) for command in commands):
        raise UnsupportedFeatureError("Unsupported Responses API feature: shell_call.action.commands is not supported")

    normalized = {}
    if isinstance(commands, list):
        normalized["commands"] = list(commands)
    for field_name in ("timeout_ms", "max_output_length"):
        value = action.get(field_name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise UnsupportedFeatureError(
                    f"Unsupported Responses API feature: shell_call.action.{field_name} is not supported"
                )
            normalized[field_name] = value

    if not normalized and payload is not None:
        fallback = _unwrap_custom_tool_payload(payload).strip()
        if fallback:
            normalized["commands"] = [fallback]
    return normalized


def _normalize_reasoning_config(reasoning):
    if not isinstance(reasoning, dict):
        return {"effort": None, "summary": None}
    summary = reasoning.get("summary")
    generate_summary = reasoning.get("generate_summary")
    if summary not in SUPPORTED_REASONING_SUMMARIES:
        raise UnsupportedFeatureError("Unsupported Responses API feature: reasoning.summary is not supported")
    if generate_summary not in SUPPORTED_REASONING_SUMMARIES:
        raise UnsupportedFeatureError("Unsupported Responses API feature: reasoning.generate_summary is not supported")
    normalized_summary = summary if summary is not None else generate_summary
    normalized = {
        "effort": reasoning.get("effort"),
        "summary": normalized_summary,
    }
    if generate_summary is not None:
        normalized["generate_summary"] = generate_summary
    return normalized


def _normalize_top_logprobs(value):
    if value is not None:
        raise UnsupportedFeatureError("Unsupported Responses API feature: top_logprobs is not supported")
    return None


def _output_text_logprobs(response_context=None):
    return None


def _output_text_part(text, response_context=None, annotations=None):
    part = {
        "type": "output_text",
        "text": text,
        "annotations": list(annotations or []),
    }
    logprobs = _output_text_logprobs(response_context)
    if logprobs is not None:
        part["logprobs"] = logprobs
    return part


def _text_done_payload(item_id, output_index, text, response_context=None):
    payload = {
        "type": "response.output_text.done",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": 0,
        "text": text,
    }
    logprobs = _output_text_logprobs(response_context)
    if logprobs is not None:
        payload["logprobs"] = logprobs
    return payload


def _text_delta_payload(item_id, output_index, delta_text, response_context=None):
    payload = {
        "type": "response.output_text.delta",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": 0,
        "delta": delta_text,
    }
    logprobs = _output_text_logprobs(response_context)
    if logprobs is not None:
        payload["logprobs"] = logprobs
    return payload


def _assistant_message_item_payload(item_id, text, status, phase="final_answer", response_context=None, annotations=None):
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "phase": phase,
        "status": status,
        "content": [_output_text_part(text, response_context=response_context, annotations=annotations)],
    }


def _assistant_blocks_have_open_tool_use(blocks):
    return any(
        isinstance(block, dict) and block.get("type") in {"tool_use", "mcp_tool_use", "server_tool_use"}
        for block in blocks
    )


def _apply_assistant_phase(content_blocks, phase):
    if phase != "commentary":
        return content_blocks
    phased_blocks = []
    for block in content_blocks:
        if block.get("type") == "text":
            phased_blocks.append({"type": "thinking", "thinking": block.get("text", "")})
        else:
            phased_blocks.append(block)
    return phased_blocks


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
        item["action"] = _normalize_shell_action(payload_obj.get("action"), payload_obj)
        environment = _normalize_shell_environment(payload_obj.get("environment"))
        if environment is None:
            environment = _default_shell_environment(name, response_context=response_context)
        if environment is not None:
            item["environment"] = environment
        return item
    item["name"] = name
    item["arguments"] = payload
    return item


def _mcp_output_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text_parts = []
        for part in value:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return "".join(text_parts)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            return value["text"]
        return json.dumps(value, ensure_ascii=False)
    return _stringify(value)


def _web_search_sources_include_enabled(response_context=None):
    include = []
    if isinstance(response_context, dict):
        include = response_context.get("include", []) or []
    return "web_search_call.action.sources" in include


def _normalize_url_source(source):
    if not isinstance(source, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call.action.sources is not supported")
    if source.get("type") not in {None, "url"}:
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call.action.sources is not supported")
    url = source.get("url")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call.action.sources is not supported")
    normalized = {"type": "url", "url": url}
    title = source.get("title")
    if title is not None:
        if not isinstance(title, str) or not title.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call.action.sources is not supported")
        normalized["title"] = title.strip()
    return normalized


def _web_search_sources_from_results(content):
    if not isinstance(content, list):
        return []
    sources = []
    seen_urls = set()
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "web_search_result":
            continue
        url = part.get("url")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        source = {"type": "url", "url": url}
        title = part.get("title")
        if isinstance(title, str) and title.strip():
            source["title"] = title.strip()
        sources.append(source)
    return sources


def _web_search_result_blocks_from_sources(sources):
    if not isinstance(sources, list):
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call.action.sources is not supported")
    results = []
    for source in sources:
        normalized = _normalize_url_source(source)
        result = {"type": "web_search_result", "url": normalized["url"]}
        if "title" in normalized:
            result["title"] = normalized["title"]
        results.append(result)
    return results


def _normalize_web_search_action(action):
    if not isinstance(action, dict):
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call.action is not supported")
    action_type = action.get("type", "search")
    if action_type != "search":
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call.action is not supported")
    query = action.get("query")
    if query is None:
        queries = action.get("queries")
        if isinstance(queries, list) and queries and isinstance(queries[0], str):
            query = queries[0]
    if not isinstance(query, str) or not query.strip():
        raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call.action is not supported")
    normalized = {"type": "search", "query": query.strip()}
    sources = action.get("sources")
    if sources is not None:
        normalized["sources"] = [_normalize_url_source(source) for source in sources]
    return normalized


def _web_search_call_item_payload(item_id, query, status, sources=None):
    item = {
        "id": item_id,
        "type": "web_search_call",
        "status": status,
        "action": {
            "type": "search",
            "query": query,
        },
    }
    if sources:
        item["action"]["sources"] = [{"type": "url", "url": source["url"]} for source in sources]
    return item


def _citation_annotation(text, citation):
    if not isinstance(text, str) or not text:
        return None
    if not isinstance(citation, dict):
        return None
    citation_type = citation.get("type")
    if citation_type in {"char_location", "page_location", "content_block_location"}:
        file_id = citation.get("file_id") or citation.get("fileId")
        if isinstance(file_id, str) and file_id:
            annotation = {
                "type": "file_citation",
                "file_id": file_id,
            }
            document_title = citation.get("document_title") or citation.get("documentTitle")
            if isinstance(document_title, str) and document_title.strip():
                annotation["filename"] = document_title.strip()
            document_index = citation.get("document_index")
            if isinstance(document_index, int):
                annotation["index"] = document_index
            return annotation
        return None
    url = None
    title = citation.get("title")
    if citation_type == "web_search_result_location":
        url = citation.get("url")
    elif citation_type == "search_result_location":
        source = citation.get("source")
        if isinstance(source, str) and source.startswith(("https://", "http://")):
            url = source
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        return None
    annotation = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": len(text),
        "url": url,
    }
    if isinstance(title, str) and title.strip():
        annotation["title"] = title.strip()
    return annotation


def _annotations_from_citations(text, citations):
    if not isinstance(citations, list):
        return []
    annotations = []
    for citation in citations:
        annotation = _citation_annotation(text, citation)
        if annotation is not None:
            annotations.append(annotation)
    return annotations


def _offset_annotations(annotations, offset):
    adjusted = []
    for annotation in annotations or []:
        if not isinstance(annotation, dict):
            continue
        item = dict(annotation)
        if item.get("type") == "url_citation":
            item["start_index"] = int(item.get("start_index", 0)) + offset
            item["end_index"] = int(item.get("end_index", 0)) + offset
        adjusted.append(item)
    return adjusted


def _mcp_call_item_payload(item_id, name, server_label, arguments, status, output=None, error=None):
    item = {
        "id": item_id,
        "type": "mcp_call",
        "name": name,
        "server_label": server_label,
        "arguments": arguments,
        "status": status,
    }
    if output is not None:
        item["output"] = output
    if error is not None:
        item["error"] = error
    return item


def _serialized_tool_payload(name, payload, response_context=None):
    item_type = _tool_type_lookup(response_context).get(name)
    if item_type is None:
        item_type = _builtin_tool_type_for_name(name) or "function_call"
    if item_type != "shell_call":
        return payload
    payload_obj = _tool_payload_object(payload)
    if "environment" in payload_obj:
        return payload
    environment = _default_shell_environment(name, response_context=response_context)
    if environment is None:
        return payload
    payload_obj["environment"] = environment
    return json.dumps(payload_obj, ensure_ascii=False)


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


def _encode_reasoning_bridge_block(block, content_text=None):
    if not isinstance(block, dict):
        return None
    block_type = block.get("type")
    if block_type not in {"thinking", "redacted_thinking"}:
        return None
    payload = {"type": block_type}
    thinking_text = content_text if isinstance(content_text, str) and content_text else None
    if thinking_text is None:
        if isinstance(block.get("thinking"), str) and block.get("thinking"):
            thinking_text = block["thinking"]
        elif isinstance(block.get("text"), str) and block.get("text"):
            thinking_text = block["text"]
    if thinking_text:
        payload["thinking"] = thinking_text
    for field_name in ("signature", "data"):
        value = block.get(field_name)
        if isinstance(value, str) and value:
            payload[field_name] = value
    if "signature" not in payload and "data" not in payload:
        return None
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).decode(
        "ascii"
    )
    return f"{REASONING_BRIDGE_PREFIX}{encoded}"


def _decode_reasoning_bridge_block(value):
    if not isinstance(value, str) or not value.startswith(REASONING_BRIDGE_PREFIX):
        raise UnsupportedFeatureError(
            "Unsupported Responses API feature: reasoning.encrypted_content replay is not supported"
        )
    encoded = value[len(REASONING_BRIDGE_PREFIX) :]
    try:
        raw = base64.b64decode(encoded, validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UnsupportedFeatureError(
            "Unsupported Responses API feature: reasoning.encrypted_content replay is not supported"
        ) from exc
    if not isinstance(payload, dict):
        raise UnsupportedFeatureError(
            "Unsupported Responses API feature: reasoning.encrypted_content replay is not supported"
        )
    block_type = payload.get("type")
    if block_type not in {"thinking", "redacted_thinking"}:
        raise UnsupportedFeatureError(
            "Unsupported Responses API feature: reasoning.encrypted_content replay is not supported"
        )
    block = {"type": block_type}
    if isinstance(payload.get("thinking"), str) and payload.get("thinking"):
        block["thinking"] = payload["thinking"]
    for field_name in ("signature", "data"):
        field_value = payload.get(field_name)
        if isinstance(field_value, str) and field_value:
            block[field_name] = field_value
    if "signature" not in block and "data" not in block:
        raise UnsupportedFeatureError(
            "Unsupported Responses API feature: reasoning.encrypted_content replay is not supported"
        )
    return block


def _reasoning_encrypted_content(block, response_context=None, content_text=None):
    include = []
    if isinstance(response_context, dict):
        include = response_context.get("include", []) or []
    if "reasoning.encrypted_content" not in include:
        return None
    return _encode_reasoning_bridge_block(block, content_text=content_text)


def _reasoning_summary_text(text, response_context=None):
    if not isinstance(text, str):
        return ""
    summary_mode = None
    if isinstance(response_context, dict):
        reasoning = response_context.get("reasoning")
        if isinstance(reasoning, dict):
            summary_mode = reasoning.get("summary", reasoning.get("generate_summary"))
    if summary_mode in {None, "auto", "detailed"}:
        return text
    if summary_mode == "concise":
        normalized = " ".join(text.split())
        if not normalized:
            return ""
        match = re.search(r"(.+?[.!?。](?:\s|$))", normalized)
        if match:
            return match.group(1).strip()
        if len(normalized) <= 160:
            return normalized
        return normalized[:157].rstrip() + "..."
    return text


def _reasoning_item_payload(item_id, summary_text, content_text=None, encrypted_content=None, status=None):
    item = {
        "id": item_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": summary_text}],
    }
    if content_text:
        item["content"] = [{"type": "reasoning_text", "text": content_text}]
    if encrypted_content is not None:
        item["encrypted_content"] = encrypted_content
    if status is not None:
        item["status"] = status
    return item


def _reasoning_input_text(item):
    content = item.get("content")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"reasoning_text", "text", "output_text"} and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return "".join(text_parts)
    summary = item.get("summary")
    if isinstance(summary, list):
        text_parts = []
        for part in summary:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "summary_text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return " ".join(text_parts)
    raise UnsupportedFeatureError("Unsupported Responses API feature: reasoning input items require textual content")


def _reasoning_input_block(item):
    if item.get("encrypted_content") is not None:
        block = _decode_reasoning_bridge_block(item.get("encrypted_content"))
        if "thinking" not in block:
            fallback_text = _reasoning_input_text(item)
            if fallback_text:
                block["thinking"] = fallback_text
        return block
    return {
        "type": "thinking",
        "thinking": _reasoning_input_text(item),
    }


def _normalize_response_tools(tools, provider_profile="minimax"):
    if tools is not None and not isinstance(tools, list):
        raise UnsupportedFeatureError("Unsupported Responses API feature: tools is not supported")
    has_supported_local_tool = _has_supported_translatable_tool(tools, provider_profile=provider_profile)
    normalized = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            raise UnsupportedFeatureError("Unsupported Responses API feature: tools is not supported")
        normalized_tool = dict(tool)
        tool_type = normalized_tool.get("type")
        tool_name = normalized_tool.get("name") or normalized_tool.get("function", {}).get("name", "")
        if _should_ignore_unnamed_hosted_tool(tool_type, tool_name, provider_profile, has_supported_local_tool):
            continue
        if tool_type not in SUPPORTED_RESPONSES_TOOL_TYPES:
            raise UnsupportedFeatureError(f"Unsupported Responses API tool type: {tool_type}")
        if tool_type == "mcp":
            normalized.append(_normalize_mcp_tool(normalized_tool, provider_profile=provider_profile))
            continue
        if tool_type == "web_search":
            normalized.append(_normalize_web_search_tool(normalized_tool, provider_profile=provider_profile))
            continue
        if tool_type in {"apply_patch", "shell"} and not normalized_tool.get("name"):
            normalized_tool["name"] = tool_type
        normalized_tool["name"] = _require_named_tool(tool_type, normalized_tool.get("name") or normalized_tool.get("function", {}).get("name", ""))
        if "description" in normalized_tool:
            normalized_tool["description"] = _normalize_tool_description(normalized_tool.get("description"))
        if tool_type == "custom" and "format" not in normalized_tool:
            normalized_tool["format"] = {"type": "text"}
        if tool_type in {None, "function"} and "strict" not in normalized_tool:
            normalized_tool["strict"] = True
        if tool_type in {None, "function"}:
            normalized_tool["parameters"] = _normalize_function_parameters(normalized_tool)
        normalized.append(normalized_tool)
    return normalized


def _custom_tool_input_value(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    raise UnsupportedFeatureError("Unsupported Responses API feature: custom tool input is not supported")


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


def _effective_response_tools(body, provider_profile="minimax"):
    tools = _normalize_response_tools(body.get("tools", []), provider_profile=provider_profile)
    tool_choice = body.get("tool_choice")
    if not isinstance(tool_choice, dict):
        return tools

    choice_type = tool_choice.get("type")
    if choice_type == "custom":
        name = tool_choice.get("name")
        if name is not None:
            name = _normalize_tool_choice_name(name, "tool_choice.custom")
        if name and all(tool.get("name") != name for tool in tools if isinstance(tool, dict)):
            tools.append({"type": "custom", "name": name, "format": {"type": "text"}})
        return tools
    if choice_type in {"apply_patch", "shell"}:
        name = choice_type
        if all(tool.get("name") != name for tool in tools if isinstance(tool, dict)):
            tools.append({"type": choice_type, "name": name})
        return tools
    if choice_type == "web_search":
        if all(tool.get("name") != "web_search" for tool in tools if isinstance(tool, dict)):
            tools.append({"type": "web_search", "name": "web_search"})
        return tools
    if choice_type == "mcp":
        return tools
    if choice_type == "tool":
        name = tool_choice.get("name")
        if name in {"apply_patch", "shell", "web_search"} and all(tool.get("name") != name for tool in tools if isinstance(tool, dict)):
            tools.append({"type": name, "name": name})
    return tools


def _response_completion_from_stop_reason(stop_reason):
    if stop_reason in RESPONSE_COMPLETED_STOP_REASONS:
        return "completed", None
    return "incomplete", RESPONSE_INCOMPLETE_STOP_REASONS.get(stop_reason)


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


def _validate_builtin_tool_output_item(item):
    item_type = item.get("type")
    status = item.get("status")
    if item_type == "apply_patch_call_output" and status not in {None, "completed", "failed"}:
        raise UnsupportedFeatureError("Unsupported Responses API feature: tool call output status is not supported")
    if item_type == "shell_call_output" and status not in {None, "in_progress", "completed", "incomplete", "failed"}:
        raise UnsupportedFeatureError("Unsupported Responses API feature: tool call output status is not supported")


def _builtin_tool_output_content(item):
    item_type = item.get("type")
    output = item.get("output", "")
    metadata = {"type": item_type} if item_type in {"apply_patch_call_output", "shell_call_output"} else {}
    if item.get("id") is not None:
        metadata["id"] = item["id"]
    if item.get("status") is not None:
        metadata["status"] = item["status"]
    if item_type == "shell_call_output" and item.get("max_output_length") is not None:
        metadata["max_output_length"] = item["max_output_length"]
    if metadata or (item_type == "shell_call_output" and isinstance(output, list)):
        payload = dict(metadata)
        payload["output"] = output
        return json.dumps(payload, ensure_ascii=False)
    return _translate_tool_result_content(output)


def _builtin_tool_output_is_error(item):
    if item.get("is_error") is True or item.get("status") == "failed":
        return True
    if item.get("type") != "shell_call_output":
        return False
    output = item.get("output")
    if not isinstance(output, list):
        return False
    for chunk in output:
        if not isinstance(chunk, dict):
            continue
        outcome = chunk.get("outcome")
        if not isinstance(outcome, dict):
            continue
        outcome_type = outcome.get("type")
        if outcome_type == "timeout":
            return True
        if outcome_type == "exit" and outcome.get("exit_code") not in (None, 0):
            return True
    return False


def build_response_context(body, model=None, provider_profile="minimax"):
    body = _require_request_object(body)
    stream = _normalize_stream(body.get("stream"))
    instructions = _normalize_instructions(body.get("instructions"))
    text_config = body.get("text")
    if not isinstance(text_config, dict):
        response_format = body.get("response_format")
        if isinstance(response_format, dict):
            text_config = {"format": response_format}
        else:
            text_config = {"format": {"type": "text"}}

    _validate_request_scalar_fields(body, provider_profile=provider_profile)
    reasoning = _normalize_reasoning_config(body.get("reasoning"))

    return {
        "model": _normalize_model(model if model is not None else body.get("model")),
        "instructions": instructions,
        "max_output_tokens": _normalize_max_output_tokens(body.get("max_output_tokens")),
        "metadata": _normalize_metadata(body.get("metadata")),
        "user": _normalize_user(body.get("user")),
        "store": body.get("store", False),
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": _effective_response_tools(body, provider_profile=provider_profile),
        "text": text_config,
        "temperature": _normalize_temperature(body.get("temperature"), provider_profile=provider_profile)
        if body.get("temperature") is not None
        else 1.0,
        "top_p": _normalize_top_p(body.get("top_p")) if body.get("top_p") is not None else 1.0,
        "parallel_tool_calls": _normalize_parallel_tool_calls(body.get("parallel_tool_calls")),
        "reasoning": reasoning,
        "previous_response_id": body.get("previous_response_id"),
        "truncation": body.get("truncation", "disabled"),
        "max_tool_calls": body.get("max_tool_calls"),
        "background": body.get("background", False),
        "include": _normalize_include(body.get("include")),
        "prompt_cache_key": body.get("prompt_cache_key"),
        "top_logprobs": _normalize_top_logprobs(body.get("top_logprobs")),
        "stream_options": _normalize_stream_options(body.get("stream_options"), stream),
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
            name = tool_choice.get("name")
            if name is None and isinstance(tool_choice.get("function"), dict):
                name = tool_choice["function"].get("name")
            name = _normalize_tool_choice_name(name, "tool_choice.function")
            return {"type": "tool", "name": name}
        if choice_type == "tool":
            name = _normalize_tool_choice_name(tool_choice.get("name"), "tool_choice.tool")
            return {"type": "tool", "name": name}
        if choice_type == "custom":
            name = _normalize_tool_choice_name(tool_choice.get("name"), "tool_choice.custom")
            return {"type": "tool", "name": name}
        if choice_type == "mcp":
            name = _normalize_tool_choice_name(tool_choice.get("name"), "tool_choice.mcp")
            server_label = _normalize_mcp_server_label(tool_choice.get("server_label"))
            return {"type": "tool", "name": name, "_mcp_server_label": server_label}
        if choice_type == "apply_patch":
            return {"type": "tool", "name": "apply_patch"}
        if choice_type == "shell":
            return {"type": "tool", "name": "shell"}
        if choice_type == "web_search":
            return {"type": "tool", "name": "web_search"}
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
    raw_tools = tool_choice.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise UnsupportedFeatureError("Unsupported Responses API feature: tool_choice.allowed_tools is not supported")
    names = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            raise UnsupportedFeatureError("Unsupported Responses API feature: tool_choice.allowed_tools is not supported")
        name = tool.get("name") or tool.get("function", {}).get("name", "")
        if not name and tool.get("type") in {"apply_patch", "shell", "web_search"}:
            name = tool["type"]
        if not isinstance(name, str) or not name.strip():
            raise UnsupportedFeatureError("Unsupported Responses API feature: tool_choice.allowed_tools is not supported")
        names.append(name.strip())
    return set(names)


def _thinking_from_reasoning(body, max_tokens):
    if isinstance(body.get("thinking"), dict):
        return dict(body["thinking"])

    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return None

    reasoning = _normalize_reasoning_config(reasoning)

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
            strict = format_spec.get("strict")
            if strict is None and isinstance(format_spec.get("json_schema"), dict):
                strict = format_spec["json_schema"].get("strict")
            if strict is None:
                strict = True
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
    raise UnsupportedFeatureError("Unsupported Responses API feature: input is not supported")


def translate_responses_request(body, provider_profile="minimax"):
    body = _require_request_object(body)
    provider_profile = _normalize_provider_profile(provider_profile)
    _validate_supported_request_fields(body)
    _validate_request_scalar_fields(body, provider_profile=provider_profile)
    stream = _normalize_stream(body.get("stream"))
    instructions = _normalize_instructions(body.get("instructions"))
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
    _normalize_stream_options(body.get("stream_options"), stream)
    _normalize_top_logprobs(body.get("top_logprobs"))

    result = {
        "model": _normalize_model(body.get("model")),
        "messages": [],
        "stream": stream,
    }
    max_output_tokens = _normalize_max_output_tokens(body.get("max_output_tokens"))
    if max_output_tokens is not None:
        result["max_tokens"] = max_output_tokens

    temperature = _normalize_temperature(body.get("temperature"), provider_profile=provider_profile)
    if temperature is not None:
        result["temperature"] = temperature
    top_p = _normalize_top_p(body.get("top_p"))
    if top_p is not None:
        result["top_p"] = top_p
    stop_sequences = _normalize_stop(body.get("stop"), provider_profile=provider_profile)
    if stop_sequences is not None:
        result["stop_sequences"] = stop_sequences
    thinking = _thinking_from_reasoning(body, result.get("max_tokens"))
    if thinking:
        result["thinking"] = thinking

    system_segments = []
    instruction_items = instructions if isinstance(instructions, list) else []
    if isinstance(instructions, str) and instructions:
        system_segments.append(instructions)
    text_format_instruction = _text_format_instruction(body)
    if text_format_instruction:
        system_segments.append(text_format_instruction)
    text_verbosity_instruction = _text_verbosity_instruction(body)
    if text_verbosity_instruction:
        system_segments.append(text_verbosity_instruction)

    response_tools = _effective_response_tools(body, provider_profile=provider_profile)
    allowed_tool_names = _allowed_tool_names(body.get("tool_choice"))
    tools, mcp_servers = _translate_tools(response_tools, provider_profile=provider_profile)
    if allowed_tool_names is not None:
        known_tool_names = _known_response_tool_names(response_tools)
        missing_names = sorted(name for name in allowed_tool_names if name not in known_tool_names)
        if missing_names:
            raise UnsupportedFeatureError(
                "Unsupported Responses API feature: tool_choice.allowed_tools references unknown tools"
            )
        filtered_response_tools = []
        for tool in response_tools:
            if tool.get("type") == "mcp":
                tool_names = set(tool.get("allowed_tools") or [])
                if not tool_names:
                    continue
                matched_names = [name for name in tool.get("allowed_tools", []) if name in allowed_tool_names]
                if not matched_names:
                    continue
                narrowed_tool = dict(tool)
                narrowed_tool["allowed_tools"] = matched_names
                filtered_response_tools.append(narrowed_tool)
                continue
            elif tool.get("name") not in allowed_tool_names:
                continue
            filtered_response_tools.append(tool)
        response_tools = filtered_response_tools
        tools, mcp_servers = _translate_tools(response_tools, provider_profile=provider_profile)
        if not tools and not mcp_servers:
            raise UnsupportedFeatureError(
                "Unsupported Responses API feature: tool_choice.allowed_tools did not match any tools"
            )

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
        and tool_choice["name"] not in _known_response_tool_names(response_tools)
    ):
        choice_type = tool_choice_input.get("type")
        if choice_type == "custom":
            response_tools.append(
                {
                    "type": choice_type,
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
            response_tools.append(
                {
                    "type": choice_type,
                    "name": tool_choice["name"],
                    "description": "",
                    "input_schema": _builtin_tool_input_schema(choice_type),
                }
            )
    if (
        isinstance(tool_choice_input, dict)
        and tool_choice_input.get("type") == "tool"
        and tool_choice
        and tool_choice.get("type") == "tool"
        and tool_choice["name"] in {"apply_patch", "shell"}
        and tool_choice["name"] not in _known_response_tool_names(response_tools)
    ):
        response_tools.append(
            {
                "type": tool_choice["name"],
                "name": tool_choice["name"],
                "description": "",
                "input_schema": _builtin_tool_input_schema(tool_choice["name"]),
            }
        )
    if (
        isinstance(tool_choice_input, dict)
        and tool_choice_input.get("type") == "mcp"
        and tool_choice
        and tool_choice.get("type") == "tool"
    ):
        target_server = tool_choice.pop("_mcp_server_label", None)
        matching_tool = None
        for tool in response_tools:
            if tool.get("type") != "mcp":
                continue
            if tool.get("server_label") != target_server:
                continue
            allowed_names = tool.get("allowed_tools") or []
            if allowed_names and tool_choice["name"] in allowed_names:
                matching_tool = tool
                break
        if matching_tool is None:
            raise UnsupportedFeatureError(
                "Unsupported Responses API feature: tool_choice references unknown tools"
            )
    if (
        tool_choice
        and tool_choice.get("type") == "tool"
        and tool_choice["name"] not in _known_response_tool_names(response_tools)
    ):
        raise UnsupportedFeatureError(
            "Unsupported Responses API feature: tool_choice references unknown tools"
        )
    tools, mcp_servers = _translate_tools(response_tools, provider_profile=provider_profile)
    if tools:
        result["tools"] = tools
    if mcp_servers:
        result["mcp_servers"] = mcp_servers
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

    for item in [*instruction_items, *_iter_input_items(body.get("input", []))]:
        if not isinstance(item, dict):
            raise UnsupportedFeatureError("Unsupported Responses API feature: input items are not supported")

        item_type = item.get("type") or ("message" if item.get("role") else None)
        if item_type == "message":
            role = item.get("role", "user")
            if role not in {"user", "assistant", "developer", "system"}:
                raise UnsupportedFeatureError("Unsupported Responses API feature: message role is not supported")
            phase = item.get("phase")
            if phase not in {None, "commentary", "final_answer"}:
                raise UnsupportedFeatureError("Unsupported Responses API feature: message phase is not supported")
            if phase is not None and role != "assistant":
                raise UnsupportedFeatureError("Unsupported Responses API feature: message phase is only supported for assistant messages")
            if role in {"developer", "system"}:
                flush_assistant()
                flush_tool_results()
                developer_blocks = _translate_content_blocks(
                    item.get("content", []),
                    allow_message_media=True,
                    provider_profile=provider_profile,
                )
                if any(block.get("type") != "text" for block in developer_blocks):
                    raise UnsupportedFeatureError(
                        "Unsupported Responses API feature: developer messages only support text content"
                    )
                developer_text = "\n\n".join(block.get("text", "") for block in developer_blocks).strip()
                if developer_text:
                    system_segments.append(developer_text)
                continue
            if role == "user":
                flush_assistant()
                flush_tool_results()
            allow_message_media = role == "user" and _provider_supports_message_media(provider_profile)
            translated_content = _translate_content_blocks(
                item.get("content", []),
                allow_message_media=allow_message_media,
                provider_profile=provider_profile,
            )
            if role == "assistant":
                flush_tool_results()
                if current_assistant_blocks and not _assistant_blocks_have_open_tool_use(current_assistant_blocks):
                    flush_assistant()
                translated_content = _apply_assistant_phase(translated_content, phase)
                current_assistant_blocks.extend(translated_content)
                continue
            result["messages"].append({"role": role, "content": translated_content})
            continue

        if item_type in {"function_call", "custom_tool_call", "apply_patch_call", "shell_call"}:
            name = (item.get("name") or "").strip()
            if item_type == "apply_patch_call":
                name = "apply_patch"
            elif item_type == "shell_call":
                name = "shell"
            if not name:
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool call name is required")
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool call call_id is required")
            flush_tool_results()
            if current_assistant_blocks and not _assistant_blocks_have_open_tool_use(current_assistant_blocks):
                flush_assistant()
            if item_type == "custom_tool_call":
                tool_input = _custom_tool_payload(item.get("input"))
            elif item_type == "apply_patch_call":
                tool_input = {"operation": _parse_apply_patch_operation(item.get("operation"))}
            elif item_type == "shell_call":
                legacy_action = {}
                if item.get("commands") is not None:
                    legacy_action["commands"] = item.get("commands")
                for field_name in ("timeout_ms", "max_output_length"):
                    if item.get(field_name) is not None:
                        legacy_action[field_name] = item[field_name]
                shell_action = _normalize_shell_action(item.get("action"), legacy_action)
                tool_input = {"action": shell_action} if shell_action else {}
                shell_environment = _normalize_shell_environment(item.get("environment"))
                if shell_environment is not None:
                    tool_input["environment"] = shell_environment
            else:
                tool_input = _parse_tool_call_arguments(item.get("arguments", item.get("input")))
            current_assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": tool_input,
                }
            )
            continue

        if item_type == "mcp_call":
            call_id = item.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise UnsupportedFeatureError("Unsupported Responses API feature: mcp_call id is required")
            name = (item.get("name") or "").strip()
            if not name:
                raise UnsupportedFeatureError("Unsupported Responses API feature: mcp_call name is required")
            server_label = item.get("server_label")
            if not isinstance(server_label, str) or not server_label.strip():
                raise UnsupportedFeatureError("Unsupported Responses API feature: mcp_call server_label is not supported")
            status = item.get("status")
            if status not in {None, "in_progress", "calling", "completed", "failed"}:
                raise UnsupportedFeatureError("Unsupported Responses API feature: mcp_call status is not supported")
            flush_tool_results()
            if current_assistant_blocks and not _assistant_blocks_have_open_tool_use(current_assistant_blocks):
                flush_assistant()
            current_assistant_blocks.append(
                {
                    "type": "mcp_tool_use",
                    "id": call_id,
                    "name": name,
                    "server_name": server_label.strip(),
                    "input": _parse_tool_call_arguments(item.get("arguments")),
                }
            )
            result_payload = None
            is_error = False
            if item.get("error") is not None:
                result_payload = item.get("error")
                is_error = True
            elif item.get("output") is not None:
                result_payload = item.get("output")
            if result_payload is not None or status in {"completed", "failed"}:
                if isinstance(result_payload, str):
                    result_content = [{"type": "text", "text": result_payload}]
                else:
                    result_content = _translate_tool_result_content(result_payload or "")
                result_block = {
                    "type": "mcp_tool_result",
                    "tool_use_id": call_id,
                    "content": result_content,
                }
                if is_error or status == "failed":
                    result_block["is_error"] = True
                current_assistant_blocks.append(result_block)
            continue

        if item_type == "web_search_call":
            call_id = item.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call id is required")
            status = item.get("status")
            if status not in {None, "searching", "completed", "failed", "incomplete"}:
                raise UnsupportedFeatureError("Unsupported Responses API feature: web_search_call status is not supported")
            action = _normalize_web_search_action(item.get("action"))
            flush_tool_results()
            if current_assistant_blocks and not _assistant_blocks_have_open_tool_use(current_assistant_blocks):
                flush_assistant()
            current_assistant_blocks.append(
                {
                    "type": "server_tool_use",
                    "id": call_id,
                    "name": "web_search",
                    "input": {"query": action["query"]},
                }
            )
            if status in {"completed", "failed"} or action.get("sources"):
                if status == "failed":
                    current_assistant_blocks.append(
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": call_id,
                            "content": {
                                "type": "web_search_tool_result_error",
                                "error_code": "unavailable",
                            },
                        }
                    )
                elif action.get("sources"):
                    current_assistant_blocks.append(
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": call_id,
                            "content": _web_search_result_blocks_from_sources(action["sources"]),
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
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool call output call_id is required")
            _validate_builtin_tool_output_item(item)
            content = (
                _builtin_tool_output_content(item)
                if item_type in {"apply_patch_call_output", "shell_call_output"}
                else _normalize_tool_call_output_value(item.get("output", ""))
            )
            tool_result = {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": content,
            }
            output_status = item.get("status")
            if (
                item_type not in {"apply_patch_call_output", "shell_call_output"}
                and output_status not in {None, "in_progress", "completed", "incomplete"}
            ):
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool call output status is not supported")
            if (
                item_type in {"apply_patch_call_output", "shell_call_output"} and _builtin_tool_output_is_error(item)
            ) or item.get("is_error") is True:
                tool_result["is_error"] = True
            pending_tool_result_blocks.append(tool_result)
            continue

        if item_type == "reasoning":
            flush_tool_results()
            if current_assistant_blocks and not _assistant_blocks_have_open_tool_use(current_assistant_blocks):
                flush_assistant()
            current_assistant_blocks.append(_reasoning_input_block(item))
            continue

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
                        raise UnsupportedFeatureError(
                            "Unsupported Responses API feature: orphan tool_result is not supported"
                        )
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
    final_text_parts = []
    parallel_tool_calls = True if response_context is None else response_context.get("parallel_tool_calls", True)
    tool_call_seen = False
    mcp_calls = {}
    web_search_calls = {}
    response_status, incomplete_details = _response_completion_from_stop_reason(body.get("stop_reason"))
    message_status = "completed" if response_status == "completed" else "incomplete"
    reasoning_status = message_status

    for index, block in enumerate(body.get("content", [])):
        block_type = block.get("type")
        if block_type in {"thinking", "redacted_thinking"}:
            content_text = block.get("thinking") or block.get("text") or ""
            summary_text = _reasoning_summary_text(content_text, response_context=response_context)
            encrypted_content = _reasoning_encrypted_content(
                block,
                response_context=response_context,
                content_text=content_text,
            )
            output.append(
                _reasoning_item_payload(
                    f"rs_{response_id}_{index}",
                    summary_text,
                    content_text=content_text,
                    encrypted_content=encrypted_content,
                    status=reasoning_status,
                )
            )
            if content_text:
                output.append(
                    _assistant_message_item_payload(
                        f"msg_{response_id}_commentary_{index}",
                        content_text,
                        reasoning_status,
                        phase="commentary",
                        response_context=response_context,
                    )
                )
        elif block_type == "text":
            text = block.get("text", "")
            annotations = _annotations_from_citations(text, block.get("citations"))
            final_text_parts.append(text)
            output.append(
                _assistant_message_item_payload(
                    f"msg_{response_id}_{index}",
                    text,
                    message_status,
                    phase="final_answer",
                    response_context=response_context,
                    annotations=annotations,
                )
            )
        elif block_type == "server_tool_use":
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise UnsupportedFeatureError("Unsupported Anthropic response block: server_tool_use.id is required")
            name = block.get("name")
            if name != "web_search":
                raise UnsupportedFeatureError("Unsupported Anthropic response block: server_tool_use is not supported")
            query = ""
            if isinstance(block.get("input"), dict) and isinstance(block["input"].get("query"), str):
                query = block["input"]["query"]
            item = _web_search_call_item_payload(call_id, query, "searching")
            web_search_calls[call_id] = item
            output.append(item)
        elif block_type == "web_search_tool_result":
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str) or not call_id or call_id not in web_search_calls:
                raise UnsupportedFeatureError("Unsupported Anthropic response block: orphan web_search_tool_result is not supported")
            item = web_search_calls[call_id]
            content = block.get("content")
            if isinstance(content, dict) and content.get("type") == "web_search_tool_result_error":
                item["status"] = "failed"
            else:
                item["status"] = "completed"
                if _web_search_sources_include_enabled(response_context):
                    sources = _web_search_sources_from_results(content)
                    if sources:
                        item["action"]["sources"] = [{"type": "url", "url": source["url"]} for source in sources]
        elif block_type == "tool_use":
            if not parallel_tool_calls and tool_call_seen:
                raise UnsupportedFeatureError(
                    "Unsupported Responses API behavior: parallel_tool_calls=false cannot accept multiple tool calls"
                )
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool call call_id is required")
            name = block.get("name")
            if not isinstance(name, str) or not name:
                raise UnsupportedFeatureError("Unsupported Responses API feature: tool call name is required")
            tool_call_seen = True
            output.append(
                _tool_item_payload(
                    call_id,
                    name,
                    json.dumps(block.get("input", {}), ensure_ascii=False),
                    "completed",
                    response_context=response_context,
                )
            )
        elif block_type == "mcp_tool_use":
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise UnsupportedFeatureError("Unsupported Anthropic response block: mcp_tool_use.id is required")
            name = block.get("name")
            if not isinstance(name, str) or not name:
                raise UnsupportedFeatureError("Unsupported Anthropic response block: mcp_tool_use.name is required")
            server_label = block.get("server_name")
            if not isinstance(server_label, str) or not server_label:
                raise UnsupportedFeatureError("Unsupported Anthropic response block: mcp_tool_use.server_name is required")
            item = _mcp_call_item_payload(
                call_id,
                name,
                server_label,
                json.dumps(block.get("input", {}), ensure_ascii=False),
                "calling",
            )
            mcp_calls[call_id] = item
            output.append(item)
        elif block_type == "mcp_tool_result":
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str) or not call_id or call_id not in mcp_calls:
                raise UnsupportedFeatureError("Unsupported Anthropic response block: orphan mcp_tool_result is not supported")
            item = mcp_calls[call_id]
            content_text = _mcp_output_text(block.get("content"))
            if block.get("is_error") is True:
                item["error"] = content_text
                item["status"] = "failed"
            else:
                item["output"] = content_text
                item["status"] = "completed"
        else:
            raise UnsupportedFeatureError(f"Unsupported Anthropic response block: {block_type} is not supported")

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
        "output_text": "".join(final_text_parts),
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
        self.message_annotations = {}
        self.text_block_targets = {}
        self.message_added = set()
        self.content_added = set()
        self.message_done = set()
        self.commentary_buffers = {}
        self.commentary_added = set()
        self.commentary_content_added = set()
        self.commentary_done = set()
        self.reasoning_buffers = {}
        self.reasoning_summary_buffers = {}
        self.reasoning_added = set()
        self.reasoning_done = set()
        self.reasoning_encrypted = {}
        self.tool_calls = {}
        self.tool_args = {}
        self.tool_seed_args = {}
        self.tool_done = set()
        self.mcp_calls = {}
        self.mcp_call_ids = {}
        self.mcp_args = {}
        self.mcp_seed_args = {}
        self.mcp_done = set()
        self.mcp_result_indexes = {}
        self.web_search_calls = {}
        self.web_search_call_ids = {}
        self.web_search_queries = {}
        self.web_search_done = set()
        self.web_search_result_indexes = {}
        self.ignored_tool_indexes = set()
        self.usage = {}
        self.stop_reason = None
        self.next_output_index = 0
        self.reasoning_index = None
        self.commentary_index = None
        self.final_message_index = None
        self.tool_output_indexes = {}
        self.content_block_types = {}
        self.reasoning_meta = {}

    def _include_obfuscation(self):
        stream_options = self.response_context.get("stream_options") or {}
        return stream_options.get("include_obfuscation", True)

    def _delta_payload(self, payload):
        data = dict(payload)
        if self._include_obfuscation():
            data["obfuscation"] = secrets.token_hex(4)
        return data

    def _allocate_output_index(self):
        output_index = self.next_output_index
        self.next_output_index += 1
        return output_index

    def _reasoning_output_index(self):
        if self.reasoning_index is None:
            self.reasoning_index = self._allocate_output_index()
        return self.reasoning_index

    def _commentary_output_index(self):
        if self.commentary_index is None:
            self.commentary_index = self._allocate_output_index()
        return self.commentary_index

    def _final_message_output_index(self):
        if self.final_message_index is None:
            self.final_message_index = self._allocate_output_index()
        return self.final_message_index

    def _tool_output_index(self, index):
        if index not in self.tool_output_indexes:
            self.tool_output_indexes[index] = self._allocate_output_index()
        return self.tool_output_indexes[index]

    def _final_response_status(self):
        return _response_completion_from_stop_reason(self.stop_reason)[0]

    def _final_incomplete_details(self):
        return _response_completion_from_stop_reason(self.stop_reason)[1]

    def _final_message_status(self):
        return "completed" if self._final_response_status() == "completed" else "incomplete"

    def _final_reasoning_status(self):
        return self._final_message_status()

    def _emit(self, event, data):
        self.sequence += 1
        payload = dict(data)
        payload["sequence_number"] = self.sequence
        return {"event": event, "data": payload}

    def _message_id(self, index):
        return f"msg_{self.response_id}_{index}"

    def _commentary_message_id(self, index):
        return f"msg_{self.response_id}_commentary_{index}"

    def _tool_item_id(self, call_id):
        return f"fc_{call_id}"

    def _build_output_items(self):
        items = []
        if self.reasoning_index in self.reasoning_added:
            full_text = self.reasoning_buffers.get(self.reasoning_index, "")
            summary_text = self.reasoning_summary_buffers.get(
                self.reasoning_index,
                _reasoning_summary_text(full_text, response_context=self.response_context),
            )
            encrypted_content = self.reasoning_encrypted.get(self.reasoning_index)
            items.append(
                (
                    self.reasoning_index,
                    _reasoning_item_payload(
                        f"rs_{self.response_id}_{self.reasoning_index}",
                        summary_text,
                        content_text=full_text,
                        encrypted_content=encrypted_content,
                        status=(
                            self._final_reasoning_status()
                            if self.reasoning_index in self.reasoning_done
                            else "in_progress"
                        ),
                    ),
                )
            )
        if self.commentary_index in self.commentary_added:
            text = self.commentary_buffers.get(self.commentary_index, "")
            items.append(
                (
                    self.commentary_index,
                    _assistant_message_item_payload(
                        self._commentary_message_id(self.commentary_index),
                        text,
                        self._final_reasoning_status() if self.commentary_index in self.commentary_done else "in_progress",
                        phase="commentary",
                        response_context=self.response_context,
                        annotations=self.message_annotations.get(self.commentary_index, []),
                    ),
                )
            )
        if self.final_message_index in self.message_added:
            text = self.text_buffers.get(self.final_message_index, "")
            items.append(
                (
                    self.final_message_index,
                    _assistant_message_item_payload(
                        self._message_id(self.final_message_index),
                        text,
                        self._final_message_status() if self.final_message_index in self.message_done else "in_progress",
                        phase="final_answer",
                        response_context=self.response_context,
                        annotations=self.message_annotations.get(self.final_message_index, []),
                    ),
                )
            )
        for index in sorted(self.tool_calls):
            call = self.tool_calls[index]
            call_id = call["call_id"]
            args = self.tool_args.get(index, "") or self.tool_seed_args.get(index, "{}")
            args = _serialized_tool_payload(call["name"], args, response_context=self.response_context)
            items.append(
                (
                    call["output_index"],
                    _tool_item_payload(
                        call_id,
                        call["name"],
                        args,
                        "completed" if index in self.tool_done else "in_progress",
                        response_context=self.response_context,
                    ),
                )
            )
        for index in sorted(self.mcp_calls):
            call = self.mcp_calls[index]
            args = self.mcp_args.get(index, "") or self.mcp_seed_args.get(index, "{}")
            items.append(
                (
                    call["output_index"],
                    _mcp_call_item_payload(
                        call["call_id"],
                        call["name"],
                        call["server_label"],
                        args,
                        call["status"] if index in self.mcp_done else "calling",
                        output=call.get("output"),
                        error=call.get("error"),
                    ),
                )
            )
        for index in sorted(self.web_search_calls):
            call = self.web_search_calls[index]
            item = _web_search_call_item_payload(
                call["call_id"],
                self.web_search_queries.get(index, ""),
                call["status"] if index in self.web_search_done else "searching",
                sources=call.get("sources"),
            )
            items.append((call["output_index"], item))
        items.sort(key=lambda pair: pair[0])
        return [item for _, item in items]

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
            if self.message_added:
                payload["output_text"] = "".join(
                    self.text_buffers.get(idx, "") for idx in sorted(self.message_added)
                )
            else:
                payload["output_text"] = ""
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

    def _ensure_message_started(self, index, phase, added_set, content_added_set, message_id):
        events = []
        if index not in added_set:
            added_set.add(index)
            events.append(
                self._emit(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": index,
                        "item": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "phase": phase,
                            "status": "in_progress",
                            "content": [],
                        },
                    },
                )
            )
        if index not in content_added_set:
            content_added_set.add(index)
            events.append(
                self._emit(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": message_id,
                        "output_index": index,
                        "content_index": 0,
                        "part": _output_text_part("", response_context=self.response_context),
                    },
                )
            )
        return events

    def _ensure_text_started(self, index):
        index = self._final_message_output_index()
        return self._ensure_message_started(
            index,
            "final_answer",
            self.message_added,
            self.content_added,
            self._message_id(index),
        )

    def _ensure_commentary_started(self, index):
        index = self._commentary_output_index()
        return self._ensure_message_started(
            index,
            "commentary",
            self.commentary_added,
            self.commentary_content_added,
            self._commentary_message_id(index),
        )

    def _close_message(self, index, phase, buffers, done_set, message_id, final_status):
        if index in done_set:
            return []
        done_set.add(index)
        text = buffers.get(index, "")
        annotations = self.message_annotations.get(index, [])
        return [
            self._emit(
                "response.output_text.done",
                _text_done_payload(message_id, index, text, response_context=self.response_context),
            ),
            self._emit(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": message_id,
                    "output_index": index,
                    "content_index": 0,
                    "part": _output_text_part(text, response_context=self.response_context, annotations=annotations),
                },
            ),
            self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": _assistant_message_item_payload(
                        message_id,
                        text,
                        final_status,
                        phase=phase,
                        response_context=self.response_context,
                        annotations=annotations,
                    ),
                },
            ),
        ]

    def _close_text(self, index):
        index = self._final_message_output_index()
        return self._close_message(
            index,
            "final_answer",
            self.text_buffers,
            self.message_done,
            self._message_id(index),
            self._final_message_status(),
        )

    def _close_commentary(self, index):
        index = self._commentary_output_index()
        return self._close_message(
            index,
            "commentary",
            self.commentary_buffers,
            self.commentary_done,
            self._commentary_message_id(index),
            self._final_reasoning_status(),
        )

    def _ensure_reasoning_started(self, index):
        index = self._reasoning_output_index()
        if index in self.reasoning_added:
            return []
        self.reasoning_added.add(index)
        item_id = f"rs_{self.response_id}_{index}"
        item = {"id": item_id, "type": "reasoning", "summary": [], "status": "in_progress"}
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
        index = self._reasoning_output_index()
        if index not in self.reasoning_added or index in self.reasoning_done:
            return []
        self.reasoning_done.add(index)
        item_id = f"rs_{self.response_id}_{index}"
        full_text = self.reasoning_buffers.get(index, "")
        text = self.reasoning_summary_buffers.get(
            index,
            _reasoning_summary_text(full_text, response_context=self.response_context),
        )
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
                    "item": _reasoning_item_payload(
                        item_id,
                        text,
                        content_text=full_text,
                        encrypted_content=encrypted_content,
                        status=self._final_reasoning_status(),
                    ),
                },
            ),
        ]

    def _close_tool(self, index):
        if index not in self.tool_calls or index in self.tool_done:
            return []
        self.tool_done.add(index)
        call = self.tool_calls[index]
        call_id = call["call_id"]
        output_index = call["output_index"]
        args = self.tool_args.get(index, "") or self.tool_seed_args.get(index, "{}")
        args = _serialized_tool_payload(call["name"], args, response_context=self.response_context)
        event_spec = _tool_stream_event_spec(call, response_context=self.response_context)
        final_value = _unwrap_custom_tool_payload(args) if event_spec["field_name"] == "input" else args
        return [
            self._emit(
                event_spec["done_event"],
                {
                    "type": event_spec["done_event"],
                    "item_id": self._tool_item_id(call_id),
                    "output_index": output_index,
                    event_spec["field_name"]: final_value,
                },
            ),
            self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
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

    def _close_mcp_call(self, index):
        if index not in self.mcp_calls or index in self.mcp_done:
            return []
        self.mcp_done.add(index)
        call = self.mcp_calls[index]
        args = self.mcp_args.get(index, "") or self.mcp_seed_args.get(index, "{}")
        item = _mcp_call_item_payload(
            call["call_id"],
            call["name"],
            call["server_label"],
            args,
            call["status"],
            output=call.get("output"),
            error=call.get("error"),
        )
        return [
            self._emit(
                "response.mcp_call.completed",
                {
                    "type": "response.mcp_call.completed",
                    "output_index": call["output_index"],
                    "item": item,
                },
            ),
            self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": call["output_index"],
                    "item": item,
                },
            ),
        ]

    def _close_web_search_call(self, index):
        if index not in self.web_search_calls or index in self.web_search_done:
            return []
        self.web_search_done.add(index)
        call = self.web_search_calls[index]
        item = _web_search_call_item_payload(
            call["call_id"],
            self.web_search_queries.get(index, ""),
            call["status"],
            sources=call.get("sources"),
        )
        return [
            self._emit(
                "response.web_search_call.completed",
                {
                    "type": "response.web_search_call.completed",
                    "item_id": call["call_id"],
                    "output_index": call["output_index"],
                    "item": item,
                },
            ),
            self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": call["output_index"],
                    "item": item,
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
            self.content_block_types[index] = block.get("type")
            if block.get("type") == "text":
                output_index = self._final_message_output_index()
                events.extend(self._ensure_text_started(index))
                start_offset = len(self.text_buffers.get(output_index, ""))
                self.text_block_targets[index] = {"output_index": output_index, "start_offset": start_offset}
                initial_text = block.get("text", "")
                if isinstance(initial_text, str) and initial_text:
                    self.text_buffers[output_index] = self.text_buffers.get(output_index, "") + initial_text
                annotations = _annotations_from_citations(initial_text, block.get("citations"))
                if annotations:
                    self.message_annotations.setdefault(output_index, []).extend(
                        _offset_annotations(annotations, start_offset)
                    )
            elif block.get("type") in {"thinking", "redacted_thinking"}:
                reasoning_index = self._reasoning_output_index()
                meta = {"type": block.get("type", "thinking")}
                for field_name in ("signature", "data"):
                    field_value = block.get(field_name)
                    if isinstance(field_value, str) and field_value:
                        meta[field_name] = field_value
                self.reasoning_meta[reasoning_index] = meta
                initial_text = block.get("thinking") or block.get("text") or ""
                if initial_text:
                    self.reasoning_buffers[reasoning_index] = self.reasoning_buffers.get(reasoning_index, "") + initial_text
                encrypted_content = _reasoning_encrypted_content(
                    meta,
                    response_context=self.response_context,
                    content_text=self.reasoning_buffers.get(reasoning_index, ""),
                )
                if encrypted_content is not None:
                    self.reasoning_encrypted[reasoning_index] = encrypted_content
                events.extend(self._ensure_reasoning_started(reasoning_index))
                if block.get("type") == "thinking" and initial_text:
                    commentary_index = self._commentary_output_index()
                    self.commentary_buffers[commentary_index] = self.commentary_buffers.get(commentary_index, "") + initial_text
                    events.extend(self._ensure_commentary_started(commentary_index))
            elif block.get("type") == "tool_use":
                if not self.response_context.get("parallel_tool_calls", True) and self.tool_calls:
                    raise UnsupportedFeatureError(
                        "Unsupported Responses API behavior: parallel_tool_calls=false cannot accept multiple tool calls"
                    )
                call_id = block.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise UnsupportedFeatureError("Unsupported Responses API feature: tool call call_id is required")
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    raise UnsupportedFeatureError("Unsupported Responses API feature: tool call name is required")
                output_index = self._tool_output_index(index)
                self.tool_calls[index] = {"call_id": call_id, "name": name, "output_index": output_index}
                self.tool_args[index] = ""
                self.tool_seed_args[index] = json.dumps(block.get("input", {}), ensure_ascii=False)
                events.append(
                    self._emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
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
            elif block.get("type") == "mcp_tool_use":
                call_id = block.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise UnsupportedFeatureError("Unsupported Anthropic response block: mcp_tool_use.id is required")
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    raise UnsupportedFeatureError("Unsupported Anthropic response block: mcp_tool_use.name is required")
                server_label = block.get("server_name")
                if not isinstance(server_label, str) or not server_label:
                    raise UnsupportedFeatureError(
                        "Unsupported Anthropic response block: mcp_tool_use.server_name is required"
                    )
                output_index = self._tool_output_index(index)
                self.mcp_calls[index] = {
                    "call_id": call_id,
                    "name": name,
                    "server_label": server_label,
                    "output_index": output_index,
                    "status": "calling",
                }
                self.mcp_call_ids[call_id] = index
                self.mcp_args[index] = ""
                self.mcp_seed_args[index] = json.dumps(block.get("input", {}), ensure_ascii=False)
                item = _mcp_call_item_payload(call_id, name, server_label, "", "calling")
                events.append(
                    self._emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": item,
                        },
                    )
                )
                events.append(
                    self._emit(
                        "response.mcp_call.in_progress",
                        {
                            "type": "response.mcp_call.in_progress",
                            "output_index": output_index,
                            "item": item,
                        },
                    )
                )
            elif block.get("type") == "mcp_tool_result":
                call_id = block.get("tool_use_id")
                if not isinstance(call_id, str) or not call_id or call_id not in self.mcp_call_ids:
                    raise UnsupportedFeatureError("Unsupported Anthropic response block: orphan mcp_tool_result is not supported")
                call_index = self.mcp_call_ids[call_id]
                self.mcp_result_indexes[index] = call_index
                call = self.mcp_calls[call_index]
                content_text = _mcp_output_text(block.get("content"))
                if block.get("is_error") is True:
                    call["error"] = content_text
                    call["status"] = "failed"
                else:
                    call["output"] = content_text
                    call["status"] = "completed"
            elif block.get("type") == "server_tool_use":
                call_id = block.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise UnsupportedFeatureError("Unsupported Anthropic response block: server_tool_use.id is required")
                name = block.get("name")
                if name != "web_search":
                    raise UnsupportedFeatureError("Unsupported Anthropic response block: server_tool_use is not supported")
                output_index = self._tool_output_index(index)
                query = ""
                if isinstance(block.get("input"), dict) and isinstance(block["input"].get("query"), str):
                    query = block["input"]["query"]
                self.web_search_calls[index] = {
                    "call_id": call_id,
                    "output_index": output_index,
                    "status": "searching",
                    "sources": [],
                }
                self.web_search_call_ids[call_id] = index
                self.web_search_queries[index] = query
                item = _web_search_call_item_payload(call_id, query, "searching")
                events.append(
                    self._emit(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": item,
                        },
                    )
                )
                events.append(
                    self._emit(
                        "response.web_search_call.searching",
                        {
                            "type": "response.web_search_call.searching",
                            "item_id": call_id,
                            "output_index": output_index,
                            "item": item,
                        },
                    )
                )
            elif block.get("type") == "web_search_tool_result":
                call_id = block.get("tool_use_id")
                if not isinstance(call_id, str) or not call_id or call_id not in self.web_search_call_ids:
                    raise UnsupportedFeatureError(
                        "Unsupported Anthropic response block: orphan web_search_tool_result is not supported"
                    )
                call_index = self.web_search_call_ids[call_id]
                self.web_search_result_indexes[index] = call_index
                call = self.web_search_calls[call_index]
                content = block.get("content")
                if isinstance(content, dict) and content.get("type") == "web_search_tool_result_error":
                    call["status"] = "failed"
                else:
                    call["status"] = "completed"
                    if _web_search_sources_include_enabled(self.response_context):
                        call["sources"] = _web_search_sources_from_results(content)
            elif block.get("type") is not None:
                raise UnsupportedFeatureError(
                    f"Unsupported Anthropic response block: {block.get('type')} is not supported"
                )
            return events

        if event_type == "content_block_delta":
            index = event.get("index", 0)
            delta = event.get("delta", {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                output_index = self._final_message_output_index()
                self.text_buffers[output_index] = self.text_buffers.get(output_index, "") + text
                events.extend(self._ensure_text_started(output_index))
                events.append(
                    self._emit(
                        "response.output_text.delta",
                        self._delta_payload(
                            _text_delta_payload(
                                self._message_id(output_index),
                                output_index,
                                text,
                                response_context=self.response_context,
                            )
                        ),
                    )
                )
                return events
            if delta_type == "citations_delta":
                target = self.text_block_targets.get(index)
                if target is None:
                    return events
                output_index = target["output_index"]
                current_text = self.text_buffers.get(output_index, "")
                start_offset = target["start_offset"]
                block_text = current_text[start_offset:]
                citation = delta.get("citation")
                annotations = _annotations_from_citations(block_text, [citation])
                if annotations:
                    self.message_annotations.setdefault(output_index, []).extend(
                        _offset_annotations(annotations, start_offset)
                    )
                return events
            if delta_type == "thinking_delta":
                text = delta.get("thinking", "")
                reasoning_index = self._reasoning_output_index()
                commentary_index = self._commentary_output_index()
                self.reasoning_buffers[reasoning_index] = self.reasoning_buffers.get(reasoning_index, "") + text
                self.commentary_buffers[commentary_index] = self.commentary_buffers.get(commentary_index, "") + text
                meta = self.reasoning_meta.setdefault(reasoning_index, {"type": "thinking"})
                encrypted_content = _reasoning_encrypted_content(
                    meta,
                    response_context=self.response_context,
                    content_text=self.reasoning_buffers[reasoning_index],
                )
                if encrypted_content is not None:
                    self.reasoning_encrypted[reasoning_index] = encrypted_content
                events.extend(self._ensure_reasoning_started(reasoning_index))
                events.extend(self._ensure_commentary_started(commentary_index))
                summary_text = _reasoning_summary_text(
                    self.reasoning_buffers[reasoning_index],
                    response_context=self.response_context,
                )
                previous_summary = self.reasoning_summary_buffers.get(reasoning_index, "")
                self.reasoning_summary_buffers[reasoning_index] = summary_text
                summary_delta = summary_text[len(previous_summary):] if summary_text.startswith(previous_summary) else summary_text
                events.append(
                    self._emit(
                        "response.output_text.delta",
                        self._delta_payload(
                            _text_delta_payload(
                                self._commentary_message_id(commentary_index),
                                commentary_index,
                                text,
                                response_context=self.response_context,
                            )
                        ),
                    )
                )
                if summary_delta:
                    events.append(
                        self._emit(
                            "response.reasoning_summary_text.delta",
                            self._delta_payload(
                                {
                                "type": "response.reasoning_summary_text.delta",
                                "item_id": f"rs_{self.response_id}_{reasoning_index}",
                                "output_index": reasoning_index,
                                "summary_index": 0,
                                "delta": summary_delta,
                                }
                            ),
                        )
                    )
                return events
            if delta_type == "signature_delta":
                reasoning_index = self._reasoning_output_index()
                signature = delta.get("signature")
                if isinstance(signature, str) and signature:
                    meta = self.reasoning_meta.setdefault(reasoning_index, {"type": "thinking"})
                    meta["signature"] = signature
                    encrypted_content = _reasoning_encrypted_content(
                        meta,
                        response_context=self.response_context,
                        content_text=self.reasoning_buffers.get(reasoning_index, ""),
                    )
                    if encrypted_content is not None:
                        self.reasoning_encrypted[reasoning_index] = encrypted_content
                return events
            if delta_type == "input_json_delta":
                if index in self.web_search_calls:
                    partial = delta.get("partial_json", "")
                    current = self.web_search_queries.get(index, "")
                    raw = current + partial
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        self.web_search_queries[index] = raw
                    else:
                        if isinstance(parsed, dict) and isinstance(parsed.get("query"), str):
                            self.web_search_queries[index] = parsed["query"]
                        else:
                            self.web_search_queries[index] = raw
                    return events
                if index in self.mcp_calls:
                    partial = delta.get("partial_json", "")
                    self.mcp_args[index] = self.mcp_args.get(index, "") + partial
                    call = self.mcp_calls[index]
                    events.append(
                        self._emit(
                            "response.mcp_call_arguments.delta",
                            self._delta_payload(
                                {
                                    "type": "response.mcp_call_arguments.delta",
                                    "item_id": call["call_id"],
                                    "output_index": call["output_index"],
                                    "delta": partial,
                                }
                            ),
                        )
                    )
                    return events
                if index in self.ignored_tool_indexes or index not in self.tool_calls:
                    return events
                partial = delta.get("partial_json", "")
                self.tool_args[index] = self.tool_args.get(index, "") + partial
                call = self.tool_calls[index]
                call_id = call["call_id"]
                event_spec = _tool_stream_event_spec(
                    call,
                    response_context=self.response_context,
                )
                events.append(
                    self._emit(
                        event_spec["delta_event"],
                        self._delta_payload(
                            {
                            "type": event_spec["delta_event"],
                            "item_id": self._tool_item_id(call_id),
                            "output_index": call["output_index"],
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
            if self.content_block_types.get(index) in {"thinking", "redacted_thinking"} and self.reasoning_index is not None:
                reasoning_index = self._reasoning_output_index()
                meta = self.reasoning_meta.get(reasoning_index, {"type": "thinking"})
                encrypted_content = _reasoning_encrypted_content(
                    meta,
                    response_context=self.response_context,
                    content_text=self.reasoning_buffers.get(reasoning_index, ""),
                )
                if encrypted_content is not None:
                    self.reasoning_encrypted[reasoning_index] = encrypted_content
            if index in self.tool_calls:
                events.extend(self._close_tool(index))
            if index in self.mcp_calls:
                call = self.mcp_calls[index]
                arguments = self.mcp_args.get(index, "") or self.mcp_seed_args.get(index, "{}")
                events.append(
                    self._emit(
                        "response.mcp_call_arguments.done",
                        {
                            "type": "response.mcp_call_arguments.done",
                            "item_id": call["call_id"],
                            "output_index": call["output_index"],
                            "arguments": arguments,
                        },
                    )
                )
            if index in self.mcp_result_indexes:
                events.extend(self._close_mcp_call(self.mcp_result_indexes[index]))
            if index in self.web_search_result_indexes:
                events.extend(self._close_web_search_call(self.web_search_result_indexes[index]))
            return events

        if event_type == "message_delta":
            self.usage = event.get("usage", {}) or {}
            delta = event.get("delta", {}) or {}
            if delta.get("stop_reason") is not None:
                self.stop_reason = delta.get("stop_reason")
            return events

        if event_type == "message_stop":
            for index in sorted(self.commentary_added):
                events.extend(self._close_commentary(index))
            for index in sorted(self.message_added):
                events.extend(self._close_text(index))
            for index in sorted(self.reasoning_added):
                events.extend(self._close_reasoning(index))
            for index in sorted(self.tool_calls):
                events.extend(self._close_tool(index))
            for index in sorted(self.mcp_calls):
                if index not in self.mcp_done:
                    events.extend(self._close_mcp_call(index))
            for index in sorted(self.web_search_calls):
                if index not in self.web_search_done:
                    events.extend(self._close_web_search_call(index))
            events.extend(self._emit_completed())
            return events

        return events

    def finish(self):
        events = []
        for index in sorted(self.commentary_added):
            events.extend(self._close_commentary(index))
        for index in sorted(self.message_added):
            events.extend(self._close_text(index))
        for index in sorted(self.reasoning_added):
            events.extend(self._close_reasoning(index))
        for index in sorted(self.tool_calls):
            events.extend(self._close_tool(index))
        for index in sorted(self.mcp_calls):
            if index not in self.mcp_done:
                events.extend(self._close_mcp_call(index))
        for index in sorted(self.web_search_calls):
            if index not in self.web_search_done:
                events.extend(self._close_web_search_call(index))
        events.extend(self._emit_completed())
        return events


def format_sse(event_name, data):
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
