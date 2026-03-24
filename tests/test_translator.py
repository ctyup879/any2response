import pytest

from app.translator import (
    UnsupportedFeatureError,
    build_response_context,
    translate_anthropic_response,
    translate_responses_request,
)


def test_translate_responses_request_to_anthropic_messages():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "instructions": "You are helpful.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"status": "ok"},
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "lookup_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "max_output_tokens": 256,
        "stream": True,
    }

    translated = translate_responses_request(request_body)

    assert translated["model"] == "codex-MiniMax-M2.7"
    assert translated["system"] == "You are helpful."
    assert translated["stream"] is True
    assert translated["max_tokens"] == 256
    assert translated["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert translated["tools"] == [
        {
            "name": "lookup_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


def test_translate_responses_request_supports_custom_tools():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        "tools": [
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch",
            }
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["tools"] == [
        {
            "name": "apply_patch",
            "description": "Apply a patch",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_translate_responses_request_skips_nameless_hosted_tools():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        "tools": [
            {"type": "web_search"},
            {
                "type": "function",
                "name": "echo",
                "description": "Echo input",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["tools"] == [
        {
            "name": "echo",
            "description": "Echo input",
            "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }
    ]


def test_translate_responses_request_rejects_named_unsupported_tools():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        "tools": [
            {"type": "web_search", "name": "search"},
        ],
    }

    with pytest.raises(UnsupportedFeatureError, match="web_search"):
        translate_responses_request(request_body)


def test_translate_responses_request_maps_developer_messages_to_system():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "instructions": "Base system",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Developer policy"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["system"] == "Base system\n\nDeveloper policy"
    assert translated["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_translate_responses_request_groups_consecutive_function_calls():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "tool_a",
                "arguments": {"x": 1},
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "tool_b",
                "arguments": {"y": 2},
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "tool_a", "input": {"x": 1}},
                {"type": "tool_use", "id": "call_2", "name": "tool_b", "input": {"y": 2}},
            ],
        }
    ]


def test_translate_responses_request_accepts_custom_tool_call_items():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": '{"path":"README.md"}',
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_patch", "name": "apply_patch", "input": {"path": "README.md"}},
            ],
        }
    ]


def test_translate_responses_request_groups_consecutive_tool_results():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "tool_a",
                "arguments": {"x": 1},
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "tool_b",
                "arguments": {"y": 2},
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"ok": 1},
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": {"ok": 2},
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "tool_a", "input": {"x": 1}},
                {"type": "tool_use", "id": "call_2", "name": "tool_b", "input": {"y": 2}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": '{"ok": 1}'},
                {"type": "tool_result", "tool_use_id": "call_2", "content": '{"ok": 2}'},
            ],
        }
    ]


def test_translate_responses_request_accepts_custom_tool_call_output_items():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": '{"path":"README.md"}',
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_patch",
                "output": [{"type": "input_text", "text": "patch applied"}],
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_patch", "name": "apply_patch", "input": {"path": "README.md"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_patch",
                    "content": [{"type": "text", "text": "patch applied"}],
                },
            ],
        },
    ]


def test_translate_responses_request_filters_orphan_tool_results():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_ok",
                "name": "tool_a",
                "arguments": {"x": 1},
            },
            {
                "type": "function_call_output",
                "call_id": "call_missing",
                "output": {"bad": True},
            },
            {
                "type": "function_call_output",
                "call_id": "call_ok",
                "output": {"good": True},
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_ok", "name": "tool_a", "input": {"x": 1}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_ok", "content": '{"good": true}'}],
        },
    ]


def test_translate_anthropic_response_maps_tool_use_to_function_call():
    body = {
        "id": "msg_tool",
        "content": [
            {"type": "text", "text": "Checking weather"},
            {"type": "tool_use", "id": "call_1", "name": "lookup_weather", "input": {"city": "Shanghai"}},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    translated = translate_anthropic_response(body, "codex-MiniMax-M2.7")

    assert translated["status"] == "completed"
    assert translated["output"][0]["type"] == "message"
    assert translated["output"][1] == {
        "id": "fc_call_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup_weather",
        "arguments": '{"city": "Shanghai"}',
        "status": "completed",
    }
    assert translated["usage"] == {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 15,
    }


def test_translate_anthropic_response_maps_custom_tool_use_to_custom_tool_call():
    body = {
        "id": "msg_tool",
        "content": [
            {"type": "tool_use", "id": "call_patch", "name": "apply_patch", "input": {"path": "README.md"}},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    translated = translate_anthropic_response(
        body,
        "codex-MiniMax-M2.7",
        response_context={"tools": [{"type": "custom", "name": "apply_patch"}]},
    )

    assert translated["output"] == [
        {
            "id": "fc_call_patch",
            "type": "custom_tool_call",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": '{"path": "README.md"}',
            "status": "completed",
        }
    ]


def test_translate_anthropic_response_includes_reasoning_encrypted_content_when_requested():
    body = {
        "id": "msg_reasoning",
        "content": [
            {"type": "thinking", "thinking": "Plan", "signature": "enc_sig_123"},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    translated = translate_anthropic_response(
        body,
        "codex-MiniMax-M2.7",
        response_context={"include": ["reasoning.encrypted_content"]},
    )

    assert translated["output"] == [
        {
            "id": "rs_resp_msg_reasoning_0",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "content": [{"type": "reasoning_text", "text": "Plan"}],
            "encrypted_content": "enc_sig_123",
        }
    ]


def test_translate_anthropic_response_includes_request_context_fields():
    body = {
        "id": "msg_ctx",
        "content": [{"type": "text", "text": "Hello"}],
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }
    response_context = {
        "instructions": "Be concise",
        "max_output_tokens": 128,
        "metadata": {"request_id": "abc"},
        "user": "user-123",
        "store": False,
        "tool_choice": "auto",
        "tools": [{"type": "function", "name": "echo"}],
        "text": {"format": {"type": "text"}},
        "temperature": 0.2,
        "top_p": 0.9,
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "cache-key-123",
    }

    translated = translate_anthropic_response(
        body,
        "codex-MiniMax-M2.7",
        response_context=response_context,
    )

    assert translated["instructions"] == "Be concise"
    assert translated["max_output_tokens"] == 128
    assert translated["metadata"] == {"request_id": "abc"}
    assert translated["user"] == "user-123"
    assert translated["store"] is False
    assert translated["tool_choice"] == "auto"
    assert translated["tools"] == [{"type": "function", "name": "echo"}]
    assert translated["text"] == {"format": {"type": "text"}}
    assert translated["temperature"] == 0.2
    assert translated["top_p"] == 0.9
    assert translated["parallel_tool_calls"] is True
    assert translated["include"] == ["reasoning.encrypted_content"]
    assert translated["prompt_cache_key"] == "cache-key-123"
    assert translated["usage"]["input_tokens_details"] == {"cached_tokens": 0}
    assert translated["usage"]["output_tokens_details"] == {"reasoning_tokens": 0}
    assert translated["output_text"] == "Hello"
    assert translated["completed_at"] >= translated["created_at"]


def test_translate_responses_request_accepts_reasoning_encrypted_content_include():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "cache-key-123",
            "parallel_tool_calls": False,
        }
    )

    assert translated["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_translate_responses_request_rejects_nonzero_top_logprobs():
    with pytest.raises(UnsupportedFeatureError, match="top_logprobs"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "top_logprobs": 3,
            }
        )


def test_build_response_context_preserves_zero_top_logprobs():
    context = build_response_context(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "top_logprobs": 0,
        }
    )

    assert context["top_logprobs"] == 0


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("conversation", "conv_123"),
        ("context_management", {"type": "auto"}),
        ("prompt", {"id": "pmpt_123"}),
        ("prompt_cache_retention", {"strategy": "session"}),
        ("safety_identifier", "safety-user"),
        ("service_tier", "auto"),
    ],
)
def test_translate_responses_request_rejects_unsupported_official_fields(field_name, field_value):
    body = {
        "model": "codex-MiniMax-M2.7",
        "input": "hello",
        field_name: field_value,
    }

    with pytest.raises(UnsupportedFeatureError, match=field_name):
        translate_responses_request(body)


def test_translate_responses_request_rejects_unknown_include_values():
    with pytest.raises(UnsupportedFeatureError, match="include"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "include": ["output_text.logprobs"],
            }
        )


def test_translate_anthropic_response_rejects_multiple_tool_calls_when_parallel_disabled():
    body = {
        "id": "msg_multi_tool",
        "content": [
            {"type": "tool_use", "id": "call_1", "name": "tool_a", "input": {"x": 1}},
            {"type": "tool_use", "id": "call_2", "name": "tool_b", "input": {"y": 2}},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    with pytest.raises(UnsupportedFeatureError, match="parallel_tool_calls"):
        translate_anthropic_response(
            body,
            "codex-MiniMax-M2.7",
            response_context={"parallel_tool_calls": False},
        )


def test_translate_responses_request_accepts_string_input():
    translated = translate_responses_request({"model": "codex-MiniMax-M2.7", "input": "hello"})

    assert translated["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_translate_responses_request_supports_multimodal_user_content():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "see attached"},
                    {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
                    {"type": "input_image", "image_url": "https://example.com/cat.png"},
                    {
                        "type": "input_file",
                        "filename": "spec.pdf",
                        "file_data": "data:application/pdf;base64,JVBERi0x",
                    },
                ],
            }
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see attached"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0x"},
                },
            ],
        }
    ]


def test_translate_responses_request_supports_inline_text_file_content():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "summarize this"},
                    {
                        "type": "input_file",
                        "filename": "notes.txt",
                        "file_data": "data:text/plain;base64,SGVsbG8gZnJvbSBmaWxl",
                    },
                ],
            }
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summarize this"},
                {"type": "text", "text": "Hello from file"},
            ],
        }
    ]


def test_translate_responses_request_supports_inline_json_file_content():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "payload.json",
                        "file_data": "data:application/json;base64,eyJoZWxsbyI6ICJ3b3JsZCJ9",
                    },
                ],
            }
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": '{"hello": "world"}'},
            ],
        }
    ]


def test_translate_responses_request_supports_text_format_and_sampling_controls():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "instructions": "Base system",
        "input": "hello",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "weather_result",
                "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        },
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["DONE"],
        "tool_choice": "none",
    }

    translated = translate_responses_request(request_body)

    assert translated["temperature"] == 0.2
    assert translated["top_p"] == 0.9
    assert translated["stop_sequences"] == ["DONE"]
    assert translated["tool_choice"] == {"type": "none"}
    assert "Base system" in translated["system"]
    assert "strictly follows this JSON schema" in translated["system"]
    assert '"city"' in translated["system"]


def test_translate_responses_request_rejects_previous_response_id():
    with pytest.raises(UnsupportedFeatureError, match="previous_response_id"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "previous_response_id": "resp_123",
                "input": "hello",
            }
        )


def test_translate_responses_request_rejects_item_reference_inputs():
    with pytest.raises(UnsupportedFeatureError, match="item_reference"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [{"type": "item_reference", "id": "msg_123"}],
            }
        )


def test_translate_responses_request_rejects_background_mode():
    with pytest.raises(UnsupportedFeatureError, match="background"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "background": True,
                "input": "hello",
            }
        )


def test_translate_responses_request_rejects_store_true():
    with pytest.raises(UnsupportedFeatureError, match="store"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "store": True,
                "input": "hello",
            }
        )


def test_translate_responses_request_rejects_non_default_truncation():
    with pytest.raises(UnsupportedFeatureError, match="truncation"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "truncation": "auto",
                "input": "hello",
            }
        )


def test_translate_responses_request_rejects_max_tool_calls():
    with pytest.raises(UnsupportedFeatureError, match="max_tool_calls"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "max_tool_calls": 1,
                "input": "hello",
            }
        )


def test_translate_responses_request_supports_embedded_tool_blocks():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Calling tool"},
                    {"type": "tool_use", "id": "call_1", "name": "lookup_weather", "input": {"city": "Shanghai"}},
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": '{"temp_c":25}',
                        "is_error": True,
                    }
                ],
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Calling tool"},
                {"type": "tool_use", "id": "call_1", "name": "lookup_weather", "input": {"city": "Shanghai"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": '{"temp_c":25}',
                    "is_error": True,
                }
            ],
        },
    ]
    assert translated["tools"] == [
        {
            "name": "lookup_weather",
            "description": "",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_translate_anthropic_response_maps_thinking_to_reasoning_output():
    body = {
        "id": "msg_reasoning",
        "content": [
            {"type": "thinking", "thinking": "Need to inspect weather first", "signature": "sig_1"},
            {"type": "text", "text": "It is sunny."},
        ],
        "usage": {"input_tokens": 7, "output_tokens": 11},
    }

    translated = translate_anthropic_response(body, "codex-MiniMax-M2.7")

    assert translated["output"][0] == {
        "id": "rs_resp_msg_reasoning_0",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Need to inspect weather first"}],
        "content": [{"type": "reasoning_text", "text": "Need to inspect weather first"}],
    }
    assert translated["output"][1]["type"] == "message"


def test_translate_responses_request_supports_allowed_tools_tool_choice():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": "hello",
        "tools": [
            {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
            {"type": "function", "name": "search_docs", "parameters": {"type": "object"}},
        ],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": "search_docs"}],
        },
    }

    translated = translate_responses_request(request_body)

    assert translated["tools"] == [
        {"name": "search_docs", "description": "", "input_schema": {"type": "object"}}
    ]
    assert translated["tool_choice"] == {"type": "any"}


def test_translate_responses_request_supports_direct_tool_choice_shape():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "tool_choice": {"type": "tool", "name": "lookup_weather"},
        }
    )

    assert translated["tool_choice"] == {"type": "tool", "name": "lookup_weather"}


def test_translate_responses_request_supports_custom_tool_choice_shape():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "tool_choice": {"type": "custom", "name": "apply_patch"},
        }
    )

    assert translated["tool_choice"] == {"type": "tool", "name": "apply_patch"}


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "mcp", "server_label": "deepwiki", "name": "search"},
        {"type": "file_search"},
        {"type": "web_search_preview"},
        {"type": "computer_use_preview"},
        {"type": "code_interpreter"},
        {"type": "image_generation"},
    ],
)
def test_translate_responses_request_rejects_unsupported_tool_choice_type(tool_choice):
    with pytest.raises(UnsupportedFeatureError, match="tool_choice"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "tool_choice": tool_choice,
            }
        )


@pytest.mark.parametrize(
    ("tool_choice", "expected_name"),
    [
        ({"type": "apply_patch"}, "apply_patch"),
        ({"type": "shell"}, "shell"),
    ],
)
def test_translate_responses_request_supports_builtin_named_tool_choice(tool_choice, expected_name):
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "tool_choice": tool_choice,
        }
    )

    assert translated["tool_choice"] == {"type": "tool", "name": expected_name}


def test_translate_responses_request_rejects_unknown_tool_choice_string():
    with pytest.raises(UnsupportedFeatureError, match="tool_choice"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "tool_choice": "banana",
            }
        )


def test_translate_responses_request_maps_reasoning_effort_to_thinking():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "max_output_tokens": 4000,
            "reasoning": {"effort": "high"},
        }
    )

    assert translated["thinking"] == {"type": "enabled", "budget_tokens": 3200}


def test_translate_responses_request_maps_reasoning_xhigh_to_thinking():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "max_output_tokens": 4000,
            "reasoning": {"effort": "xhigh"},
        }
    )

    assert translated["thinking"] == {"type": "enabled", "budget_tokens": 3800}


def test_translate_responses_request_omits_thinking_for_reasoning_none():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "reasoning": {"effort": "none"},
        }
    )

    assert "thinking" not in translated


def test_translate_responses_request_rejects_reasoning_summary_control():
    with pytest.raises(UnsupportedFeatureError, match="reasoning.summary"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "reasoning": {"summary": "concise"},
            }
        )


def test_translate_responses_request_rejects_invalid_reasoning_effort():
    with pytest.raises(UnsupportedFeatureError, match="reasoning.effort"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "reasoning": {"effort": "ultra"},
            }
        )


def test_translate_responses_request_rejects_reasoning_input_items():
    with pytest.raises(UnsupportedFeatureError, match="reasoning input items"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "Plan"}]}],
            }
        )


def test_translate_responses_request_rejects_non_text_developer_content():
    with pytest.raises(UnsupportedFeatureError, match="developer"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_image", "image_url": "https://example.com/cat.png"}],
                    }
                ],
            }
        )


def test_translate_responses_request_applies_text_verbosity_instruction():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "text": {"format": {"type": "text"}, "verbosity": "low"},
        }
    )

    assert "Keep the response concise." in translated["system"]


def test_translate_responses_request_rejects_invalid_text_verbosity():
    with pytest.raises(UnsupportedFeatureError, match="text.verbosity"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "text": {"format": {"type": "text"}, "verbosity": "ultra"},
            }
        )


def test_translate_responses_request_rejects_unknown_text_format_type():
    with pytest.raises(UnsupportedFeatureError, match="text.format"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "text": {"format": {"type": "xml"}},
            }
        )


def test_translate_responses_request_respects_non_strict_json_schema_format():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "strict": False,
                    "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            },
        }
    )

    assert "strictly follows this JSON schema" not in translated["system"]
    assert "matches this JSON schema as closely as possible" in translated["system"]


def test_translate_responses_request_rejects_function_tool_strict_false():
    with pytest.raises(UnsupportedFeatureError, match="strict"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_weather",
                        "strict": False,
                        "parameters": {"type": "object"},
                    }
                ],
            }
        )


def test_translate_responses_request_rejects_unknown_stream_options():
    with pytest.raises(UnsupportedFeatureError, match="stream_options"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "stream_options": {"bogus": True},
            }
        )


def test_translate_responses_request_rejects_stream_options_when_stream_disabled():
    with pytest.raises(UnsupportedFeatureError, match="stream_options"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "stream": False,
                "stream_options": {"include_obfuscation": False},
            }
        )
