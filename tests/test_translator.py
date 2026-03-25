import json

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
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup_weather",
                "arguments": {"city": "Shanghai"},
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
    assert translated["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "lookup_weather", "input": {"city": "Shanghai"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": '{"status": "ok"}'}],
        },
    ]
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
            "input_schema": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            },
        }
    ]


def test_translate_responses_request_supports_custom_tools_with_grammar_format():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        "tools": [
            {
                "type": "custom",
                "name": "matcher",
                "description": "Match custom input",
                "format": {
                    "type": "grammar",
                    "syntax": "regex",
                    "definition": "^[a-z]+$",
                },
            }
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["tools"] == [
        {
            "name": "matcher",
            "description": "Match custom input",
            "input_schema": {
                "type": "object",
                "properties": {"input": {"type": "string", "pattern": "^[a-z]+$"}},
                "required": ["input"],
                "additionalProperties": False,
            },
        }
    ]


def test_translate_responses_request_rejects_lark_custom_tool_format():
    with pytest.raises(UnsupportedFeatureError, match="custom tool format"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
                "tools": [
                    {
                        "type": "custom",
                        "name": "matcher",
                        "description": "Match a token",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": "start: WORD",
                        },
                    }
                ],
            }
        )


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


def test_translate_responses_request_rejects_function_tool_without_name():
    with pytest.raises(UnsupportedFeatureError, match="tool name"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "tools": [{"type": "function", "parameters": {"type": "object"}}],
            }
        )


def test_translate_responses_request_rejects_custom_tool_without_name():
    with pytest.raises(UnsupportedFeatureError, match="tool name"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "tools": [{"type": "custom", "format": {"type": "text"}}],
            }
        )


def test_translate_responses_request_rejects_function_tool_with_non_object_parameters():
    with pytest.raises(UnsupportedFeatureError, match="tool parameters"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "tools": [{"type": "function", "name": "echo", "parameters": "not-a-schema"}],
            }
        )


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


def test_translate_responses_request_rejects_invalid_message_phase():
    with pytest.raises(UnsupportedFeatureError, match="phase"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "side_channel",
                        "content": [{"type": "output_text", "text": "Thinking"}],
                    }
                ],
            }
        )


def test_translate_responses_request_preserves_assistant_commentary_phase_best_effort():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Plan the edit first"}],
                }
            ],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "Plan the edit first"}],
        }
    ]


def test_translate_anthropic_response_emits_commentary_message_for_thinking_blocks():
    translated = translate_anthropic_response(
        {
            "id": "msg_phase",
            "content": [{"type": "thinking", "thinking": "Inspect files before editing."}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        "codex-MiniMax-M2.7",
    )

    commentary_messages = [
        item for item in translated["output"] if item.get("type") == "message" and item.get("phase") == "commentary"
    ]

    assert commentary_messages == [
        {
            "id": "msg_resp_msg_phase_commentary_0",
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Inspect files before editing.",
                    "annotations": [],
                }
            ],
        }
    ]


def test_translate_responses_request_round_trips_commentary_message_from_proxy_response():
    response = translate_anthropic_response(
        {
            "id": "msg_phase",
            "content": [{"type": "thinking", "thinking": "Inspect files before editing."}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        "codex-MiniMax-M2.7",
    )

    commentary_message = next(
        item for item in response["output"] if item.get("type") == "message" and item.get("phase") == "commentary"
    )
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": [commentary_message],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "Inspect files before editing."}],
        }
    ]


def test_translate_responses_request_preserves_consecutive_assistant_message_boundaries_without_tool_calls():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "First commentary"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "Second answer"}],
                },
            ],
        }
    )

    assert translated["messages"] == [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "First commentary"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Second answer"}]},
    ]


def test_translate_responses_request_keeps_tool_result_adjacent_when_commentary_and_reasoning_follow_tool_call():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": {"cmd": "head -1 README.md"},
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Checking the README heading."}],
                },
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "The tool result should follow the tool call."}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "# MiniMax Python Responses Proxy",
                },
            ],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "exec_command", "input": {"cmd": "head -1 README.md"}},
                {"type": "thinking", "thinking": "Checking the README heading."},
                {"type": "thinking", "thinking": "The tool result should follow the tool call."},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "# MiniMax Python Responses Proxy",
                }
            ],
        },
    ]


def test_translate_responses_request_rejects_non_assistant_message_phase():
    with pytest.raises(UnsupportedFeatureError, match="phase"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "phase": "commentary",
                        "content": [{"type": "input_text", "text": "hello"}],
                    }
                ],
            }
        )


def test_translate_responses_request_rejects_unknown_message_role():
    with pytest.raises(UnsupportedFeatureError, match="role"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [
                    {
                        "type": "message",
                        "role": "hacker",
                        "content": [{"type": "input_text", "text": "hello"}],
                    }
                ],
            }
        )


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


def test_translate_responses_request_rejects_function_call_without_name():
    with pytest.raises(UnsupportedFeatureError, match="tool call name"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [{"type": "function_call", "call_id": "call_1", "arguments": {"x": 1}}],
            }
        )


def test_translate_responses_request_rejects_tool_output_without_call_id():
    with pytest.raises(UnsupportedFeatureError, match="call_id"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [{"type": "function_call_output", "output": {"ok": True}}],
            }
        )


def test_translate_responses_request_accepts_custom_tool_call_items():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** End Patch\n",
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_patch",
                    "name": "apply_patch",
                    "input": {"input": "*** Begin Patch\n*** End Patch\n"},
                },
            ],
        }
    ]


def test_translate_responses_request_accepts_apply_patch_call_items():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "apply_patch_call",
                "call_id": "call_patch",
                "operation": {
                    "type": "update_file",
                    "path": "README.md",
                    "diff": "*** Begin Patch\n*** End Patch\n",
                },
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_patch",
                    "name": "apply_patch",
                    "input": {
                        "operation": {
                            "type": "update_file",
                            "path": "README.md",
                            "diff": "*** Begin Patch\n*** End Patch\n",
                        }
                    },
                },
            ],
        }
    ]


def test_translate_responses_request_accepts_shell_call_items():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "shell_call",
                "call_id": "call_shell",
                "action": {
                    "commands": ["pwd"],
                    "timeout_ms": 1000,
                    "max_output_length": 4096,
                },
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_shell",
                    "name": "shell",
                    "input": {
                        "action": {
                            "commands": ["pwd"],
                            "timeout_ms": 1000,
                            "max_output_length": 4096,
                        }
                    },
                },
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
                "input": "*** Begin Patch\n*** End Patch\n",
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
                {
                    "type": "tool_use",
                    "id": "call_patch",
                    "name": "apply_patch",
                    "input": {"input": "*** Begin Patch\n*** End Patch\n"},
                },
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


def test_translate_responses_request_accepts_apply_patch_call_output_items():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "apply_patch_call",
                "call_id": "call_patch",
                "operation": {
                    "type": "update_file",
                    "path": "README.md",
                    "diff": "*** Begin Patch\n*** End Patch\n",
                },
            },
            {
                "type": "apply_patch_call_output",
                "call_id": "call_patch",
                "output": "patch applied",
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_patch",
                    "name": "apply_patch",
                    "input": {
                        "operation": {
                            "type": "update_file",
                            "path": "README.md",
                            "diff": "*** Begin Patch\n*** End Patch\n",
                        }
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_patch",
                    "content": '{"type": "apply_patch_call_output", "output": "patch applied"}',
                }
            ],
        },
    ]


def test_translate_responses_request_accepts_shell_call_output_items():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "shell_call",
                "call_id": "call_shell",
                "action": {"commands": ["pwd"]},
            },
            {
                "type": "shell_call_output",
                "call_id": "call_shell",
                "output": "stdout: /root/minimaxdemo",
            },
        ],
    }

    translated = translate_responses_request(request_body)

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_shell",
                    "name": "shell",
                    "input": {
                        "action": {"commands": ["pwd"]},
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_shell",
                    "content": '{"type": "shell_call_output", "output": "stdout: /root/minimaxdemo"}',
                }
            ],
        },
    ]


def test_translate_responses_request_supports_shell_tool_environment():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "tools": [
                {
                    "type": "shell",
                    "name": "shell",
                    "description": "Run shell commands",
                    "environment": {"type": "local", "skills": [{"name": "python", "path": "/tmp/python"}]},
                }
            ],
            "input": "hello",
        }
    )

    tool = translated["tools"][0]

    assert tool["name"] == "shell"
    assert tool["description"] == "Run shell commands"
    assert tool["input_schema"]["properties"]["environment"]["default"] == {
        "type": "local",
        "skills": [{"name": "python", "path": "/tmp/python"}],
    }
    assert tool["input_schema"]["properties"]["environment"]["oneOf"][0]["properties"]["type"]["const"] == "local"
    assert (
        tool["input_schema"]["properties"]["environment"]["oneOf"][1]["properties"]["type"]["const"]
        == "container_reference"
    )


def test_translate_responses_request_preserves_shell_call_environment():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": [
                {
                    "type": "shell_call",
                    "call_id": "call_shell",
                    "action": {"commands": ["pwd"], "timeout_ms": 1000},
                    "environment": {"type": "local", "skills": [{"name": "python", "path": "/tmp/python"}]},
                }
            ],
        }
    )

    assert translated["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_shell",
                    "name": "shell",
                    "input": {
                        "action": {"commands": ["pwd"], "timeout_ms": 1000},
                        "environment": {"type": "local", "skills": [{"name": "python", "path": "/tmp/python"}]},
                    },
                }
            ],
        }
    ]


def test_translate_responses_request_maps_failed_apply_patch_call_output_to_error_tool_result():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "apply_patch_call",
                "call_id": "call_patch",
                "operation": {
                    "type": "update_file",
                    "path": "README.md",
                    "diff": "*** Begin Patch\n*** End Patch\n",
                },
            },
            {
                "type": "apply_patch_call_output",
                "call_id": "call_patch",
                "status": "failed",
                "output": "patch failed",
            },
        ],
    }

    translated = translate_responses_request(request_body)

    tool_result = translated["messages"][1]["content"][0]

    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_patch"
    assert tool_result["is_error"] is True
    assert json.loads(tool_result["content"]) == {
        "type": "apply_patch_call_output",
        "status": "failed",
        "output": "patch failed",
    }


def test_translate_responses_request_preserves_apply_patch_call_output_metadata():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": [
                {
                    "type": "apply_patch_call",
                    "call_id": "call_patch",
                    "operation": {"type": "update_file", "path": "README.md", "diff": "*** Begin Patch\n*** End Patch\n"},
                },
                {
                    "type": "apply_patch_call_output",
                    "id": "apo_123",
                    "call_id": "call_patch",
                    "status": "completed",
                    "output": "patch applied",
                },
            ],
        }
    )

    tool_result = translated["messages"][1]["content"][0]

    assert tool_result["tool_use_id"] == "call_patch"
    assert json.loads(tool_result["content"]) == {
        "type": "apply_patch_call_output",
        "id": "apo_123",
        "status": "completed",
        "output": "patch applied",
    }


def test_translate_responses_request_maps_failed_shell_call_output_to_error_tool_result():
    request_body = {
        "model": "codex-MiniMax-M2.7",
        "input": [
            {
                "type": "shell_call",
                "call_id": "call_shell",
                "action": {"commands": ["pwd"]},
            },
            {
                "type": "shell_call_output",
                "call_id": "call_shell",
                "status": "failed",
                "output": "permission denied",
            },
        ],
    }

    translated = translate_responses_request(request_body)

    tool_result = translated["messages"][1]["content"][0]

    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_shell"
    assert tool_result["is_error"] is True
    assert json.loads(tool_result["content"]) == {
        "type": "shell_call_output",
        "status": "failed",
        "output": "permission denied",
    }


def test_translate_responses_request_preserves_shell_call_output_metadata_and_chunks():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": [
                {
                    "type": "shell_call",
                    "call_id": "call_shell",
                    "action": {"commands": ["pwd"]},
                },
                {
                    "type": "shell_call_output",
                    "id": "sho_123",
                    "call_id": "call_shell",
                    "status": "completed",
                    "max_output_length": 4096,
                    "output": [
                        {
                            "stdout": "/root/minimaxdemo\n",
                            "stderr": "",
                            "outcome": {"type": "exit", "exit_code": 0},
                        }
                    ],
                },
            ],
        }
    )

    tool_result = translated["messages"][1]["content"][0]

    assert tool_result["tool_use_id"] == "call_shell"
    assert json.loads(tool_result["content"]) == {
        "type": "shell_call_output",
        "id": "sho_123",
        "status": "completed",
        "max_output_length": 4096,
        "output": [
            {
                "stdout": "/root/minimaxdemo\n",
                "stderr": "",
                "outcome": {"type": "exit", "exit_code": 0},
            }
        ],
    }


def test_translate_responses_request_maps_nonzero_shell_exit_to_error_tool_result():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": [
                {
                    "type": "shell_call",
                    "call_id": "call_shell",
                    "action": {"commands": ["false"]},
                },
                {
                    "type": "shell_call_output",
                    "call_id": "call_shell",
                    "status": "completed",
                    "output": [
                        {
                            "stdout": "",
                            "stderr": "permission denied\n",
                            "outcome": {"type": "exit", "exit_code": 1},
                        }
                    ],
                },
            ],
        }
    )

    tool_result = translated["messages"][1]["content"][0]

    assert tool_result["is_error"] is True
    assert json.loads(tool_result["content"]) == {
        "type": "shell_call_output",
        "status": "completed",
        "output": [
            {
                "stdout": "",
                "stderr": "permission denied\n",
                "outcome": {"type": "exit", "exit_code": 1},
            }
        ],
    }


def test_translate_responses_request_rejects_orphan_tool_results():
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

    with pytest.raises(UnsupportedFeatureError, match="orphan tool_result"):
        translate_responses_request(request_body)


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
            {"type": "tool_use", "id": "call_patch", "name": "apply_patch", "input": {"input": "*** Begin Patch\n*** End Patch\n"}},
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
            "input": "*** Begin Patch\n*** End Patch\n",
            "status": "completed",
        }
    ]


def test_translate_anthropic_response_maps_apply_patch_tool_use_to_builtin_call():
    body = {
        "id": "msg_tool",
        "content": [
            {
                "type": "tool_use",
                "id": "call_patch",
                "name": "apply_patch",
                "input": {
                    "operation": {
                        "type": "update_file",
                        "path": "README.md",
                        "diff": "*** Begin Patch\n*** End Patch\n",
                    }
                },
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    translated = translate_anthropic_response(
        body,
        "codex-MiniMax-M2.7",
        response_context={"tools": [{"type": "apply_patch", "name": "apply_patch"}]},
    )

    assert translated["output"] == [
        {
            "id": "fc_call_patch",
            "type": "apply_patch_call",
            "call_id": "call_patch",
            "status": "completed",
            "operation": {
                "type": "update_file",
                "path": "README.md",
                "diff": "*** Begin Patch\n*** End Patch\n",
            },
        }
    ]


def test_translate_anthropic_response_maps_shell_tool_use_to_builtin_call():
    body = {
        "id": "msg_tool",
        "content": [
            {
                "type": "tool_use",
                "id": "call_shell",
                "name": "shell",
                "input": {"action": {"commands": ["pwd"], "timeout_ms": 1000, "max_output_length": 4096}},
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    translated = translate_anthropic_response(
        body,
        "codex-MiniMax-M2.7",
        response_context={"tools": [{"type": "shell", "name": "shell"}]},
    )

    assert translated["output"] == [
        {
            "id": "fc_call_shell",
            "type": "shell_call",
            "call_id": "call_shell",
            "action": {"commands": ["pwd"], "timeout_ms": 1000, "max_output_length": 4096},
            "status": "completed",
        }
    ]


def test_translate_anthropic_response_maps_shell_environment_to_builtin_call():
    body = {
        "id": "msg_tool",
        "content": [
            {
                "type": "tool_use",
                "id": "call_shell",
                "name": "shell",
                "input": {
                    "action": {"commands": ["pwd"]},
                    "environment": {"type": "local", "skills": [{"name": "python", "path": "/tmp/python"}]},
                },
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    translated = translate_anthropic_response(
        body,
        "codex-MiniMax-M2.7",
        response_context={"tools": [{"type": "shell", "name": "shell"}]},
    )

    assert translated["output"] == [
        {
            "id": "fc_call_shell",
            "type": "shell_call",
            "call_id": "call_shell",
            "action": {"commands": ["pwd"]},
            "environment": {"type": "local", "skills": [{"name": "python", "path": "/tmp/python"}]},
            "status": "completed",
        }
    ]


def test_translate_anthropic_response_falls_back_to_shell_tool_environment_definition():
    body = {
        "id": "msg_tool",
        "content": [
            {
                "type": "tool_use",
                "id": "call_shell",
                "name": "shell",
                "input": {
                    "action": {"commands": ["pwd"]},
                },
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    translated = translate_anthropic_response(
        body,
        "codex-MiniMax-M2.7",
        response_context={
            "tools": [
                {
                    "type": "shell",
                    "name": "shell",
                    "environment": {"type": "local", "skills": [{"name": "python", "path": "/tmp/python"}]},
                }
            ]
        },
    )

    assert translated["output"] == [
        {
            "id": "fc_call_shell",
            "type": "shell_call",
            "call_id": "call_shell",
            "action": {"commands": ["pwd"]},
            "environment": {"type": "local", "skills": [{"name": "python", "path": "/tmp/python"}]},
            "status": "completed",
        }
    ]


def test_translate_anthropic_response_marks_message_phase_and_incomplete_details():
    translated = translate_anthropic_response(
        {
            "id": "msg_text",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "codex-MiniMax-M2.7",
    )

    assert translated["incomplete_details"] is None
    assert translated["output"][0]["phase"] == "final_answer"


def test_translate_anthropic_response_marks_reasoning_items_completed():
    translated = translate_anthropic_response(
        {
            "id": "msg_reasoning",
            "content": [{"type": "thinking", "thinking": "Plan"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "codex-MiniMax-M2.7",
    )

    assert translated["output"][0]["status"] == "completed"


def test_translate_anthropic_response_marks_max_tokens_stop_as_incomplete():
    translated = translate_anthropic_response(
        {
            "id": "msg_text",
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "Partial"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "codex-MiniMax-M2.7",
    )

    assert translated["status"] == "incomplete"
    assert translated["incomplete_details"] == {"reason": "max_output_tokens"}
    assert translated["output"][0]["status"] == "incomplete"


def test_translate_anthropic_response_omits_reasoning_encrypted_content_even_if_requested():
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

    reasoning_item = translated["output"][0]

    assert reasoning_item["type"] == "reasoning"
    assert "encrypted_content" not in reasoning_item


def test_translate_anthropic_response_omits_reasoning_encrypted_content_when_upstream_missing():
    body = {
        "id": "msg_reasoning",
        "content": [
            {"type": "thinking", "thinking": "Plan carefully before acting"},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    translated = translate_anthropic_response(
        body,
        "codex-MiniMax-M2.7",
        response_context={"include": ["reasoning.encrypted_content"]},
    )

    reasoning_item = translated["output"][0]

    assert reasoning_item["type"] == "reasoning"
    assert "encrypted_content" not in reasoning_item


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
        "include": [],
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
    assert translated["include"] == []
    assert translated["prompt_cache_key"] == "cache-key-123"
    assert translated["usage"]["input_tokens_details"] == {"cached_tokens": 0}
    assert translated["usage"]["output_tokens_details"] == {"reasoning_tokens": 0}
    assert translated["output_text"] == "Hello"
    assert translated["completed_at"] >= translated["created_at"]


def test_build_response_context_sets_default_function_tool_strict_true():
    context = build_response_context(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "tools": [{"type": "function", "name": "echo", "parameters": {"type": "object"}}],
        }
    )

    assert context["tools"] == [
        {"type": "function", "name": "echo", "parameters": {"type": "object"}, "strict": True}
    ]


def test_translate_responses_request_accepts_deprecated_reasoning_generate_summary():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "max_output_tokens": 4000,
            "reasoning": {"effort": "high", "generate_summary": "concise"},
        }
    )

    response_context = build_response_context(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "reasoning": {"effort": "high", "generate_summary": "concise"},
        }
    )

    assert translated["thinking"] == {"type": "enabled", "budget_tokens": 3200}
    assert response_context["reasoning"] == {
        "effort": "high",
        "generate_summary": "concise",
        "summary": "concise",
    }


def test_translate_anthropic_response_shapes_reasoning_summary_when_generate_summary_requested():
    translated = translate_anthropic_response(
        {
            "id": "msg_reasoning",
            "content": [{"type": "thinking", "thinking": "First sentence. Second sentence with more detail."}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "codex-MiniMax-M2.7",
        response_context={"reasoning": {"effort": "high", "generate_summary": "concise", "summary": "concise"}},
    )

    assert translated["output"][0]["summary"] == [{"type": "summary_text", "text": "First sentence."}]


def test_translate_responses_request_does_not_default_max_output_tokens():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
        }
    )

    assert "max_tokens" not in translated


def test_translate_responses_request_rejects_reasoning_encrypted_content_include():
    with pytest.raises(UnsupportedFeatureError, match="include"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "include": ["reasoning.encrypted_content"],
                "prompt_cache_key": "cache-key-123",
                "parallel_tool_calls": False,
            }
        )


def test_translate_responses_request_rejects_nonzero_top_logprobs():
    with pytest.raises(UnsupportedFeatureError, match="top_logprobs"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "top_logprobs": 3,
            }
        )


def test_build_response_context_rejects_zero_top_logprobs():
    with pytest.raises(UnsupportedFeatureError, match="top_logprobs"):
        build_response_context(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "top_logprobs": 0,
            }
        )


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


def test_translate_responses_request_rejects_unsupported_logprobs_include():
    with pytest.raises(UnsupportedFeatureError, match="include"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "include": ["message.output_text.logprobs"],
            }
        )


def test_translate_anthropic_response_omits_logprobs_when_not_requested():
    translated = translate_anthropic_response(
        {
            "id": "msg_text",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "codex-MiniMax-M2.7",
    )

    assert "logprobs" not in translated["output"][0]["content"][0]


def test_translate_anthropic_response_omits_logprobs_when_zero_requested():
    translated = translate_anthropic_response(
        {
            "id": "msg_text",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "codex-MiniMax-M2.7",
        response_context={"top_logprobs": 0},
    )

    assert "logprobs" not in translated["output"][0]["content"][0]


def test_translate_anthropic_response_does_not_fallback_output_text_to_commentary():
    translated = translate_anthropic_response(
        {
            "id": "msg_commentary_only",
            "content": [{"type": "thinking", "thinking": "Inspecting files."}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "codex-MiniMax-M2.7",
    )

    assert translated["output_text"] == ""


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


def test_translate_responses_request_rejects_unsupported_tool_output_content_blocks():
    with pytest.raises(UnsupportedFeatureError, match="input_image.detail"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": [
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/cat.png",
                                "detail": "high",
                            }
                        ],
                    }
                ],
            }
        )


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


def test_translate_responses_request_rejects_unsupported_image_detail():
    with pytest.raises(UnsupportedFeatureError, match="input_image.detail"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/cat.png",
                                "detail": "high",
                            }
                        ],
                    }
                ],
            }
        )


def test_translate_responses_request_rejects_input_image_file_id():
    with pytest.raises(UnsupportedFeatureError, match="input_image.file_id"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "file_id": "file_123"}],
                    }
                ],
            }
        )


def test_translate_responses_request_rejects_invalid_shell_tool_environment_shape():
    with pytest.raises(UnsupportedFeatureError, match="shell environment"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "tools": [
                    {
                        "type": "shell",
                        "name": "shell",
                        "environment": {"skills": [{"name": "python", "path": "/tmp/python"}]},
                    }
                ],
                "input": "hello",
            }
        )


def test_translate_responses_request_supports_container_reference_shell_environment():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "tools": [
                {
                    "type": "shell",
                    "name": "shell",
                    "environment": {"type": "container_reference", "container_id": "ctr_123"},
                }
            ],
            "input": "hello",
        }
    )

    assert translated["tools"][0]["input_schema"]["properties"]["environment"]["default"] == {
        "type": "container_reference",
        "container_id": "ctr_123",
    }


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
        "status": "completed",
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


def test_translate_responses_request_rejects_allowed_tools_with_invalid_entries():
    with pytest.raises(UnsupportedFeatureError, match="allowed_tools"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "tools": [{"type": "function", "name": "search_docs", "parameters": {"type": "object"}}],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [{}],
                },
            }
        )


def test_translate_responses_request_rejects_allowed_tools_with_unknown_tool_names():
    with pytest.raises(UnsupportedFeatureError, match="allowed_tools"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "tools": [{"type": "function", "name": "search_docs", "parameters": {"type": "object"}}],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [{"type": "function", "name": "missing_tool"}],
                },
            }
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("max_output_tokens", 0, "max_output_tokens"),
        ("max_output_tokens", "100", "max_output_tokens"),
        ("temperature", -0.1, "temperature"),
        ("temperature", 2.5, "temperature"),
        ("temperature", "warm", "temperature"),
        ("top_p", -0.1, "top_p"),
        ("top_p", 1.5, "top_p"),
        ("top_p", "wide", "top_p"),
        ("parallel_tool_calls", "false", "parallel_tool_calls"),
        ("metadata", [], "metadata"),
        ("metadata", {1: "bad"}, "metadata"),
        ("user", 123, "user"),
    ],
)
def test_translate_responses_request_rejects_invalid_request_field_shapes(field_name, field_value, message):
    with pytest.raises(UnsupportedFeatureError, match=message):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                field_name: field_value,
            }
        )


@pytest.mark.parametrize(
    "stop_value",
    [
        123,
        [123],
        ["ok", 123],
        {"value": "stop"},
    ],
)
def test_translate_responses_request_rejects_invalid_stop_shapes(stop_value):
    with pytest.raises(UnsupportedFeatureError, match="stop"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "stop": stop_value,
            }
        )


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


def test_translate_responses_request_synthesizes_builtin_tool_choice_definition():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "tool_choice": {"type": "apply_patch"},
        }
    )

    assert translated["tools"] == [
        {
            "name": "apply_patch",
            "description": "",
            "input_schema": {
                "type": "object",
                "properties": {"operation": {"type": "object"}},
                "required": ["operation"],
                "additionalProperties": False,
            },
        }
    ]


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


def test_translate_responses_request_accepts_reasoning_summary_control():
    translated = translate_responses_request(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "max_output_tokens": 4000,
            "reasoning": {"effort": "high", "summary": "concise"},
        }
    )

    response_context = build_response_context(
        {
            "model": "codex-MiniMax-M2.7",
            "input": "hello",
            "reasoning": {"effort": "high", "summary": "concise"},
        }
    )

    assert translated["thinking"] == {"type": "enabled", "budget_tokens": 3200}
    assert response_context["reasoning"] == {"effort": "high", "summary": "concise"}


def test_translate_anthropic_response_shapes_reasoning_summary_when_concise_requested():
    translated = translate_anthropic_response(
        {
            "id": "msg_reasoning",
            "content": [{"type": "thinking", "thinking": "First sentence. Second sentence with more detail."}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "codex-MiniMax-M2.7",
        response_context={"reasoning": {"effort": "high", "summary": "concise"}},
    )

    assert translated["output"][0]["summary"] == [{"type": "summary_text", "text": "First sentence."}]
    assert translated["output"][0]["content"] == [
        {"type": "reasoning_text", "text": "First sentence. Second sentence with more detail."}
    ]


def test_translate_responses_request_rejects_invalid_reasoning_effort():
    with pytest.raises(UnsupportedFeatureError, match="reasoning.effort"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": "hello",
                "reasoning": {"effort": "ultra"},
            }
        )


def test_translate_responses_request_rejects_reasoning_input_items_with_encrypted_content():
    with pytest.raises(UnsupportedFeatureError, match="reasoning.encrypted_content"):
        translate_responses_request(
            {
                "model": "codex-MiniMax-M2.7",
                "input": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Plan"}],
                        "encrypted_content": "proxy_reasoning_v1:SW5zcGVjdCBmaWxlcyBiZWZvcmUgZWRpdGluZy4=",
                    }
                ],
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


def test_translate_responses_request_accepts_function_tool_strict_false():
    translated = translate_responses_request(
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

    assert translated["tools"] == [
        {
            "name": "lookup_weather",
            "description": "",
            "input_schema": {"type": "object"},
        }
    ]


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
