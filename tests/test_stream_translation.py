import pytest

from app.translator import ResponsesEventTranslator, UnsupportedFeatureError


def test_translate_anthropic_sse_to_responses_events():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    event_types = [event["event"] for event in events]

    assert event_types[:2] == ["response.created", "response.in_progress"]
    assert "response.output_item.added" in event_types
    assert "response.content_part.added" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.output_text.done" in event_types
    assert "response.output_item.done" in event_types
    assert event_types[-1] == "response.completed"

    delta_event = next(event for event in events if event["event"] == "response.output_text.delta")
    assert delta_event["data"]["delta"] == "Hello"
    assert "obfuscation" in delta_event["data"]
    message_done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "message"
    )
    assert message_done["data"]["item"]["phase"] == "final_answer"

    completed_event = events[-1]
    assert completed_event["data"]["response"]["status"] == "completed"


def test_translate_anthropic_sse_includes_request_context_fields():
    translator = ResponsesEventTranslator(
        model="codex-MiniMax-M2.7",
        response_context={
            "instructions": "Be concise",
            "metadata": {"request_id": "abc"},
            "user": "user-123",
            "store": False,
            "tool_choice": "auto",
            "tools": [{"type": "function", "name": "echo"}],
            "text": {"format": {"type": "text"}},
            "temperature": 0.2,
            "top_p": 0.9,
            "parallel_tool_calls": True,
            "max_output_tokens": 128,
            "include": [],
            "prompt_cache_key": "cache-key-123",
        },
    )

    events = translator.feed({"type": "message_start", "message": {"id": "msg_ctx"}})
    created = next(event for event in events if event["event"] == "response.created")
    in_progress = next(event for event in events if event["event"] == "response.in_progress")

    assert created["data"]["response"]["instructions"] == "Be concise"
    assert created["data"]["response"]["metadata"] == {"request_id": "abc"}
    assert created["data"]["response"]["user"] == "user-123"
    assert created["data"]["response"]["store"] is False
    assert created["data"]["response"]["tool_choice"] == "auto"
    assert created["data"]["response"]["tools"] == [{"type": "function", "name": "echo"}]
    assert created["data"]["response"]["text"] == {"format": {"type": "text"}}
    assert created["data"]["response"]["temperature"] == 0.2
    assert created["data"]["response"]["top_p"] == 0.9
    assert created["data"]["response"]["parallel_tool_calls"] is True
    assert created["data"]["response"]["max_output_tokens"] == 128
    assert created["data"]["response"]["include"] == []
    assert created["data"]["response"]["prompt_cache_key"] == "cache-key-123"
    assert created["data"]["response"]["incomplete_details"] is None
    assert in_progress["data"]["response"]["metadata"] == {"request_id": "abc"}


def test_translate_anthropic_thinking_block_to_reasoning_events():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Planning"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    event_types = [event["event"] for event in events]

    assert "response.reasoning_summary_part.added" in event_types
    assert "response.reasoning_summary_text.delta" in event_types
    assert "response.reasoning_summary_text.done" in event_types
    assert "response.output_item.done" in event_types


def test_translate_anthropic_thinking_block_emits_commentary_message_events():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Planning"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    commentary_added = next(
        event
        for event in events
        if event["event"] == "response.output_item.added"
        and event["data"]["item"]["type"] == "message"
        and event["data"]["item"].get("phase") == "commentary"
    )
    commentary_delta = next(
        event
        for event in events
        if event["event"] == "response.output_text.delta" and event["data"]["item_id"] == commentary_added["data"]["item"]["id"]
    )
    commentary_done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done"
        and event["data"]["item"]["type"] == "message"
        and event["data"]["item"].get("phase") == "commentary"
    )
    completed = next(event for event in events if event["event"] == "response.completed")

    assert commentary_delta["data"]["delta"] == "Planning"
    assert commentary_done["data"]["item"]["content"][0]["text"] == "Planning"
    commentary_messages = [
        item
        for item in completed["data"]["response"]["output"]
        if item.get("type") == "message" and item.get("phase") == "commentary"
    ]
    assert len(commentary_messages) == 1
    assert commentary_messages[0]["content"][0]["text"] == "Planning"
    assert completed["data"]["response"]["output_text"] == ""


def test_translate_anthropic_thinking_block_uses_concise_summary_when_requested():
    translator = ResponsesEventTranslator(
        response_context={"reasoning": {"effort": "high", "summary": "concise"}},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "First sentence. Second sentence with more detail."},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    summary_done = next(event for event in events if event["event"] == "response.reasoning_summary_text.done")

    assert summary_done["data"]["text"] == "First sentence."


def test_translate_anthropic_thinking_block_includes_reasoning_encrypted_content_when_signature_available():
    translator = ResponsesEventTranslator(
        response_context={"include": ["reasoning.encrypted_content"]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Planning"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "enc_sig_456"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "reasoning"
    )
    completed = next(event for event in events if event["event"] == "response.completed")

    assert done["data"]["item"]["encrypted_content"].startswith("a2r_reasoning_v1:")
    assert completed["data"]["response"]["output"][0]["encrypted_content"].startswith("a2r_reasoning_v1:")


def test_translate_anthropic_thinking_block_omits_reasoning_encrypted_content_when_upstream_missing():
    translator = ResponsesEventTranslator(
        response_context={"include": ["reasoning.encrypted_content"]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Plan carefully before acting."},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "reasoning"
    )

    assert "encrypted_content" not in done["data"]["item"]


def test_translate_anthropic_redacted_thinking_block_emits_reasoning_without_empty_commentary():
    translator = ResponsesEventTranslator(
        response_context={"include": ["reasoning.encrypted_content"]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "redacted_blob_456"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Answer"},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    done_items = [
        event["data"]["item"]
        for event in events
        if event["event"] == "response.output_item.done"
    ]

    assert sorted(item["type"] for item in done_items) == ["message", "reasoning"]
    reasoning_item = next(item for item in done_items if item["type"] == "reasoning")
    message_item = next(item for item in done_items if item["type"] == "message")
    assert reasoning_item["encrypted_content"].startswith("a2r_reasoning_v1:")
    assert message_item["phase"] == "final_answer"


def test_translate_anthropic_mcp_tool_use_and_result_streams_as_mcp_call():
    translator = ResponsesEventTranslator(model="claude-sonnet-4-20250514")
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_mcp"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "mcp_tool_use",
                "id": "mcptoolu_123",
                "name": "ask_question",
                "server_name": "deepwiki",
                "input": {"repoName": "modelcontextprotocol/modelcontextprotocol"},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": "{\"repoName\":\"modelcontextprotocol/modelcontextprotocol\"}",
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "mcp_tool_result",
                "tool_use_id": "mcptoolu_123",
                "content": [{"type": "text", "text": "Supported transports include Streamable HTTP and SSE."}],
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 12, "output_tokens": 8}},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    event_names = [event["event"] for event in events]
    assert "response.mcp_call.in_progress" in event_names
    assert "response.mcp_call_arguments.delta" in event_names
    assert "response.mcp_call_arguments.done" in event_names
    assert "response.mcp_call.completed" in event_names

    completed_event = next(event for event in events if event["event"] == "response.mcp_call.completed")
    assert completed_event["data"]["item"] == {
        "id": "mcptoolu_123",
        "type": "mcp_call",
        "name": "ask_question",
        "server_label": "deepwiki",
        "arguments": "{\"repoName\":\"modelcontextprotocol/modelcontextprotocol\"}",
        "output": "Supported transports include Streamable HTTP and SSE.",
        "status": "completed",
    }


def test_translate_anthropic_thinking_block_omits_proxy_encrypted_content_when_requested_without_upstream_value():
    translator = ResponsesEventTranslator(
        response_context={"include": ["reasoning.encrypted_content"]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Plan carefully before acting."},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "reasoning"
    )

    assert "encrypted_content" not in done["data"]["item"]


def test_translate_anthropic_reasoning_commentary_and_final_answer_use_distinct_output_indexes():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Planning"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    reasoning_added = next(event for event in events if event["event"] == "response.output_item.added")
    message_events = [
        event
        for event in events
        if event["event"] == "response.output_item.added" and event["data"]["item"]["type"] == "message"
    ]
    commentary_added = next(event for event in message_events if event["data"]["item"].get("phase") == "commentary")
    final_added = next(event for event in message_events if event["data"]["item"].get("phase") == "final_answer")
    text_deltas = [event for event in events if event["event"] == "response.output_text.delta"]

    assert reasoning_added["data"]["output_index"] == 0
    assert commentary_added["data"]["output_index"] == 1
    assert final_added["data"]["output_index"] == 2
    assert [event["data"]["output_index"] for event in text_deltas] == [1, 2]


def test_translate_anthropic_stream_omits_obfuscation_when_disabled():
    translator = ResponsesEventTranslator(response_context={"stream_options": {"include_obfuscation": False}})
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    delta_event = next(event for event in events if event["event"] == "response.output_text.delta")

    assert "obfuscation" not in delta_event["data"]


def test_translate_anthropic_tool_use_stream_to_function_call_events():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "call_1", "name": "lookup_weather", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":"Sha'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": 'nghai"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    added = next(
        event
        for event in events
        if event["event"] == "response.output_item.added" and event["data"]["item"]["type"] == "function_call"
    )
    arg_deltas = [
        event["data"]["delta"] for event in events if event["event"] == "response.function_call_arguments.delta"
    ]
    args_done = next(event for event in events if event["event"] == "response.function_call_arguments.done")
    item_done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "function_call"
    )

    assert added["data"]["item"]["call_id"] == "call_1"
    assert arg_deltas == ['{"city":"Sha', 'nghai"}']
    assert args_done["data"]["arguments"] == '{"city":"Shanghai"}'
    assert item_done["data"]["item"]["arguments"] == '{"city":"Shanghai"}'


def test_translate_anthropic_tool_use_stream_to_custom_tool_call_events():
    translator = ResponsesEventTranslator(
        response_context={"tools": [{"type": "custom", "name": "apply_patch"}]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "call_patch",
                "name": "apply_patch",
                "input": {"input": "*** Begin Patch\n*** End Patch\n"},
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    added = next(
        event
        for event in events
        if event["event"] == "response.output_item.added" and event["data"]["item"]["call_id"] == "call_patch"
    )
    input_done = next(event for event in events if event["event"] == "response.custom_tool_call_input.done")
    done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["call_id"] == "call_patch"
    )
    completed = next(event for event in events if event["event"] == "response.completed")
    output_item = next(item for item in completed["data"]["response"]["output"] if item["call_id"] == "call_patch")

    assert added["data"]["item"]["type"] == "custom_tool_call"
    assert input_done["data"]["input"] == "*** Begin Patch\n*** End Patch\n"
    assert done["data"]["item"]["type"] == "custom_tool_call"
    assert output_item["type"] == "custom_tool_call"
    assert output_item["input"] == "*** Begin Patch\n*** End Patch\n"


def test_translate_anthropic_tool_use_stream_to_apply_patch_call_events():
    translator = ResponsesEventTranslator(
        response_context={"tools": [{"type": "apply_patch", "name": "apply_patch"}]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
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
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    added = next(
        event
        for event in events
        if event["event"] == "response.output_item.added" and event["data"]["item"]["call_id"] == "call_patch"
    )
    done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["call_id"] == "call_patch"
    )
    completed = next(event for event in events if event["event"] == "response.completed")
    output_item = next(item for item in completed["data"]["response"]["output"] if item["call_id"] == "call_patch")

    assert added["data"]["item"]["type"] == "apply_patch_call"
    assert done["data"]["item"]["type"] == "apply_patch_call"
    assert output_item["type"] == "apply_patch_call"
    assert output_item["operation"]["path"] == "README.md"


def test_translate_anthropic_tool_use_stream_to_shell_call_events():
    translator = ResponsesEventTranslator(
        response_context={"tools": [{"type": "shell", "name": "shell"}]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "call_shell",
                "name": "shell",
                "input": {"action": {"commands": ["pwd"], "timeout_ms": 1000}},
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    added = next(
        event
        for event in events
        if event["event"] == "response.output_item.added" and event["data"]["item"]["call_id"] == "call_shell"
    )
    done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["call_id"] == "call_shell"
    )
    completed = next(event for event in events if event["event"] == "response.completed")
    output_item = next(item for item in completed["data"]["response"]["output"] if item["call_id"] == "call_shell")

    assert added["data"]["item"]["type"] == "shell_call"
    assert done["data"]["item"]["type"] == "shell_call"
    assert output_item["type"] == "shell_call"
    assert output_item["action"] == {"commands": ["pwd"], "timeout_ms": 1000}


def test_translate_anthropic_web_search_stream_to_web_search_call_and_url_citations():
    translator = ResponsesEventTranslator(
        response_context={"include": ["web_search_call.action.sources"]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_search"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "server_tool_use", "id": "ws_1", "name": "web_search"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"query\":\"claude shannon birth date\"}"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "ws_1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                        "title": "Claude Shannon - Wikipedia",
                        "encrypted_content": "enc_123",
                    }
                ],
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "Claude Shannon was born on April 30, 1916."},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                    "title": "Claude Shannon - Wikipedia",
                    "encrypted_index": "idx_123",
                    "cited_text": "Claude Elwood Shannon (April 30, 1916",
                },
            },
        },
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 1, "output_tokens": 1}},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    searching = next(event for event in events if event["event"] == "response.web_search_call.searching")
    completed_call = next(event for event in events if event["event"] == "response.web_search_call.completed")
    message_done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "message"
    )
    response_done = next(event for event in events if event["event"] == "response.completed")
    web_search_item = next(
        item for item in response_done["data"]["response"]["output"] if item["type"] == "web_search_call"
    )

    assert searching["data"]["item_id"] == "ws_1"
    assert completed_call["data"]["item"]["action"] == {
        "type": "search",
        "query": "claude shannon birth date",
        "sources": [{"type": "url", "url": "https://en.wikipedia.org/wiki/Claude_Shannon"}],
    }
    assert message_done["data"]["item"]["content"][0]["annotations"] == [
        {
            "type": "url_citation",
            "start_index": 0,
            "end_index": 42,
            "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
            "title": "Claude Shannon - Wikipedia",
        }
    ]
    assert web_search_item["action"]["sources"] == [{"type": "url", "url": "https://en.wikipedia.org/wiki/Claude_Shannon"}]


def test_translate_anthropic_shell_stream_preserves_environment_and_argument_payload():
    translator = ResponsesEventTranslator(
        response_context={"tools": [{"type": "shell", "name": "shell"}]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "call_shell",
                "name": "shell",
                "input": {
                    "action": {"commands": ["pwd"], "timeout_ms": 1000},
                    "environment": {"type": "local", "skills": [{"name": "python", "path": "/tmp/python"}]},
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    args_done = next(event for event in events if event["event"] == "response.function_call_arguments.done")
    done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["call_id"] == "call_shell"
    )
    output_item = next(
        item
        for item in next(event for event in events if event["event"] == "response.completed")["data"]["response"]["output"]
        if item["call_id"] == "call_shell"
    )

    assert '"environment": {"type": "local", "skills": [{"name": "python", "path": "/tmp/python"}]}' in args_done["data"]["arguments"]
    assert done["data"]["item"]["environment"] == {
        "type": "local",
        "skills": [{"name": "python", "path": "/tmp/python"}],
    }
    assert output_item["environment"] == {
        "type": "local",
        "skills": [{"name": "python", "path": "/tmp/python"}],
    }


def test_translate_anthropic_shell_stream_falls_back_to_tool_environment_definition():
    translator = ResponsesEventTranslator(
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
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "call_shell",
                "name": "shell",
                "input": {
                    "action": {"commands": ["pwd"]},
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["call_id"] == "call_shell"
    )
    output_item = next(
        item
        for item in next(event for event in events if event["event"] == "response.completed")["data"]["response"]["output"]
        if item["call_id"] == "call_shell"
    )

    assert done["data"]["item"]["environment"] == {
        "type": "local",
        "skills": [{"name": "python", "path": "/tmp/python"}],
    }
    assert output_item["environment"] == {
        "type": "local",
        "skills": [{"name": "python", "path": "/tmp/python"}],
    }


def test_translate_anthropic_tool_use_with_start_input_closes_with_full_arguments():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_2",
                "name": "lookup_weather",
                "input": {"city": "Shanghai"},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    arg_deltas = [event for event in events if event["event"] == "response.function_call_arguments.delta"]
    args_done = next(event for event in events if event["event"] == "response.function_call_arguments.done")

    assert arg_deltas == []
    assert args_done["data"]["arguments"] == '{"city": "Shanghai"}'


def test_translate_anthropic_custom_tool_stream_uses_custom_input_delta_events():
    translator = ResponsesEventTranslator(
        response_context={"tools": [{"type": "custom", "name": "apply_patch"}]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "call_patch", "name": "apply_patch", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"REA'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": 'DME.md"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    delta_events = [event for event in events if event["event"] == "response.custom_tool_call_input.delta"]
    done_event = next(event for event in events if event["event"] == "response.custom_tool_call_input.done")

    assert [event["data"]["delta"] for event in delta_events] == ['{"path":"REA', 'DME.md"}']
    assert done_event["data"]["input"] == '{"path":"README.md"}'


def test_translate_anthropic_stream_completed_includes_usage_details():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    completed = next(event for event in events if event["event"] == "response.completed")

    assert completed["data"]["response"]["usage"]["input_tokens_details"] == {"cached_tokens": 0}
    assert completed["data"]["response"]["usage"]["output_tokens_details"] == {"reasoning_tokens": 0}


def test_translate_anthropic_stream_rejects_tool_use_without_call_id():
    translator = ResponsesEventTranslator(response_context={})

    translator.feed({"type": "message_start", "message": {"id": "msg_stream"}})

    with pytest.raises(UnsupportedFeatureError, match="tool call call_id"):
        translator.feed(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "name": "lookup_weather", "input": {"city": "Shanghai"}},
            }
        )


def test_translate_anthropic_stream_omits_logprobs_when_not_requested():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    part_added = next(event for event in events if event["event"] == "response.content_part.added")
    text_done = next(event for event in events if event["event"] == "response.output_text.done")

    assert "logprobs" not in part_added["data"]["part"]
    assert "logprobs" not in text_done["data"]


def test_translate_anthropic_stream_omits_logprobs_when_zero_requested():
    translator = ResponsesEventTranslator(response_context={"top_logprobs": 0})
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    part_added = next(event for event in events if event["event"] == "response.content_part.added")
    text_done = next(event for event in events if event["event"] == "response.output_text.done")

    assert "logprobs" not in part_added["data"]["part"]
    assert "logprobs" not in text_done["data"]


def test_translate_anthropic_stream_marks_max_tokens_stop_as_incomplete():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Partial"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    message_done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "message"
    )
    completed = next(event for event in events if event["event"] == "response.completed")

    assert message_done["data"]["item"]["status"] == "incomplete"
    assert completed["data"]["response"]["status"] == "incomplete"
    assert completed["data"]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


@pytest.mark.parametrize("stop_reason", ["pause_turn", "model_context_window_exceeded"])
def test_translate_anthropic_stream_omits_nonstandard_incomplete_reason_values(stop_reason):
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Partial"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    completed = next(event for event in events if event["event"] == "response.completed")

    assert completed["data"]["response"]["status"] == "incomplete"
    assert completed["data"]["response"]["incomplete_details"] in (None, {})


def test_translate_anthropic_reasoning_done_item_includes_reasoning_text_content():
    translator = ResponsesEventTranslator()
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_reasoning"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Planning"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    for item in anthropic_events:
        events.extend(translator.feed(item))

    done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "reasoning"
    )
    added = next(
        event
        for event in events
        if event["event"] == "response.output_item.added" and event["data"]["item"]["type"] == "reasoning"
    )
    completed = next(event for event in events if event["event"] == "response.completed")

    assert added["data"]["item"]["status"] == "in_progress"
    assert done["data"]["item"]["status"] == "completed"
    assert done["data"]["item"]["content"] == [{"type": "reasoning_text", "text": "Planning"}]
    assert completed["data"]["response"]["output"][0]["content"] == [
        {"type": "reasoning_text", "text": "Planning"}
    ]


def test_translate_anthropic_stream_rejects_multiple_tool_calls_when_parallel_disabled():
    translator = ResponsesEventTranslator(
        response_context={"parallel_tool_calls": False},
    )
    translator.feed({"type": "message_start", "message": {"id": "msg_123"}})
    translator.feed(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "call_1", "name": "tool_a", "input": {"x": 1}},
        }
    )

    with pytest.raises(UnsupportedFeatureError, match="parallel_tool_calls"):
        translator.feed(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "call_2", "name": "tool_b", "input": {"y": 2}},
            }
        )
