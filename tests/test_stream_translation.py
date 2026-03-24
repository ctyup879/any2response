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
            "include": ["reasoning.encrypted_content"],
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
    assert created["data"]["response"]["include"] == ["reasoning.encrypted_content"]
    assert created["data"]["response"]["prompt_cache_key"] == "cache-key-123"
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


def test_translate_anthropic_thinking_block_emits_reasoning_encrypted_content_when_requested():
    translator = ResponsesEventTranslator(
        response_context={"include": ["reasoning.encrypted_content"]},
    )
    events = []

    anthropic_events = [
        {"type": "message_start", "message": {"id": "msg_123"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": "enc_sig_456"},
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
    completed = next(event for event in events if event["event"] == "response.completed")

    assert done["data"]["item"]["encrypted_content"] == "enc_sig_456"
    assert completed["data"]["response"]["output"][0]["encrypted_content"] == "enc_sig_456"


def test_translate_anthropic_reasoning_and_text_share_same_output_index():
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
    message_added = next(
        event
        for event in events
        if event["event"] == "response.output_item.added" and event["data"]["item"]["type"] == "message"
    )
    text_delta = next(event for event in events if event["event"] == "response.output_text.delta")

    assert reasoning_added["data"]["output_index"] == 0
    assert message_added["data"]["output_index"] == 0
    assert text_delta["data"]["output_index"] == 0


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
                "input": {"path": "README.md"},
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
    assert input_done["data"]["input"] == '{"path": "README.md"}'
    assert done["data"]["item"]["type"] == "custom_tool_call"
    assert output_item["type"] == "custom_tool_call"
    assert output_item["input"] == '{"path": "README.md"}'


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
