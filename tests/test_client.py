from app.client import parse_sse_events, parse_upstream_error


def test_parse_sse_events_supports_multiline_data_and_event_headers():
    lines = [
        "event: content_block_delta",
        'data: {"type":"content_block_delta",',
        'data: "index":0,"delta":{"type":"text_delta","text":"Hi"}}',
        "",
        "data: [DONE]",
        "",
    ]

    events = list(parse_sse_events(lines))

    assert events == [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}]


def test_parse_sse_events_ignores_comments_and_blank_lines():
    lines = [
        ": keep-alive",
        "",
        'data: {"type":"message_start","message":{"id":"msg_1"}}',
        "",
    ]

    events = list(parse_sse_events(lines))

    assert events == [{"type": "message_start", "message": {"id": "msg_1"}}]


def test_parse_upstream_error_extracts_nested_message():
    body = b'{"type":"error","error":{"type":"api_error","message":"unknown error (1000)"},"request_id":"req_1"}'

    message = parse_upstream_error(body)

    assert message == "unknown error (1000)"
