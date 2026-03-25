from types import SimpleNamespace

from app.client import MiniMaxClient, parse_sse_events, parse_upstream_error


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


def test_minimax_client_adds_files_and_mcp_beta_headers_when_payload_requires_them():
    client = MiniMaxClient(
        SimpleNamespace(
            minimax_api_key="minimax-secret",
            anthropic_version="2023-06-01",
            anthropic_beta="claude-code-20250219,interleaved-thinking-2025-05-14",
        )
    )

    headers = client._headers(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "file", "file_id": "file_img_123"}}],
                }
            ],
            "mcp_servers": [{"type": "url", "name": "deepwiki", "url": "https://mcp.deepwiki.com/mcp"}],
        }
    )

    assert headers["Anthropic-Beta"] == ",".join(
        [
            "claude-code-20250219",
            "interleaved-thinking-2025-05-14",
            "files-api-2025-04-14",
            "mcp-client-2025-11-20",
        ]
    )
