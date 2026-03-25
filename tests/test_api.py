import json

from fastapi.testclient import TestClient

from app.main import create_app


class FakeUpstreamClient:
    def __init__(self):
        self.last_payload = None

    async def create_message(self, payload):
        self.last_payload = payload
        return {
            "id": "msg_fake",
            "role": "assistant",
            "type": "message",
            "content": [{"type": "text", "text": "Hello from MiniMax"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 4},
        }

    async def stream_messages(self, payload):
        self.last_payload = payload
        for item in [
            {"type": "message_start", "message": {"id": "msg_stream"}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello from stream"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
            {"type": "message_stop"},
        ]:
            yield item


class MultiToolStreamClient(FakeUpstreamClient):
    async def stream_messages(self, payload):
        self.last_payload = payload
        for item in [
            {"type": "message_start", "message": {"id": "msg_stream"}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call_1", "name": "tool_a", "input": {"x": 1}},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "call_2", "name": "tool_b", "input": {"y": 2}},
            },
            {"type": "content_block_stop", "index": 1},
        ]:
            yield item


def test_post_responses_requires_auth():
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
        },
        upstream_client=FakeUpstreamClient(),
    )
    client = TestClient(app)

    response = client.post("/v1/responses", json={"model": "codex-MiniMax-M2.7", "input": []})

    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Unauthorized"


def test_post_responses_non_stream():
    upstream = FakeUpstreamClient()
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
        },
        upstream_client=upstream,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer proxy-secret"},
        json={
            "model": "codex-MiniMax-M2.7",
            "stream": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert upstream.last_payload["messages"][0]["content"][0]["text"] == "hello"
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["usage"]["input_tokens_details"] == {"cached_tokens": 0}
    assert data["usage"]["output_tokens_details"] == {"reasoning_tokens": 0}
    assert data["output"][0]["content"][0]["text"] == "Hello from MiniMax"


def test_post_responses_writes_request_log_to_configured_path(tmp_path):
    upstream = FakeUpstreamClient()
    log_path = tmp_path / "logs" / "last_request.json"
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
            "request_log_path": str(log_path),
        },
        upstream_client=upstream,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer proxy-secret"},
        json={
            "model": "codex-MiniMax-M2.7",
            "stream": False,
            "input": "hello",
        },
    )

    assert response.status_code == 200
    assert log_path.exists()
    assert json.loads(log_path.read_text(encoding="utf-8"))["input"] == "hello"


def test_post_responses_stream():
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
        },
        upstream_client=FakeUpstreamClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer proxy-secret"},
        json={
            "model": "codex-MiniMax-M2.7",
            "stream": True,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: response.output_text.delta" in body
    assert "Hello from stream" in body
    assert "event: response.completed" in body
    assert '"input_tokens_details":{"cached_tokens":0}' in body
    assert '"output_tokens_details":{"reasoning_tokens":0}' in body
    assert "data: [DONE]" in body


def test_post_responses_forwards_replayed_tool_turns_to_upstream():
    upstream = FakeUpstreamClient()
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
        },
        upstream_client=upstream,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer proxy-secret"},
        json={
            "model": "codex-MiniMax-M2.7",
            "stream": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "check project"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": {"cmd": "pwd"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "/root/minimaxdemo",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert upstream.last_payload is not None
    assert upstream.last_payload["messages"][-2]["content"][0]["type"] == "tool_use"
    assert upstream.last_payload["messages"][-1]["content"][0]["type"] == "tool_result"


def test_post_responses_stream_emits_error_for_parallel_tool_call_violation():
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
        },
        upstream_client=MultiToolStreamClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer proxy-secret"},
        json={
            "model": "codex-MiniMax-M2.7",
            "stream": True,
            "parallel_tool_calls": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "parallel_tool_calls" in response.text
    assert "data: [DONE]" in response.text


def test_post_responses_rejects_nonempty_include_requests():
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
        },
        upstream_client=FakeUpstreamClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer proxy-secret"},
        json={
            "model": "codex-MiniMax-M2.7",
            "stream": False,
            "include": ["reasoning.encrypted_content"],
            "input": "hello",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsupported_feature"
    assert "include" in response.json()["error"]["message"]


def test_post_responses_rejects_zero_top_logprobs():
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
        },
        upstream_client=FakeUpstreamClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer proxy-secret"},
        json={
            "model": "codex-MiniMax-M2.7",
            "stream": False,
            "top_logprobs": 0,
            "input": "hello",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsupported_feature"
    assert "top_logprobs" in response.json()["error"]["message"]


def test_post_responses_rejects_non_object_request_body():
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
        },
        upstream_client=FakeUpstreamClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer proxy-secret"},
        json=["not", "an", "object"],
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsupported_feature"
    assert "request body" in response.json()["error"]["message"]


def test_post_responses_rejects_invalid_json_body():
    app = create_app(
        {
            "proxy_api_key": "proxy-secret",
            "minimax_api_key": "minimax-secret",
        },
        upstream_client=FakeUpstreamClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        headers={
            "Authorization": "Bearer proxy-secret",
            "Content-Type": "application/json",
        },
        content="{",
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "Invalid JSON" in response.json()["error"]["message"]
