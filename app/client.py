import json

import httpx


class UpstreamHTTPError(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def parse_upstream_error(body):
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
    else:
        text = str(body)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if payload.get("message"):
            return str(payload["message"])
    return text


def parse_sse_events(lines):
    event_name = None
    data_lines = []

    def flush():
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        data = "\n".join(data_lines).strip()
        event_name = None
        data_lines = []
        if not data or data == "[DONE]":
            return None
        return json.loads(data)

    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            event = flush()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or None
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue

    event = flush()
    if event is not None:
        yield event


class MiniMaxClient:
    def __init__(self, settings):
        self.settings = settings

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-api-key": self.settings.minimax_api_key,
            "Anthropic-Version": self.settings.anthropic_version,
            "Anthropic-Beta": self.settings.anthropic_beta,
        }

    async def create_message(self, payload):
        request_payload = dict(payload)
        request_payload["stream"] = False
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            response = await client.post(
                self.settings.upstream_base_url,
                headers=self._headers(),
                json=request_payload,
            )
        if response.status_code >= 400:
            raise UpstreamHTTPError(response.status_code, parse_upstream_error(response.text))
        return response.json()

    async def stream_messages(self, payload):
        request_payload = dict(payload)
        request_payload["stream"] = True
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            async with client.stream(
                "POST",
                self.settings.upstream_base_url,
                headers=self._headers(),
                json=request_payload,
            ) as response:
                if response.status_code >= 400:
                    raise UpstreamHTTPError(
                        response.status_code,
                        parse_upstream_error(await response.aread()),
                    )

                buffered_lines = []
                async for line in response.aiter_lines():
                    buffered_lines.append(line)
                    if line == "":
                        for event in parse_sse_events(buffered_lines):
                            yield event
                        buffered_lines = []
                if buffered_lines:
                    for event in parse_sse_events(buffered_lines):
                        yield event
