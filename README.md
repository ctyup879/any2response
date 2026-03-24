# MiniMax Python Responses Proxy

Local Python proxy that exposes `/v1/responses` for `codex` and forwards requests to MiniMax's Anthropic-compatible API.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
./start_proxy.sh
```

## Run As a User Service

Install the user unit:

```bash
python3 scripts/install_user_service.py
systemctl --user daemon-reload
systemctl --user enable --now minimaxdemo-proxy.service
```

Inspect it:

```bash
systemctl --user status minimaxdemo-proxy.service --no-pager
journalctl --user -u minimaxdemo-proxy.service -n 50 --no-pager
```

Stop or disable it:

```bash
systemctl --user stop minimaxdemo-proxy.service
systemctl --user disable minimaxdemo-proxy.service
```

## Test

```bash
.venv/bin/pytest -v
```

## Codex

Profile name: `m128py`

Install local model metadata once so Codex does not fall back to generic metadata:

```bash
python3 scripts/install_codex_metadata.py
```

## Protocol Coverage

This proxy implements the subset of the OpenAI Responses API needed for `codex` and related function-calling flows. It is not yet a full OpenAI Responses parity layer.

Currently supported on `/v1/responses`:

- Text input via top-level string input or message items
- Developer/system instructions folded into Anthropic `system`
- Function tools, `tool_choice` (`auto`, `none`, `required`, named function)
- `tool_choice.allowed_tools` via proxy-side tool filtering
- Function call / function call output turn translation
- Embedded `tool_use` / `tool_result` content blocks inside messages
- `custom_tool_call` / `custom_tool_call_output` input items mapped onto Anthropic tool turns
- Responses output typing for both `function_call` and `custom_tool_call`
- Text/image/PDF user content blocks:
  - `input_text`
  - `input_image` with `data:` URL or `http(s)` URL
  - `input_file` with inline `data:` for text, JSON, images, and PDFs
- Sampling and output controls:
  - `max_output_tokens`
  - `temperature`
  - `top_p`
  - `stop`
  - `text.format` and legacy `response_format` for JSON-mode instruction injection
  - `prompt_cache_key` retained in proxy response context
  - `include: ["reasoning.encrypted_content"]` accepted as a no-op compatibility flag
  - `parallel_tool_calls: false` enforced proxy-side by rejecting upstream turns that contain multiple tool calls
- Streaming translation for:
  - text deltas
  - reasoning/thinking deltas
  - function call argument deltas
- SSE parsing with standard multiline `data:` support and `event:` header tolerance
- Non-stream Anthropic `thinking` blocks mapped into Responses `reasoning` output items

Explicitly unsupported today:

- `background`
- `previous_response_id`
- `item_reference`
- `conversation`
- `context_management`
- `prompt`
- `prompt_cache_retention`
- `safety_identifier`
- `service_tier`
- `input_file.file_id`
- remote text-file fetching and most non-text/non-image/non-PDF file media types
- named OpenAI hosted tool types beyond plain function tools, such as `file_search`, `web_search`, and `code_interpreter`
- full OpenAI reasoning item replay semantics
- annotations/logprobs/citations style response extras

Compatibility notes:

- `custom` tools from Codex are accepted and exposed upstream as plain callable tools with an object schema.
- `include: ["reasoning.encrypted_content"]` is still accepted because `codex` sends it today, but MiniMax does not expose an upstream equivalent that this proxy can map into OpenAI-style `encrypted_content`.
- nameless hosted tools from Codex are ignored instead of failing the request, so Codex CLI can continue to operate when it sends local-only tool descriptors the proxy cannot translate upstream.
