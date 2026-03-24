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

## Test

```bash
.venv/bin/pytest -v
```

## Codex

Profile name: `m128py`

## Protocol Coverage

This proxy implements the subset of the OpenAI Responses API needed for `codex` and related function-calling flows. It is not yet a full OpenAI Responses parity layer.

Currently supported on `/v1/responses`:

- Text input via top-level string input or message items
- Developer/system instructions folded into Anthropic `system`
- Function tools, `tool_choice` (`auto`, `none`, `required`, named function)
- `tool_choice.allowed_tools` via proxy-side tool filtering
- Function call / function call output turn translation
- Embedded `tool_use` / `tool_result` content blocks inside messages
- Text/image/PDF user content blocks:
  - `input_text`
  - `input_image` with `data:` URL or `http(s)` URL
  - `input_file` with inline `data:` for images/PDFs
- Sampling and output controls:
  - `max_output_tokens`
  - `temperature`
  - `top_p`
  - `stop`
  - `text.format` and legacy `response_format` for JSON-mode instruction injection
- Streaming translation for:
  - text deltas
  - reasoning/thinking deltas
  - function call argument deltas
- SSE parsing with standard multiline `data:` support and `event:` header tolerance
- Non-stream Anthropic `thinking` blocks mapped into Responses `reasoning` output items

Explicitly unsupported today:

- `background`
- `previous_response_id`
- `input_file.file_id`
- non-image/non-PDF file media types
- named OpenAI hosted tool types beyond plain function tools, such as `file_search`, `web_search`, and `code_interpreter`
- full OpenAI reasoning item replay semantics
- annotations/logprobs/citations style response extras

Compatibility notes:

- `custom` tools from Codex are accepted and exposed upstream as plain callable tools with an object schema.
- nameless hosted tools from Codex are ignored instead of failing the request, so Codex CLI can continue to operate when it sends local-only tool descriptors the proxy cannot translate upstream.
