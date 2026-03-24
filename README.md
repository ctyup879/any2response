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
- `tool_choice` best-effort support for named `custom`, `apply_patch`, and `shell` selections
- built-in `apply_patch` / `shell` selection synthesized into proxy-side tool definitions when needed
- Function call / function call output turn translation
- Embedded `tool_use` / `tool_result` content blocks inside messages
- `custom_tool_call` / `custom_tool_call_output` input items mapped onto Anthropic tool turns
- custom tools exposed upstream through a string-input wrapper object so text inputs remain usable
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
  - best-effort `text.verbosity` via injected system guidance
  - `prompt_cache_key` retained in proxy response context
  - best-effort `include: ["reasoning.encrypted_content"]` when the upstream reasoning block exposes a compatible field
  - `parallel_tool_calls: false` enforced proxy-side by rejecting upstream turns that contain multiple tool calls
  - `stream_options.include_obfuscation` on streaming delta events
- Reasoning controls:
  - `reasoning.effort` including `xhigh` mapped approximately onto Anthropic thinking budgets
- Streaming translation for:
  - text deltas
  - reasoning/thinking deltas
  - function call argument deltas
- SSE parsing with standard multiline `data:` support and `event:` header tolerance
- Non-stream Anthropic `thinking` blocks mapped into Responses `reasoning` output items

Explicitly unsupported today:

- `background`
- `store=true`
- `previous_response_id`
- `truncation` values other than `disabled`
- `max_tool_calls`
- input message `phase`
- `item_reference`
- `reasoning.summary`
- reasoning input replay items
- `conversation`
- `context_management`
- `prompt`
- `prompt_cache_retention`
- `safety_identifier`
- `service_tier`
- `input_file.file_id`
- remote text-file fetching and most non-text/non-image/non-PDF file media types
- `input_image.detail` values other than `auto`
- named OpenAI hosted or remote tool types beyond plain function/custom tools, such as `file_search`, `web_search`, `computer_use`, `code_interpreter`, and `mcp`
- full OpenAI reasoning item replay semantics
- annotations/logprobs/citations style response extras

Compatibility notes:

- `custom` tools from Codex are accepted and exposed upstream as plain callable tools with an object schema.
- assistant text outputs are labeled with `phase: "final_answer"`; input message `phase` is rejected instead of being dropped.
- function tools with `strict=false` are rejected instead of being silently weakened.
- function tools omitted `strict` on input are echoed back with the OpenAI default `strict: true`.
- developer/system messages are text-only; non-text content is rejected instead of being dropped.
- `include: ["reasoning.encrypted_content"]` is best-effort only: the proxy emits `encrypted_content` when MiniMax reasoning blocks expose a compatible `data` or `signature` field, and otherwise returns a normal reasoning item without that field.
- nameless hosted tools from Codex are ignored instead of failing the request, so Codex CLI can continue to operate when it sends local-only tool descriptors the proxy cannot translate upstream.
