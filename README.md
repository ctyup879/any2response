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
- `tool_choice.allowed_tools` via validated proxy-side tool filtering
- `tool_choice` best-effort support for named `custom`, `apply_patch`, and `shell` selections
- built-in `apply_patch` / `shell` selection synthesized into proxy-side tool definitions when needed
- Function call / function call output turn translation
- built-in `apply_patch_call` / `apply_patch_call_output` turn translation
- built-in `shell_call` / `shell_call_output` turn translation
- Embedded `tool_use` / `tool_result` content blocks inside messages
- `custom_tool_call` / `custom_tool_call_output` input items mapped onto Anthropic tool turns
- custom tools exposed upstream through a string-input wrapper object so text inputs remain usable
- custom tool `grammar` format with `regex` syntax mapped into a string pattern constraint
- Responses output typing for `function_call`, `custom_tool_call`, `apply_patch_call`, and `shell_call`
- Text/image/PDF user content blocks:
  - `input_text`
  - `input_image` with `data:` URL or `http(s)` URL
  - `input_image.detail` values other than `auto` are explicitly rejected
  - `input_file` with inline `data:` for text, JSON, images, and PDFs
- Sampling and output controls:
  - `max_output_tokens` when explicitly provided
  - `temperature`
  - `top_p`
  - `stop`
  - `text.format` and legacy `response_format` for JSON-mode instruction injection
  - best-effort `text.verbosity` via injected system guidance
  - `prompt_cache_key` retained in proxy response context
  - `parallel_tool_calls: false` enforced proxy-side by rejecting upstream turns that contain multiple tool calls
  - `stream_options.include_obfuscation` on streaming delta events
- Reasoning controls:
  - `reasoning.effort` including `xhigh` mapped approximately onto Anthropic thinking budgets
  - `reasoning.summary`, with proxy-side summary shaping for `concise`
  - deprecated `reasoning.generate_summary`, normalized onto the same summary behavior
- Streaming translation for:
  - text deltas
  - reasoning/thinking deltas
  - function call argument deltas
- SSE parsing with standard multiline `data:` support and `event:` header tolerance
- Non-stream Anthropic `thinking` blocks mapped into Responses `reasoning` output items
- Anthropic `thinking` blocks also surfaced as assistant `message` items with `phase: "commentary"` so follow-up requests can preserve assistant phase semantics
- reasoning output items carry `status` on both streaming and non-streaming paths
- Anthropic `stop_reason` mapped into Responses `status` / `incomplete_details` for truncated turns

Explicitly unsupported today:

- `background`
- `store=true`
- `previous_response_id`
- `truncation` values other than `disabled`
- `max_tool_calls`
- `item_reference`
- `conversation`
- `context_management`
- `prompt`
- `prompt_cache_retention`
- `safety_identifier`
- `service_tier`
- `input_file.file_id`
- `input_image.file_id`
- `input_image.detail` values other than `auto`
- remote text-file fetching and most non-text/non-image/non-PDF file media types
- named OpenAI hosted or remote tool types beyond plain function/custom tools, such as `file_search`, `web_search`, `computer_use`, `code_interpreter`, and `mcp`
- full OpenAI reasoning item replay semantics
- non-empty `include` requests, including `reasoning.encrypted_content`, `message.output_text.logprobs`, and `message.input_image.image_url`
- all `top_logprobs` requests
- custom tool grammars with `syntax: "lark"`
- annotations/citations style response extras

Compatibility notes:

- `custom` tools from Codex are accepted and exposed upstream as plain callable tools with an object schema.
- assistant text outputs are labeled with `phase: "final_answer"`.
- `output_text` only reflects assistant `final_answer` text; commentary / reasoning text is not folded into `output_text`.
- assistant message `phase` is bridged proxy-side rather than preserved as a native upstream field:
  - input `phase: "commentary"` maps onto Anthropic `thinking` blocks
  - input `phase: "final_answer"` remains normal assistant text
  - Anthropic `thinking` output is surfaced both as a `reasoning` item and as an assistant `message` item with `phase: "commentary"` for multi-turn replay
  - non-assistant messages with `phase` are rejected
  - consecutive assistant messages remain distinct unless they must be merged to keep a pending `tool_result` adjacent to its `tool_use` for MiniMax compatibility
- function tools with `strict=false` are accepted for Codex compatibility; the flag is preserved in proxy response context but not enforced by the Anthropic upstream.
- function tools omitted `strict` on input are echoed back with the OpenAI default `strict: true`.
- developer/system messages are text-only; non-text content is rejected instead of being dropped.
- `reasoning` input items are accepted only when they contain textual reasoning content; encrypted reasoning replay payloads are explicitly unsupported.
- `reasoning.summary: "concise"` now shapes reasoning summaries proxy-side instead of merely echoing the request field back.
- built-in `apply_patch` and `shell` calls are modeled with their OpenAI item types on the Responses side, while still being proxied upstream as Anthropic-style tool calls.
- `shell_call.environment` is preserved structurally when it matches the supported OpenAI shapes (`local` or `container_reference`), and top-level `tools[].environment` is carried as a shell input-schema default plus a response-side fallback when the upstream tool call omits it.
- `shell_call` is normalized to the OpenAI `action` object shape on the Responses side; legacy flat shell payloads are still accepted on input for compatibility.
- custom tool grammars with `syntax: "regex"` are mapped structurally; `syntax: "lark"` is explicitly rejected.
- built-in tool output replay preserves `type` / `id` / `status` / `max_output_length` inside the serialized tool result payload, and still maps failure-like statuses and shell nonzero exits/timeouts onto `is_error: true`.
- failed `apply_patch_call_output` and `shell_call_output` items are translated into Anthropic `tool_result` blocks with `is_error: true`.
- unsupported hosted or remote tool descriptors, including nameless hosted tools from Codex, are rejected explicitly instead of being ignored or silently dropped.
- malformed function/custom tool definitions and malformed tool-call replay items are rejected explicitly instead of being skipped or silently normalized.
- unsupported multimodal content inside tool outputs is rejected explicitly instead of being stringified.
- orphan `tool_result` blocks are rejected explicitly instead of being silently filtered out.
- malformed `tool_choice.allowed_tools` entries, unknown allowed tool names, and invalid scalar request controls are rejected locally before upstream dispatch.
- invalid `stream` shapes are rejected locally instead of falling through to runtime type errors.
- `instructions` supports either a string or a list of Responses input items; invalid instruction item shapes are rejected locally.
- non-object `tools` entries and non-object `input` items are rejected explicitly instead of being silently ignored or coerced.
- non-object request bodies and invalid `model` values are rejected locally before upstream dispatch.
- `metadata` now follows the documented OpenAI scalar value types (`string`/`number`/`boolean`) and rejects out-of-range key/value counts and lengths locally.
- malformed message content arrays and malformed tool-call payloads are rejected explicitly instead of being stringified or wrapped into proxy-invented objects.
- non-built-in function/custom tool outputs must be strings or valid multimodal content arrays, matching the documented Responses schema.
