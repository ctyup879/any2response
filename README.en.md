# any2response

[English](README.en.md) | [简体中文](README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`any2response` is a lightweight local proxy that translates upstream model APIs into the OpenAI Responses protocol.

The current implementation focuses on one practical path:

- input: OpenAI Responses-style requests
- upstream: MiniMax Anthropic-compatible Messages API
- output: OpenAI Responses-compatible responses and streaming events

This makes it possible to plug tools such as `codex` into a single local `/v1/responses` endpoint while keeping the upstream provider behind a translation layer.

## Why

- Standardize downstream integrations on the OpenAI Responses API
- Hide provider-specific request and streaming differences behind one proxy
- Keep the deployment model simple: one local Python service, one HTTP endpoint
- Provide a base that can be extended with more upstream adapters later

## Current Scope

| Input protocol | Status | Notes |
| --- | --- | --- |
| OpenAI Responses | Supported | Main supported path, translated to MiniMax Anthropic-compatible Messages API |
| OpenAI Chat Completions | Not implemented | Not available in this repository today |
| Anthropic Messages as input | Not implemented | Current proxy is not a bidirectional protocol converter |

## Architecture

```text
client / codex
    -> local /v1/responses
    -> Python translator
    -> MiniMax Anthropic-compatible API
    -> Python translator
    -> OpenAI Responses output
```

Main components:

- `app/main.py`: FastAPI entrypoint and HTTP surface
- `app/translator.py`: Responses <-> MiniMax/Anthropic translation logic
- `app/client.py`: upstream HTTP client
- `app/config.py`: environment-based settings
- `scripts/install_user_service.py`: install a `systemd --user` service
- `scripts/install_codex_metadata.py`: install local Codex model metadata

## Core Features

- Responses request translation to MiniMax Anthropic-compatible messages
- Responses-style non-streaming and streaming output
- Function tools and Codex-oriented tool-call replay handling
- Built-in `apply_patch` and `shell` call translation
- Text, image, and PDF input coverage for the supported subset
- Local proxy authentication with a separate proxy key
- Optional `systemd --user` service installation for persistent local runs

## Quick Start

### Clone

```bash
git clone https://github.com/ctyup879/any2response.git
cd any2response
```

### Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Update `.env` with your real values:

```bash
MINIMAX_API_KEY=your-real-minimax-key
PROXY_API_KEY=your-local-proxy-key
HOST=127.0.0.1
PORT=8765
UPSTREAM_BASE_URL=https://api.minimaxi.com/anthropic/v1/messages?beta=true
```

### Run

```bash
./start_proxy.sh
```

The proxy listens on:

```text
http://127.0.0.1:8765/v1/responses
```

## Run as a User Service

Install and enable the `systemd --user` unit:

```bash
python3 scripts/install_user_service.py
systemctl --user daemon-reload
systemctl --user enable --now minimaxdemo-proxy.service
```

Inspect service status:

```bash
systemctl --user status minimaxdemo-proxy.service --no-pager
journalctl --user -u minimaxdemo-proxy.service -n 50 --no-pager
```

Stop or disable it:

```bash
systemctl --user stop minimaxdemo-proxy.service
systemctl --user disable minimaxdemo-proxy.service
```

## Example Request

```bash
curl http://127.0.0.1:8765/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${PROXY_API_KEY}" \
  -d '{
    "model": "codex-MiniMax-M2.7",
    "input": "hello",
    "stream": false
  }'
```

## Codex Integration

Local examples in this repository use the profile name `m128py`.

First export the environment variable that Codex will use. Its value should match the `PROXY_API_KEY` configured in the proxy `.env` file:

```bash
export MINIMAX_PROXY_API_KEY=your-local-proxy-key
```

Then add the provider and profile to `~/.codex/config.toml`:

```toml
[model_providers.minimax_py]
name = "MiniMax Python Responses Proxy"
base_url = "http://127.0.0.1:8765/v1"
env_key = "MINIMAX_PROXY_API_KEY"
wire_api = "responses"
requires_openai_auth = false
request_max_retries = 4
stream_max_retries = 10
stream_idle_timeout_ms = 300000

[profiles.m128py]
model = "codex-MiniMax-M2.7"
model_provider = "minimax_py"
```

Install local model metadata once:

```bash
python3 scripts/install_codex_metadata.py
```

After your local Codex config points to this proxy, a smoke test looks like:

```bash
codex exec --profile m128py "Reply with exactly OK and nothing else."
```

Additional examples:

```bash
codex exec --profile m128py "hello"
codex exec --profile m128py "Read README.md, then reply with only the first heading including the leading #."
```

## Protocol Coverage

This project does not implement full OpenAI Responses parity. It implements the subset needed for `codex` and adjacent tool-calling workflows, plus explicit validation around unsupported fields so requests are rejected instead of silently rewritten.

Supported today:

- top-level string input and message-item input
- developer and system instructions folded into upstream `system`
- function tools
- validated `tool_choice.allowed_tools`
- named `custom`, `apply_patch`, and `shell` tool selection
- `function_call`, `custom_tool_call`, `apply_patch_call`, `shell_call`
- corresponding tool output items
- embedded `tool_use` and `tool_result` blocks inside messages
- text input
- image input by `data:` URL or `http(s)` URL
- inline text, JSON, image, and PDF file inputs
- `max_output_tokens` when explicitly provided
- `temperature`
- `top_p`
- `stop`
- `text.format` and legacy `response_format`
- best-effort `text.verbosity`
- `prompt_cache_key` response-context retention
- proxy-side enforcement for `parallel_tool_calls: false`
- `reasoning.effort`
- `reasoning.summary`
- deprecated `reasoning.generate_summary`
- streaming text deltas
- streaming reasoning deltas
- streaming tool argument deltas
- SSE parsing with multiline `data:` support

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
- hosted and remote OpenAI tools such as `file_search`, `web_search`, `computer_use`, `code_interpreter`, and `mcp`
- full OpenAI reasoning replay semantics
- non-empty `include`
- all `top_logprobs` requests
- custom tool grammars with `syntax: "lark"`
- annotations and citations style response extras

Compatibility notes:

- assistant `phase` is bridged proxy-side rather than preserved as a native upstream field
- `commentary` is translated onto Anthropic `thinking` blocks
- `output_text` only reflects assistant `final_answer` text
- function tools with `strict=false` are accepted for Codex compatibility but not enforced upstream
- malformed request shapes are rejected locally instead of being silently normalized

## Security Notes

- `.env` is ignored by git and should never be committed
- `last_request.json` is ignored by git and may contain request payloads from local debugging
- keep local secret files at restrictive permissions such as `chmod 600 .env last_request.json`
- rotate provider keys before publishing if a real key was ever used outside ignored files
- review your git author identity before pushing to a public remote

## Development

Run the test suite:

```bash
.venv/bin/pytest -q
```

## Repository

- GitHub: <https://github.com/ctyup879/any2response>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
