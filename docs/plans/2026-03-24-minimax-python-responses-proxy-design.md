# MiniMax Python Responses Proxy Design

**Goal:** Build a standalone Python service in `/root/minimaxdemo` that accepts Codex `/v1/responses` requests, translates them to MiniMax's Anthropic-compatible Messages API, and returns Codex-compatible Responses output.

**Context**

- Existing working reference: `/root/omniroute`
- Target upstream: `https://api.minimaxi.com/anthropic/v1/messages?beta=true`
- Local client: `codex` via a new profile in `/root/.codex/config.toml`
- Security model:
  - MiniMax upstream key is stored only in the Python service configuration
  - Codex authenticates to the local Python proxy using a separate local proxy key

## Scope

The service will implement the minimum Responses API subset required for Codex interactive prompts:

- `POST /v1/responses`
- `GET /health`
- OpenAI Responses request fields:
  - `model`
  - `instructions`
  - `input`
  - `tools`
  - `tool_choice`
  - `stream`
  - `max_output_tokens`
- Responses output:
  - streaming SSE in Codex-compatible event order
  - non-stream JSON response for direct testing
- OpenAI-style error JSON for upstream failures

## Out of Scope

- OmniRoute dashboard features
- multi-provider routing
- OAuth
- model discovery endpoints
- background tasks
- unsupported Responses tools like `file_search`

## Architecture

The system has three layers:

1. FastAPI HTTP layer
   - validates incoming auth
   - parses `/v1/responses` requests
   - returns SSE or JSON

2. Translation layer
   - converts OpenAI Responses input to Anthropic Messages payload
   - converts Anthropic streaming events back to Responses SSE events
   - aggregates final output for non-stream requests

3. Upstream client layer
   - sends requests to MiniMax with `x-api-key`
   - sets `Anthropic-Version: 2023-06-01`
   - sets `Anthropic-Beta: claude-code-20250219,interleaved-thinking-2025-05-14`

## Request Translation

Incoming Responses requests will be normalized into Anthropic Messages requests:

- `instructions` -> `system`
- `input[].type == "message"` -> Anthropic `messages[]`
- `input_text` -> `{ "type": "text", "text": ... }`
- `function_call_output` -> user message content with `tool_result`
- `tools` -> Anthropic `tools`
- `max_output_tokens` -> `max_tokens`
- `stream` -> `stream`

The service will reject unsupported features with HTTP 400, rather than silently degrading behavior.

## Response Translation

For stream mode:

- emit `response.created`
- emit `response.in_progress`
- create one assistant output item
- map Anthropic text deltas to:
  - `response.output_item.added`
  - `response.content_part.added`
  - `response.output_text.delta`
- on completion emit:
  - `response.output_text.done`
  - `response.content_part.done`
  - `response.output_item.done`
  - `response.completed`

For non-stream mode:

- buffer Anthropic SSE text deltas
- return a compact OpenAI Responses JSON object with one assistant output message

## Configuration

Project files:

- `requirements.txt`
- `README.md`
- `.env.example`
- `app/`
- `tests/`

Runtime settings:

- `MINIMAX_API_KEY`
- `PROXY_API_KEY`
- `HOST`
- `PORT`
- `UPSTREAM_BASE_URL`

## Codex Integration

Local Codex config will add:

- one `model_provider` pointing to `http://127.0.0.1:<port>/v1`
- one new `profile` using that provider
- one environment variable for the proxy auth key

The success condition is concrete:

- `codex --profile <new-profile> "hello"` reaches the local Python proxy
- the proxy reaches MiniMax
- Codex receives a valid answer

## Testing

Tests will cover:

- request translation
- Anthropic stream to Responses SSE translation
- `/v1/responses` endpoint auth handling
- `/v1/responses` non-stream output
- `/v1/responses` stream output

The final verification will include a live Codex smoke test against MiniMax.
