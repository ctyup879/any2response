# Responses Protocol Shape Fixes Design

## Scope

Tighten four remaining protocol mismatches in the Python Responses proxy:

- preserve assistant message `phase` semantics instead of silently dropping them
- align `shell_call` request and response shape to the OpenAI `action` object
- populate `reasoning` item `status` on streaming and non-streaming responses
- replace best-effort custom grammar guidance with stricter behavior

## Chosen Approach

### Assistant phase

Anthropic messages do not expose a native equivalent for OpenAI assistant `phase`.
The proxy will preserve assistant `phase` best-effort by prepending a synthetic text
block such as `[assistant phase: commentary]` before the translated assistant content.
This keeps the phase visible to the upstream model without inventing unsupported
request fields. Non-assistant messages with `phase` remain invalid and will be
rejected explicitly.

### Shell call shape

Responses-side `shell_call` items will use the OpenAI `action` object shape:

- `shell_call.action.commands`
- `shell_call.action.timeout_ms`
- `shell_call.action.max_output_length`

The proxy will continue to accept legacy flat shell inputs on request translation so
existing local tests and earlier transcripts do not break unnecessarily.

### Reasoning item status

`reasoning` output items should carry `status` like other returned items. The proxy
will emit:

- `in_progress` on streaming `response.output_item.added`
- `completed` or `incomplete` on streaming `response.output_item.done`
- `completed` or `incomplete` on non-streaming translated responses

The final reasoning status follows the same completion mapping already used for
message items from Anthropic `stop_reason`.

### Custom grammar formats

Only `regex` grammars have a straightforward structural mapping into the Anthropic
tool schema wrapper: the proxy can map them to a JSON Schema `pattern` on the
wrapped string input.

`lark` grammars have no equivalent upstream constraint. Instead of pretending to
support them via descriptions, the proxy will reject them explicitly.

## Verification

Add red-green tests for:

- assistant `phase` preservation
- `shell_call.action` request and response shape
- reasoning `status` in both streaming and non-streaming flows
- regex grammar mapping and explicit `lark` rejection
