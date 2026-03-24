# Responses Protocol Hardening Design

**Goal:** Tighten the MiniMax proxy so it no longer advertises or silently degrades Responses API features that it cannot faithfully implement.

## Scope

This change covers four protocol gaps identified in the current proxy:

- `include=["reasoning.encrypted_content"]` is accepted today but not implemented
- `parallel_tool_calls=false` silently drops additional tool calls
- `custom` tools are translated back as `function_call` instead of `custom_tool_call`
- several official request fields are silently ignored instead of being explicitly rejected

## Design

### 1. Include handling

The proxy will reject `include=["reasoning.encrypted_content"]` until MiniMax exposes a real upstream equivalent that can be mapped into Responses reasoning items. Accepting the field without returning `encrypted_content` is a protocol lie and is worse than a clear error.

The implementation will keep `include` validation centralized so future supported values can be added in one place.

### 2. Parallel tool calls

When the client sends `parallel_tool_calls=false`, the proxy must not drop extra tool calls. Instead:

- non-stream translation will raise `UnsupportedFeatureError` if MiniMax returns more than one tool call
- stream translation will emit a terminal error when a second tool call appears

This preserves correctness and makes the limitation explicit.

### 3. Custom tool output typing

The proxy already accepts `custom` tools and custom tool call inputs. The response path will be extended so translated tool calls preserve the correct Responses item type:

- `function` tools -> `function_call`
- `custom` tools -> `custom_tool_call`

Tool metadata from the request context will be used to determine the emitted item type in both stream and non-stream responses.

### 4. Unsupported field policy

The request translator will add a white-list style validation step for known Responses create fields that the proxy still cannot implement, including:

- `conversation`
- `context_management`
- `prompt`
- `prompt_cache_retention`
- `safety_identifier`
- `service_tier`

If any of these are present, the proxy will return a clear `unsupported_feature` error instead of silently ignoring them.

## Testing

The work will follow TDD:

- add failing translator tests for each new rejection path
- add failing response translation tests for custom tool typing and multi-tool rejection
- verify the new tests fail for the expected reason
- implement the minimal translator changes
- run targeted tests, full pytest, and a real `codex exec --profile m128py` smoke test
