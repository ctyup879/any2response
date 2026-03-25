# Final Protocol Tightening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the last misleading protocol-parity claims by tightening unsupported behavior, clarifying bridge-only semantics, and locking the behavior with tests.

**Architecture:** Keep the MiniMax proxy stable for real Codex traffic while avoiding fake parity. For features that cannot be represented faithfully against the Anthropic-compatible upstream, prefer explicit unsupported behavior or clearly documented proxy-side bridging. Only make code changes where the translator can become more honest or more structurally faithful without breaking Codex.

**Tech Stack:** FastAPI, Python, pytest, uvicorn, Codex CLI

---

### Task 1: Lock the remaining semantics with failing tests

**Files:**
- Modify: `tests/test_translator.py`
- Modify: `tests/test_stream_translation.py`
- Modify: `tests/test_api.py`

**Step 1: Write the failing tests**

Add tests for:
- bridge-only `phase` behavior being documented rather than claimed as exact preservation
- built-in tool output replay preserving only the explicitly supported envelope shape
- API-level rejection paths for unsupported protocol controls staying explicit

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_translator.py tests/test_stream_translation.py tests/test_api.py -k 'phase or tool_output or include or top_logprobs or file_id'`

Expected: at least one failure caused by current translator/doc semantics not matching the tightened contract.

**Step 3: Write minimal implementation**

Only change translator/doc behavior required to make the tests pass.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_translator.py tests/test_stream_translation.py tests/test_api.py -k 'phase or tool_output or include or top_logprobs or file_id'`

Expected: PASS

### Task 2: Tighten translator semantics

**Files:**
- Modify: `app/translator.py`
- Modify: `app/main.py`

**Step 1: Write the failing test**

Use the tests from Task 1 to drive:
- no hidden parity claims for `phase`
- no hidden replay guarantees beyond the supported built-in tool-result envelope
- explicit error handling for unsupported request controls

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_translator.py tests/test_stream_translation.py tests/test_api.py -k 'phase or tool_output or include or top_logprobs or file_id'`

Expected: FAIL

**Step 3: Write minimal implementation**

Implement only:
- translator tightening needed by the tests
- route-level unsupported-feature handling if missing

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_translator.py tests/test_stream_translation.py tests/test_api.py -k 'phase or tool_output or include or top_logprobs or file_id'`

Expected: PASS

### Task 3: Align documentation with actual behavior

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

Treat the test as a doc audit checklist:
- `phase` described as bridge-only semantics
- `include` / `top_logprobs` described as explicitly unsupported
- `input_image.file_id` listed as unsupported
- built-in tool output replay described as serialized-envelope compatibility, not full protocol parity

**Step 2: Run test to verify it fails**

Run: `rg -n 'phase|include|top_logprobs|input_image.file_id|tool output replay' README.md`

Expected: wording still overclaims or omits one of the boundary notes.

**Step 3: Write minimal implementation**

Update wording only; avoid expanding scope.

**Step 4: Run test to verify it passes**

Run: `rg -n 'phase|include|top_logprobs|input_image.file_id|tool output replay' README.md`

Expected: all boundary notes present and accurate.

### Task 4: Verify end-to-end behavior

**Files:**
- Verify only

**Step 1: Run targeted tests**

Run: `.venv/bin/pytest -q tests/test_translator.py tests/test_stream_translation.py tests/test_api.py`

Expected: PASS

**Step 2: Run full test suite**

Run: `.venv/bin/pytest -q`

Expected: PASS

**Step 3: Restart the proxy**

Run: `systemctl --user restart minimaxdemo-proxy.service`

Expected: service restarts cleanly.

**Step 4: Verify service status**

Run: `systemctl --user status minimaxdemo-proxy.service --no-pager`

Expected: `active (running)`

**Step 5: Verify real Codex text path**

Run: `source ~/.bashrc && codex exec --profile m128py "Reply with exactly OK and nothing else."`

Expected: `OK`

**Step 6: Verify real Codex tool path**

Run: `source ~/.bashrc && codex exec --profile m128py "Read README.md, then reply with only the first heading including the leading #."`

Expected: `# MiniMax Python Responses Proxy`
