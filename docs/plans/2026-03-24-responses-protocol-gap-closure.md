# Responses Protocol Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining high-value OpenAI Responses protocol mismatches in the MiniMax proxy without regressing Codex compatibility.

**Architecture:** Keep the proxy conservative: add best-effort mappings where the upstream can support them, and convert the rest from silent degradation into explicit `unsupported_feature` errors. Extend the translator and streaming layer only where the output shape can be improved safely.

**Tech Stack:** FastAPI, Python translator layer, pytest

---

### Task 1: Cover request-side silent degradations with tests

**Files:**
- Modify: `tests/test_translator.py`

**Step 1: Write the failing test**

Add focused tests for:
- `tool_choice={"type":"custom","name":"..."}` support
- unsupported `tool_choice` variants (`mcp`, hosted tools, `apply_patch`, `shell`) being rejected instead of downgraded
- `reasoning.summary` rejection
- `reasoning.effort="xhigh"` support
- `text.verbosity` support
- `stream_options` handling
- function tool `strict=false` rejection
- `json_schema.strict=false` no longer forcing strict wording

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_translator.py -k 'tool_choice or reasoning or verbosity or stream_options or strict'`

Expected: FAIL on the uncovered protocol mismatches.

**Step 3: Write minimal implementation**

Update `app/translator.py` to:
- translate `tool_choice.custom`
- explicitly reject unsupported `tool_choice` types
- reject unsupported `reasoning.summary`
- map `xhigh`
- carry `text.verbosity`
- reject unsupported `stream_options`
- reject `strict=false` on function tools
- make `json_schema.strict` wording conditional

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_translator.py -k 'tool_choice or reasoning or verbosity or stream_options or strict'`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_translator.py app/translator.py
git commit -m "feat: close remaining request protocol gaps"
```

### Task 2: Improve reasoning response parity

**Files:**
- Modify: `tests/test_translator.py`
- Modify: `tests/test_stream_translation.py`
- Modify: `app/translator.py`

**Step 1: Write the failing test**

Add tests that require reasoning output items to include `content` with `reasoning_text` alongside existing summary output.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_translator.py tests/test_stream_translation.py -k reasoning_text`

Expected: FAIL because reasoning items omit `content`.

**Step 3: Write minimal implementation**

Update reasoning output builders so non-stream and final stream payloads expose `content: [{"type":"reasoning_text","text": ...}]` when reasoning text is present.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_translator.py tests/test_stream_translation.py -k reasoning_text`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_translator.py tests/test_stream_translation.py app/translator.py
git commit -m "feat: enrich reasoning response payloads"
```

### Task 3: Update documentation and verify end to end

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

No automated doc test. Treat the doc diff as the check artifact.

**Step 2: Run verification**

Run:
- `pytest -q`
- `source ~/.bashrc && codex exec --profile m128py "Say only: protocol-gap-closure-check"`

Expected:
- pytest passes
- Codex returns `protocol-gap-closure-check`

**Step 3: Write minimal documentation**

Update `README.md` to describe the new support and explicit unsupported behavior for:
- `tool_choice`
- `text.verbosity`
- `stream_options`
- reasoning controls
- function tool `strict`

**Step 4: Commit**

```bash
git add README.md app/translator.py tests/test_translator.py tests/test_stream_translation.py
git commit -m "feat: align remaining responses protocol semantics"
```
