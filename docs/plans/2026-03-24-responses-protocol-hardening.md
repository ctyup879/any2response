# Responses Protocol Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove misleading partial support from the MiniMax Responses proxy and replace it with correct output typing or explicit unsupported-feature errors.

**Architecture:** The change stays inside the translation layer. Request validation will reject unsupported official fields early, response translation will derive tool item types from request context, and both stream and non-stream translators will fail fast instead of silently truncating tool calls.

**Tech Stack:** Python 3, FastAPI, httpx, pytest

---

### Task 1: Document the protocol hardening rules

**Files:**
- Create: `docs/plans/2026-03-24-responses-protocol-hardening-design.md`
- Create: `docs/plans/2026-03-24-responses-protocol-hardening.md`

**Step 1: Write the failing test**

Manual verification target:
- the design and implementation plan do not exist yet

**Step 2: Run test to verify it fails**

Run: `rg --files docs/plans | rg 'responses-protocol-hardening'`
Expected: no matches

**Step 3: Write minimal implementation**

Create the design and plan documents describing scope, behavior changes, and testing strategy.

**Step 4: Run test to verify it passes**

Run: `rg --files docs/plans | rg 'responses-protocol-hardening'`
Expected: both files are listed

### Task 2: Add failing translator tests

**Files:**
- Modify: `tests/test_translator.py`
- Modify: `tests/test_stream_translation.py`

**Step 1: Write the failing test**

Add tests for:
- rejecting `include=["reasoning.encrypted_content"]`
- rejecting unsupported official request fields
- returning `custom_tool_call` for custom tools
- failing when `parallel_tool_calls=false` and multiple tool calls appear

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_translator.py tests/test_stream_translation.py -q`
Expected: FAIL with assertion mismatches for the new protocol expectations

**Step 3: Write minimal implementation**

Only after the failures are confirmed, update translator behavior to satisfy the new expectations.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_translator.py tests/test_stream_translation.py -q`
Expected: PASS

### Task 3: Implement translator hardening

**Files:**
- Modify: `app/translator.py`

**Step 1: Write the failing test**

Covered by Task 2.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_translator.py tests/test_stream_translation.py -q`
Expected: FAIL before translator changes

**Step 3: Write minimal implementation**

Implement:
- explicit include rejection
- explicit unsupported-field validation
- custom tool output typing
- multi-tool rejection for `parallel_tool_calls=false`

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_translator.py tests/test_stream_translation.py -q`
Expected: PASS

### Task 4: Verify end-to-end behavior

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

Manual verification target:
- documentation still overstates compatibility

**Step 2: Run test to verify it fails**

Run: `rg -n 'encrypted_content|parallel_tool_calls|custom_tool_call|unsupported' README.md`
Expected: missing or outdated protocol boundary notes

**Step 3: Write minimal implementation**

Update README protocol support notes, then run:

```bash
.venv/bin/pytest -q
source ~/.bashrc && codex exec --profile m128py "Say only: protocol-hardening-check"
```

**Step 4: Run test to verify it passes**

Expected:
- pytest passes
- Codex smoke test succeeds through the local proxy
