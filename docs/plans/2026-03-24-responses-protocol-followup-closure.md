# Responses Protocol Follow-up Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the next layer of OpenAI Responses protocol mismatches in the MiniMax proxy without regressing the verified Codex path.

**Architecture:** Convert silent protocol drift into either explicit support or explicit `unsupported_feature` errors. Improve response shape parity where it is safe to do so locally, and keep upstream translation conservative for fields MiniMax cannot represent directly.

**Tech Stack:** FastAPI, Python translator layer, pytest

---

### Task 1: Add request-side regression tests

**Files:**
- Modify: `tests/test_translator.py`

**Step 1: Write the failing test**

Add focused tests for:
- rejecting input message `phase`
- rejecting invalid message `role`
- preserving built-in message `phase` on output items
- custom tool text/grammar format translation
- custom tool call string input wrapping
- built-in `apply_patch`/`shell` tool choice synthesizing tools
- rejecting unsupported image `detail`
- response `incomplete_details` and default function `strict=true`

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_translator.py tests/test_stream_translation.py -k 'phase or custom or detail or incomplete_details or strict or role'`

Expected: FAIL on uncovered protocol gaps.

**Step 3: Write minimal implementation**

Update `app/translator.py` to:
- validate input roles and phases
- add output message phase
- normalize response tool metadata
- translate custom tools as string-input wrappers
- synthesize chosen built-in tools when needed
- reject unsupported image detail
- add `incomplete_details: null`

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_translator.py tests/test_stream_translation.py -k 'phase or custom or detail or incomplete_details or strict or role'`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_translator.py tests/test_stream_translation.py app/translator.py README.md
git commit -m "feat: close follow-up responses protocol gaps"
```

### Task 2: Full verification

**Files:**
- Modify: `README.md`

**Step 1: Run verification**

Run:
- `pytest -q`
- `source ~/.bashrc && codex exec --profile m128py "Say only: followup-protocol-check"`
- `source ~/.bashrc && codex exec --profile m128py "Read start_proxy.sh, then reply with only the host and port as host:port."`

Expected:
- pytest passes
- Codex returns `followup-protocol-check`
- Codex returns `127.0.0.1:8765`

**Step 2: Update docs**

Document the new behavior for:
- message `phase`
- custom tool text input handling
- built-in `apply_patch` / `shell` tool selection
- image detail limitations
- response shape parity additions

**Step 3: Commit**

```bash
git add README.md docs/plans/2026-03-24-responses-protocol-followup-closure.md app/translator.py tests/test_translator.py tests/test_stream_translation.py
git commit -m "feat: close follow-up responses protocol gaps"
```
