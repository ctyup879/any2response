# MiniMax Python Responses Proxy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python proxy that lets Codex use a new profile against a local `/v1/responses` endpoint which forwards to MiniMax's Anthropic-compatible API.

**Architecture:** A FastAPI service exposes `/v1/responses` and `/health`, translates OpenAI Responses requests to Anthropic Messages requests, forwards them with `httpx`, and converts Anthropic SSE or final responses back into Codex-compatible Responses output. Local shell and Codex config changes wire a new profile to the service and protect it with a separate proxy API key.

**Tech Stack:** Python 3, FastAPI, httpx, uvicorn, pytest

---

### Task 1: Bootstrap repository files

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`
- Create: `app/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Write the failing test**

```python
def test_repository_bootstrap_files_exist():
    assert False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bootstrap.py -v`
Expected: FAIL with missing file assertions

**Step 3: Write minimal implementation**

Create the base repository files and package directories.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bootstrap.py -v`
Expected: PASS

### Task 2: Implement request translation

**Files:**
- Create: `app/translator.py`
- Test: `tests/test_translator.py`

**Step 1: Write the failing test**

```python
def test_translate_responses_request_to_anthropic_messages():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_translator.py::test_translate_responses_request_to_anthropic_messages -v`
Expected: FAIL because translator does not exist yet

**Step 3: Write minimal implementation**

Implement conversion for:
- `instructions`
- `input` message items
- tool result items
- `tools`
- `max_output_tokens`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_translator.py::test_translate_responses_request_to_anthropic_messages -v`
Expected: PASS

### Task 3: Implement stream translation

**Files:**
- Modify: `app/translator.py`
- Test: `tests/test_stream_translation.py`

**Step 1: Write the failing test**

```python
def test_translate_anthropic_sse_to_responses_events():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_translation.py::test_translate_anthropic_sse_to_responses_events -v`
Expected: FAIL because the SSE converter is incomplete

**Step 3: Write minimal implementation**

Implement Anthropic event parsing and Responses SSE emission for the text path used by Codex.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_translation.py::test_translate_anthropic_sse_to_responses_events -v`
Expected: PASS

### Task 4: Implement FastAPI service

**Files:**
- Create: `app/config.py`
- Create: `app/client.py`
- Create: `app/main.py`
- Test: `tests/test_api.py`

**Step 1: Write the failing test**

```python
def test_post_responses_requires_auth():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_post_responses_requires_auth -v`
Expected: FAIL because the API does not exist yet

**Step 3: Write minimal implementation**

Implement:
- config loading from env
- auth validation
- `/health`
- `/v1/responses`
- upstream forwarding

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS

### Task 5: Wire Codex profile and shell environment

**Files:**
- Modify: `/root/.codex/config.toml`
- Modify: `/root/.bashrc`

**Step 1: Write the failing test**

Manual verification target:
- new provider missing from Codex config
- proxy auth env missing from shell config

**Step 2: Run test to verify it fails**

Run: `rg -n "minimax_py|m128py|MINIMAX_PROXY_API_KEY" /root/.codex/config.toml /root/.bashrc`
Expected: no matching lines

**Step 3: Write minimal implementation**

Add the new provider, profile, and exported proxy key.

**Step 4: Run test to verify it passes**

Run: `rg -n "minimax_py|m128py|MINIMAX_PROXY_API_KEY" /root/.codex/config.toml /root/.bashrc`
Expected: matches in both files

### Task 6: End-to-end verification

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

Manual failure case:
- service not running
- Codex smoke test cannot reach local proxy

**Step 2: Run test to verify it fails**

Run: `curl -sS http://127.0.0.1:8765/health`
Expected: connection failure before service start

**Step 3: Write minimal implementation**

Start the service, run pytest, run direct curl tests, then run:

```bash
codex --profile m128py "hello"
```

**Step 4: Run test to verify it passes**

Run the same commands again.
Expected:
- health endpoint returns success
- pytest passes
- direct `/v1/responses` request succeeds
- Codex smoke test succeeds
