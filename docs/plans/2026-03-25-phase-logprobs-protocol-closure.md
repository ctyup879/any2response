# Responses Phase And Logprobs Protocol Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining high-value Responses protocol gaps around assistant `phase`, output-text `logprobs`, proxy-replayable `encrypted_content`, `input_image.detail`, and custom tool `lark` grammar handling.

**Architecture:** Keep the proxy as a strict OpenAI Responses facade over MiniMax's Anthropic-compatible endpoint. Where MiniMax lacks a structural equivalent, prefer a stable proxy-side representation that can round-trip through this proxy on later turns, and only fall back to best-effort instruction injection when no lossless transport exists.

**Tech Stack:** FastAPI, httpx, pytest, MiniMax Anthropic-compatible Messages API

---

### Task 1: Assistant phase round-trip

**Files:**
- Modify: `app/translator.py`
- Test: `tests/test_translator.py`
- Test: `tests/test_stream_translation.py`

**Step 1: Write the failing test**

```python
def test_translate_responses_request_round_trips_commentary_message_from_proxy_response():
    response = translate_anthropic_response(...)
    commentary_message = next(item for item in response["output"] if item.get("phase") == "commentary")
    translated = translate_responses_request({"model": "...", "input": [commentary_message]})
    assert translated["messages"] == [{"role": "assistant", "content": [{"type": "thinking", "thinking": "..."}]}]
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_translator.py -k phase`
Expected: FAIL because the proxy did not emit commentary-phase assistant messages from Anthropic thinking blocks.

**Step 3: Write minimal implementation**

```python
# Represent Anthropic thinking both as a Responses reasoning item and as an
# assistant message item with phase="commentary" so follow-up requests can
# preserve phase semantics through the proxy.
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_translator.py -k phase`
Expected: PASS

**Step 5: Commit**

```bash
git add app/translator.py tests/test_translator.py tests/test_stream_translation.py
git commit -m "feat: preserve assistant commentary phase round-trip"
```

### Task 2: Logprobs request/response shape

**Files:**
- Modify: `app/translator.py`
- Test: `tests/test_translator.py`

**Step 1: Write the failing test**

```python
def test_translate_responses_request_accepts_output_logprobs_include():
    translated = translate_responses_request({"model": "...", "input": "hello", "include": ["message.output_text.logprobs"]})
    assert translated["messages"]
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_translator.py -k logprobs`
Expected: FAIL because the proxy rejects official logprobs request knobs.

**Step 3: Write minimal implementation**

```python
# Accept include=["message.output_text.logprobs"] and integer top_logprobs.
# Emit empty logprobs arrays because MiniMax does not expose token scores.
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_translator.py -k logprobs`
Expected: PASS

**Step 5: Commit**

```bash
git add app/translator.py tests/test_translator.py
git commit -m "feat: accept responses logprobs request fields"
```

### Task 3: Proxy replayable reasoning payloads

**Files:**
- Modify: `app/translator.py`
- Test: `tests/test_translator.py`
- Test: `tests/test_stream_translation.py`

**Step 1: Write the failing test**

```python
def test_translate_responses_request_accepts_reasoning_input_items_with_proxy_encrypted_content():
    translated = translate_responses_request({"model": "...", "input": [{"type": "reasoning", "encrypted_content": "proxy_reasoning_v1:..."}]})
    assert translated["messages"][0]["content"][0]["type"] == "thinking"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_translator.py -k reasoning_input`
Expected: FAIL because reasoning input items are rejected.

**Step 3: Write minimal implementation**

```python
# Encode missing encrypted_content as a proxy-local opaque blob and decode it on
# later reasoning input replay.
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_translator.py -k reasoning_input`
Expected: PASS

**Step 5: Commit**

```bash
git add app/translator.py tests/test_translator.py tests/test_stream_translation.py
git commit -m "feat: add proxy reasoning replay payloads"
```

### Task 4: Best-effort image detail and lark grammar

**Files:**
- Modify: `app/translator.py`
- Modify: `README.md`
- Test: `tests/test_translator.py`

**Step 1: Write the failing test**

```python
def test_translate_responses_request_supports_image_detail_best_effort():
    translated = translate_responses_request(...)
    assert translated["messages"][0]["content"][0]["text"].startswith("Analyze the following image")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_translator.py -k 'image_detail or lark'`
Expected: FAIL because these fields are rejected.

**Step 3: Write minimal implementation**

```python
# Map image detail onto injected guidance text and carry lark grammar as a
# string-schema description when no structural Anthropic equivalent exists.
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_translator.py -k 'image_detail or lark'`
Expected: PASS

**Step 5: Commit**

```bash
git add app/translator.py README.md tests/test_translator.py
git commit -m "feat: add best-effort detail and grammar support"
```
