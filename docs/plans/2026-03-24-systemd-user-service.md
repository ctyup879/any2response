# Systemd User Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the MiniMax proxy run as a persistent `systemd --user` service so the `m128py` Codex profile remains available after shell sessions end.

**Architecture:** Keep `start_proxy.sh` as the one runtime entrypoint. Add a small Python module to render the service unit and a small installer script to place it in the user's systemd directory. Validate with both unit tests and a live `systemctl --user` run.

**Tech Stack:** Python 3, pytest, systemd user services, existing bash launcher

---

### Task 1: Add failing tests for service rendering

**Files:**
- Create: `tests/test_service_unit.py`
- Create: `app/service_unit.py`

**Step 1: Write the failing test**

Add tests for:
- canonical unit name
- rendered unit content includes `ExecStart`, `WorkingDirectory`, `Restart=always`
- install path resolves under `~/.config/systemd/user`

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_service_unit.py`

**Step 3: Write minimal implementation**

Implement rendering helpers in `app/service_unit.py`.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_service_unit.py`

### Task 2: Add installer script

**Files:**
- Create: `scripts/install_user_service.py`
- Modify: `app/service_unit.py`
- Modify: `tests/test_service_unit.py`

**Step 1: Write the failing test**

Add a test that writes the rendered unit into a fake home directory.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_service_unit.py -k install`

**Step 3: Write minimal implementation**

Implement installer entrypoint that writes the unit file.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_service_unit.py`

### Task 3: Document service lifecycle

**Files:**
- Modify: `README.md`

**Step 1: Add docs**

Document install/start/status/logs/stop commands for `systemd --user`.

**Step 2: Verify docs reference real files and commands**

Run: `sed -n '1,260p' README.md`

### Task 4: Real service verification

**Files:**
- No code changes expected

**Step 1: Reload and start the user service**

Run:
- `python3 scripts/install_user_service.py`
- `systemctl --user daemon-reload`
- `systemctl --user enable --now minimaxdemo-proxy.service`

**Step 2: Verify service health**

Run:
- `systemctl --user status minimaxdemo-proxy.service --no-pager`
- `ss -ltnp | rg ':8765'`

**Step 3: Verify end-to-end Codex**

Run:
- `source ~/.bashrc && codex exec --profile m128py "Say only: service-check"`

### Task 5: Final regression verification

**Files:**
- No additional files

**Step 1: Run full tests**

Run: `.venv/bin/pytest -q`

**Step 2: Commit**

Run:
- `git add app/service_unit.py scripts/install_user_service.py tests/test_service_unit.py README.md docs/plans/2026-03-24-systemd-user-service-design.md docs/plans/2026-03-24-systemd-user-service.md`
- `git commit -m "feat: add systemd user service support"`
