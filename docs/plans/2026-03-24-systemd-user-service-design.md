# Systemd User Service Design

**Goal:** Run the MiniMax Python responses proxy as a persistent `systemd --user` service so `codex --profile m128py` does not depend on a temporary shell session.

## Approach

Use `start_proxy.sh` as the single runtime entrypoint for both manual launches and `systemd`. Add a small Python-backed installer that renders a user service unit with absolute paths for the current repo, writes it into `~/.config/systemd/user/`, and prints the `systemctl --user` commands needed to enable and start it.

This keeps the service logic simple:

- the proxy process still reads `.env` in one place
- the systemd unit stays thin and repo-local
- the install step remains deterministic and testable

## Alternatives Considered

1. Plain `nohup`/pidfile shell scripts
   - simpler to sketch
   - weaker restart behavior and worse observability

2. System-wide `/etc/systemd/system` service
   - more persistent than a user unit
   - unnecessary privilege scope for a repo-local developer service

## Components

- `app/service_unit.py`
  - render the user service unit content
  - expose the canonical unit name and install path helpers
- `scripts/install_user_service.py`
  - write the rendered unit into `~/.config/systemd/user/`
  - avoid direct shell templating mistakes
- `systemd` lifecycle docs in `README.md`
  - install
  - enable/start
  - status/logs
  - stop/disable

## Behavior

- Unit uses `ExecStart=<repo>/start_proxy.sh`
- Unit restarts automatically on failure
- Unit starts in the repo working directory
- Unit remains compatible with the existing `.env` file
- Manual `./start_proxy.sh` remains unchanged

## Testing

- Unit rendering test verifies:
  - service name
  - `ExecStart`
  - `WorkingDirectory`
  - restart policy
- Installer test verifies target path and written content
- Real verification runs:
  - `systemctl --user daemon-reload`
  - `systemctl --user enable --now <unit>`
  - `systemctl --user status <unit>`
  - `codex exec --profile m128py ...`
