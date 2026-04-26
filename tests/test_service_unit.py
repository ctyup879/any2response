from pathlib import Path
import os
import subprocess
import sys

from app.service_unit import (
    SERVICE_NAME,
    default_unit_path,
    install_unit_file,
    render_service_unit,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_render_service_unit_contains_expected_fields():
    project_root = PROJECT_ROOT

    rendered = render_service_unit(project_root)

    assert "[Unit]" in rendered
    assert "Description=MiniMax Python Responses Proxy" in rendered
    assert f"WorkingDirectory={project_root}" in rendered
    assert f"ExecStart={project_root / 'start_proxy.sh'}" in rendered
    assert "Restart=always" in rendered
    assert "RestartSec=2" in rendered


def test_default_unit_path_uses_systemd_user_directory():
    home = Path("/tmp/example-home")

    unit_path = default_unit_path(home)

    assert unit_path == home / ".config/systemd/user" / SERVICE_NAME


def test_install_unit_file_writes_rendered_content(tmp_path):
    project_root = PROJECT_ROOT
    home = tmp_path / "home"
    home.mkdir()

    unit_path = install_unit_file(project_root, home=home)

    assert unit_path == home / ".config/systemd/user" / SERVICE_NAME
    assert unit_path.exists()
    content = unit_path.read_text(encoding="utf-8")
    assert f"ExecStart={project_root / 'start_proxy.sh'}" in content


def test_install_script_writes_unit_file(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(home)

    result = subprocess.run(
        [sys.executable, "scripts/install_user_service.py"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    unit_path = home / ".config/systemd/user" / SERVICE_NAME
    assert unit_path.exists()
