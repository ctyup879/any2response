from pathlib import Path


SERVICE_NAME = "minimaxdemo-proxy.service"


def render_service_unit(project_root: Path) -> str:
    root = Path(project_root).resolve()
    return "\n".join(
        [
            "[Unit]",
            "Description=MiniMax Python Responses Proxy",
            "After=default.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={root}",
            f"ExecStart={root / 'start_proxy.sh'}",
            "Restart=always",
            "RestartSec=2",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def default_unit_path(home: Path | None = None) -> Path:
    home_dir = Path.home() if home is None else Path(home)
    return home_dir / ".config/systemd/user" / SERVICE_NAME


def install_unit_file(project_root: Path, home: Path | None = None) -> Path:
    unit_path = default_unit_path(home)
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(render_service_unit(project_root), encoding="utf-8")
    return unit_path
