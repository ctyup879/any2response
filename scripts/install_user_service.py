#!/usr/bin/env python3
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.service_unit import SERVICE_NAME, install_unit_file


def main():
    unit_path = install_unit_file(PROJECT_ROOT)
    print(f"Installed {SERVICE_NAME} to {unit_path}")
    print("Next commands:")
    print("  systemctl --user daemon-reload")
    print(f"  systemctl --user enable --now {SERVICE_NAME}")


if __name__ == "__main__":
    main()
