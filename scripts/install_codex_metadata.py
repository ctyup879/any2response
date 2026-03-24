#!/usr/bin/env python3
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.codex_metadata import DEFAULT_CUSTOM_MODEL, install_custom_model_metadata


def main():
    cache_path = Path.home() / ".codex/models_cache.json"
    install_custom_model_metadata(cache_path, custom_slug=DEFAULT_CUSTOM_MODEL)
    print(f"Installed metadata for {DEFAULT_CUSTOM_MODEL} into {cache_path}")


if __name__ == "__main__":
    main()
