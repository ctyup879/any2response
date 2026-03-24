import json
from copy import deepcopy
from pathlib import Path


DEFAULT_BASE_MODEL = "gpt-5.4"
DEFAULT_CUSTOM_MODEL = "codex-MiniMax-M2.7"


def ensure_custom_model_metadata(cache, custom_slug=DEFAULT_CUSTOM_MODEL, base_slug=DEFAULT_BASE_MODEL):
    models = cache.setdefault("models", [])
    if any(model.get("slug") == custom_slug for model in models if isinstance(model, dict)):
        return cache

    base_model = next(
        (model for model in models if isinstance(model, dict) and model.get("slug") == base_slug),
        None,
    )
    if base_model is None:
        raise ValueError(f"Base model metadata not found: {base_slug}")

    cloned = deepcopy(base_model)
    cloned["slug"] = custom_slug
    cloned["display_name"] = custom_slug
    cloned["description"] = "MiniMax Codex-compatible coding model."
    cloned["priority"] = cloned.get("priority", 0) + 100
    models.append(cloned)
    return cache


def install_custom_model_metadata(
    cache_path: Path,
    custom_slug=DEFAULT_CUSTOM_MODEL,
    base_slug=DEFAULT_BASE_MODEL,
):
    path = Path(cache_path)
    cache = json.loads(path.read_text(encoding="utf-8"))
    updated = ensure_custom_model_metadata(cache, custom_slug=custom_slug, base_slug=base_slug)
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
