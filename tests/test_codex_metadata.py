from app.codex_metadata import ensure_custom_model_metadata


def test_ensure_custom_model_metadata_clones_base_model():
    cache = {
        "models": [
            {
                "slug": "gpt-5.4",
                "display_name": "gpt-5.4",
                "description": "Latest frontier agentic coding model.",
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [{"effort": "low"}],
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 0,
                "availability_nux": None,
                "upgrade": None,
                "base_instructions": "base",
            }
        ]
    }

    updated = ensure_custom_model_metadata(cache, "codex-MiniMax-M2.7", base_slug="gpt-5.4")

    model = next(item for item in updated["models"] if item["slug"] == "codex-MiniMax-M2.7")
    assert model["display_name"] == "codex-MiniMax-M2.7"
    assert model["description"] == "MiniMax Codex-compatible coding model."
    assert model["shell_type"] == "shell_command"


def test_ensure_custom_model_metadata_is_idempotent():
    cache = {
        "models": [
            {"slug": "gpt-5.4", "display_name": "gpt-5.4"},
            {"slug": "codex-MiniMax-M2.7", "display_name": "codex-MiniMax-M2.7"},
        ]
    }

    updated = ensure_custom_model_metadata(cache, "codex-MiniMax-M2.7", base_slug="gpt-5.4")

    assert [item["slug"] for item in updated["models"]].count("codex-MiniMax-M2.7") == 1
