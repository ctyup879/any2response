import os
from dataclasses import dataclass


@dataclass
class Settings:
    minimax_api_key: str
    proxy_api_key: str
    host: str = "127.0.0.1"
    port: int = 8765
    upstream_base_url: str = "https://api.minimaxi.com/anthropic/v1/messages?beta=true"
    anthropic_version: str = "2023-06-01"
    anthropic_beta: str = "claude-code-20250219,interleaved-thinking-2025-05-14"
    request_timeout: float = 300.0


def load_settings(source: dict | None = None) -> Settings:
    env = dict(os.environ)
    if source:
        env.update(source)

    return Settings(
        minimax_api_key=env.get("MINIMAX_API_KEY") or env.get("minimax_api_key", ""),
        proxy_api_key=env.get("PROXY_API_KEY") or env.get("proxy_api_key", ""),
        host=env.get("HOST") or env.get("host", "127.0.0.1"),
        port=int(env.get("PORT") or env.get("port", 8765)),
        upstream_base_url=env.get("UPSTREAM_BASE_URL")
        or env.get("upstream_base_url", "https://api.minimaxi.com/anthropic/v1/messages?beta=true"),
        anthropic_version=env.get("ANTHROPIC_VERSION")
        or env.get("anthropic_version", "2023-06-01"),
        anthropic_beta=env.get("ANTHROPIC_BETA")
        or env.get("anthropic_beta", "claude-code-20250219,interleaved-thinking-2025-05-14"),
        request_timeout=float(env.get("REQUEST_TIMEOUT") or env.get("request_timeout", 300.0)),
    )
