"""Optional OpenAI-compatible high-level JSON client using existing env config."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def _load_local_env() -> None:
    """Load the repo .env without overriding variables already set in the shell."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)


class ManagerClient:
    """Small boundary client; callers always validate/fallback locally."""

    def __init__(self) -> None:
        _load_local_env()
        self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = os.getenv("VLM_MODEL", "qwen-vl-max")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def json_completion(self, system: str, user: str, timeout_s: float = 20.0) -> dict:
        if not self.available:
            raise RuntimeError("manager API key is not configured")
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }).encode("utf-8")
        request = Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        return json.loads(body["choices"][0]["message"]["content"])
