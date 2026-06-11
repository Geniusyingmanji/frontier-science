"""LLM client for Frontier-Science.

Supports two wire formats behind one interface:

- ``chat``      : OpenAI-compatible ``POST {base_url}/chat/completions``.
                  This is the public, vendor-neutral path (base_url + api_key + model).
- ``responses`` : OpenAI ``POST {base_url}/responses`` (Responses API). Used by our
                  local keyless proxy, which injects auth itself — selected only via a
                  git-ignored local config.

The client is configured from a YAML/dict; see ``conf/llm/openai_compatible.example.yaml``.
No endpoint or credential is hard-coded here.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional


def _expand(value: Any) -> Any:
    """Expand ``${ENV_VAR}`` references in strings using os.environ."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


@dataclass
class LLMConfig:
    wire: str = "chat"  # "chat" | "responses"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    max_output_tokens: int = 8000
    temperature: Optional[float] = 0.7
    reasoning_effort: Optional[str] = None  # for reasoning models on the responses wire
    timeout_seconds: float = 600.0
    extra_headers: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        d = {k: _expand(v) for k, v in (d or {}).items()}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    # ---- public API -----------------------------------------------------
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        if self.config.wire == "responses":
            return self._complete_responses(prompt, system)
        return self._complete_chat(prompt, system)

    # ---- wire: chat completions ----------------------------------------
    def _complete_chat(self, prompt: str, system: Optional[str]) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": int(self.config.max_output_tokens),
        }
        if self.config.temperature is not None:
            payload["temperature"] = float(self.config.temperature)
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.extra_headers)
        data = self._post(url, payload, headers)
        return data["choices"][0]["message"]["content"] or ""

    # ---- wire: responses API -------------------------------------------
    def _complete_responses(self, prompt: str, system: Optional[str]) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": prompt,
            "max_output_tokens": int(self.config.max_output_tokens),
        }
        if system:
            payload["instructions"] = system
        if self.config.temperature is not None:
            payload["temperature"] = float(self.config.temperature)
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        url = self.config.base_url.rstrip("/") + "/responses"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.extra_headers)
        data = self._post(url, payload, headers)
        return self._extract_responses_text(data)

    @staticmethod
    def _extract_responses_text(data: dict) -> str:
        direct = data.get("output_text")
        if isinstance(direct, str) and direct:
            return direct
        chunks: list[str] = []
        for item in data.get("output", []) or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        if not chunks and data.get("error"):
            raise RuntimeError(f"LLM error: {data['error']}")
        return "".join(chunks)

    # ---- transport ------------------------------------------------------
    def _post(self, url: str, payload: dict, headers: dict, retries: int = 3) -> dict:
        body = json.dumps(payload).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001 - surface after retries
                last_err = exc
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {retries} attempts: {last_err}")
