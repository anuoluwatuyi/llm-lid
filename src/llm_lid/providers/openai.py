from __future__ import annotations

from typing import Optional

from .base import call_with_temperature, missing_sdk


class OpenAIProvider:
    """OpenAI, and any OpenAI-compatible endpoint via base_url (OpenRouter, Groq, Together, Ollama, vLLM...)."""

    name = "openai"

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 temperature: Optional[float] = 0.0, json_mode: bool = True,
                 request_options: Optional[dict] = None, **client_kwargs):
        try:
            import openai
        except ImportError as exc:
            raise missing_sdk("openai", "openai") from exc
        self.model = model
        self.temperature = temperature
        self.json_mode = json_mode
        self.request_options = dict(request_options or {})
        if base_url:
            self.name = "openai_compatible"
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, **client_kwargs)  # None falls back to OPENAI_API_KEY

    def complete(self, system: str, user: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self.request_options)

        resp = call_with_temperature(lambda **extra: self._client.chat.completions.create(**kwargs, **extra),
                                     self.temperature)
        return resp.choices[0].message.content or ""
