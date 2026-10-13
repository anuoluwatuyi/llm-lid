from __future__ import annotations

from typing import Optional

from .base import call_with_temperature, missing_sdk


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: Optional[str] = None, temperature: Optional[float] = None,
                 max_tokens: int = 4096, request_options: Optional[dict] = None, **client_kwargs):
        try:
            import anthropic
        except ImportError as exc:
            raise missing_sdk("anthropic", "anthropic") from exc
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_options = dict(request_options or {})
        self._client = anthropic.Anthropic(api_key=api_key, **client_kwargs)  # None falls back to ANTHROPIC_API_KEY

    def complete(self, system: str, user: str) -> str:
        def call(**extra):
            return self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                **self.request_options,
                **extra,
            )

        resp = call_with_temperature(call, self.temperature)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
