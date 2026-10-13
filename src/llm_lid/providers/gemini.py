from __future__ import annotations

from typing import Optional

from .base import call_with_temperature, missing_sdk


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, api_key: Optional[str] = None, temperature: Optional[float] = 0.0,
                 json_mode: bool = True, request_options: Optional[dict] = None, **client_kwargs):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise missing_sdk("google-genai", "gemini") from exc
        self.model = model
        self.temperature = temperature
        self.json_mode = json_mode
        self.request_options = dict(request_options or {})
        self._types = types
        self._client = genai.Client(api_key=api_key, **client_kwargs)  # None falls back to GEMINI_API_KEY

    def complete(self, system: str, user: str) -> str:
        def call(**extra):
            config = self._types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json" if self.json_mode else None,
                **{**self.request_options, **extra},
            )
            return self._client.models.generate_content(model=self.model, contents=user, config=config)

        resp = call_with_temperature(call, self.temperature)
        return resp.text or ""
