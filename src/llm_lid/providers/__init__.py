from __future__ import annotations

import importlib
from typing import Optional

from .base import CallableProvider, Provider

_REGISTRY = {
    "anthropic": ("anthropic", "AnthropicProvider"),
    "claude": ("anthropic", "AnthropicProvider"),
    "openai": ("openai", "OpenAIProvider"),
    "openai_compatible": ("openai", "OpenAIProvider"),
    "gemini": ("gemini", "GeminiProvider"),
    "google": ("gemini", "GeminiProvider"),
}

PROVIDERS = sorted(_REGISTRY)


def make_provider(provider, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> Provider:
    """Build a provider from a name, or pass through a Provider object or a function fn(system, user) -> str."""
    if isinstance(provider, str):
        key = provider.strip().lower().replace("-", "_")
        if key not in _REGISTRY:
            raise ValueError(f"Unknown provider {provider!r}. Choose from {PROVIDERS}, or pass your own.")
        if not model:
            raise ValueError("model is required, e.g. model='claude-sonnet-5-5'.")
        if key == "openai_compatible" and not kwargs.get("base_url"):
            raise ValueError("openai_compatible needs base_url, e.g. 'https://openrouter.ai/api/v1'.")
        module_name, cls_name = _REGISTRY[key]
        cls = getattr(importlib.import_module(f".{module_name}", __name__), cls_name)
        return cls(model=model, api_key=api_key, **kwargs)
    if isinstance(provider, Provider):
        return provider
    if callable(provider):
        return CallableProvider(provider, model or "custom")
    raise TypeError("provider must be a provider name, a Provider object, or a function fn(system, user) -> str.")


__all__ = ["Provider", "CallableProvider", "make_provider", "PROVIDERS"]
