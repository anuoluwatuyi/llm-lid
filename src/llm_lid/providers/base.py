from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """Anything with a name, a model and a complete(system, user) -> str method."""

    name: str
    model: str

    def complete(self, system: str, user: str) -> str: ...


class CallableProvider:
    """Wraps a plain function fn(system, user) -> str."""

    name = "custom"

    def __init__(self, fn: Callable[[str, str], str], model: str = "custom"):
        self._fn = fn
        self.model = model

    def complete(self, system: str, user: str) -> str:
        return self._fn(system, user)


def call_with_temperature(call: Callable, temperature: Optional[float]):
    """Send temperature if set; retry without it if the model rejects the parameter."""
    if temperature is None:
        return call()
    try:
        return call(temperature=temperature)
    except Exception as exc:  # SDK error types differ across providers
        if "temperature" in str(exc).lower():
            return call()
        raise


def missing_sdk(package: str, extra: str) -> ImportError:
    return ImportError(f"The {package} package is required. Install it with: pip install 'llm-lid[{extra}]'")
