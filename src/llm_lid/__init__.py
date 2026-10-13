"""llm-lid: token-level language identification with any LLM provider."""
from .errors import LabelingError, LLMLIDError, ResponseError
from .labels import SPECIAL_LABELS, LabelSet
from .providers import PROVIDERS, CallableProvider, Provider, make_provider
from .result import TagResult, TokenLabel
from .tagger import LanguageTagger
from .tokenize import Token, is_universal, tokenize

__version__ = "0.1.0"

__all__ = [
    "LanguageTagger", "LabelSet", "SPECIAL_LABELS", "TagResult", "TokenLabel",
    "Token", "tokenize", "is_universal", "Provider", "CallableProvider", "make_provider",
    "PROVIDERS", "LLMLIDError", "ResponseError", "LabelingError",
]
