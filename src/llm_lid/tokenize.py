"""Offset-preserving tokenizer for social media and code-switched text."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Combining marks, so tonal diacritics (e.g. Yoruba ọ̀) stay inside the word.
_MARKS = "\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f"
_WORD = rf"[\w{_MARKS}]+(?:['\u2019\-][\w{_MARKS}]+)*"
_URL_END = r"(?=[.,!?;:)\]]*(?:\s|$))"

_TOKEN_RE = re.compile(
    rf"""
      (?P<url>(?:https?://|www\.)\S+?{_URL_END})
    | (?P<email>[\w.+-]+@[\w-]+\.[\w.-]*\w)
    | (?P<mention>@\w+)
    | (?P<hashtag>\#{_WORD})
    | (?P<number>\d+(?:[.,:/]\d+)+)(?![\w{_MARKS}])
    | (?P<word>{_WORD})
    | (?P<other>[^\w\s](?:[\ufe0f\U0001F3FB-\U0001F3FF]|\u200d[^\w\s])*)
    """,
    re.VERBOSE,
)

_NON_LINGUISTIC = re.compile(r"^(?:https?://|www\.)|^[\w.+-]+@[\w-]+\.|^@\w+$")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    """Split text into tokens with character offsets into the original string."""
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def is_universal(token: str) -> bool:
    """True for tokens with no language: URLs, e-mails, @mentions, and tokens with no letters."""
    if _NON_LINGUISTIC.search(token):
        return True
    return not any(ch.isalpha() for ch in token)
