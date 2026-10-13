"""Result objects."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional


@dataclass
class TokenLabel:
    token: str
    label: Optional[str]
    start: Optional[int] = None
    end: Optional[int] = None
    source: str = "model"  # "model", "rule" or "failed"


@dataclass
class TagResult:
    tokens: List[TokenLabel]
    text: Optional[str]
    provider: str
    model: str
    non_language: frozenset = field(default_factory=frozenset, repr=False)
    raw_responses: List[str] = field(default_factory=list, repr=False)

    @property
    def words(self) -> list:
        return [t.token for t in self.tokens]

    @property
    def labels(self) -> list:
        return [t.label for t in self.tokens]

    def pairs(self) -> list:
        return [(t.token, t.label) for t in self.tokens]

    def _language_positions(self, ignore: Optional[Iterable[str]]):
        skip = set(self.non_language if ignore is None else ignore)
        return [(i, t.label) for i, t in enumerate(self.tokens) if t.label is not None and t.label not in skip]

    def switch_points(self, ignore: Optional[Iterable[str]] = None) -> list:
        """Indices of tokens whose language differs from the previous language-bearing token."""
        seq = self._language_positions(ignore)
        return [i for (i, lab), (_, prev) in zip(seq[1:], seq) if lab != prev]

    def cmi(self, ignore: Optional[Iterable[str]] = None) -> float:
        """Code-Mixing Index (Das and Gamback, 2014): 100 * (1 - max_lang / (n - u))."""
        langs = [lab for _, lab in self._language_positions(ignore)]
        if not langs:
            return 0.0
        return 100.0 * (1 - Counter(langs).most_common(1)[0][1] / len(langs))

    def spans(self) -> list:
        """Runs of consecutive tokens with the same label."""
        out = []
        for i, t in enumerate(self.tokens):
            if out and out[-1]["label"] == t.label:
                out[-1]["end_token"] = i
                out[-1]["tokens"].append(t.token)
            else:
                out.append({"label": t.label, "start_token": i, "end_token": i, "tokens": [t.token]})
        return out

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "tokens": [asdict(t) for t in self.tokens],
        }

    def to_conll(self) -> str:
        return "\n".join(f"{t.token}\t{t.label}" for t in self.tokens)
