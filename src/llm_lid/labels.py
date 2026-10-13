"""Label sets: an open ISO 639-3 set by default, or a closed set you define."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Union

SPECIAL_LABELS = {
    "mixed": "One token that combines material from two languages, e.g. an English stem with a Yoruba affix.",
    "ne": "Named entity: a person, place, organisation, brand or product name.",
    "other": "Punctuation, numbers, emoji, symbols, URLs, e-mail addresses and @mentions.",
    "unk": "A word whose language cannot be determined.",
}

_ISO3 = re.compile(r"^[a-z]{3}$")
_AUTO = object()


@dataclass(frozen=True)
class LabelSet:
    labels: Optional[Mapping[str, str]]  # None means open ISO 639-3
    universal: Optional[str] = "other"
    non_language: frozenset = field(default_factory=lambda: frozenset({"other", "ne", "unk"}))

    @classmethod
    def open(cls) -> "LabelSet":
        """Any ISO 639-3 code (eng, pcm, yor, hau, ibo, ...) plus the special labels."""
        return cls(labels=None)

    @classmethod
    def closed(
        cls,
        labels: Union[Mapping[str, str], Iterable[str]],
        universal=_AUTO,
        non_language: Optional[Iterable[str]] = None,
    ) -> "LabelSet":
        """A fixed label set. Pass a dict to give each label a description for the prompt.

        universal: label given to punctuation, numbers, emoji and URLs by rule.
            Defaults to "other" if it is in the set, otherwise no rule is applied.
        non_language: labels ignored by switch_points() and cmi().
        """
        mapping = dict(labels) if isinstance(labels, Mapping) else {lab: "" for lab in labels}
        if not mapping:
            raise ValueError("A closed label set needs at least one label.")
        if universal is _AUTO:
            universal = "other" if "other" in mapping else None
        elif universal is not None and universal not in mapping:
            raise ValueError(f"universal label {universal!r} is not in the label set.")
        if non_language is None:
            non_language = {"other", "ne", "unk"} & mapping.keys()
            if universal:
                non_language.add(universal)
        return cls(labels=mapping, universal=universal, non_language=frozenset(non_language))

    def normalize(self, label) -> Optional[str]:
        """Return the canonical form of a label, or None if it is not allowed."""
        if not isinstance(label, str):
            return None
        label = label.strip()
        if self.labels is None:
            label = label.lower()
            return label if (_ISO3.match(label) or label in SPECIAL_LABELS) else None
        if label in self.labels:
            return label
        folded = {lab.lower(): lab for lab in self.labels}
        return folded.get(label.lower())

    def describe(self) -> str:
        """Label section of the system prompt."""
        if self.labels is None:
            lines = ["- An ISO 639-3 code for the token's language, e.g. eng (English), pcm (Nigerian Pidgin), yor (Yoruba), hau (Hausa), ibo (Igbo)."]
            lines += [f"- {k}: {v}" for k, v in SPECIAL_LABELS.items()]
        else:
            lines = [f"- {k}: {v}" if v else f"- {k}" for k, v in self.labels.items()]
        return "\n".join(lines)
