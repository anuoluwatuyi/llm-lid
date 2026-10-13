"""Prompt construction and reply parsing."""
from __future__ import annotations

import json
import re
from typing import Optional, Sequence, Tuple

from .errors import ResponseError
from .labels import LabelSet

_SYSTEM = """You are an expert annotator for token-level language identification in code-switched text.

You receive a text and its tokens, each with an index. Assign exactly one label to every token.
- Judge each token in context. A word that exists in more than one language takes the language of the phrase around it.
- Do not skip, merge or split tokens.
- Use only the labels listed below.

Labels:
{labels}
{instructions}{examples}
Reply with only a JSON object, no other text, in this form:
{{"labels": {{"0": "<label>", "1": "<label>", ...}}}}
It must have one entry for every index from 0 to N-1."""

Example = Tuple[Sequence[str], Sequence[str]]


def build_system_prompt(
    label_set: LabelSet,
    instructions: Optional[str] = None,
    examples: Optional[Sequence[Example]] = None,
) -> str:
    instr = f"\nAnnotation guidelines:\n{instructions.strip()}\n" if instructions else ""
    ex = ""
    if examples:
        blocks = []
        for tokens, labels in examples:
            if len(tokens) != len(labels):
                raise ValueError("Each example needs one label per token.")
            blocks.append(
                "Tokens: " + json.dumps(list(tokens), ensure_ascii=False)
                + "\nLabels: " + json.dumps(list(labels), ensure_ascii=False)
            )
        ex = "\nExamples:\n" + "\n\n".join(blocks) + "\n"
    return _SYSTEM.format(labels=label_set.describe(), instructions=instr, examples=ex)


def build_user_message(tokens: Sequence[str], text: Optional[str] = None) -> str:
    indexed = json.dumps([[i, t] for i, t in enumerate(tokens)], ensure_ascii=False)
    context = text if text is not None else " ".join(tokens)
    return f"Text: {context}\n\nTokens (N = {len(tokens)}):\n{indexed}"


def _extract_json(raw: str):
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ResponseError("Reply was not valid JSON.")


def parse_labels(raw: str, n: int, label_set: LabelSet) -> list:
    """Parse a reply into exactly n valid labels, or raise ResponseError."""
    obj = _extract_json(raw)
    labels = obj.get("labels") if isinstance(obj, dict) else obj
    if isinstance(labels, list):
        mapping = {str(i): v for i, v in enumerate(labels)}
    elif isinstance(labels, dict):
        mapping = {str(k).strip(): v for k, v in labels.items()}
    else:
        raise ResponseError('Reply has no "labels" object.')

    expected = {str(i) for i in range(n)}
    missing = sorted(expected - mapping.keys(), key=int)
    extra = sorted(mapping.keys() - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing indices {missing[:20]}")
        if extra:
            parts.append(f"unexpected keys {extra[:20]}")
        raise ResponseError("Labels do not match the tokens: " + "; ".join(parts) + ".")

    out, invalid = [], {}
    for i in range(n):
        norm = label_set.normalize(mapping[str(i)])
        if norm is None:
            invalid[i] = mapping[str(i)]
        out.append(norm)
    if invalid:
        sample = dict(list(invalid.items())[:10])
        raise ResponseError(f"Labels not in the allowed set at these indices: {sample}.")
    return out
