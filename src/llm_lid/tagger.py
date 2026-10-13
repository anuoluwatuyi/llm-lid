"""The main entry point: LanguageTagger."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Optional, Sequence, Union

from .errors import LabelingError, ResponseError
from .labels import LabelSet
from .prompt import Example, build_system_prompt, build_user_message, parse_labels
from .providers import make_provider
from .result import TagResult, TokenLabel
from .tokenize import Token, is_universal, tokenize

_SENTENCE_END = {".", "!", "?", "\u2026"}


class LanguageTagger:
    """Token-level language identification with an LLM.

    provider: "anthropic", "openai", "openai_compatible" or "gemini"; or your own
        Provider object or function fn(system, user) -> str.
    model: the provider's model name.
    api_key: your key for that provider. If omitted, the provider SDK reads its usual
        environment variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY).
    labels: None for open ISO 639-3 labels, a list or dict of your own labels, or a LabelSet.
    instructions: extra annotation guidelines added to the prompt.
    examples: few-shot examples as (tokens, labels) pairs.
    chunk_size: maximum tokens per request; longer texts are split near sentence ends.
    max_retries: extra attempts when a reply is misaligned or uses unknown labels.
    auto_universal: label punctuation, numbers, emoji, URLs and @mentions by rule.
    strict: raise LabelingError if retries run out; if False, those tokens get label None.
    Any other keyword (base_url, temperature, max_tokens, json_mode, timeout...) goes to the provider.
    """

    def __init__(
        self,
        provider,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        labels: Union[None, Sequence[str], Mapping[str, str], LabelSet] = None,
        instructions: Optional[str] = None,
        examples: Optional[Sequence[Example]] = None,
        chunk_size: int = 200,
        max_retries: int = 2,
        auto_universal: bool = True,
        strict: bool = True,
        **provider_kwargs,
    ):
        self.provider = make_provider(provider, model=model, api_key=api_key, **provider_kwargs)
        if labels is None:
            self.label_set = LabelSet.open()
        elif isinstance(labels, LabelSet):
            self.label_set = labels
        else:
            self.label_set = LabelSet.closed(labels)
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.auto_universal = auto_universal and self.label_set.universal is not None
        self.strict = strict
        self.system_prompt = build_system_prompt(self.label_set, instructions, examples)

    # Public API ---------------------------------------------------------

    def tag(self, text: Union[str, Sequence[str]]) -> TagResult:
        """Label a string (tokenized for you) or a list of pre-tokenized words."""
        if isinstance(text, str):
            toks = tokenize(text)
            original = text
        else:
            toks = [Token(str(w), None, None) for w in text]
            original = None

        out, raws = [], []
        for lo, hi in self._chunks([t.text for t in toks]):
            chunk = toks[lo:hi]
            labels, sources, chunk_raws = self._label_chunk(chunk, original)
            raws.extend(chunk_raws)
            out.extend(TokenLabel(t.text, lab, t.start, t.end, src) for t, lab, src in zip(chunk, labels, sources))

        return TagResult(
            tokens=out,
            text=original,
            provider=self.provider.name,
            model=self.provider.model,
            non_language=self.label_set.non_language,
            raw_responses=raws,
        )

    def tag_word(self, word: str) -> TokenLabel:
        """Label a single word with no context. Words shared across languages are ambiguous here."""
        return self.tag([word]).tokens[0]

    def tag_batch(self, texts: Sequence[Union[str, Sequence[str]]], max_workers: int = 4) -> list:
        """Label many texts in parallel threads. Results keep input order."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(self.tag, texts))

    # Internals ----------------------------------------------------------

    def _chunks(self, words: Sequence[str]):
        n, size = len(words), self.chunk_size
        lo = 0
        while lo < n:
            hi = min(lo + size, n)
            if hi < n:
                floor = lo + max(1, (size * 3) // 4)
                for j in range(hi - 1, floor - 1, -1):
                    if words[j] in _SENTENCE_END:
                        hi = j + 1
                        break
            yield lo, hi
            lo = hi

    def _label_chunk(self, chunk: Sequence[Token], original: Optional[str]):
        words = [t.text for t in chunk]
        rule = [self.auto_universal and is_universal(w) for w in words]
        universal = self.label_set.universal

        if all(rule):  # nothing for the model to decide
            return [universal] * len(words), ["rule"] * len(words), []

        context = None
        if original is not None and chunk[0].start is not None:
            context = original[chunk[0].start : chunk[-1].end]
        user = build_user_message(words, context)

        raws, note, last_err = [], "", None
        for _ in range(self.max_retries + 1):
            raw = self.provider.complete(self.system_prompt, user + note)
            raws.append(raw)
            try:
                labels = parse_labels(raw, len(words), self.label_set)
                break
            except ResponseError as err:
                last_err = err
                note = (f"\n\nYour previous reply was invalid: {err} "
                        f"Reply again with one allowed label for each of the {len(words)} tokens.")
        else:
            if self.strict:
                raise LabelingError(f"No valid labels after {self.max_retries + 1} attempts: {last_err}", raws)
            return ([universal if r else None for r in rule],
                    ["rule" if r else "failed" for r in rule], raws)

        final = [universal if r else lab for r, lab in zip(rule, labels)]
        sources = ["rule" if r else "model" for r in rule]
        return final, sources, raws
