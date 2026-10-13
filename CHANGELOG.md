# Changelog

## 0.1.0 (2026-10-13)

First release.

- `LanguageTagger` with `tag`, `tag_word` and `tag_batch`
- Providers: Anthropic, OpenAI, Gemini, any OpenAI-compatible endpoint, or your own function
- Open ISO 639-3 labels by default, or a closed label set with descriptions, guidelines and few-shot examples
- Local tokenizer that keeps character offsets and tonal diacritics
- Validation of model replies with automatic retries
- Rule-based labels for punctuation, numbers, emoji, URLs and @mentions
- Chunking of long texts near sentence ends
- Switch points, Code-Mixing Index, spans, JSON and CoNLL output
