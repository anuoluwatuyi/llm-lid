# llm-lid

Token-level language identification with any LLM. Bring your own provider, model and API key, pass a text or a word, get one language label per token.

## Install

```bash
pip install "llm-lid[anthropic]"   # or [openai], [gemini], [all]
```

## Quick start

```python
from llm_lid import LanguageTagger

tagger = LanguageTagger(provider="anthropic", model="claude-sonnet-5-5", api_key="sk-ant-...")

result = tagger.tag("I no fit come today because rain dey fall for Lagos.")
result.pairs()
# [('I', 'eng'), ('no', 'pcm'), ('fit', 'pcm'), ('come', 'eng'), ..., ('Lagos', 'ne'), ('.', 'other')]

tagger.tag_word("abeg").label          # 'pcm'
tagger.tag(["abeg", "come", "o"])      # pre-tokenized input, labels align with your tokens
tagger.tag_batch(list_of_texts, max_workers=8)
```

If `api_key` is omitted, the provider SDK reads its usual environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`).

## Providers

```python
LanguageTagger("openai", model="<model>", api_key="...")
LanguageTagger("gemini", model="<model>", api_key="...")

# Any OpenAI-compatible endpoint: OpenRouter, Groq, Together, DeepSeek, vLLM, Ollama
LanguageTagger("openai_compatible", model="<model>", api_key="...", base_url="https://openrouter.ai/api/v1")
LanguageTagger("openai_compatible", model="llama3.1", api_key="ollama", base_url="http://localhost:11434/v1")

# Your own function
LanguageTagger(lambda system, user: my_client.generate(system, user), model="my-model")
```

Provider options pass straight through: `temperature` (default 0 for OpenAI and Gemini; not sent to Anthropic unless you set it), `max_tokens` (Anthropic), `json_mode` (OpenAI and Gemini, default on), `request_options` (extra fields for the underlying request, e.g. `{"output_config": {"effort": "low"}}` for Anthropic), and client settings such as `timeout`. If a model rejects `temperature`, the call is retried without it.

## Labels

By default labels are ISO 639-3 codes (`eng`, `pcm`, `yor`, `hau`, `ibo`, ...) plus `mixed`, `ne`, `other` and `unk`.

For a fixed scheme, pass your own labels. Descriptions go into the prompt.

```python
tagger = LanguageTagger(
    "anthropic", model="claude-sonnet-5-5",
    labels={
        "en": "English",
        "pcm": "Nigerian Pidgin",
        "yo": "Yoruba",
        "mixed": "Intra-word mixing",
        "ne": "Named entity",
        "other": "Punctuation, numbers, emoji, URLs, @mentions",
    },
    instructions="Words shared by English and Pidgin take the language of the surrounding clause.",
    examples=[(["Wetin", "you", "dey", "do", "?"], ["pcm", "pcm", "pcm", "pcm", "other"])],
)
```

`other` is assigned by rule to punctuation, numbers, emoji, URLs and @mentions. Change it with `LabelSet.closed(labels, universal="univ")`, or turn it off with `auto_universal=False`.

## Output

```python
result.labels          # ['eng', 'pcm', ...]
result.tokens          # TokenLabel(token, label, start, end, source) with character offsets
result.switch_points() # indices where the language changes
result.cmi()           # Code-Mixing Index (Das and Gamback, 2014)
result.spans()         # runs of the same label
result.to_dict()       # JSON-ready
result.to_conll()      # token<TAB>label lines
result.raw_responses   # raw model replies, for auditing
```

## How it works

1. The text is tokenized locally, keeping character offsets and tonal diacritics.
2. Tokens are sent to the model with their indices. The model returns a label per index as JSON.
3. The reply is checked: every index present once, every label in the allowed set. If not, the error is sent back and the model retries (`max_retries`, default 2). After that it raises `LabelingError`, or with `strict=False` marks those tokens `None`.
4. Texts longer than `chunk_size` tokens (default 200) are split near sentence ends.

A word on its own has no context, so words shared between languages (e.g. English and Pidgin) will be ambiguous with `tag_word`. Use `tag` on the sentence where possible.

## Development

```bash
git clone https://github.com/anuoluwatuyi/llm-lid
cd llm-lid
pip install -e ".[dev]" && pytest
```

## Citation

If you use llm-lid in research, please cite it using the metadata in [CITATION.cff](CITATION.cff), or the "Cite this repository" button on GitHub.

## License

MIT
