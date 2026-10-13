import json
import re

import pytest

from llm_lid import LabelingError, LabelSet, LanguageTagger, is_universal, tokenize
from llm_lid.prompt import parse_labels

LEXICON = {"i": "eng", "no": "pcm", "fit": "pcm", "come": "eng", "today": "eng", "because": "eng",
           "rain": "eng", "dey": "pcm", "fall": "eng", "for": "pcm", "lagos": "ne", "abeg": "pcm"}


def tokens_from(user):
    return [t for _, t in json.loads(re.search(r"\[\[.*\]\]", user, re.S).group())]


def fake_model(system, user):
    return json.dumps({"labels": {str(i): LEXICON.get(t.lower(), "other") for i, t in enumerate(tokens_from(user))}})


def test_tokenize_offsets_and_diacritics():
    text = "Ọ̀rọ̀ yẹn don't make sense 😂👍🏽, check https://x.com/a. @debby #EndSARS 2,000"
    toks = tokenize(text)
    assert all(text[t.start:t.end] == t.text for t in toks)
    words = [t.text for t in toks]
    assert "Ọ̀rọ̀" in words and "don't" in words and "https://x.com/a" in words
    assert "👍🏽" in words and "#EndSARS" in words and "2,000" in words


def test_is_universal():
    assert is_universal("2,000") and is_universal("@debby") and is_universal("?")
    assert not is_universal("#EndSARS") and not is_universal("2day")


def test_tag_text():
    tagger = LanguageTagger(fake_model)
    res = tagger.tag("I no fit come today because rain dey fall for Lagos.")
    assert res.labels == ["eng", "pcm", "pcm", "eng", "eng", "eng", "eng", "pcm", "eng", "pcm", "ne", "other"]
    assert res.tokens[-1].source == "rule"
    assert res.switch_points() == [1, 3, 7, 8, 9]
    assert round(res.cmi(), 2) == 40.0
    assert res.tokens[0].start == 0


def test_pretokenized_and_word():
    tagger = LanguageTagger(fake_model)
    assert tagger.tag(["abeg", "come"]).labels == ["pcm", "eng"]
    assert tagger.tag_word("abeg").label == "pcm"


def test_retry_then_success():
    calls = []

    def flaky(system, user):
        calls.append(user)
        if len(calls) == 1:
            return '{"labels": {"0": "eng"}}'  # misaligned
        return fake_model(system, user)

    res = LanguageTagger(flaky).tag("abeg come")
    assert res.labels == ["pcm", "eng"] and len(calls) == 2
    assert "previous reply was invalid" in calls[1]


def test_strict_and_lenient_failure():
    bad = lambda s, u: "not json"
    with pytest.raises(LabelingError):
        LanguageTagger(bad, max_retries=1).tag("abeg come !")
    res = LanguageTagger(bad, max_retries=0, strict=False).tag("abeg come !")
    assert res.labels == [None, None, "other"]


def test_closed_labels_case_folding():
    model = lambda s, u: '{"labels": ["EN", "pcm", "Other"]}'
    tagger = LanguageTagger(model, labels={"en": "English", "pcm": "Nigerian Pidgin", "other": "Non-linguistic"})
    assert tagger.tag(["come", "dey", "!"]).labels == ["en", "pcm", "other"]
    assert "Nigerian Pidgin" in tagger.system_prompt


def test_closed_rejects_unknown_label():
    with pytest.raises(Exception):
        parse_labels('{"labels": ["fr"]}', 1, LabelSet.closed(["en", "pcm"]))


def test_chunking_splits_on_sentence_end():
    seen = []

    def model(system, user):
        seen.append(len(tokens_from(user)))
        return fake_model(system, user)

    text = " ".join(["abeg come ."] * 10)  # 30 tokens
    res = LanguageTagger(model, chunk_size=8).tag(text)
    assert len(res.tokens) == 30 and all(n <= 8 for n in seen) and sum(seen) == 30


def test_fenced_json_and_unknown_provider():
    assert parse_labels('```json\n{"labels": {"0": "eng"}}\n```', 1, LabelSet.open()) == ["eng"]
    with pytest.raises(ValueError):
        LanguageTagger("nope", model="x")
