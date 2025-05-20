import sys, types

# Stub modules not required for unit tests
for name in ['httpx', 'trafilatura', 'yaml', 'slugify', 'dateparser', 'openai', 'tqdm']:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

def _slugify(text, **kwargs):
    return text
sys.modules['slugify'].slugify = _slugify

# stub langdetect
langdetect_stub = types.ModuleType('langdetect')
class LangDetectException(Exception):
    pass

def detect(text):
    if not text.strip():
        raise LangDetectException('empty')
    if any('\u3040' <= c <= '\u30ff' for c in text):
        return 'ja'
    return 'en'
langdetect_stub.detect = detect
langdetect_stub.LangDetectException = LangDetectException
sys.modules['langdetect'] = langdetect_stub

# stub dateutil.parser
parser_stub = types.ModuleType('parser')
parser_stub.parse = lambda x: x

dateutil_stub = types.ModuleType('dateutil')
dateutil_stub.parser = parser_stub
sys.modules['dateutil'] = dateutil_stub

import generate_cards


def test_translate_full_chunking(monkeypatch):
    chunks = ['first', 'second', 'third']
    monkeypatch.setattr(generate_cards, 'chunk_text', lambda text: chunks)

    calls = []
    def fake_ask(prompt, *a, **k):
        calls.append(prompt)
        return f"ja-{len(calls)}"
    monkeypatch.setattr(generate_cards, 'ask_openai', fake_ask)

    result = generate_cards.translate_full('ignored')

    assert result == 'ja-1\n\nja-2\n\nja-3'
    base = '次の文章を日本語に正確に全文翻訳してください。\n\n'
    assert calls == [base + c for c in chunks]


def test_classify_ndc_llm_valid(monkeypatch):
    monkeypatch.setattr(generate_cards, 'ask_openai', lambda *a, **k: '007')
    code = generate_cards.classify_ndc_llm('t', 's')
    assert code == '007'


def test_classify_ndc_llm_retry(monkeypatch):
    responses = iter(['998', '007'])
    def fake_ask(prompt, *a, **k):
        return next(responses)
    monkeypatch.setattr(generate_cards, 'ask_openai', fake_ask)
    code = generate_cards.classify_ndc_llm('t', 's')
    assert code == '007'


def test_classify_ndc_llm_all_invalid(monkeypatch):
    responses = iter(['998', '998'])
    def fake_ask(prompt, *a, **k):
        return next(responses)
    monkeypatch.setattr(generate_cards, 'ask_openai', fake_ask)
    code = generate_cards.classify_ndc_llm('t', 's')
    assert code == ''
