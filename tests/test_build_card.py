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


def test_build_card_with_translation(monkeypatch):
    meta = {
        'title': 'Example',
        'publication_date': '2024-01-01',
        'author_family': '',
        'author_given': '',
        'keywords': [],
        'text': 'Hello world'
    }
    url = 'https://example.com'
    access_date = '2024-06-01'

    monkeypatch.setattr(generate_cards, 'ask_openai', lambda *a, **k: 'summary')
    monkeypatch.setattr(generate_cards, 'classify_ndc_llm', lambda *a, **k: '000')
    monkeypatch.setattr(generate_cards, 'extract_keywords_llm', lambda *a, **k: [])
    monkeypatch.setattr(generate_cards, 'translate_full', lambda text: 'こんにちは世界')

    card = generate_cards.build_card(meta, url, access_date, skip_translation=False)

    assert '## Translation （和訳）' in card
    assert '## Original Text' in card
