import sys, types

# Stub modules not required for unit tests
for name in ['httpx', 'trafilatura', 'yaml', 'slugify', 'dateparser', 'openai', 'tqdm']:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

# minimal slugify.slugify
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


def test_chunk_text():
    text = "Hello world. This is test. Another."
    chunks = generate_cards.chunk_text(text, max_chars=30)
    assert chunks == ["Hello world.", "This is test.", "Another."]


def test_tidy_markdown_para():
    txt = "Sentence one.\nSentence two."
    assert generate_cards.tidy_markdown_para(txt) == "Sentence one.\n\nSentence two."

    txt_multi = "A.\n\n\n\nB."
    assert generate_cards.tidy_markdown_para(txt_multi) == "A.\n\nB."


def test_detect_lang():
    assert generate_cards.detect_lang("Hello") == "en"
    assert generate_cards.detect_lang("こんにちは") == "ja"
    assert generate_cards.detect_lang("") == "unknown"
