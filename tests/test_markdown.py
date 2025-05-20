import sys, types, datetime, pathlib

# Stub modules not required for unit tests
for name in ['httpx', 'trafilatura', 'yaml', 'slugify', 'dateparser', 'openai', 'tqdm']:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

sys.modules['slugify'].slugify = lambda text, **kwargs: text

langdetect_stub = types.ModuleType('langdetect')
class LangDetectException(Exception):
    pass
langdetect_stub.detect = lambda text: 'ja'
langdetect_stub.LangDetectException = LangDetectException
sys.modules['langdetect'] = langdetect_stub

parser_stub = types.ModuleType('parser')
parser_stub.parse = lambda x: datetime.datetime.fromisoformat(x)

dateutil_stub = types.ModuleType('dateutil')
dateutil_stub.parser = parser_stub
sys.modules['dateutil'] = dateutil_stub

import generate_cards


def test_process_markdown_creates_digest(monkeypatch, tmp_path):
    lib_dir = tmp_path / 'Library'
    digest_dir = lib_dir / '_digests'
    lib_dir.mkdir()
    digest_dir.mkdir()

    monkeypatch.setattr(generate_cards, 'LIBRARY_DIR', lib_dir)
    monkeypatch.setattr(generate_cards, 'DIGEST_DIR', digest_dir)

    md = tmp_path / 'sample.md'
    md.write_text('hello world', encoding='utf-8')

    monkeypatch.setattr(generate_cards, 'build_card', lambda meta, url, ad, skip_translation=False: 'body')
    monkeypatch.setattr(generate_cards, 'save_card', lambda c, m: lib_dir / 'card.md')

    generate_cards._process_markdown_files([str(md)])

    latest_link = lib_dir / '_daily_digest.md'
    assert latest_link.is_symlink()
    expected = pathlib.Path('_digests') / f"{datetime.date.today().isoformat()}.md"
    assert latest_link.readlink() == expected
