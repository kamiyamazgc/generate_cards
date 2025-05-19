import sys, types, datetime, pathlib

# Stub modules not required for unit tests
for name in ['httpx', 'trafilatura', 'yaml', 'slugify', 'dateparser', 'openai', 'tqdm']:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

# minimal slugify.slugify stub
sys.modules['slugify'].slugify = lambda text, **kwargs: text

# stub langdetect (not used but kept for parity)
langdetect_stub = types.ModuleType('langdetect')
class LangDetectException(Exception):
    pass
langdetect_stub.detect = lambda text: 'ja'
langdetect_stub.LangDetectException = LangDetectException
sys.modules['langdetect'] = langdetect_stub

# stub dateutil.parser to return datetime objects
parser_stub = types.ModuleType('parser')
parser_stub.parse = lambda x: datetime.datetime.fromisoformat(x)

dateutil_stub = types.ModuleType('dateutil')
dateutil_stub.parser = parser_stub
sys.modules['dateutil'] = dateutil_stub

import generate_cards


def test_process_urls_creates_digest_symlink(monkeypatch, tmp_path):
    lib_dir = tmp_path / 'Library'
    digest_dir = lib_dir / '_digests'
    lib_dir.mkdir()
    digest_dir.mkdir()

    monkeypatch.setattr(generate_cards, 'LIBRARY_DIR', lib_dir)
    monkeypatch.setattr(generate_cards, 'DIGEST_DIR', digest_dir)

    monkeypatch.setattr(generate_cards, 'fetch_html', lambda url: '<html>')

    def fake_extract_meta(url, html):
        return {
            'title': 'Example',
            'publication_date': '2024-01-01',
            'author_family': '',
            'author_given': '',
            'keywords': [],
            'text': 'dummy'
        }
    monkeypatch.setattr(generate_cards, 'extract_meta', fake_extract_meta)
    monkeypatch.setattr(generate_cards, 'build_card', lambda meta, url, ad, skip_translation=False: 'content')

    def fake_save_card(content, meta):
        path = lib_dir / 'example.md'
        path.write_text(content)
        return path
    monkeypatch.setattr(generate_cards, 'save_card', fake_save_card)

    generate_cards._process_urls(['http://example.com'])

    latest_link = lib_dir / '_daily_digest.md'
    assert latest_link.is_symlink()
    expected = pathlib.Path('_digests') / f"{datetime.date.today().isoformat()}.md"
    assert latest_link.readlink() == expected


def test_process_urls_handles_duplicate_digest(monkeypatch, tmp_path):
    lib_dir = tmp_path / 'Library'
    digest_dir = lib_dir / '_digests'
    lib_dir.mkdir()
    digest_dir.mkdir()

    monkeypatch.setattr(generate_cards, 'LIBRARY_DIR', lib_dir)
    monkeypatch.setattr(generate_cards, 'DIGEST_DIR', digest_dir)

    monkeypatch.setattr(generate_cards, 'fetch_html', lambda url: '<html>')

    def fake_extract_meta(url, html):
        return {
            'title': 'Example',
            'publication_date': '2024-01-01',
            'author_family': '',
            'author_given': '',
            'keywords': [],
            'text': 'dummy'
        }
    monkeypatch.setattr(generate_cards, 'extract_meta', fake_extract_meta)
    monkeypatch.setattr(generate_cards, 'build_card', lambda meta, url, ad, skip_translation=False: 'content')

    def fake_save_card(content, meta):
        path = lib_dir / 'example.md'
        path.write_text(content)
        return path
    monkeypatch.setattr(generate_cards, 'save_card', fake_save_card)

    # create an existing digest for today
    existing = digest_dir / f"{datetime.date.today().isoformat()}.md"
    existing.write_text('old')

    generate_cards._process_urls(['http://example.com'])

    new_digest = digest_dir / f"{datetime.date.today().isoformat()}-1.md"
    assert new_digest.exists()
    latest_link = lib_dir / '_daily_digest.md'
    assert latest_link.is_symlink()
    expected = pathlib.Path('_digests') / new_digest.name
    assert latest_link.readlink() == expected
