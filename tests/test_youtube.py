import sys, types, pathlib, datetime

# Stub modules not required for unit tests
for name in ['yt_dlp', 'whisper', 'httpx', 'trafilatura', 'yaml', 'slugify', 'dateparser', 'openai', 'tqdm']:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

sys.modules['slugify'].slugify = lambda text, **k: text

import generate_cards


def test_process_youtube_moves_audio(monkeypatch, tmp_path):
    lib_dir = tmp_path / 'Library'
    digest_dir = lib_dir / '_digests'
    lib_dir.mkdir()
    digest_dir.mkdir()

    monkeypatch.setattr(generate_cards, 'LIBRARY_DIR', lib_dir)
    monkeypatch.setattr(generate_cards, 'DIGEST_DIR', digest_dir)

    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'data')
    meta = {
        'title': 'Video',
        'publication_date': '2024-01-01',
        'author_family': '',
        'author_given': '',
        'keywords': [],
        'text': 'hello world'
    }
    monkeypatch.setattr(generate_cards, 'is_youtube_url', lambda url: True)
    monkeypatch.setattr(generate_cards, 'process_youtube_url', lambda url: (meta, audio))
    monkeypatch.setattr(generate_cards, 'build_card', lambda m, u, ad, skip_translation=False: 'body')
    monkeypatch.setattr(generate_cards, 'save_card', lambda c, m: lib_dir / 'card.md')

    generate_cards._process_urls(['https://youtu.be/xyz'])

    assert (lib_dir / 'card.mp3').exists()
