import types, pathlib
import generate_cards


def make_torch(cuda=False, mps=False):
    cuda_stub = types.SimpleNamespace(is_available=lambda: cuda)
    mps_stub = types.SimpleNamespace(is_available=lambda: mps)
    backends = types.SimpleNamespace(mps=mps_stub)
    return types.SimpleNamespace(cuda=cuda_stub, backends=backends)


def make_whisper(record):
    class Model:
        def transcribe(self, path, **_):
            record['path'] = path
            return {'language': 'en', 'text': 'hi'}

        def float(self):
            return self

    def load_model(name, device):
        record['device'] = device
        assert name == 'base'
        return Model()

    return types.SimpleNamespace(load_model=load_model)


def test_transcribe_audio_uses_cuda(monkeypatch, tmp_path):
    record = {}
    monkeypatch.setattr(generate_cards, 'torch', make_torch(cuda=True))
    monkeypatch.setattr(generate_cards, 'whisper', make_whisper(record))
    generate_cards._transcribe_audio(tmp_path / 'a.mp3')
    assert record['device'] == 'cuda'


def test_transcribe_audio_uses_mps(monkeypatch, tmp_path):
    record = {}
    monkeypatch.setattr(generate_cards, 'torch', make_torch(mps=True))
    monkeypatch.setattr(generate_cards, 'whisper', make_whisper(record))
    generate_cards._transcribe_audio(tmp_path / 'a.mp3')
    assert record['device'] == 'mps'

