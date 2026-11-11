import pytest


@pytest.fixture(autouse=True)
def banco_temporario(tmp_path, monkeypatch):
    from app import config
    cfg = config.obter_config()
    monkeypatch.setattr(cfg, "banco", tmp_path / "registro.sqlite")
    yield
