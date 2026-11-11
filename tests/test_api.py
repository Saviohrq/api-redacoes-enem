import sqlite3

from fastapi.testclient import TestClient

from app.config import obter_config
from app.main import app

cliente = TestClient(app)
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100


def test_avaliar_simulado_registra_chamada():
    r = cliente.post("/avaliar", json={"texto": "Uma redação qualquer.", "proposta": {"titulo": "Tema"},
                                       "provedor": "simulado"})
    assert r.status_code == 200, r.text
    d = r.json()
    notas = [d["avaliacao"][c]["nota"] for c in ("c1", "c2", "c3", "c4", "c5")]
    assert all(n in (0, 40, 80, 120, 160, 200) for n in notas)
    assert d["nota_total"] == sum(notas)
    con = sqlite3.connect(obter_config().banco)
    assert con.execute("select count(*), sum(sucesso) from chamadas").fetchone() == (1, 1)


def test_transcrever_e_pre_analise():
    r = cliente.post("/transcrever", files={"imagem": ("r.png", PNG, "image/png")}, data={"provedor": "simulado"})
    assert r.status_code == 200 and r.json()["texto"]
    r = cliente.post("/pre-analise", files={"imagem": ("r.png", PNG, "image/png")},
                     data={"tema": "Tema", "provedor": "simulado"})
    assert r.status_code == 200, r.text
    assert r.json()["modo"] == "unico"


def test_imagem_formato_invalido():
    r = cliente.post("/transcrever", files={"imagem": ("r.pdf", b"x", "application/pdf")},
                     data={"provedor": "simulado"})
    assert r.status_code == 415


def test_separado_sem_provedor_transcricao():
    r = cliente.post("/pre-analise", files={"imagem": ("r.png", PNG, "image/png")},
                     data={"tema": "Tema", "provedor": "simulado", "modo": "separado"})
    assert r.status_code == 422
