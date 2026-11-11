"""Testa a recuperação com um corpus pequeno e um 'modelo de embeddings' falso (sem download)."""
import numpy as np
import pandas as pd
import pytest

from app.rag import recuperacao as rec


class EmbedderFalso:
    def encode(self, textos, **_):
        v = np.array([[t.count(p) for p in ("sol", "mar", "rio", "lua")] for t in textos], float) + 1e-6
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def acervo(tmp_path, monkeypatch):
    textos = ["sol sol", "mar mar", "rio rio", "lua lua", "sol mar", "mar rio"]
    df = pd.DataFrame({"id": range(6), "prompt": [0, 0, 1, 1, 0, 1], "texto": textos,
                       **{c: [120] * 6 for c in ("c1", "c2", "c3", "c4", "c5")}, "score": [600] * 6})
    from app import config
    monkeypatch.setattr(config.obter_config(), "dados_dir", tmp_path)
    monkeypatch.setattr(rec, "carregar_redacoes", lambda: df)
    monkeypatch.setattr(rec, "ids_do_split", lambda nome: [0, 1, 2, 3, 4, 5])
    monkeypatch.setattr(rec, "carregar_propostas",
                        lambda: pd.DataFrame({"title": ["A", "B"]}, index=[0, 1]))
    monkeypatch.setattr(rec, "_modelo_embeddings", lambda: EmbedderFalso())
    rec.construir_indice()
    yield
    rec._indice.cache_clear()


def test_similares_exclui_a_propria_redacao(acervo):
    assert rec.recuperar("sol sol", 1, "similares", redacao_id=0) == [4]
    assert rec.recuperar("sol sol", 2, "similares")[0] == 0


def test_mesma_proposta(acervo):
    assert set(rec.recuperar("sol", 3, "similares", proposta_id=1)) == {2, 3, 5}


def test_aleatorios_reprodutivel(acervo):
    a = rec.recuperar("x", 3, "aleatorios", redacao_id=99, semente=42)
    assert a == rec.recuperar("x", 3, "aleatorios", redacao_id=99, semente=42)
    assert len(set(a)) == 3


def test_avaliar_com_k3(acervo):
    from app.esquemas import ConfigRecuperacao, PedidoAvaliacao, Proposta
    from app.servicos import avaliar
    r = avaliar(PedidoAvaliacao(texto="mar e sol", proposta=Proposta(titulo="T"), provedor="simulado",
                                recuperacao=ConfigRecuperacao(k=3)))
    assert len(r.ids_exemplos) == 3
