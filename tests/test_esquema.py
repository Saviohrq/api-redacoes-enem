import json

import pytest

from app.esquemas import esquema_json_avaliacao
from app.servicos import _interpretar


def _saida(nota="120"):
    return json.dumps({f"c{i}": {"justificativa": "x", "nota": nota} for i in range(1, 6)})


def test_interpreta_notas_validas():
    a = _interpretar(_saida("160"))
    assert a.total == 800


def test_rejeita_nota_fora_da_escala():
    with pytest.raises(Exception):
        _interpretar(_saida("150"))


def test_esquema_tem_cinco_competencias():
    e = esquema_json_avaliacao()
    assert e["required"] == ["c1", "c2", "c3", "c4", "c5"]
