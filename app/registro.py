"""Registro de todas as chamadas aos modelos em SQLite (dados/registro.sqlite).

Permite recalcular métricas sem repetir chamadas, estimar custo e tempo, e
reproduzir os experimentos (modelo exato, data, versão do prompt, configuração).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import obter_config

_trava = threading.Lock()

ESQUEMA = """
CREATE TABLE IF NOT EXISTS chamadas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em TEXT NOT NULL,
    operacao TEXT NOT NULL,            -- avaliar | transcrever
    experimento TEXT,                  -- nome da rodada (NULL quando veio da API)
    redacao_id TEXT,                   -- id da redação no corpus (Essay-BR ou BRESSAY)
    repeticao INTEGER,
    provedor TEXT NOT NULL,
    modelo_pedido TEXT,
    modelo_resposta TEXT,
    versao_prompt TEXT,
    config TEXT,                       -- JSON: k, modo, parâmetros de geração etc.
    ids_exemplos TEXT,                 -- JSON
    tokens_entrada INTEGER,
    tokens_saida INTEGER,
    latencia_ms INTEGER,
    sucesso INTEGER NOT NULL,
    erro TEXT,
    resposta_bruta TEXT,
    resultado TEXT                     -- JSON validado (notas) ou texto transcrito
);
CREATE INDEX IF NOT EXISTS ix_exp ON chamadas(experimento, redacao_id, repeticao);
"""


def conectar(caminho: Path | None = None) -> sqlite3.Connection:
    caminho = Path(caminho or obter_config().banco)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(caminho, check_same_thread=False)
    con.executescript(ESQUEMA)
    return con


def registrar(**campos) -> int:
    campos.setdefault("criado_em", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    for k in ("config", "ids_exemplos"):
        if k in campos and not isinstance(campos[k], str) and campos[k] is not None:
            campos[k] = json.dumps(campos[k], ensure_ascii=False)
    campos["sucesso"] = int(bool(campos.get("sucesso")))
    cols = ", ".join(campos)
    marcas = ", ".join("?" for _ in campos)
    with _trava:
        con = conectar()
        try:
            cur = con.execute(f"INSERT INTO chamadas ({cols}) VALUES ({marcas})", list(campos.values()))
            con.commit()
            return cur.lastrowid
        finally:
            con.close()
