from experimentos.bressay import contar_ilegiveis, normalizar_hipotese, normalizar_referencia
from experimentos.metricas_base import cer


def test_riscado_e_removido():
    assert normalizar_referencia("em --infraestru-- espaços") == "em espaços"
    assert normalizar_referencia("texto --xxx-- final") == "texto final"


def test_insercao_mantem_texto():
    assert normalizar_referencia("o ##governo## deve") == "o governo deve"
    assert normalizar_referencia("o $$novo$$ plano") == "o novo plano"


def test_insercao_riscada_e_removida():
    assert normalizar_referencia("a ##--palavra--## b") == "a b"
    assert normalizar_referencia("a $$--xxx--$$ b") == "a b"


def test_ilegivel_e_removido_e_contado():
    bruto = "uma @@???@@ palavra ##@@???@@##"
    assert normalizar_referencia(bruto) == "uma palavra"
    assert contar_ilegiveis(bruto) == 2


def test_hifenizacao_e_quebras():
    assert normalizar_referencia("um maior entendimento de co-\nmo era\n\nNovo parágrafo") == \
        "um maior entendimento de como era Novo parágrafo"
    assert normalizar_hipotese("de co-\nmo era") == "de como era"


def test_hifen_dentro_da_linha_preservado():
    assert normalizar_referencia("parceria público-privada") == "parceria público-privada"


def test_cer_zero_quando_igual_apos_normalizar():
    ref = normalizar_referencia("Os filmes, ##então,## possuem a fun-\nção --social--")
    hip = normalizar_hipotese("Os filmes, então, possuem a função")
    assert cer(ref, hip) == 0
