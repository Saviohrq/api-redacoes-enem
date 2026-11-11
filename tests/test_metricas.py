import numpy as np
from sklearn.metrics import cohen_kappa_score

from experimentos.metricas_base import NIVEIS_COMP, cer, metricas_total, qwk, wer


def test_qwk_igual_ao_sklearn():
    rng = np.random.default_rng(0)
    y = rng.choice(NIVEIS_COMP, 200)
    p = np.clip(y + rng.choice([-40, 0, 40], 200), 0, 200)
    esperado = cohen_kappa_score(y, p, weights="quadratic", labels=NIVEIS_COMP)
    assert abs(qwk(y, p) - esperado) < 1e-9


def test_qwk_perfeito():
    y = np.array([0, 40, 120, 200, 160])
    assert qwk(y, y) == 1.0


def test_discrepancia_inep():
    Y = np.array([[120] * 5, [120] * 5])
    P = np.array([[120, 120, 120, 120, 0], [160] * 5])  # 1ª: dif 120 na C5; 2ª: total dif 200
    assert metricas_total(Y, P)["Discrepancia_INEP"] == 1.0


def test_cer_wer():
    assert cer("casa", "caza") == 0.25
    assert wer("a casa azul", "a caza azul") == 1 / 3
    assert cer("abc", "abc") == 0
