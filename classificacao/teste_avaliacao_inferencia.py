
import pytest

from classificacao.preparacao_dados import ESTRATOS
from classificacao.avaliacao import carregar_modelos
from classificacao.inferencia import (
    identificar_estrato,
    classificar_paciente,
)


@pytest.fixture(scope="module")
def modelos():
    """Carrega os três modelos uma única vez."""
    return carregar_modelos()


def test_carregamento_modelos(modelos):
    """Confere os três classificadores."""
    assert len(modelos) == 3

    for estrato in ESTRATOS:
        assert estrato in modelos


def test_quantidade_classes(modelos):
    """Confere a quantidade de perfis por estrato."""
    esperados = [2, 6, 2]

    for estrato, esperado in zip(ESTRATOS, esperados):
        assert len(modelos[estrato].classes_) == esperado


@pytest.mark.parametrize(
    "hba1c, esperado",
    [
        (5.2, ESTRATOS[0]),
        (5.7, ESTRATOS[1]),
        (6.1, ESTRATOS[1]),
        (6.5, ESTRATOS[2]),
        (8.0, ESTRATOS[2]),
    ],
)
def test_identificar_estrato(hba1c, esperado):
    """Confere os limites de HbA1c."""
    assert identificar_estrato(hba1c) == esperado


def test_hba1c_invalida():
    """Rejeita HbA1c inválida."""
    with pytest.raises(ValueError):
        identificar_estrato(-1)


def test_classificar_paciente(modelos):
    """Verifica a classificação de um paciente fictício."""

    paciente = {
        "hba1c": 6.1,
        "idade": 54,
        "imc": 29.4,
        "pressao_sistolica": 138,
        "pressao_diastolica": 88,
        "colesterol_total": 215,
    }

    resultado = classificar_paciente(paciente, modelos)

    assert resultado["estrato_hba1c"] == ESTRATOS[1]

    assert resultado["perfil_id"] in modelos[
        ESTRATOS[1]
    ].classes_

    assert 0 <= resultado["confianca_modelo"] <= 1


def test_paciente_incompleto(modelos):
    """Rejeita dados com features ausentes."""

    paciente = {
        "hba1c": 5.4,
        "idade": 40,
    }

    with pytest.raises(ValueError):
        classificar_paciente(paciente, modelos)
