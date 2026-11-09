
import math

import pandas as pd

from classificacao.preparacao_dados import (
    ESTRATOS,
    FEATURES,
)

from classificacao.avaliacao import carregar_modelos


def identificar_estrato(hba1c):
    """Identifica o estrato glicêmico do paciente."""

    hba1c = float(hba1c)

    if not math.isfinite(hba1c) or hba1c <= 0:
        raise ValueError("HbA1c inválida.")

    if hba1c < 5.7:
        return ESTRATOS[0]

    if hba1c < 6.5:
        return ESTRATOS[1]

    return ESTRATOS[2]


def classificar_paciente(dados_paciente, modelos):
    """Classifica um paciente usando o modelo do estrato."""

    if "hba1c" not in dados_paciente:
        raise ValueError("Campo obrigatório ausente: hba1c")

    estrato = identificar_estrato(
        dados_paciente["hba1c"]
    )

    faltantes = [
        nome for nome in FEATURES
        if nome not in dados_paciente
    ]

    if faltantes:
        raise ValueError(
            f"Features ausentes: {faltantes}"
        )

    valores = {}

    for nome in FEATURES:

        valor = float(dados_paciente[nome])

        if not math.isfinite(valor) or valor <= 0:
            raise ValueError(
                f"Valor inválido para {nome}: {valor}"
            )

        valores[nome] = valor


    X_novo = pd.DataFrame(
        [valores],
        columns=FEATURES,
    )

    modelo = modelos[estrato]

    perfil_id = str(modelo.predict(X_novo)[0])

    probabilidades = modelo.predict_proba(X_novo)[0]

    classes = list(modelo.classes_)
    indice = classes.index(perfil_id)

    confianca = float(probabilidades[indice])

    return {
        "estrato_hba1c": estrato,
        "perfil_id": perfil_id,
        "confianca_modelo": round(confianca, 4),
    }


if __name__ == "__main__":

    modelos = carregar_modelos()

    paciente = {
        "hba1c": 6.1,
        "idade": 54,
        "imc": 29.4,
        "pressao_sistolica": 138,
        "pressao_diastolica": 88,
        "colesterol_total": 215,
    }

    resultado = classificar_paciente(
        paciente,
        modelos,
    )

    print("\nRESULTADO DA CLASSIFICAÇÃO")
    print(resultado)
