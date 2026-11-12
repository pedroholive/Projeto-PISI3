
"""
HealthSync - L5
Auditoria da qualidade dos dados.

Responsabilidade: Integrante 1
Este arquivo nao modifica o dataset nem treina modelos.
"""

from pathlib import Path

import pandas as pd

from classificacao_l5.preparacao_l5 import (
    FEATURES_AMPLIADAS,
    carregar_dados,
    preparar_dados_l5,
    dividir_treino_teste,
)


RAIZ = Path(__file__).resolve().parents[1]

PASTA_RELATORIOS = (
    RAIZ / "classificacao_l5" / "relatorios"
)

CATEGORICAS = {
    "sexo",
    "hipertensao_informada",
}


def executar_auditoria():
    print("\n===== AUDITORIA L5 =====")

    # 1. Carregar os dados com a preparacao oficial
    df = carregar_dados()

    X_A, X_B, y, resumo = preparar_dados_l5(df)

    print("\nPARTICIPANTES")
    print("Base original:", len(df))
    print("Populacao elegivel:", len(y))

    # 2. Distribuicao das classes
    print("\nDISTRIBUICAO DAS CLASSES")

    for classe, quantidade in (
        y.value_counts().sort_index().items()
    ):
        percentual = quantidade / len(y) * 100

        print(
            f"Classe {classe}: {quantidade} "
            f"({percentual:.2f}%)"
        )

    # 3. Verificacao das variaveis
    print("\nQUALIDADE DAS VARIAVEIS")

    resultados = []

    for coluna in FEATURES_AMPLIADAS:
        serie = X_B[coluna]

        ausentes = int(serie.isna().sum())
        percentual = ausentes / len(serie) * 100

        resultado = {
            "variavel": coluna,
            "tipo": (
                "categorica"
                if coluna in CATEGORICAS
                else "numerica"
            ),
            "ausentes": ausentes,
            "percentual_ausente": round(
                percentual, 4
            ),
            "minimo": None,
            "mediana": None,
            "maximo": None,
            "outliers_iqr": None,
            "categorias": None,
        }

        if coluna in CATEGORICAS:
            resultado["categorias"] = str(
                serie.value_counts(
                    dropna=False
                ).to_dict()
            )

        else:
            valores = pd.to_numeric(
                serie,
                errors="coerce",
            )

            resultado["minimo"] = valores.min()
            resultado["mediana"] = valores.median()
            resultado["maximo"] = valores.max()

            # Metodo IQR: identifica extremos estatisticos
            q1 = valores.quantile(0.25)
            q3 = valores.quantile(0.75)

            iqr = q3 - q1

            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr

            extremos = (
                (valores < limite_inferior)
                | (valores > limite_superior)
            )

            resultado["outliers_iqr"] = int(
                extremos.sum()
            )

        resultados.append(resultado)

    relatorio = pd.DataFrame(resultados)

    print(
        relatorio[
            [
                "variavel",
                "tipo",
                "ausentes",
                "percentual_ausente",
                "outliers_iqr",
            ]
        ].to_string(index=False)
    )

    # 4. Verificar divisao treino/teste
    splits = dividir_treino_teste(
        X_A, X_B, y
    )

    treino = splits["X_train_A"]
    teste = splits["X_test_A"]

    sobreposicao = set(treino.index).intersection(
        set(teste.index)
    )

    print("\nDIVISAO DOS DADOS")
    print("Treinamento:", len(treino))
    print("Teste:", len(teste))
    print(
        "Participantes presentes nos dois grupos:",
        len(sobreposicao)
    )

    if sobreposicao:
        raise AssertionError(
            "Ha indices repetidos entre treino e teste."
        )

    # 5. Verificar IDs duplicados
    duplicados = 0

    if "id_participante" in df.columns:
        ids = df.loc[
            X_B.index, "id_participante"
        ]

        duplicados = int(
            ids.dropna().duplicated().sum()
        )

    print("IDs duplicados:", duplicados)

    # 6. Exportar o relatorio
    PASTA_RELATORIOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    caminho_csv = (
        PASTA_RELATORIOS
        / "auditoria_variaveis_l5.csv"
    )

    relatorio.to_csv(
        caminho_csv,
        index=False,
        encoding="utf-8-sig",
    )

    caminho_resumo = (
        PASTA_RELATORIOS
        / "resumo_auditoria_l5.txt"
    )

    texto_resumo = (
        "AUDITORIA DA BASE - HEALTHSYNC L5\n"
        f"Participantes originais: {len(df)}\n"
        f"Participantes elegiveis: {len(y)}\n"
        f"Classe 0: {int((y == 0).sum())}\n"
        f"Classe 1: {int((y == 1).sum())}\n"
        f"Treinamento: {len(treino)}\n"
        f"Teste: {len(teste)}\n"
        f"Sobreposicao de indices: {len(sobreposicao)}\n"
        f"IDs duplicados: {duplicados}\n"
        "\nValores extremos identificados pelo IQR "
        "devem ser investigados, nao removidos "
        "automaticamente.\n"
    )

    caminho_resumo.write_text(
        texto_resumo,
        encoding="utf-8",
    )

    print("\nRELATORIOS GERADOS")
    print(caminho_csv)
    print(caminho_resumo)

    print("\nAuditoria concluida com sucesso!")


if __name__ == "__main__":
    executar_auditoria()
