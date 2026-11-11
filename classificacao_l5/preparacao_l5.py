
"""HealthSync — L5: preparação dos dados para a triagem experimental de HbA1c.

Alvo binário: 0 se HbA1c < 5,7%; 1 se HbA1c >= 5,7%.
Somente participantes com diabetes_informado == "Nao".
Este módulo não ajusta transformações nem treina modelos.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# 1. CONFIGURAÇÕES
# =========================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

CAMINHO_DATASET = (
    RAIZ_PROJETO
    / "dataset"
    / "healthsync_nhanes_2021_2023_core.csv"
)

ALVO_ORIGINAL = "hba1c"
ALVO_BINARIO = "hba1c_binaria"
LIMIAR_HBA1C = 5.7

FEATURES_BASE = [
    "idade",
    "imc",
    "pressao_sistolica",
    "pressao_diastolica",
    "colesterol_total",
    "sexo",
]

FEATURES_AMPLIADAS = FEATURES_BASE + [
    "hipertensao_informada"
]

NUMERICAS = [
    col for col in FEATURES_AMPLIADAS
    if col != "sexo"
]


# =========================================================
# 2. CARREGAMENTO DOS DADOS
# =========================================================

def carregar_dados(caminho=CAMINHO_DATASET):
    """Lê a base CORE; não usa o CSV clusterizado nem o FASTING."""

    caminho = Path(caminho)

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Base CORE não encontrada: {caminho}"
        )

    return pd.read_csv(
        caminho,
        low_memory=False,
    )


# =========================================================
# 3. PREPARAÇÃO DO ALVO E DAS FEATURES
# =========================================================

def preparar_dados_l5(df):
    """Prepara as versões A e B sem ajustar transformações.

    Retorna (X_A, X_B, y, resumo) com índices alinhados.
    """

    obrigatorias = {
        "diabetes_informado",
        ALVO_ORIGINAL,
        *FEATURES_AMPLIADAS,
    }

    ausentes = sorted(
        obrigatorias - set(df.columns)
    )

    if ausentes:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {ausentes}"
        )

    # Mantém apenas participantes sem diabetes informado.
    # "Sim", "Borderline" e ausentes não são incluídos.

    resposta = (
        df["diabetes_informado"]
        .astype("string")
        .str.strip()
    )

    filtrado = df.loc[
        resposta.eq("Nao").fillna(False)
    ].copy()

    total_filtrado = len(filtrado)

    if total_filtrado == 0:
        raise ValueError(
            'Nenhum participante com diabetes_informado == "Nao".'
        )

    # HbA1c participa EXCLUSIVAMENTE da construção do alvo.

    filtrado[ALVO_ORIGINAL] = pd.to_numeric(
        filtrado[ALVO_ORIGINAL],
        errors="coerce",
    )

    hba1c_valida = (
        filtrado[ALVO_ORIGINAL].notna()
        & np.isfinite(filtrado[ALVO_ORIGINAL])
    )

    filtrado = filtrado.loc[
        hba1c_valida
    ].copy()

    if filtrado.empty:
        raise ValueError(
            "Não há valores válidos de HbA1c após o filtro."
        )

    # Converte indicadores numéricos.
    # Valores ausentes serão tratados dentro dos pipelines
    # de treinamento, evitando vazamento de dados.

    for coluna in NUMERICAS:

        filtrado[coluna] = pd.to_numeric(
            filtrado[coluna],
            errors="raise",
        )

        filtrado[coluna] = filtrado[coluna].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    # Preserva sexo como variável categórica.
    # OneHotEncoder será utilizado posteriormente,
    # com ajuste somente nos dados de treinamento.

    filtrado["sexo"] = filtrado["sexo"].map(
        lambda valor: (
            valor.strip()
            if isinstance(valor, str)
            else np.nan
        )
    ).replace("", np.nan)

    # -----------------------------------------------------
    # CRIAÇÃO DO ALVO BINÁRIO
    # -----------------------------------------------------

    # Classe 0: HbA1c < 5,7
    # Classe 1: HbA1c >= 5,7

    y = (
        filtrado[ALVO_ORIGINAL] >= LIMIAR_HBA1C
    ).astype("int8")

    if y.nunique() != 2:
        raise ValueError(
            "O alvo L5 precisa conter as duas classes (0 e 1)."
        )

    # -----------------------------------------------------
    # VERSÃO A — FEATURES BASE
    # -----------------------------------------------------

    X_A = filtrado[
        FEATURES_BASE
    ].copy()

    # -----------------------------------------------------
    # VERSÃO B — FEATURES AMPLIADAS
    # -----------------------------------------------------

    X_B = filtrado[
        FEATURES_AMPLIADAS
    ].copy()

    # -----------------------------------------------------
    # VERIFICAÇÃO CONTRA VAZAMENTO DE DADOS
    # -----------------------------------------------------

    proibidas = {
        "hba1c",
        "hba1c_binaria",
        "perfil_id",
        "diabetes_informado",
        "id_participante",
        "glicose_jejum",
        "estrato_hba1c",
    }

    if (
        proibidas.intersection(X_A.columns)
        or proibidas.intersection(X_B.columns)
    ):
        raise AssertionError(
            "Foi detectada uma feature proibida no L5."
        )

    # -----------------------------------------------------
    # RESUMO DA PREPARAÇÃO
    # -----------------------------------------------------

    resumo = {
        "participantes_antes_filtro": len(df),

        "participantes_sem_diabetes_informado":
            total_filtrado,

        "participantes_com_alvo_valido":
            len(filtrado),

        "classe_0":
            int((y == 0).sum()),

        "classe_1":
            int((y == 1).sum()),

        "ausentes_hipertensao_informada":
            int(
                X_B["hipertensao_informada"]
                .isna()
                .sum()
            ),
    }

    return X_A, X_B, y, resumo


# =========================================================
# 4. DIVISÃO DE TREINO E TESTE
# =========================================================

def dividir_treino_teste(
    X_A,
    X_B,
    y,
    test_size=0.20,
    random_state=42,
):
    """Usa os mesmos participantes nos splits das versões A e B.

    Nenhum scaler ou imputador é ajustado nesta etapa.
    O pré-processamento ocorrerá dentro dos pipelines
    de treinamento e validação cruzada.
    """

    if (
        not X_A.index.equals(X_B.index)
        or not X_A.index.equals(y.index)
    ):
        raise ValueError(
            "Índices de X_A, X_B e y precisam estar alinhados."
        )

    indices_treino, indices_teste = train_test_split(
        y.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return {
        "X_train_A":
            X_A.loc[indices_treino].copy(),

        "X_test_A":
            X_A.loc[indices_teste].copy(),

        "X_train_B":
            X_B.loc[indices_treino].copy(),

        "X_test_B":
            X_B.loc[indices_teste].copy(),

        "y_train":
            y.loc[indices_treino].copy(),

        "y_test":
            y.loc[indices_teste].copy(),
    }


# =========================================================
# 5. EXECUÇÃO
# =========================================================

def executar_preparacao():

    df = carregar_dados()

    X_A, X_B, y, resumo = preparar_dados_l5(df)

    splits = dividir_treino_teste(
        X_A,
        X_B,
        y,
    )

    print(
        "\nL5 — PREPARAÇÃO DA TRIAGEM EXPERIMENTAL DE HbA1c"
    )

    for chave, valor in resumo.items():
        print(f"{chave}: {valor}")

    print(
        "\nConjunto de treinamento:",
        len(splits["y_train"]),
    )

    print(
        "Conjunto de teste:",
        len(splits["y_test"]),
    )

    print(
        "Distribuição treino:",
        splits["y_train"].value_counts().to_dict(),
    )

    print(
        "Distribuição teste:",
        splits["y_test"].value_counts().to_dict(),
    )

    return splits


if __name__ == "__main__":
    executar_preparacao()
