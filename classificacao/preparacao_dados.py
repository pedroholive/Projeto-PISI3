
"""
HealthSync - Preparação de dados para classificação.

Etapas:
1. Carregamento e validação dos dados.
2. Separação dos estratos de HbA1c.
3. Preparação da variável alvo.
4. Divisão estratificada de treino e teste.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# Caminho padrão da base clusterizada.
CAMINHO_DATASET = (
    Path(__file__).resolve().parents[1]
    / "modelos"
    / "clusterizacao"
    / "healthsync_perfis_cardiometabolicos.csv"
)

# Cinco variáveis utilizadas na classificação.
FEATURES = [
    "idade",
    "imc",
    "pressao_sistolica",
    "pressao_diastolica",
    "colesterol_total",
]

# Variável alvo criada pela clusterização.
TARGET = "perfil_id"

# Nomes exatos dos estratos presentes no CSV.
ESTRATOS = [
    "HbA1c < 5.7",
    "5.7 <= HbA1c < 6.5",
    "HbA1c >= 6.5",
]

# Quantidade esperada de clusters.
CLUSTERS_ESPERADOS = {
    "HbA1c < 5.7": 2,
    "5.7 <= HbA1c < 6.5": 6,
    "HbA1c >= 6.5": 2,
}


def carregar_dados(caminho=CAMINHO_DATASET):
    """Carrega e valida o dataset clusterizado."""

    # Converte o caminho para Path.
    caminho = Path(caminho)

    # Confere se o arquivo existe.
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {caminho}"
        )

    # Lê o arquivo CSV.
    df = pd.read_csv(caminho)

    # Reúne as colunas obrigatórias.
    obrigatorias = FEATURES + [
        "estrato_hba1c",
        TARGET,
    ]

    # Identifica eventuais colunas ausentes.
    faltantes = [
        coluna
        for coluna in obrigatorias
        if coluna not in df.columns
    ]

    # Impede a execução com uma estrutura incorreta.
    if faltantes:
        raise ValueError(
            f"Colunas ausentes: {faltantes}"
        )

    # Confere a existência de valores ausentes.
    if df[obrigatorias].isna().any().any():
        raise ValueError(
            "Existem valores ausentes nas colunas obrigatórias."
        )

    # Converte as features para números.
    for coluna in FEATURES:
        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="raise",
        )

    # Verifica valores não finitos.
    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError(
            "Existem valores numéricos inválidos."
        )

    # Apresenta informações básicas.
    print("\nDATASET CARREGADO")
    print(f"Participantes: {len(df)}")
    print(f"Colunas: {len(df.columns)}")

    # Retorna os dados validados.
    return df


def preparar_estratos(df):
    """Separa os dados nos três estratos de HbA1c."""

    # Protege o DataFrame original.
    df = df.copy()

    # Valida os nomes dos estratos.
    invalidos = set(df["estrato_hba1c"]) - set(ESTRATOS)

    if invalidos:
        raise ValueError(
            f"Estratos não reconhecidos: {invalidos}"
        )

    # Cria identificadores únicos entre os estratos.
    df["perfil_global"] = (
        df["estrato_hba1c"].astype(str)
        + " | "
        + df[TARGET].astype(str)
    )

    # Dicionário para armazenar cada estrato.
    dados_estratos = {}

    # Processa os estratos individualmente.
    for estrato in ESTRATOS:

        # Seleciona os participantes do estrato.
        subset = df.loc[
            df["estrato_hba1c"] == estrato
        ].copy()

        # Verifica se o grupo está vazio.
        if subset.empty:
            raise ValueError(
                f"Estrato vazio: {estrato}"
            )

        # Conta os perfis encontrados.
        encontrados = subset["perfil_global"].nunique()

        # Recupera a quantidade esperada.
        esperados = CLUSTERS_ESPERADOS[estrato]

        # Confere os clusters.
        if encontrados != esperados:
            raise ValueError(
                f"{estrato}: esperados {esperados} "
                f"perfis; encontrados {encontrados}."
            )

        # Armazena o subconjunto validado.
        dados_estratos[estrato] = subset

        # Apresenta a distribuição dos perfis.
        print(f"\nESTRATO: {estrato}")
        print(f"Participantes: {len(subset)}")
        print(subset["perfil_global"].value_counts())

    # Confere os dez perfis globais.
    total_perfis = df["perfil_global"].nunique()

    if total_perfis != 10:
        raise ValueError(
            f"Esperados 10 perfis; encontrados {total_perfis}."
        )

    # Garante que nenhum registro foi perdido.
    total = sum(
        len(subset)
        for subset in dados_estratos.values()
    )

    if total != len(df):
        raise ValueError(
            "A separação dos estratos perdeu participantes."
        )

    # Retorna os três subconjuntos.
    return dados_estratos


def dividir_treino_teste(
    dados_estratos,
    test_size=0.20,
    random_state=42,
):
    """Gera treino e teste estratificados por perfil."""

    # Armazena os conjuntos de cada estrato.
    splits = {}

    # Percorre os três estratos.
    for estrato in ESTRATOS:

        # Recupera o subconjunto atual.
        subset = dados_estratos[estrato]

        # Seleciona as cinco features.
        X = subset[FEATURES].copy()

        # Utiliza o identificador original do perfil cardiometabólico.
        y = subset["perfil_id"].copy()

        # Verifica se alguma classe tem poucas amostras.
        if y.value_counts().min() < 2:
            raise ValueError(
                f"{estrato}: classe com menos de 2 amostras."
            )

        # Divide os dados preservando as proporções.
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        )

        # Verifica a presença de todas as classes.
        classes = set(y)

        if set(y_train) != classes:
            raise ValueError(
                f"{estrato}: classes ausentes no treino."
            )

        if set(y_test) != classes:
            raise ValueError(
                f"{estrato}: classes ausentes no teste."
            )

        # Guarda os conjuntos para o Integrante 2.
        splits[estrato] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        # Mostra o resultado da divisão.
        print(f"\nSPLIT: {estrato}")
        print(f"Treino: {len(X_train)}")
        print(f"Teste: {len(X_test)}")

        # Compara as proporções das classes.
        proporcoes = pd.DataFrame({
            "Original": y.value_counts(normalize=True),
            "Treino": y_train.value_counts(normalize=True),
            "Teste": y_test.value_counts(normalize=True),
        }).fillna(0)

        print((proporcoes * 100).round(2))

    # Retorna os dados preparados para treinamento.
    return splits


if __name__ == "__main__":

    # Carrega o CSV final da clusterização.
    df = carregar_dados()

    # Separa os dados nos três estratos.
    dados_estratos = preparar_estratos(df)

    # Divide cada estrato em treino e teste.
    splits = dividir_treino_teste(dados_estratos)

    # Confirma que a preparação foi concluída.
    print("\nPREPARAÇÃO CONCLUÍDA COM SUCESSO!")
