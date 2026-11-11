
"""
HealthSync - Avaliação da atribuição de subperfil.

Responsabilidade:
Medir a capacidade dos modelos Random Forest de reproduzir
a partição de perfis cardiometabólicos gerada pelo K-Means,
comparando-os com duas linhas de base simples:

1. Regressão Logística treinada no mesmo split;
2. KMeans.predict direto sobre as entradas escalonadas.

Como o alvo (perfil_id) foi criado pelo próprio K-Means
usando as mesmas variáveis de entrada, métricas elevadas
demonstram reprodução da partição, e não previsão de um
dado novo.
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from classificacao.preparacao_dados import (
    ESTRATOS,
    carregar_dados,
    preparar_estratos,
    dividir_treino_teste,
)

from classificacao.treinamento import DIRETORIO_MODELOS

DIRETORIO_RESULTADOS = (
    Path(__file__).resolve().parents[1]
    / "modelos"
    / "avaliacao_classificacao"
)

# Artefatos do K-Means usados para gerar o alvo (perfil_id).
DIRETORIO_MODELOS_KMEANS = (
    Path(__file__).resolve().parents[1]
    / "modelos"
    / "modelosclusterizacao"
)


def carregar_modelos(diretorio=DIRETORIO_MODELOS):

    diretorio = Path(diretorio)

    modelos = {}

    for indice, estrato in enumerate(ESTRATOS, start=1):

        caminho = (
            diretorio / f"modelo_estrato_{indice}.joblib"
        )

        if not caminho.is_file():
            raise FileNotFoundError(
                f"Modelo não encontrado: {caminho}"
            )

        modelos[estrato] = joblib.load(caminho)

    return modelos


def avaliar_modelos(
    modelos,
    splits,
    diretorio=DIRETORIO_RESULTADOS,
):
    """Avalia a reprodução da partição K-Means nos testes."""

    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)

    resultados = []

    for indice, estrato in enumerate(ESTRATOS, start=1):

        modelo = modelos[estrato]

        X_test = splits[estrato]["X_test"]
        y_test = splits[estrato]["y_test"]

        y_pred = modelo.predict(X_test)

        classes = list(modelo.classes_)

        metricas = {
            "estrato": estrato,
            "participantes_teste": len(y_test),
            "accuracy_rf": accuracy_score(y_test, y_pred),
            "precision_macro_rf": precision_score(
                y_test, y_pred,
                average="macro",
                zero_division=0,
            ),
            "recall_macro_rf": recall_score(
                y_test, y_pred,
                average="macro",
                zero_division=0,
            ),
            "f1_macro_rf": f1_score(
                y_test, y_pred,
                average="macro",
                zero_division=0,
            ),
            "f1_weighted_rf": f1_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0,
            ),
        }

        resultados.append(metricas)

        print(f"\n{'=' * 60}")
        print(f"ATRIBUIÇÃO DE SUBPERFIL - ESTRATO {indice}")
        print(estrato)
        print(f"{'=' * 60}")

        print(
            classification_report(
                y_test,
                y_pred,
                labels=classes,
                zero_division=0,
            )
        )

        fig, ax = plt.subplots(figsize=(9, 7))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=classes,
            display_labels=classes,
            cmap="Blues",
            values_format="d",
            ax=ax,
        )

        ax.set_title(
            f"HealthSync - Matriz de Confusão - Estrato {indice}"
        )

        plt.tight_layout()

        caminho_imagem = (
            diretorio / f"matriz_estrato_{indice}.png"
        )

        fig.savefig(
            caminho_imagem,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Matriz salva em: {caminho_imagem}")

    return pd.DataFrame(resultados)


def carregar_kmeans(diretorio=DIRETORIO_MODELOS_KMEANS):
    """Carrega os artefatos do K-Means que geraram o alvo."""

    diretorio = Path(diretorio)

    artefatos = {}

    for indice, estrato in enumerate(ESTRATOS, start=1):

        caminho = (
            diretorio / f"estrato_{indice}_kmeans.joblib"
        )

        if not caminho.is_file():
            raise FileNotFoundError(
                f"Modelo K-Means não encontrado: {caminho}"
            )

        artefatos[estrato] = joblib.load(caminho)

    return artefatos


def criar_regressao_logistica(random_state=42):
    """Cria a linha de base linear com RobustScaler."""

    return Pipeline([

        # Mesmo escalonamento robusto do pipeline principal.
        ("scaler", RobustScaler()),

        # Linha de base linear simples.
        ("classificador", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ))
    ])


def avaliar_regressao_logistica(splits, random_state=42):
    """Treina e avalia a Regressão Logística no mesmo split."""

    resultados = {}

    for estrato in ESTRATOS:

        dados = splits[estrato]

        baseline = criar_regressao_logistica(random_state)

        # Mesmo treino e mesmo teste do Random Forest.
        baseline.fit(dados["X_train"], dados["y_train"])

        y_pred = baseline.predict(dados["X_test"])

        resultados[estrato] = {
            "accuracy_regressao_logistica": accuracy_score(
                dados["y_test"], y_pred
            ),
            "f1_macro_regressao_logistica": f1_score(
                dados["y_test"], y_pred,
                average="macro",
                zero_division=0,
            ),
        }

    return resultados


def avaliar_kmeans_direto(artefatos, splits, mapeamentos):
    """Reproduz o alvo com KMeans.predict direto nas entradas."""

    resultados = {}

    for estrato in ESTRATOS:

        artefato = artefatos[estrato]

        X_test = splits[estrato]["X_test"]
        y_test = splits[estrato]["y_test"]

        # Escalona as entradas com o scaler original do K-Means.
        X_scaled = artefato["scaler"].transform(
            X_test[artefato["features"]]
        )

        # Reatribui os clusters sem nenhum treinamento novo.
        clusters = artefato["modelo"].predict(X_scaled)

        # Converte o índice do cluster no perfil_id original.
        y_pred = [
            mapeamentos[estrato][int(cluster)]
            for cluster in clusters
        ]

        resultados[estrato] = {
            "concordancia_kmeans_direto": accuracy_score(
                y_test, y_pred
            ),
        }

    return resultados


def montar_tabela_comparativa(
    df_rf,
    resultados_regressao,
    resultados_kmeans,
    diretorio=DIRETORIO_RESULTADOS,
):
    """Reúne Random Forest e linhas de base lado a lado."""

    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)

    linhas = []

    for _, linha_rf in df_rf.iterrows():

        estrato = linha_rf["estrato"]

        linhas.append({
            **linha_rf.to_dict(),
            **resultados_regressao[estrato],
            **resultados_kmeans[estrato],
        })

    df_comparativo = pd.DataFrame(linhas)

    caminho_csv = diretorio / "metricas_classificacao.csv"

    df_comparativo.to_csv(
        caminho_csv,
        index=False,
    )

    print("\nCOMPARAÇÃO FINAL - RANDOM FOREST x LINHAS DE BASE")
    print(df_comparativo.round(4).to_string(index=False))

    print(
        "\nCOMO INTERPRETAR\n"
        "O alvo (perfil_id) foi gerado pelo próprio K-Means\n"
        "usando as mesmas variáveis de entrada. Métricas\n"
        "elevadas do Random Forest demonstram reprodução da\n"
        "partição, e não previsão de um dado novo. A\n"
        "concordância do K-Means direto e o desempenho da\n"
        "Regressão Logística evidenciam que o alvo deriva\n"
        "das entradas."
    )

    print(f"\nMétricas salvas em: {caminho_csv}")

    return df_comparativo


def executar_avaliacao():

    df = carregar_dados()

    dados_estratos = preparar_estratos(df)

    # Mapeia o índice de cada cluster ao perfil_id original.
    mapeamentos = {
        estrato: {
            int(cluster): perfil
            for cluster, perfil in zip(
                dados_estratos[estrato]["cluster"],
                dados_estratos[estrato]["perfil_id"],
            )
        }
        for estrato in ESTRATOS
    }

    splits = dividir_treino_teste(
        dados_estratos,
        test_size=0.20,
        random_state=42,
    )

    modelos = carregar_modelos()

    # Desempenho do Random Forest na reprodução da partição.
    df_rf = avaliar_modelos(modelos, splits)

    # Linha de base 1: Regressão Logística no mesmo split.
    resultados_regressao = avaliar_regressao_logistica(splits)

    # Linha de base 2: KMeans.predict direto nas entradas.
    artefatos = carregar_kmeans()

    resultados_kmeans = avaliar_kmeans_direto(
        artefatos,
        splits,
        mapeamentos,
    )

    # Tabela final com os modelos lado a lado.
    return montar_tabela_comparativa(
        df_rf,
        resultados_regressao,
        resultados_kmeans,
    )


if __name__ == "__main__":
    executar_avaliacao()
