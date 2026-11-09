
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
    """Avalia os classificadores nos conjuntos de teste."""

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
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(
                y_test, y_pred,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_test, y_pred,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_test, y_pred,
                average="macro",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0,
            ),
        }

        resultados.append(metricas)

        print(f"\n{'=' * 60}")
        print(f"AVALIAÇÃO DO ESTRATO {indice}")
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

    df_resultados = pd.DataFrame(resultados)

    caminho_csv = diretorio / "metricas_classificacao.csv"

    df_resultados.to_csv(
        caminho_csv,
        index=False,
    )

    print("\nCOMPARAÇÃO FINAL DOS MODELOS")
    print(df_resultados.round(4).to_string(index=False))

    return df_resultados


def executar_avaliacao():

    df = carregar_dados()

    dados_estratos = preparar_estratos(df)

    splits = dividir_treino_teste(
        dados_estratos,
        test_size=0.20,
        random_state=42,
    )

    modelos = carregar_modelos()

    return avaliar_modelos(modelos, splits)


if __name__ == "__main__":
    executar_avaliacao()
