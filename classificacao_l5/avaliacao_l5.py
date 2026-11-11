
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from classificacao_l5.preparacao_l5 import (
    FEATURES_BASE,
    FEATURES_AMPLIADAS,
    carregar_dados,
    dividir_treino_teste,
    preparar_dados_l5,
)


# =========================================================
# 1. CONFIGURAÇÕES
# =========================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

PASTA_MODELOS = (
    RAIZ_PROJETO / "modelos" / "classificacao_L5"
)

PASTA_AVALIACAO = (
    RAIZ_PROJETO / "modelos" / "avaliacao_L5"
)

PASTA_MATRIZES = (
    PASTA_AVALIACAO / "matrizes_confusao"
)

PASTA_IMPORTANCIAS = (
    PASTA_AVALIACAO / "importancia_variaveis"
)

SEMENTE = 42

VERSOES = ("A", "B")

ALGORITMOS = (
    "regressao_logistica",
    "random_forest",
)

NOMES_ALGORITMOS = {
    "regressao_logistica": "Regressão Logística",
    "random_forest": "Random Forest",
}

NOMES_CLASSES = [
    "0 — HbA1c < 5,7%",
    "1 — HbA1c ≥ 5,7%",
]


# =========================================================
# 2. RECUPERAÇÃO DO CONJUNTO DE TESTE
# =========================================================

def recuperar_dados_teste():
    """Reconstrói o mesmo split utilizado no treinamento."""

    df = carregar_dados()

    X_A, X_B, y, resumo = preparar_dados_l5(df)

    splits = dividir_treino_teste(
        X_A,
        X_B,
        y,
        test_size=0.20,
        random_state=SEMENTE,
    )

    arquivo_metadados = (
        PASTA_MODELOS / "metadados_l5.json"
    )

    if not arquivo_metadados.is_file():
        raise FileNotFoundError(
            f"Metadados não encontrados: {arquivo_metadados}"
        )

    metadados = json.loads(
        arquivo_metadados.read_text(encoding="utf-8")
    )

    if metadados.get("random_state") != SEMENTE:
        raise ValueError(
            "A semente difere da utilizada no treinamento."
        )

    if (
        int(metadados.get("n_teste_reservado", -1))
        != len(splits["y_test"])
    ):
        raise ValueError(
            "A quantidade de participantes do teste mudou."
        )

    if (
        int(metadados.get("n_treino", -1))
        != len(splits["y_train"])
    ):
        raise ValueError(
            "A quantidade de participantes do treino mudou."
        )

    return splits, resumo


# =========================================================
# 3. CARREGAMENTO DOS MODELOS
# =========================================================

def carregar_modelo(versao, algoritmo, X_teste):
    """Carrega o pipeline salvo no treinamento L5."""

    caminho = (
        PASTA_MODELOS
        / f"versao_{versao}_{algoritmo}.joblib"
    )

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho}"
        )

    # Carregue somente modelos .joblib gerados pela
    # equipe e provenientes de uma fonte confiável.
    modelo = joblib.load(caminho)

    if not hasattr(modelo, "predict_proba"):
        raise TypeError(
            "O modelo não fornece probabilidades."
        )

    features_esperadas = (
        FEATURES_BASE
        if versao == "A"
        else FEATURES_AMPLIADAS
    )

    if list(X_teste.columns) != list(features_esperadas):
        raise ValueError(
            "As features do teste estão incorretas."
        )

    features_ajustadas = getattr(
        modelo,
        "feature_names_in_",
        None,
    )

    if features_ajustadas is not None:
        if list(features_ajustadas) != list(X_teste.columns):
            raise ValueError(
                "As features do modelo diferem das "
                "utilizadas no treinamento."
            )

    if list(modelo.classes_) != [0, 1]:
        raise ValueError(
            f"Classes inesperadas: {modelo.classes_}"
        )

    return modelo


# =========================================================
# 4. MATRIZ DE CONFUSÃO
# =========================================================

def salvar_matriz(y_verdadeiro, y_previsto, nome):
    """Exporta matriz de confusão em CSV e PNG."""

    matriz = confusion_matrix(
        y_verdadeiro,
        y_previsto,
        labels=[0, 1],
    )

    tabela = pd.DataFrame(
        matriz,
        index=["real_0", "real_1"],
        columns=["previsto_0", "previsto_1"],
    )

    tabela.to_csv(
        PASTA_MATRIZES / f"matriz_{nome}.csv",
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=NOMES_CLASSES,
    ).plot(
        ax=ax,
        values_format="d",
        colorbar=False,
        xticks_rotation=15,
    )

    ax.set_title(
        f"L5 — Matriz de confusão: {nome}"
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_MATRIZES / f"matriz_{nome}.png",
        dpi=160,
    )

    plt.close(fig)

    return matriz


# =========================================================
# 5. IMPORTÂNCIA DAS VARIÁVEIS
# =========================================================

def calcular_importancia(
    modelo,
    X_teste,
    y_teste,
    nome,
):
    """Calcula importância exploratória por permutação.

    Mede a redução da AUC quando uma variável é
    embaralhada, mantendo as outras inalteradas.

    Os valores não indicam causalidade e não devem
    orientar novo ajuste no mesmo conjunto de teste.
    """

    resultado = permutation_importance(
        modelo,
        X_teste,
        y_teste,
        scoring="roc_auc",
        n_repeats=5,
        random_state=SEMENTE,
        n_jobs=1,
    )

    tabela = pd.DataFrame({
        "modelo": nome,
        "variavel": X_teste.columns,
        "queda_auc_media": resultado.importances_mean,
        "queda_auc_desvio": resultado.importances_std,
    })

    tabela = tabela.sort_values(
        "queda_auc_media",
        ascending=False,
    )

    tabela.to_csv(
        PASTA_IMPORTANCIAS / f"importancia_{nome}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ordenada = tabela.sort_values(
        "queda_auc_media"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        ordenada["variavel"],
        ordenada["queda_auc_media"],
        xerr=ordenada["queda_auc_desvio"],
        capsize=3,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Queda média de AUC ao permutar a variável"
    )

    ax.set_title(
        f"L5 — Importância exploratória: {nome}"
    )

    fig.tight_layout()

    fig.savefig(
        PASTA_IMPORTANCIAS / f"importancia_{nome}.png",
        dpi=160,
    )

    plt.close(fig)

    return tabela


# =========================================================
# 6. AVALIAÇÃO FINAL
# =========================================================

def avaliar_modelos_l5():
    """Avalia os quatro modelos no teste reservado."""

    splits, resumo = recuperar_dados_teste()

    y_teste = splits["y_test"]

    if set(y_teste.unique()) != {0, 1}:
        raise ValueError(
            "O conjunto de teste precisa conter "
            "as classes 0 e 1."
        )

    # Criação das pastas de exportação.
    for pasta in (
        PASTA_AVALIACAO,
        PASTA_MATRIZES,
        PASTA_IMPORTANCIAS,
    ):
        pasta.mkdir(
            parents=True,
            exist_ok=True,
        )

    linhas = []
    pontos_roc = []
    importancias = []

    fig_roc, ax_roc = plt.subplots(
        figsize=(8, 6)
    )

    try:

        # Avalia ambas as versões.
        for versao in VERSOES:

            X_teste = splits[
                f"X_test_{versao}"
            ]

            # Avalia os dois algoritmos.
            for algoritmo in ALGORITMOS:

                nome = (
                    f"versao_{versao}_{algoritmo}"
                )

                modelo = carregar_modelo(
                    versao,
                    algoritmo,
                    X_teste,
                )

                # Limiar de decisão do modelo: 0,5.
                # Não é ajustado sobre os dados de teste.
                y_pred = modelo.predict(
                    X_teste
                )

                proba_classe_1 = (
                    modelo.predict_proba(X_teste)[:, 1]
                )

                # Matriz de confusão.
                matriz = salvar_matriz(
                    y_teste,
                    y_pred,
                    nome,
                )

                tn, fp, fn, tp = matriz.ravel()

                # AUC-ROC.
                auc = roc_auc_score(
                    y_teste,
                    proba_classe_1,
                )

                # Pontos da curva ROC.
                fpr, tpr, thresholds = roc_curve(
                    y_teste,
                    proba_classe_1,
                )

                # Armazena todas as métricas.
                linhas.append({
                    "versao": versao,
                    "modelo": algoritmo,

                    "participantes_teste": int(
                        len(y_teste)
                    ),

                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),

                    "auc_roc": float(auc),

                    "accuracy": float(
                        accuracy_score(
                            y_teste,
                            y_pred,
                        )
                    ),

                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            y_teste,
                            y_pred,
                        )
                    ),

                    "precision": float(
                        precision_score(
                            y_teste,
                            y_pred,
                            zero_division=0,
                        )
                    ),

                    "recall_sensibilidade": float(
                        recall_score(
                            y_teste,
                            y_pred,
                            zero_division=0,
                        )
                    ),

                    "especificidade": float(
                        tn / (tn + fp)
                    ),

                    "f1": float(
                        f1_score(
                            y_teste,
                            y_pred,
                            zero_division=0,
                        )
                    ),
                })

                # Curva ROC do modelo.
                ax_roc.plot(
                    fpr,
                    tpr,
                    label=(
                        f"Versão {versao} — "
                        f"{NOMES_ALGORITMOS[algoritmo]} "
                        f"(AUC={auc:.3f})"
                    ),
                )

                # Exporta pontos da curva.
                for x, y, limiar in zip(
                    fpr,
                    tpr,
                    thresholds,
                ):
                    pontos_roc.append({
                        "versao": versao,
                        "modelo": algoritmo,

                        "taxa_falsos_positivos": float(x),
                        "sensibilidade": float(y),
                        "limiar": float(limiar),
                    })

                # Importância das variáveis.
                importancias.append(
                    calcular_importancia(
                        modelo,
                        X_teste,
                        y_teste,
                        nome,
                    )
                )

                print(
                    f"{nome}: "
                    f"AUC={auc:.4f}, "
                    f"F1={linhas[-1]['f1']:.4f}"
                )

        # -------------------------------------------------
        # CURVA ROC COMPARATIVA
        # -------------------------------------------------

        ax_roc.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Acaso (AUC=0,5)",
        )

        ax_roc.set(
            xlabel=(
                "Taxa de falsos positivos "
                "(1 − especificidade)"
            ),
            ylabel="Sensibilidade",
            title="L5 — Curvas ROC no conjunto de teste",
            xlim=(0, 1),
            ylim=(0, 1),
        )

        ax_roc.legend(
            loc="lower right"
        )

        ax_roc.grid(alpha=0.2)

        fig_roc.tight_layout()

        fig_roc.savefig(
            PASTA_AVALIACAO / "curvas_roc_l5.png",
            dpi=160,
        )

    finally:
        plt.close(fig_roc)

    # =====================================================
    # 7. EXPORTAÇÃO DOS RESULTADOS
    # =====================================================

    metricas = pd.DataFrame(linhas)

    metricas.to_csv(
        PASTA_AVALIACAO / "metricas_teste_l5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        pontos_roc
    ).to_csv(
        PASTA_AVALIACAO / "curvas_roc_pontos_l5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        importancias,
        ignore_index=True,
    ).to_csv(
        PASTA_AVALIACAO / "importancia_variaveis_l5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # 8. RESUMO DA AVALIAÇÃO
    # =====================================================

    # A seleção do melhor modelo deve considerar a
    # validação cruzada de treino, não o teste final.
    caminho_cv = (
        PASTA_AVALIACAO / "validacao_cruzada_l5.csv"
    )

    if caminho_cv.is_file():

        cv = pd.read_csv(caminho_cv)

        melhor_cv = cv.sort_values(
            "cv_auc_roc_media",
            ascending=False,
        ).iloc[0]

        print(
            "\nMaior AUC na validação cruzada de treino: "
            f"versão {melhor_cv['versao']}, "
            f"{melhor_cv['modelo']} "
            f"(AUC média="
            f"{melhor_cv['cv_auc_roc_media']:.4f})"
        )

    print(
        "\nAVALIAÇÃO FINAL L5 — CONJUNTO DE TESTE"
    )

    print(
        metricas[
            [
                "versao",
                "modelo",
                "auc_roc",
                "recall_sensibilidade",
                "especificidade",
                "f1",
                "accuracy",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        f"\nResultados exportados em: "
        f"{PASTA_AVALIACAO}"
    )

    print(
        "Aviso: a tarefa estima a faixa de HbA1c, "
        "não confirma diagnóstico nem valida "
        "uso clínico no aplicativo."
    )

    return metricas


# =========================================================
# 9. EXECUÇÃO
# =========================================================

if __name__ == "__main__":
    avaliar_modelos_l5()
