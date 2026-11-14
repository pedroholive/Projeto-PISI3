
"""
HealthSync - L5
Integrante 2: comparacao de algoritmos SEM balanceamento.

Execucao:
    python -m classificacao_l5.comparacao_algoritmos_l5

Usa somente o conjunto de treinamento.
Nao sobrescreve os modelos e resultados originais.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from classificacao_l5.preparacao_l5 import (
    FEATURES_BASE,
    carregar_dados,
    preparar_dados_l5,
    dividir_treino_teste,
)
from classificacao_l5.treinamento_l5 import criar_preprocessador


# =========================================================
# 1. CONFIGURACOES
# =========================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

PASTA_SAIDA = (
    RAIZ_PROJETO
    / "modelos"
    / "avaliacao_L5"
    / "comparacao_integrante2"
)

SEMENTE = 42
NUMERO_FOLDS = 5
VERSAO = "A"

METRICAS = {
    "auc_roc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "precisao": make_scorer(
        precision_score,
        zero_division=0,
    ),
    "recall": "recall",
    "acuracia": "accuracy",
    "acuracia_balanceada": "balanced_accuracy",
}


# =========================================================
# 2. ALGORITMOS SEM BALANCEAMENTO
# =========================================================

def criar_algoritmos():
    return {
        "dummy": DummyClassifier(
            strategy="prior",
        ),

        "regressao_logistica": LogisticRegression(
            max_iter=2000,
            random_state=SEMENTE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight=None,
            n_jobs=-1,
            random_state=SEMENTE,
        ),

        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight=None,
            random_state=SEMENTE,
        ),

        "hist_gradient_boosting":
            HistGradientBoostingClassifier(
                max_iter=100,
                early_stopping=False,
                random_state=SEMENTE,
            ),
    }


# =========================================================
# 3. PIPELINE
# =========================================================

def criar_pipeline_comparacao(estimador):

    # Reutiliza o preprocessamento oficial do L5.
    preprocessador = criar_preprocessador(VERSAO)

    # Saida densa para compatibilidade com boosting.
    preprocessador.set_params(
        sparse_threshold=0.0
    )

    return Pipeline([
        (
            "preprocessador",
            preprocessador,
        ),
        (
            "modelo",
            estimador,
        ),
    ])


# =========================================================
# 4. EXECUCAO DA COMPARACAO
# =========================================================

def executar_comparacao():

    df = carregar_dados()

    X_A, X_B, y, _ = preparar_dados_l5(df)

    splits = dividir_treino_teste(
        X_A,
        X_B,
        y,
        test_size=0.20,
        random_state=SEMENTE,
    )

    X_treino = splits["X_train_A"]
    y_treino = splits["y_train"]

    # Verificacoes de integridade
    if list(X_treino.columns) != list(FEATURES_BASE):
        raise ValueError(
            "Features diferentes da versao A oficial."
        )

    if not X_treino.index.equals(y_treino.index):
        raise ValueError(
            "Indices de X e y nao estao alinhados."
        )

    if set(X_treino.index) & set(
        splits["X_test_A"].index
    ):
        raise AssertionError(
            "Sobreposicao entre treino e teste."
        )

    if y_treino.nunique() != 2:
        raise ValueError(
            "Treinamento precisa conter ambas as classes."
        )

    # Validação cruzada estratificada
    cv = StratifiedKFold(
        n_splits=NUMERO_FOLDS,
        shuffle=True,
        random_state=SEMENTE,
    )

    linhas_resumo = []
    linhas_folds = []

    algoritmos = criar_algoritmos()

    print(
        "\n=== L5: COMPARACAO SEM BALANCEAMENTO ===",
        flush=True,
    )

    print(
        f"Versao: {VERSAO} | "
        f"Treino: {len(y_treino)} | "
        f"Teste reservado: {len(splits['y_test'])}",
        flush=True,
    )

    print(
        "Distribuicao treino:",
        y_treino.value_counts().sort_index().to_dict(),
        flush=True,
    )

    # -----------------------------------------------------
    # AVALIACAO DE CADA ALGORITMO
    # -----------------------------------------------------

    for nome, estimador in algoritmos.items():

        print(
            f"\nValidando {nome}...",
            flush=True,
        )

        pipeline = criar_pipeline_comparacao(
            estimador
        )

        resultados = cross_validate(
            pipeline,
            X_treino,
            y_treino,
            cv=cv,
            scoring=METRICAS,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )

        linha = {
            "versao": VERSAO,
            "algoritmo": nome,
            "balanceamento": "nenhum",
            "n_treino": len(y_treino),
            "n_folds": NUMERO_FOLDS,
        }

        # Medias, desvios e resultados por fold
        for metrica in METRICAS:

            valores = resultados[
                f"test_{metrica}"
            ]

            linha[
                f"cv_{metrica}_media"
            ] = float(np.mean(valores))

            linha[
                f"cv_{metrica}_desvio"
            ] = float(np.std(valores, ddof=0))

            for indice, valor in enumerate(
                valores,
                start=1,
            ):
                linhas_folds.append({
                    "versao": VERSAO,
                    "algoritmo": nome,
                    "balanceamento": "nenhum",
                    "fold": indice,
                    "metrica": metrica,
                    "valor": float(valor),
                })

        linhas_resumo.append(linha)

        print(
            f"  AUC={linha['cv_auc_roc_media']:.3f} | "
            f"AP={linha['cv_average_precision_media']:.3f} | "
            f"Precision={linha['cv_precisao_media']:.3f} | "
            f"Recall={linha['cv_recall_media']:.3f} | "
            f"F1={linha['cv_f1_media']:.3f}",
            flush=True,
        )

    # =====================================================
    # 5. EXPORTACAO DOS RESULTADOS
    # =====================================================

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumo = pd.DataFrame(
        linhas_resumo
    )

    folds = pd.DataFrame(
        linhas_folds
    )

    arquivo_resumo = (
        PASTA_SAIDA
        / "comparacao_sem_balanceamento_l5.csv"
    )

    arquivo_folds = (
        PASTA_SAIDA
        / "comparacao_sem_balanceamento_folds_l5.csv"
    )

    arquivo_meta = (
        PASTA_SAIDA
        / "metadados_comparacao_l5.json"
    )

    resumo.to_csv(
        arquivo_resumo,
        index=False,
        encoding="utf-8-sig",
    )

    folds.to_csv(
        arquivo_folds,
        index=False,
        encoding="utf-8-sig",
    )

    # Metadados para reprodutibilidade
    metadados = {
        "descricao": (
            "Comparacao L5 de algoritmos "
            "sem balanceamento"
        ),
        "populacao": (
            "diabetes_informado == 'Nao'"
        ),
        "alvo": (
            "Classe 0: HbA1c < 5.7; "
            "Classe 1: HbA1c >= 5.7"
        ),
        "versao_features": VERSAO,
        "features": list(FEATURES_BASE),
        "n_treino": len(y_treino),
        "n_teste_reservado": len(
            splits["y_test"]
        ),
        "teste_utilizado": False,
        "cv": "StratifiedKFold",
        "folds": NUMERO_FOLDS,
        "shuffle": True,
        "random_state": SEMENTE,
        "balanceamento": "nenhum",
        "algoritmos": list(algoritmos),
        "metricas": list(METRICAS),
        "limiar_predicao": (
            "padrao de cada estimador"
        ),
        "sklearn_versao": sklearn.__version__,
        "observacao": (
            "Resultados apenas da validacao cruzada "
            "no treinamento. Nao representam "
            "validacao clinica."
        ),
    }

    arquivo_meta.write_text(
        json.dumps(
            metadados,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # 6. EXIBICAO DOS RESULTADOS
    # =====================================================

    print(
        "\n=== RESUMO DA VALIDACAO CRUZADA ==="
    )

    colunas = [
        "algoritmo",
        "cv_auc_roc_media",
        "cv_average_precision_media",
        "cv_precisao_media",
        "cv_recall_media",
        "cv_f1_media",
    ]

    print(
        resumo[colunas]
        .round(4)
        .to_string(index=False)
    )

    print("\nArquivos gerados:")
    print(arquivo_resumo)
    print(arquivo_folds)
    print(arquivo_meta)

    print(
        "\nTeste preservado. "
        "Modelos originais nao alterados."
    )

    return resumo


if __name__ == "__main__":
    executar_comparacao()
