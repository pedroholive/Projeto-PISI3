"""L5 | Integrante 2: compara quatro algoritmos COM pesos de classes.

Executar na raiz do repositorio:
    python -m classificacao_l5.comparacao_pesos_l5

Usa somente o treinamento e preserva resultados/modelos anteriores.
"""

import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate

from classificacao_l5.preparacao_l5 import (
    FEATURES_BASE, carregar_dados, preparar_dados_l5, dividir_treino_teste
)
from classificacao_l5.comparacao_algoritmos_l5 import (
    PASTA_SAIDA, SEMENTE, NUMERO_FOLDS, VERSAO, METRICAS,
    criar_pipeline_comparacao,
)


def criar_algoritmos_com_pesos():
    return {
        "regressao_logistica": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEMENTE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=SEMENTE
        ),
        "svm_rbf": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", random_state=SEMENTE
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, early_stopping=False,
            class_weight="balanced", random_state=SEMENTE
        ),
    }


def executar_comparacao_pesos():
    if VERSAO != "A":
        raise ValueError("Comparacao configurada para a versao A.")

    df = carregar_dados()
    X_A, X_B, y, _ = preparar_dados_l5(df)
    splits = dividir_treino_teste(
        X_A, X_B, y, test_size=0.20, random_state=SEMENTE
    )
    X_treino = splits["X_train_A"]
    y_treino = splits["y_train"]
    if list(X_treino.columns) != list(FEATURES_BASE):
        raise ValueError("As features da versao A foram alteradas.")
    if not X_treino.index.equals(y_treino.index):
        raise ValueError("Indices de X e y desalinhados.")
    if set(X_treino.index).intersection(splits["X_test_A"].index):
        raise AssertionError("Treino e teste com indices sobrepostos!")

    cv = StratifiedKFold(
        n_splits=NUMERO_FOLDS, shuffle=True, random_state=SEMENTE
    )
    linhas, linhas_folds = [], []
    algoritmos = criar_algoritmos_com_pesos()

    print("\n=== L5: BALANCEAMENTO POR PESOS ===", flush=True)
    print(f"Treino: {len(y_treino)} | Teste reservado: {len(splits['y_test'])}")

    for nome, estimador in algoritmos.items():
        print(f"\nValidando {nome}...", flush=True)
        pipeline = criar_pipeline_comparacao(estimador)
        resultado = cross_validate(
            pipeline, X_treino, y_treino,
            cv=cv, scoring=METRICAS, n_jobs=1,
            return_train_score=False, error_score="raise"
        )
        linha = {
            "versao": VERSAO, "algoritmo": nome,
            "balanceamento": "pesos_de_classes",
            "n_treino": len(y_treino), "n_folds": NUMERO_FOLDS,
        }
        for metrica in METRICAS:
            valores = resultado[f"test_{metrica}"]
            linha[f"cv_{metrica}_media"] = float(np.mean(valores))
            linha[f"cv_{metrica}_desvio"] = float(np.std(valores, ddof=0))
            for fold, valor in enumerate(valores, start=1):
                linhas_folds.append({
                    "versao": VERSAO, "algoritmo": nome,
                    "balanceamento": "pesos_de_classes",
                    "fold": fold, "metrica": metrica, "valor": float(valor)
                })
        linhas.append(linha)
        print(
            f"  AUC={linha['cv_auc_roc_media']:.3f} | "
            f"AP={linha['cv_average_precision_media']:.3f} | "
            f"Precision={linha['cv_precisao_media']:.3f} | "
            f"Recall={linha['cv_recall_media']:.3f} | "
            f"F1={linha['cv_f1_media']:.3f}", flush=True
        )

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(
        PASTA_SAIDA / "comparacao_com_pesos_l5.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(linhas_folds).to_csv(
        PASTA_SAIDA / "comparacao_com_pesos_folds_l5.csv",
        index=False, encoding="utf-8-sig"
    )
    metadados = {
        "descricao": "Versao A, balanceamento por pesos de classes",
        "features": list(FEATURES_BASE), "n_treino": len(y_treino),
        "n_teste_reservado": len(splits["y_test"]),
        "teste_utilizado": False, "cv_folds": NUMERO_FOLDS,
        "random_state": SEMENTE, "sklearn_versao": sklearn.__version__,
        "metricas": list(METRICAS), "algoritmos": list(algoritmos),
        "nota": "Os pesos sao ajustados dentro de cada fold, sem SMOTE."
    }
    (PASTA_SAIDA / "metadados_pesos_l5.json").write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sem_balanceamento = PASTA_SAIDA / "comparacao_sem_balanceamento_l5.csv"
    if sem_balanceamento.exists():
        comparacao = pd.concat(
            [pd.read_csv(sem_balanceamento), resumo], ignore_index=True
        )
        colunas = [
            "algoritmo", "balanceamento", "cv_auc_roc_media",
            "cv_average_precision_media", "cv_precisao_media",
            "cv_recall_media", "cv_f1_media"
        ]
        comparacao = comparacao[colunas]
        comparacao.to_csv(
            PASTA_SAIDA / "comparacao_pesos_vs_sem_balanceamento_l5.csv",
            index=False, encoding="utf-8-sig"
        )
        print("\n=== COMPARACAO DOS RESULTADOS ===")
        print(comparacao.round(3).to_string(index=False))
    else:
        print("\nSem arquivo anterior; apenas resultados com pesos salvos.")

    print(f"\nArquivos gerados em: {PASTA_SAIDA}")
    print("Teste reservado e modelos originais preservados.")
    return resumo


if __name__ == "__main__":
    executar_comparacao_pesos()
