"""HealthSync / PISI3 — L5 — Integrante 2

Compara configuracoes padrao e otimizadas de Regressao Logistica e SVM RBF
com pesos de classe. Ajusta hiperparametros por GridSearchCV e usa validacao
cruzada aninhada para estimar desempenho sem reutilizar a validacao interna.

IMPORTANTE:
- Apenas X_train_A / y_train entram nas buscas e avaliacoes.
- Os 872 participantes do holdout final NAO sao avaliados nem utilizados.
- Pipelines incluem toda imputacao, codificacao e escalonamento.
- Nao altera modelos .joblib nem metricas anteriores.
- Escolha dos hiperparametros: Average Precision (AP), sem depender de limiar.
- Precision/Recall/F1 usam o limiar padrao de cada classificador;
  se um modelo for escolhido, o limiar devera ser ajustado novamente.

Executar na raiz do projeto:
    python -m classificacao_l5.otimizacao_hiperparametros_l5
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from classificacao_l5.preparacao_l5 import (
    FEATURES_BASE,
    carregar_dados,
    dividir_treino_teste,
    preparar_dados_l5,
)
from classificacao_l5.comparacao_algoritmos_l5 import (
    PASTA_SAIDA,
    SEMENTE,
    criar_pipeline_comparacao,
)

FOLDS_EXTERNOS = 3
FOLDS_INTERNOS = 3


def configuracoes():
    """Cada grade inclui o parametro padrao para comparacao justa."""
    return {
        "regressao_logistica_pesos": {
            "pipeline": criar_pipeline_comparacao(
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEMENTE,
                )
            ),
            "grade": {
                "modelo__C": [0.05, 0.2, 1.0, 5.0],
            },
        },
        "svm_rbf_pesos": {
            "pipeline": criar_pipeline_comparacao(
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=SEMENTE,
                )
            ),
            "grade": {
                "modelo__C": [0.3, 1.0, 3.0],
                "modelo__gamma": ["scale", 0.03, 0.1],
            },
        },
    }


def obter_scores_e_predicoes(pipeline, X):
    """Usa probabilidade ou margem de decisao para metricas sem limiar."""
    pred = np.asarray(pipeline.predict(X), dtype=int)
    if hasattr(pipeline, "predict_proba"):
        score = pipeline.predict_proba(X)[:, 1]
    else:
        score = pipeline.decision_function(X)
    return np.asarray(score, dtype=float), pred


def avaliar(pipeline, X, y):
    score, pred = obter_scores_e_predicoes(pipeline, X)
    return {
        "average_precision": float(average_precision_score(y, score)),
        "auc_roc": float(roc_auc_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "acuracia": float(accuracy_score(y, pred)),
        "acuracia_balanceada": float(balanced_accuracy_score(y, pred)),
    }


def executar_otimizacao():
    df = carregar_dados()
    X_A, X_B, y, _ = preparar_dados_l5(df)
    splits = dividir_treino_teste(
        X_A, X_B, y, test_size=0.20, random_state=SEMENTE
    )
    X_treino, y_treino = splits["X_train_A"], splits["y_train"]

    if list(X_treino.columns) != list(FEATURES_BASE):
        raise ValueError("As variaveis da versao A foram modificadas.")
    if not X_treino.index.equals(y_treino.index):
        raise ValueError("Indices de X e y nao estao alinhados.")
    if set(X_treino.index).intersection(splits["X_test_A"].index):
        raise AssertionError("Sobreposicao entre treino e teste.")

    cv_externo = StratifiedKFold(
        n_splits=FOLDS_EXTERNOS, shuffle=True, random_state=SEMENTE
    )

    linhas_folds = []
    linhas_resumo = []
    parametros_finais = []
    candidatos = configuracoes()

    print("\n=== L5: OTIMIZACAO DE HIPERPARAMETROS (SOMENTE TREINO) ===", flush=True)
    print(f"Treino: {len(y_treino)} | Teste reservado: {len(splits['y_test'])}", flush=True)
    print("Otimizacao interna: Average Precision (AP)", flush=True)
    print(f"Validacao aninhada: {FOLDS_EXTERNOS} externos x {FOLDS_INTERNOS} internos", flush=True)

    for nome, config in candidatos.items():
        print(f"\nAlgoritmo: {nome}", flush=True)
        grade = config["grade"]
        pipeline_base = config["pipeline"]

        for fold, (idx_ajuste, idx_validacao) in enumerate(
            cv_externo.split(X_treino, y_treino), start=1
        ):
            X_ajuste = X_treino.iloc[idx_ajuste]
            y_ajuste = y_treino.iloc[idx_ajuste]
            X_validacao = X_treino.iloc[idx_validacao]
            y_validacao = y_treino.iloc[idx_validacao]

            cv_interno = StratifiedKFold(
                n_splits=FOLDS_INTERNOS,
                shuffle=True,
                random_state=SEMENTE + fold,
            )
            busca = GridSearchCV(
                estimator=clone(pipeline_base),
                param_grid=grade,
                scoring="average_precision",
                cv=cv_interno,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            busca.fit(X_ajuste, y_ajuste)

            # Comparacao base vs otimizado no MESMO fold externo.
            modelo_base = clone(pipeline_base).fit(X_ajuste, y_ajuste)
            metricas_base = avaliar(modelo_base, X_validacao, y_validacao)
            metricas_otim = avaliar(busca.best_estimator_, X_validacao, y_validacao)

            for tipo, metricas in (
                ("padrao", metricas_base),
                ("otimizado", metricas_otim),
            ):
                linhas_folds.append({
                    "algoritmo": nome,
                    "fold_externo": fold,
                    "configuracao": tipo,
                    "n_validacao": len(y_validacao),
                    "melhor_ap_interna": float(busca.best_score_) if tipo == "otimizado" else np.nan,
                    "melhores_parametros_internos": (
                        json.dumps(busca.best_params_, ensure_ascii=False)
                        if tipo == "otimizado" else "padrao"
                    ),
                    **metricas,
                })

            print(
                f"  Fold {fold}: parametros={busca.best_params_} | "
                f"AP padrao={metricas_base['average_precision']:.3f} "
                f"-> otimizado={metricas_otim['average_precision']:.3f} | "
                f"Recall={metricas_otim['recall']:.3f}",
                flush=True,
            )

        # Grades sao reajustadas somente no TREINAMENTO INTEIRO.
        busca_final = GridSearchCV(
            estimator=clone(pipeline_base),
            param_grid=grade,
            scoring="average_precision",
            cv=StratifiedKFold(
                n_splits=FOLDS_INTERNOS, shuffle=True, random_state=SEMENTE
            ),
            n_jobs=1,
            refit=False,  # Nao salva um modelo nem o ajusta no holdout.
            error_score="raise",
        )
        busca_final.fit(X_treino, y_treino)
        parametros_finais.append({
            "algoritmo": nome,
            "melhor_ap_cv_interna": float(busca_final.best_score_),
            "parametros_candidatos": json.dumps(
                busca_final.best_params_, ensure_ascii=False
            ),
        })

    folds = pd.DataFrame(linhas_folds)
    for (nome, tipo), dados in folds.groupby(["algoritmo", "configuracao"], sort=False):
        linha = {
            "algoritmo": nome,
            "configuracao": tipo,
            "n_folds_externos": FOLDS_EXTERNOS,
        }
        for metrica in (
            "average_precision", "auc_roc", "precision", "recall", "f1",
            "acuracia", "acuracia_balanceada"
        ):
            linha[f"cv_externa_{metrica}_media"] = float(dados[metrica].mean())
            linha[f"cv_externa_{metrica}_dp"] = float(dados[metrica].std(ddof=0))
        linhas_resumo.append(linha)

    resumo = pd.DataFrame(linhas_resumo)
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    arq_resumo = PASTA_SAIDA / "hiperparametros_resumo_l5.csv"
    arq_folds = PASTA_SAIDA / "hiperparametros_folds_l5.csv"
    arq_params = PASTA_SAIDA / "hiperparametros_candidatos_l5.csv"
    arq_meta = PASTA_SAIDA / "metadados_hiperparametros_l5.json"

    resumo.to_csv(arq_resumo, index=False, encoding="utf-8-sig")
    folds.to_csv(arq_folds, index=False, encoding="utf-8-sig")
    pd.DataFrame(parametros_finais).to_csv(arq_params, index=False, encoding="utf-8-sig")
    metadados = {
        "descricao": "Busca de hiperparametros com validacao cruzada aninhada; versao A",
        "modelos": list(candidatos),
        "n_treino": len(y_treino),
        "n_teste_reservado": len(splits["y_test"]),
        "teste_utilizado": False,
        "cv_externo": FOLDS_EXTERNOS,
        "cv_interno": FOLDS_INTERNOS,
        "scoring_busca": "average_precision",
        "random_state": SEMENTE,
        "sklearn_versao": sklearn.__version__,
        "criterio": "Melhora em AP de validacao; precision/recall usam limiar padrao",
        "observacao": (
            "Busca de parametros e comparacoes exploratorias. "
            "Se um modelo otimizado for escolhido, seu limiar deve ser "
            "reajustado com dados de treinamento. Nenhuma avaliacao clinica."
        ),
        "grades": {nome: config["grade"] for nome, config in candidatos.items()},
    }
    arq_meta.write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== RESUMO (VALIDACAO EXTERNA) ===")
    print(resumo[[
        "algoritmo", "configuracao", "cv_externa_average_precision_media",
        "cv_externa_auc_roc_media", "cv_externa_precision_media",
        "cv_externa_recall_media", "cv_externa_f1_media",
    ]].round(4).to_string(index=False))
    print("\n=== PARAMETROS CANDIDATOS (TODA A BASE DE TREINAMENTO) ===")
    print(pd.DataFrame(parametros_finais).to_string(index=False))
    print("\nArquivos gerados:")
    for arquivo in (arq_resumo, arq_folds, arq_params, arq_meta):
        print(arquivo)
    print("\nTeste final preservado. Nenhum modelo anterior foi sobrescrito.")
    return resumo


if __name__ == "__main__":
    executar_otimizacao()
