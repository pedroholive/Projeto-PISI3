"""
HealthSync / PISI3 - L5 - Integrante 2
Otimiza hiperparametros E limiar de decisao, sem usar o teste final.

Metodologia:
- Treinamento oficial (80%) e holdout (20%) preservados.
- Tres folds externos: estimativa de desempenho.
- Em cada fold externo, tres folds internos selecionam hiperparametros
  pela Average Precision.
- Com os hiperparametros selecionados, preditos OOF no treino externo
  escolhem limiar que maximiza Precision com Recall >= 65% (meta exploratoria).
- Desempenho avaliado no fold externo, nao usado para escolher limiar.
- Depois, o treinamento completo e usado para gerar DOIS modelos candidatos
  e limiares, SEM consultar o conjunto de teste.

Executar a partir da raiz do projeto:
    python -m classificacao_l5.otimizacao_conjunta_l5
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from classificacao_l5.preparacao_l5 import (
    FEATURES_BASE,
    carregar_dados,
    dividir_treino_teste,
    preparar_dados_l5,
)
from classificacao_l5.otimizacao_hiperparametros_l5 import configuracoes
from classificacao_l5.otimizacao_limiar_l5 import (
    avaliar_limiar,
    escolher_limiar,
    limiar_padrao,
    pontuacoes,
    pontuacoes_oof,
    tipo_score,
)
from classificacao_l5.comparacao_algoritmos_l5 import PASTA_SAIDA, SEMENTE

RAIZ = Path(__file__).resolve().parents[1]
PASTA_MODELOS = RAIZ / "modelos" / "classificacao_L5" / "candidatos_integrante2"
RECALL_MINIMO = 0.65  # Objetivo exploratorio, nao requisito clinico
FOLDS_EXTERNOS = 3
FOLDS_INTERNOS = 3
FOLDS_LIMIAR_FINAL = 3


def criar_cv(n_splits, seed):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def nova_busca(pipeline, grade, cv):
    return GridSearchCV(
        estimator=clone(pipeline),
        param_grid=grade,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )


def resumo_por_modelo(df):
    resultados = []
    metricas = (
        "precision", "recall", "f1", "acuracia", "acuracia_balanceada",
        "average_precision", "auc_roc", "precision_padrao", "recall_padrao",
        "f1_padrao",
    )
    for nome, grupo in df.groupby("modelo", sort=False):
        linha = {
            "modelo": nome,
            "n_folds_externos": len(grupo),
            "folds_recall_abaixo_meta": int((grupo["recall"] < RECALL_MINIMO).sum()),
        }
        for metrica in metricas:
            linha[f"cv_externa_{metrica}_media"] = float(grupo[metrica].mean())
            linha[f"cv_externa_{metrica}_dp"] = float(grupo[metrica].std(ddof=0))
        resultados.append(linha)
    return pd.DataFrame(resultados)


def executar_otimizacao_conjunta():
    df = carregar_dados()
    X_A, X_B, y, _ = preparar_dados_l5(df)
    divisao = dividir_treino_teste(X_A, X_B, y, test_size=0.20, random_state=SEMENTE)
    X_treino, y_treino = divisao["X_train_A"], divisao["y_train"]

    if list(X_treino.columns) != list(FEATURES_BASE):
        raise ValueError("Variaveis da versao A nao correspondem a preparacao oficial.")
    if not X_treino.index.equals(y_treino.index):
        raise ValueError("Indices de X_treino e y_treino nao correspondem.")
    if set(X_treino.index) & set(divisao["X_test_A"].index):
        raise AssertionError("Sobreposicao entre indices de treino e teste.")
    if y_treino.nunique() != 2:
        raise ValueError("O treinamento precisa conter as duas classes.")

    candidatos = configuracoes()
    cv_externo = criar_cv(FOLDS_EXTERNOS, SEMENTE)
    linhas_folds = []
    candidatos_finais = []

    print("\n=== L5: HIPERPARAMETROS + LIMIAR (SOMENTE TREINAMENTO) ===", flush=True)
    print(f"Treino: {len(y_treino)} | Teste reservado: {len(divisao['y_test'])}", flush=True)
    print(f"Meta exploratoria: maior Precision com Recall >= {RECALL_MINIMO:.0%} na validacao interna", flush=True)

    for nome, cfg in candidatos.items():
        base = cfg["pipeline"]
        grade = cfg["grade"]
        print(f"\nModelo: {nome}", flush=True)

        for fold, (idx_desenv, idx_validacao) in enumerate(
            cv_externo.split(X_treino, y_treino), start=1
        ):
            X_dev, y_dev = X_treino.iloc[idx_desenv], y_treino.iloc[idx_desenv]
            X_val, y_val = X_treino.iloc[idx_validacao], y_treino.iloc[idx_validacao]

            cv_interno = criar_cv(FOLDS_INTERNOS, SEMENTE + fold)
            busca = nova_busca(base, grade, cv_interno)
            busca.fit(X_dev, y_dev)

            # Hiperparametros aprendidos somente no treino externo.
            melhor = busca.best_estimator_
            tipo = tipo_score(melhor)
            scores_oof = pontuacoes_oof(melhor, X_dev, y_dev, cv_interno)
            limiar, _ = escolher_limiar(
                y_dev, scores_oof, tipo, recall_minimo=RECALL_MINIMO
            )

            # O fold externo so e consultado DEPOIS das escolhas.
            scores_val = pontuacoes(melhor, X_val)
            novo = avaliar_limiar(y_val, scores_val, limiar)
            padrao = avaliar_limiar(y_val, scores_val, limiar_padrao(tipo))

            linhas_folds.append({
                "modelo": nome,
                "fold_externo": fold,
                "n_validacao": len(y_val),
                "hiperparametros": json.dumps(busca.best_params_, ensure_ascii=False),
                "ap_selecao_interna": float(busca.best_score_),
                "tipo_score": tipo,
                "limiar_escolhido": float(limiar),
                "limiar_padrao": limiar_padrao(tipo),
                "average_precision": float(average_precision_score(y_val, scores_val)),
                "auc_roc": float(roc_auc_score(y_val, scores_val)),
                "precision": novo["precision"],
                "recall": novo["recall"],
                "f1": novo["f1"],
                "acuracia": novo["acuracia"],
                "acuracia_balanceada": novo["acuracia_balanceada"],
                "precision_padrao": padrao["precision"],
                "recall_padrao": padrao["recall"],
                "f1_padrao": padrao["f1"],
                "vp": novo["vp"], "fp": novo["fp"],
                "vn": novo["vn"], "fn": novo["fn"],
            })
            print(
                f" Fold {fold}: parametros={busca.best_params_} | "
                f"limiar={limiar:.4f} | Precision={novo['precision']:.3f} "
                f"Recall={novo['recall']:.3f} F1={novo['f1']:.3f}",
                flush=True,
            )

        # Reutiliza os parametros candidatos ja selecionados na etapa
        # anterior de GridSearchCV, SEM repetir a busca completa.
        # A validacao externa acima selecionou parametros novamente dentro
        # de cada fold, preservando a estimativa independente da nova etapa.
        parametros_candidatos = {
            "regressao_logistica_pesos": {"modelo__C": 0.05},
            "svm_rbf_pesos": {"modelo__C": 0.3, "modelo__gamma": 0.03},
        }
        parametros = parametros_candidatos[nome]
        modelo_final = clone(base).set_params(**parametros)
        tipo_final = tipo_score(modelo_final)
        cv_limiar_final = criar_cv(FOLDS_LIMIAR_FINAL, SEMENTE)
        scores_oof_finais = pontuacoes_oof(
            modelo_final, X_treino, y_treino, cv_limiar_final
        )
        limiar_final, _ = escolher_limiar(
            y_treino, scores_oof_finais, tipo_final, recall_minimo=RECALL_MINIMO
        )
        # Ajusta o candidato definitivo apenas nos 3488 registros de treino.
        modelo_final.fit(X_treino, y_treino)

        PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
        arq_modelo = PASTA_MODELOS / f"{nome}_otimizado.joblib"
        joblib.dump(modelo_final, arq_modelo)
        candidatos_finais.append({
            "modelo": nome,
            "parametros": json.dumps(parametros, ensure_ascii=False),
            "origem_parametros": "otimizacao_hiperparametros_l5",
            "tipo_score": tipo_final,
            "limiar_candidato": float(limiar_final),
            "precision_oof_ajuste": float(
                avaliar_limiar(y_treino, scores_oof_finais, limiar_final)["precision"]
            ),
            "recall_oof_ajuste": float(
                avaliar_limiar(y_treino, scores_oof_finais, limiar_final)["recall"]
            ),
            "arquivo_joblib": str(arq_modelo.relative_to(RAIZ)),
        })

    folds = pd.DataFrame(linhas_folds)
    resumo = resumo_por_modelo(folds)
    finais = pd.DataFrame(candidatos_finais)
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    arquivos = {
        "folds": PASTA_SAIDA / "otimizacao_conjunta_folds_l5.csv",
        "resumo": PASTA_SAIDA / "otimizacao_conjunta_resumo_l5.csv",
        "candidatos": PASTA_SAIDA / "otimizacao_conjunta_candidatos_l5.csv",
        "metadados": PASTA_SAIDA / "metadados_otimizacao_conjunta_l5.json",
    }
    folds.to_csv(arquivos["folds"], index=False, encoding="utf-8-sig")
    resumo.to_csv(arquivos["resumo"], index=False, encoding="utf-8-sig")
    finais.to_csv(arquivos["candidatos"], index=False, encoding="utf-8-sig")

    metadados = {
        "descricao": "Selecao de hiperparametros e limiar em etapas, com avaliacao externa",
        "n_treino": int(len(y_treino)),
        "n_teste_reservado": int(len(divisao["y_test"])),
        "teste_utilizado": False,
        "feature_set": "A",
        "features": list(FEATURES_BASE),
        "n_folds_externos": FOLDS_EXTERNOS,
        "n_folds_internos": FOLDS_INTERNOS,
        "n_folds_limiar_final": FOLDS_LIMIAR_FINAL,
        "semente": SEMENTE,
        "objetivo_busca": "average_precision",
        "objetivo_limiar": "max_precision_com_recall_minimo_em_oof",
        "recall_minimo_exploratorio": RECALL_MINIMO,
        "versao_sklearn": sklearn.__version__,
        "observacoes": [
            "SVM utiliza margem de decisao; limiares nao sao probabilidades.",
            "Parametros finais foram reutilizados da busca anterior realizada somente no treino.",
            "Precisao e Recall OOF do ajuste nao sao metricas de avaliacao independente.",
            "A meta de Recall aplicada internamente nao e garantida nos folds externos.",
            "Comparar/selecionar candidatos por CV e realizar avaliacao final uma unica vez.",
            "Os modelos joblib foram ajustados somente com os 3488 exemplos de treinamento.",
        ],
    }
    arquivos["metadados"].write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== RESULTADOS DE VALIDACAO EXTERNA ===")
    print(resumo[[
        "modelo", "cv_externa_precision_media", "cv_externa_recall_media",
        "cv_externa_f1_media", "cv_externa_average_precision_media",
        "cv_externa_auc_roc_media", "folds_recall_abaixo_meta",
    ]].round(4).to_string(index=False))
    print("\n=== MODELOS/ LIMIARES CANDIDATOS (TREINAMENTO) ===")
    print(finais[["modelo", "parametros", "tipo_score", "limiar_candidato"]].to_string(index=False))
    print("\nArquivos criados:")
    for caminho in arquivos.values():
        print(caminho)
    print("\nModelos candidatos em:", PASTA_MODELOS)
    print("Teste de 872 participantes: PRESERVADO.")
    return resumo


if __name__ == "__main__":
    executar_otimizacao_conjunta()
