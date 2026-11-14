"""
HealthSync / PISI3 - L5 - Integrante 2
Otimiza o limiar de classificacao usando SOMENTE o conjunto de treino.

Candidatos: Regressao Logistica + pesos, Regressao Logistica + SMOTENC,
SVM RBF + pesos. A meta experimental e maximizar a precision da classe 1
sujeita a recall >= 65% na validacao interna (nao e regra clinica).

Para estimar desempenho sem avaliar nos mesmos dados usados para escolher
limiares, utiliza validacao cruzada aninhada (3 folds externos x 3 internos).
Apos essa avaliacao, ajusta um limiar candidato usando OOF de 5 folds
sobre todo o treinamento; o teste reservado de 872 pessoas NAO e acessado.

Execucao a partir da raiz do repositorio:
    python -m classificacao_l5.otimizacao_limiar_l5
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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
from classificacao_l5.comparacao_pesos_l5 import criar_algoritmos_com_pesos
from classificacao_l5.comparacao_reamostragem_l5 import criar_pipeline as criar_pipeline_reamostragem


RECALL_MINIMO = 0.65  # Meta de estudo, nao parametro clinicamente validado
FOLDS_EXTERNOS = 3
FOLDS_INTERNOS = 3
FOLDS_LIMIAR_FINAL = 5


# =========================================================
# 1. CONFIGURACOES DOS MODELOS EXISTENTES
# =========================================================

def criar_candidatos():
    """Mantem os hiperparametros dos experimentos ja realizados."""
    modelos_pesos = criar_algoritmos_com_pesos()

    return {
        "regressao_logistica_pesos": criar_pipeline_comparacao(
            modelos_pesos["regressao_logistica"]
        ),
        "regressao_logistica_smotenc": criar_pipeline_reamostragem(
            "smotenc",
            LogisticRegression(max_iter=2000, random_state=SEMENTE),
        ),
        "svm_rbf_pesos": criar_pipeline_comparacao(
            modelos_pesos["svm_rbf"]
        ),
    }


def tipo_score(modelo):
    """Regressao Logistica usa probabilidade; SVM usa margem de decisao."""
    return "probabilidade" if hasattr(modelo, "predict_proba") else "margem_svm"


def pontuacoes(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return np.asarray(modelo.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(modelo.decision_function(X), dtype=float)


def pontuacoes_oof(modelo, X, y, cv):
    """Cada score vem de um modelo que nao treinou naquela observacao."""
    metodo = "predict_proba" if hasattr(modelo, "predict_proba") else "decision_function"
    resultados = cross_val_predict(
        clone(modelo),
        X,
        y,
        cv=cv,
        method=metodo,
        n_jobs=1,
    )
    if metodo == "predict_proba":
        return np.asarray(resultados[:, 1], dtype=float)
    return np.asarray(resultados, dtype=float)


def limiar_padrao(tipo):
    return 0.5 if tipo == "probabilidade" else 0.0


# =========================================================
# 2. ESCOLHA DE LIMIAR
# =========================================================

def avaliar_limiar(y, scores, limiar):
    pred = (scores >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "acuracia": float(accuracy_score(y, pred)),
        "acuracia_balanceada": float(balanced_accuracy_score(y, pred)),
        "vp": int(tp),
        "fp": int(fp),
        "vn": int(tn),
        "fn": int(fn),
        "positivos_previstos": int(pred.sum()),
    }


def escolher_limiar(y, scores, tipo, recall_minimo=RECALL_MINIMO):
    """Prioriza maior Precision entre limiares com Recall >= meta.

    Os limiares sao quantis dos scores OOF e incluem o ponto padrao.
    Assim funciona tanto para probabilidades quanto para margens SVM.
    """
    quantis = np.linspace(0, 1, 201)
    candidatos = np.unique(
        np.r_[np.quantile(scores, quantis), limiar_padrao(tipo)]
    )
    linhas = []

    for valor in candidatos:
        m = avaliar_limiar(y, scores, float(valor))
        linhas.append({
            "limiar": float(valor),
            "elegivel": bool(m["recall"] >= recall_minimo),
            **m,
        })

    curva = pd.DataFrame(linhas)
    elegiveis = curva.loc[curva["elegivel"]].copy()
    if elegiveis.empty:
        raise RuntimeError("Nenhum limiar atingiu o recall minimo. Verifique os scores.")

    # Maior Precision; desempate pelo Recall e depois pelo F1.
    escolhido = elegiveis.sort_values(
        ["precision", "recall", "f1", "limiar"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return float(escolhido["limiar"]), curva


# =========================================================
# 3. VALIDACAO CRUZADA ANINHADA
# =========================================================

def executar_otimizacao():
    df = carregar_dados()
    X_A, X_B, y, _ = preparar_dados_l5(df)
    splits = dividir_treino_teste(
        X_A, X_B, y, test_size=0.20, random_state=SEMENTE
    )
    X_train = splits["X_train_A"]
    y_train = splits["y_train"]

    if list(X_train.columns) != list(FEATURES_BASE):
        raise ValueError("As variaveis da versao A foram alteradas.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Indices de treino desalinhados.")
    if set(X_train.index) & set(splits["X_test_A"].index):
        raise AssertionError("Ha sobreposicao entre treino e teste!")

    candidatos = criar_candidatos()
    cv_externo = StratifiedKFold(
        n_splits=FOLDS_EXTERNOS, shuffle=True, random_state=SEMENTE
    )
    linhas_folds = []
    linhas_finais = []
    curvas_finais = []

    print("\n=== L5: OTIMIZACAO DE LIMIAR (SOMENTE TREINO) ===", flush=True)
    print(f"Treino: {len(y_train)} | Teste reservado: {len(splits['y_test'])}")
    print(f"Meta exploratoria: maior Precision com Recall >= {RECALL_MINIMO:.0%} na validacao interna")
    print(f"Validacao aninhada: {FOLDS_EXTERNOS} folds externos x {FOLDS_INTERNOS} internos")

    for nome, modelo in candidatos.items():
        tipo = tipo_score(modelo)
        print(f"\nModelo: {nome} | score: {tipo}", flush=True)

        for fold, (idx_dev, idx_val) in enumerate(cv_externo.split(X_train, y_train), start=1):
            X_dev, y_dev = X_train.iloc[idx_dev], y_train.iloc[idx_dev]
            X_val, y_val = X_train.iloc[idx_val], y_train.iloc[idx_val]

            cv_interno = StratifiedKFold(
                n_splits=FOLDS_INTERNOS,
                shuffle=True,
                random_state=SEMENTE + fold,
            )

            scores_internos = pontuacoes_oof(modelo, X_dev, y_dev, cv_interno)
            limiar, _ = escolher_limiar(y_dev, scores_internos, tipo)

            # Ajusta SOMENTE no treino externo; avalia em participantes nao vistos.
            ajustado = clone(modelo).fit(X_dev, y_dev)
            scores_val = pontuacoes(ajustado, X_val)
            metrica_ajustada = avaliar_limiar(y_val, scores_val, limiar)
            metrica_padrao = avaliar_limiar(
                y_val, scores_val, limiar_padrao(tipo)
            )

            linhas_folds.append({
                "modelo": nome,
                "tipo_score": tipo,
                "fold_externo": fold,
                "n_validacao_externa": len(y_val),
                "limiar_escolhido_no_treino": limiar,
                "limiar_padrao": limiar_padrao(tipo),
                "precision": metrica_ajustada["precision"],
                "recall": metrica_ajustada["recall"],
                "f1": metrica_ajustada["f1"],
                "acuracia": metrica_ajustada["acuracia"],
                "acuracia_balanceada": metrica_ajustada["acuracia_balanceada"],
                "precision_padrao": metrica_padrao["precision"],
                "recall_padrao": metrica_padrao["recall"],
                "f1_padrao": metrica_padrao["f1"],
                "auc_roc": float(roc_auc_score(y_val, scores_val)),
                "average_precision": float(average_precision_score(y_val, scores_val)),
                "vp": metrica_ajustada["vp"],
                "fp": metrica_ajustada["fp"],
                "vn": metrica_ajustada["vn"],
                "fn": metrica_ajustada["fn"],
            })
            print(
                f"  Fold externo {fold}: limiar={limiar:.4f} | "
                f"Precision={metrica_ajustada['precision']:.3f} "
                f"(padrao={metrica_padrao['precision']:.3f}) | "
                f"Recall={metrica_ajustada['recall']:.3f} "
                f"(padrao={metrica_padrao['recall']:.3f})",
                flush=True,
            )

        # Um limiar candidato para uso futuro, ajustado nos scores OOF
        # do TREINAMENTO inteiro; sem acesso aos 872 participantes do teste.
        cv_final = StratifiedKFold(
            n_splits=FOLDS_LIMIAR_FINAL, shuffle=True, random_state=SEMENTE
        )
        scores_final = pontuacoes_oof(modelo, X_train, y_train, cv_final)
        limiar_final, curva = escolher_limiar(y_train, scores_final, tipo)
        curva.insert(0, "modelo", nome)
        curva.insert(1, "tipo_score", tipo)
        curvas_finais.append(curva)
        linhas_finais.append({
            "modelo": nome,
            "tipo_score": tipo,
            "limiar_sugerido": limiar_final,
            "limiar_padrao": limiar_padrao(tipo),
            "precision_oof_ajuste": avaliar_limiar(y_train, scores_final, limiar_final)["precision"],
            "recall_oof_ajuste": avaliar_limiar(y_train, scores_final, limiar_final)["recall"],
        })

    folds = pd.DataFrame(linhas_folds)
    resumos = []
    colunas_medias = [
        "precision", "recall", "f1", "auc_roc", "average_precision",
        "acuracia", "acuracia_balanceada",
        "precision_padrao", "recall_padrao", "f1_padrao",
    ]
    for nome in candidatos:
        dados = folds.loc[folds["modelo"] == nome]
        registro = {"modelo": nome, "folds_externos": FOLDS_EXTERNOS}
        for coluna in colunas_medias:
            registro[f"cv_externa_{coluna}_media"] = float(dados[coluna].mean())
            registro[f"cv_externa_{coluna}_dp"] = float(dados[coluna].std(ddof=0))
        registro["folds_com_recall_abaixo_da_meta"] = int(
            (dados["recall"] < RECALL_MINIMO).sum()
        )
        resumos.append(registro)

    resumo = pd.DataFrame(resumos)
    finais = pd.DataFrame(linhas_finais)
    curvas = pd.concat(curvas_finais, ignore_index=True)

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    caminhos = {
        "resumo": PASTA_SAIDA / "otimizacao_limiar_resumo_l5.csv",
        "folds": PASTA_SAIDA / "otimizacao_limiar_folds_l5.csv",
        "limiares": PASTA_SAIDA / "otimizacao_limiar_final_treino_l5.csv",
        "curvas": PASTA_SAIDA / "otimizacao_limiar_candidatos_l5.csv",
        "metadados": PASTA_SAIDA / "metadados_otimizacao_limiar_l5.json",
    }
    resumo.to_csv(caminhos["resumo"], index=False, encoding="utf-8-sig")
    folds.to_csv(caminhos["folds"], index=False, encoding="utf-8-sig")
    finais.to_csv(caminhos["limiares"], index=False, encoding="utf-8-sig")
    curvas.to_csv(caminhos["curvas"], index=False, encoding="utf-8-sig")

    metadados = {
        "objetivo": "Maximizar Precision mantendo Recall minimo na validacao interna",
        "recall_minimo_exploratorio": RECALL_MINIMO,
        "populacao": "sem diabetes autorrelatado",
        "versao": "A",
        "features": list(FEATURES_BASE),
        "n_treino": len(y_train),
        "n_teste_reservado": len(splits["y_test"]),
        "teste_usado": False,
        "validacao": "CV aninhada estratificada: limiar no fold interno, avaliacao no fold externo",
        "folds_externos": FOLDS_EXTERNOS,
        "folds_internos": FOLDS_INTERNOS,
        "folds_limiar_final": FOLDS_LIMIAR_FINAL,
        "semente": SEMENTE,
        "modelos": list(candidatos),
        "score": {nome: tipo_score(modelo) for nome, modelo in candidatos.items()},
        "versao_sklearn": sklearn.__version__,
        "nota": (
            "Precision/Recall OOF do arquivo de limiares refletem o ajuste do limiar "
            "e podem ser otimistas. Para avaliar o efeito do ajuste, consultar "
            "as metricas de CV externa. Os thresholds SVM sao margens, "
            "NAO probabilidades. Nenhuma validacao clinica foi realizada."
        ),
    }
    caminhos["metadados"].write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== RESUMO: VALIDACAO EXTERNA ANINHADA ===")
    tabela = resumo[[
        "modelo", "cv_externa_precision_media", "cv_externa_recall_media",
        "cv_externa_f1_media", "cv_externa_auc_roc_media",
        "cv_externa_precision_padrao_media", "cv_externa_recall_padrao_media",
    ]]
    print(tabela.round(3).to_string(index=False))
    print("\n=== LIMIARES PROPOSTOS COM TODO O TREINAMENTO ===")
    print(finais.round(4).to_string(index=False))
    print("\nArquivos criados:")
    for caminho in caminhos.values():
        print(" ", caminho)
    print("\nConjunto de teste final e modelos anteriores PRESERVADOS.")
    return resumo


if __name__ == "__main__":
    executar_otimizacao()
