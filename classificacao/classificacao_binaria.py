"""
HealthSync - L5: classificação binária de HbA1c (< 5,7 contra >= 5,7).

Diferente da atribuição de subperfil, o alvo aqui NÃO é derivado das
variáveis de entrada: a HbA1c é um exame de laboratório e não entra como
entrada. Isso permite medir capacidade real de triagem.

Cuidados contra vazamento:
- hba1c, diabetes_informado, perfil_id e cluster nunca são entradas;
- imputação e escalonamento ficam dentro do Pipeline, ajustados só no treino
  (em cada dobra da validação cruzada e no holdout);
- o holdout 80/20 é estratificado e separado antes de qualquer ajuste.

Experimentos (cada um em duas populações):
  E1 essencial        idade, sexo, IMC, pressão sistólica/diastólica, colesterol
  E2 + estilo de vida + atividade moderada/vigorosa e tempo sedentário
  E3 + informadas     + hipertensão informada e colesterol alto informado
  E4 ampliada         E1 + estilo de vida + informadas
  E5 jejum sem glicose  (arquivo fasting) igual a E1, mesmas pessoas do E6
  E6 jejum com glicose  (arquivo fasting) E1 + glicose em jejum

Populações:
  todos            toda a base do arquivo
  sem_diagnostico  apenas diabetes_informado == "Nao" (triagem de quem
                   ainda não sabe que tem diabetes)

Uso (na raiz do repositório):
    python -m classificacao.classificacao_binaria
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RAIZ = Path(__file__).resolve().parents[1]
CSV_CORE = RAIZ / "dataset" / "healthsync_nhanes_2021_2023_core.csv"
CSV_JEJUM = RAIZ / "dataset" / "healthsync_nhanes_2021_2023_fasting.csv"
SAIDA = RAIZ / "modelos" / "classificacao_binaria"

LIMIAR_HBA1C = 5.7
SEMENTE = 42

BASE = ["idade", "sexo_homem", "imc", "pressao_sistolica",
        "pressao_diastolica", "colesterol_total"]
VIDA = ["atv_moderada_min_semana", "atv_vigorosa_min_semana",
        "minutos_sedentarios_dia"]
INFORMADAS = ["hipertensao_informada", "colesterol_alto_informado"]

# nome, arquivo, variáveis
EXPERIMENTOS = [
    ("E1 essencial", "core", BASE),
    ("E2 + estilo de vida", "core", BASE + VIDA),
    ("E3 + informadas", "core", BASE + INFORMADAS),
    ("E4 ampliada", "core", BASE + VIDA + INFORMADAS),
    ("E5 jejum sem glicose", "jejum", BASE),
    ("E6 jejum com glicose", "jejum", BASE + ["glicose_jejum"]),
]
POPULACOES = ["todos", "sem_diagnostico"]


def carregar(arquivo, populacao):
    df = pd.read_csv(CSV_CORE if arquivo == "core" else CSV_JEJUM,
                     encoding="utf-8-sig")
    df["sexo_homem"] = (df["sexo"] == "Homem").astype(float)
    if populacao == "sem_diagnostico":
        df = df[df["diabetes_informado"] == "Nao"]
    y = (df["hba1c"] >= LIMIAR_HBA1C).astype(int)
    return df.reset_index(drop=True), y.reset_index(drop=True)


def modelos():
    def pipe(clf):
        return Pipeline([
            ("imputador", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", clf),
        ])
    return {
        "Dummy (prevalência)": pipe(DummyClassifier(strategy="prior")),
        "Regressão Logística": pipe(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEMENTE)),
        "Random Forest": pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=SEMENTE)),
    }


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas, importancias, curvas = [], [], {}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMENTE)

    for populacao in POPULACOES:
        for nome_exp, arquivo, variaveis in EXPERIMENTOS:
            df, y = carregar(arquivo, populacao)
            X = df[variaveis]
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.20, stratify=y, random_state=SEMENTE)

            for nome_mod, modelo in modelos().items():
                # Validação cruzada repetida (ajusta tudo só no treino da dobra)
                res = cross_validate(
                    modelo, X, y, cv=cv, n_jobs=1,
                    scoring={"auc": "roc_auc", "ap": "average_precision"})

                # Holdout 80/20
                modelo.fit(Xtr, ytr)
                prob = modelo.predict_proba(Xte)[:, 1]
                pred = (prob >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()

                linhas.append({
                    "populacao": populacao,
                    "experimento": nome_exp,
                    "modelo": nome_mod,
                    "n": len(df),
                    "prevalencia_pos": round(y.mean(), 4),
                    "n_variaveis": len(variaveis),
                    "cv_auc_media": res["test_auc"].mean(),
                    "cv_auc_dp": res["test_auc"].std(),
                    "cv_ap_media": res["test_ap"].mean(),
                    "teste_auc": roc_auc_score(yte, prob),
                    "teste_ap": average_precision_score(yte, prob),
                    "teste_f1_macro": f1_score(yte, pred, average="macro"),
                    "teste_precisao_pos": precision_score(
                        yte, pred, zero_division=0),
                    "teste_recall_pos": recall_score(yte, pred),
                    "teste_especificidade": tn / (tn + fp),
                    "vp": tp, "fp": fp, "vn": tn, "fn": fn,
                })

                if nome_mod == "Random Forest":
                    curvas[(populacao, nome_exp)] = roc_curve(yte, prob)[:2]
                if nome_mod != "Dummy (prevalência)":
                    imp = permutation_importance(
                        modelo, Xte, yte, scoring="roc_auc",
                        n_repeats=10, random_state=SEMENTE)
                    for v, m, d in zip(variaveis, imp.importances_mean,
                                       imp.importances_std):
                        importancias.append({
                            "populacao": populacao, "experimento": nome_exp,
                            "modelo": nome_mod, "variavel": v,
                            "queda_auc_media": m, "queda_auc_dp": d})

    df_res = pd.DataFrame(linhas)
    df_res.to_csv(SAIDA / "metricas_binaria.csv", index=False)
    pd.DataFrame(importancias).to_csv(
        SAIDA / "importancia_variaveis.csv", index=False)

    # Curvas ROC (Random Forest, holdout)
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for eixo, populacao in zip(eixos, POPULACOES):
        for nome_exp, _, _ in EXPERIMENTOS:
            fpr, tpr = curvas[(populacao, nome_exp)]
            auc = df_res[(df_res.populacao == populacao)
                         & (df_res.experimento == nome_exp)
                         & (df_res.modelo == "Random Forest")].teste_auc.iloc[0]
            eixo.plot(fpr, tpr, label=f"{nome_exp} (AUC {auc:.3f})")
        eixo.plot([0, 1], [0, 1], "k--", lw=1)
        eixo.set_title(f"População: {populacao}")
        eixo.set_xlabel("Taxa de falsos positivos")
        eixo.legend(loc="lower right", fontsize=8)
    eixos[0].set_ylabel("Taxa de verdadeiros positivos")
    fig.suptitle("HbA1c >= 5,7 contra < 5,7 - Random Forest, conjunto de teste")
    fig.tight_layout()
    fig.savefig(SAIDA / "curvas_roc.png", dpi=150)
    plt.close(fig)

    colunas = ["populacao", "experimento", "modelo", "n", "cv_auc_media",
               "cv_auc_dp", "teste_auc", "teste_recall_pos",
               "teste_especificidade"]
    print(df_res[colunas].round(3).to_string(index=False))
    print(f"\nArquivos salvos em: {SAIDA}")


if __name__ == "__main__":
    main()