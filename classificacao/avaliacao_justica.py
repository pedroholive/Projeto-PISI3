"""
HealthSync — auditoria exploratória de justiça do experimento E7.

Compara desempenho por sexo informado e faixas etárias no holdout 20%,
sem escolher limiares com base no teste. Usa os modelos já definidos em
classificacao_binaria.py. Diferenças não demonstram discriminação por si só.

Executar na raiz:
    python -m classificacao.avaliacao_justica
    python -m classificacao.avaliacao_justica --populacao todos
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from classificacao.classificacao_binaria import (
    EXPERIMENTOS, RAIZ, SEMENTE, carregar, modelos,
)

PROIBIDAS = {"hba1c", "diabetes_informado", "perfil_id", "cluster"}
LIMIAR_PROBABILIDADE = 0.5


def divisao_segura(a, b):
    return float(a / b) if b else float("nan")


def wilson_95(acertos, total):
    """Intervalo de Wilson aproximado (95%) para proporção binomial."""
    if total == 0:
        return float("nan"), float("nan")
    z = 1.96
    p = acertos / total
    den = 1 + z*z / total
    centro = (p + z*z / (2 * total)) / den
    margem = z * np.sqrt(p * (1-p) / total + z*z / (4*total*total)) / den
    return float(centro-margem), float(centro+margem)


def metricas_grupo(y, prob, limiar=0.5):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= limiar).astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    positivos, negativos = tp + fn, tn + fp
    ic_recall = wilson_95(tp, positivos)
    ic_fpr = wilson_95(fp, negativos)
    auc = float(roc_auc_score(y, prob)) if positivos and negativos else float("nan")

    return {
        "n": int(len(y)),
        "n_positivos_reais": positivos,
        "n_negativos_reais": negativos,
        "prevalencia_positiva": divisao_segura(positivos, len(y)),
        "proporcao_prevista_positiva": divisao_segura(tp+fp, len(y)),
        "precisao_positiva": divisao_segura(tp, tp+fp),
        "recall_sensibilidade": divisao_segura(tp, positivos),
        "recall_ic95_inf": ic_recall[0],
        "recall_ic95_sup": ic_recall[1],
        "especificidade": divisao_segura(tn, negativos),
        "taxa_falsos_positivos": divisao_segura(fp, negativos),
        "fpr_ic95_inf": ic_fpr[0],
        "fpr_ic95_sup": ic_fpr[1],
        "auc": auc,
        "vp": tp, "fp": fp, "vn": tn, "fn": fn,
    }


def medir_por_subgrupo(nome_modelo, yte, prob, grupos):
    resultados = []
    for eixo in ("sexo", "faixa_etaria"):
        for grupo in grupos[eixo].dropna().unique():
            mascara = grupos[eixo].eq(grupo).to_numpy()
            medidas = metricas_grupo(yte.to_numpy()[mascara], prob[mascara], LIMIAR_PROBABILIDADE)
            resultados.append({
                "modelo": nome_modelo,
                "eixo": eixo,
                "grupo": str(grupo),
                "limiar": LIMIAR_PROBABILIDADE,
                **medidas,
            })
    return resultados


def resumir_diferencas(df):
    linhas = []
    for (modelo, eixo), recorte in df.groupby(["modelo", "eixo"], sort=False):
        for metrica in ("recall_sensibilidade", "taxa_falsos_positivos", "precisao_positiva"):
            validos = recorte[metrica].dropna()
            linhas.append({
                "modelo": modelo,
                "eixo": eixo,
                "metrica": metrica,
                "gap_max_menos_min": float(validos.max()-validos.min()) if len(validos)>=2 else float("nan"),
                "n_grupos_validos": int(len(validos)),
            })
    return pd.DataFrame(linhas)


def grafico(df, modelo, saida):
    recorte = df[df["modelo"] == modelo].copy()
    recorte["rotulo"] = recorte["eixo"].replace({"sexo": "Sexo", "faixa_etaria": "Idade"}) + ": " + recorte["grupo"]
    nomes = recorte["rotulo"].tolist()
    pos = np.arange(len(nomes))
    largura = 0.36

    fig, eixo = plt.subplots(figsize=(max(9, len(nomes)*1.5), 5.5))
    eixo.bar(pos-largura/2, recorte["recall_sensibilidade"], largura, label="Sensibilidade (recall)")
    eixo.bar(pos+largura/2, recorte["taxa_falsos_positivos"], largura, label="Taxa de falsos positivos")
    eixo.set_ylim(0, 1)
    eixo.set_xticks(pos, nomes, rotation=20, ha="right")
    eixo.set_ylabel("Taxa no conjunto de teste")
    eixo.set_title(f"E7 — justiça por subgrupos · {modelo}")
    eixo.legend()
    fig.tight_layout()
    fig.savefig(saida, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Auditoria de justiça do E7")
    parser.add_argument(
        "--populacao", choices=["sem_diagnostico", "todos"],
        default="sem_diagnostico",
    )
    args = parser.parse_args()

    exp, arquivo, variaveis = next(
        (e for e in EXPERIMENTOS if e[0].startswith("E7 ")), None
    )
    vazamento = PROIBIDAS.intersection(variaveis)
    if vazamento:
        raise ValueError(f"Variáveis indevidas no E7: {sorted(vazamento)}")

    df, y = carregar(arquivo, args.populacao)
    X = df[variaveis]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEMENTE
    )

    # As categorias vêm do dataframe original; não são deduzidas das previsões.
    grupos = df.loc[Xte.index, ["sexo", "idade"]].copy()
    grupos["faixa_etaria"] = pd.cut(
        grupos["idade"], bins=[-np.inf, 40, 60, np.inf],
        labels=["< 40", "40–59", "60+"], right=False,
    ).astype("object")

    modelos_avaliados = ["Regressão Logística", "Random Forest"]
    linhas = []
    resumos_modelos = {}

    for nome in modelos_avaliados:
        modelo = modelos()[nome]
        modelo.fit(Xtr, ytr)
        prob = modelo.predict_proba(Xte)[:, 1]
        resumos_modelos[nome] = {
            "teste_auc": float(roc_auc_score(yte, prob)),
            **metricas_grupo(yte, prob, LIMIAR_PROBABILIDADE),
        }
        linhas += medir_por_subgrupo(nome, yte, prob, grupos)

    saida = RAIZ / "modelos" / "avaliacao_justica" / args.populacao
    saida.mkdir(parents=True, exist_ok=True)
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(saida / "justica_por_grupos.csv", index=False)
    gaps = resumir_diferencas(tabela)
    gaps.to_csv(saida / "diferencas_entre_grupos.csv", index=False)

    for nome in modelos_avaliados:
        identificador = "regressao_logistica" if nome == "Regressão Logística" else "random_forest"
        grafico(tabela, nome, saida / f"justica_{identificador}.png")

    resumo = {
        "experimento": exp,
        "populacao": args.populacao,
        "alvo_positivo": "HbA1c >= 5,7",
        "limiar_probabilidade": LIMIAR_PROBABILIDADE,
        "n_treino": int(len(Xtr)),
        "n_teste": int(len(Xte)),
        "modelos": resumos_modelos,
        "notas": [
            "Análise descritiva de justiça, não prova de ausência/presença de discriminação.",
            "Sexo é usado como preditor no E7; idade também. Avaliam-se diferenças de desempenho entre grupos.",
            "Intervalos de Wilson são aproximados e não incorporam desenho amostral complexo do NHANES.",
            "As métricas não foram ponderadas pelos pesos amostrais do NHANES.",
            "O limiar 0,5 é fixo; não foi otimizado usando o conjunto de teste.",
        ],
    }
    (saida / "resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    colunas = ["modelo", "eixo", "grupo", "n", "n_positivos_reais", "recall_sensibilidade", "precisao_positiva", "taxa_falsos_positivos", "auc"]
    print("\nJustiça do E7 no holdout (limiar 0,5):")
    print(tabela[colunas].round(3).to_string(index=False))
    print("\nMaiores diferenças entre grupos:")
    print(gaps.round(3).to_string(index=False))
    print(f"\nArquivos em: {saida}")


if __name__ == "__main__":
    main()
