"""
HealthSync — explicabilidade SHAP do E7 (HbA1c >= 5,7).

Explica a probabilidade da classe positiva prevista pela Random Forest.
Usa exatamente as variáveis, pipeline e holdout do classificacao_binaria.py.
O SHAP analisa observações do teste; não treina ou escolhe o modelo por elas.

Executar na raiz:
    pip install shap
    python -m classificacao.explicabilidade_shap
    python -m classificacao.explicabilidade_shap --populacao todos --amostras 100
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from classificacao.classificacao_binaria import (
    EXPERIMENTOS, RAIZ, SEMENTE, carregar, modelos,
)

PROIBIDAS = {"hba1c", "diabetes_informado", "perfil_id", "cluster"}


def preparar_e7(populacao):
    nome, arquivo, variaveis = next(
        (e for e in EXPERIMENTOS if e[0].startswith("E7 ")), None
    )
    vazamento = PROIBIDAS.intersection(variaveis)
    if vazamento:
        raise ValueError(f"Variáveis indevidas no E7: {sorted(vazamento)}")

    df, y = carregar(arquivo, populacao)
    X = df[variaveis]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEMENTE
    )
    return nome, variaveis, Xtr, Xte, ytr, yte


def extrair_classe_positiva(explicacao, n_amostras, n_variaveis):
    """Compatibilidade com versões de SHAP que retornam 1 ou 2 classes."""
    valores = np.asarray(explicacao.values)
    bases = np.asarray(explicacao.base_values)

    if valores.shape == (n_amostras, n_variaveis, 2):
        valores = valores[:, :, 1]
        if bases.shape == (n_amostras, 2):
            bases = bases[:, 1]
        elif bases.shape == (2,):
            bases = np.full(n_amostras, bases[1])
    elif valores.shape == (n_amostras, n_variaveis):
        if bases.shape == (n_amostras,):
            pass
        elif bases.ndim == 0:
            bases = np.full(n_amostras, float(bases))
        else:
            raise ValueError(f"Formato inesperado de valores-base: {bases.shape}")
    else:
        raise ValueError(f"Formato SHAP não suportado: {valores.shape}")

    if bases.shape != (n_amostras,):
        raise ValueError(f"Formato inesperado de valores-base: {bases.shape}")
    return valores, bases


def main():
    parser = argparse.ArgumentParser(description="SHAP para o experimento E7")
    parser.add_argument(
        "--populacao", choices=["sem_diagnostico", "todos"],
        default="sem_diagnostico",
    )
    parser.add_argument(
        "--amostras", type=int, default=120,
        help="Número de participantes do teste a explicar (padrão: 120)",
    )
    args = parser.parse_args()
    if args.amostras < 1:
        parser.error("--amostras precisa ser maior ou igual a 1")

    exp, variaveis, Xtr, Xte, ytr, yte = preparar_e7(args.populacao)
    modelo = modelos()["Random Forest"]
    modelo.fit(Xtr, ytr)

    # Seleção aleatória reproduzível, feita APÓS ajustar o modelo apenas no treino.
    rng = np.random.default_rng(SEMENTE)
    posicoes = np.sort(rng.choice(
        len(Xte), size=min(args.amostras, len(Xte)), replace=False
    ))
    Xamostra = Xte.iloc[posicoes].copy()
    Xtratado = modelo[:-1].transform(Xamostra)

    # O classificador recebe as variáveis já imputadas e escalonadas.
    # Para mostrar valores legíveis, os gráficos usam a escala original.
    explicador = shap.TreeExplainer(modelo.named_steps["clf"])
    explicacao = explicador(Xtratado)
    valores, bases = extrair_classe_positiva(
        explicacao, len(Xamostra), len(variaveis)
    )

    probabilidades = modelo.predict_proba(Xamostra)[:, 1]
    erro_max = float(np.max(np.abs(bases + valores.sum(axis=1) - probabilidades)))
    if erro_max > 1e-3:
        raise RuntimeError(
            f"SHAP não reproduziu a probabilidade da classe positiva "
            f"(erro máximo={erro_max:.5g}). Verifique a versão do SHAP."
        )

    saida = RAIZ / "modelos" / "explicabilidade_shap" / args.populacao
    saida.mkdir(parents=True, exist_ok=True)

    # Valores originais (podem incluir ausências, exibidas como NaN).
    explicacao_positiva = shap.Explanation(
        values=valores,
        base_values=bases,
        data=Xamostra.to_numpy(dtype=float),
        feature_names=variaveis,
    )

    # Ranking global: importância média absoluta, em pontos de probabilidade.
    ranking = pd.DataFrame({
        "variavel": variaveis,
        "shap_abs_medio": np.abs(valores).mean(axis=0),
        "shap_medio_sinalizado": valores.mean(axis=0),
    }).sort_values("shap_abs_medio", ascending=False)
    ranking.to_csv(saida / "importancia_shap.csv", index=False)

    plt.figure(figsize=(10, 6))
    ordem = ranking.iloc[::-1]
    plt.barh(ordem["variavel"], ordem["shap_abs_medio"])
    plt.xlabel("SHAP absoluto médio (probabilidade da classe positiva)")
    plt.title("E7 — importância global SHAP · Random Forest")
    plt.tight_layout()
    plt.savefig(saida / "shap_global.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(
        explicacao_positiva,
        max_display=min(len(variaveis), 15), show=False,
    )
    plt.title("E7 — impacto e direção das variáveis na previsão")
    plt.tight_layout()
    plt.savefig(saida / "shap_resumo.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Explicação de um caso de ALTA probabilidade prevista no subconjunto.
    # Isso não é diagnóstico nem exemplo aleatório representativo da população.
    posicao_local = int(np.argmax(probabilidades))
    shap.plots.waterfall(
        explicacao_positiva[posicao_local],
        max_display=min(len(variaveis), 15), show=False,
    )
    plt.title("E7 — exemplo individual (maior risco previsto na amostra)")
    plt.tight_layout()
    plt.savefig(saida / "shap_individual.png", dpi=160, bbox_inches="tight")
    plt.close()

    resumo = {
        "experimento": exp,
        "modelo": "Random Forest",
        "populacao": args.populacao,
        "alvo_positivo": "HbA1c >= 5,7",
        "n_treino": int(len(Xtr)),
        "n_teste": int(len(Xte)),
        "n_explicados": int(len(Xamostra)),
        "teste_auc": float(roc_auc_score(yte, modelo.predict_proba(Xte)[:, 1])),
        "erro_maximo_aditividade_shap": erro_max,
        "exemplo_individual_probabilidade": float(probabilidades[posicao_local]),
        "observacao": "Amostra do teste; explicações referem-se ao modelo, não a causalidade ou diagnóstico.",
    }
    (saida / "resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSHAP concluído:")
    print(ranking.head(8).to_string(index=False))
    print(f"\nAUC do E7 no holdout: {resumo['teste_auc']:.3f}")
    print(f"Participantes explicados: {len(Xamostra)}")
    print(f"Arquivos em: {saida}")


if __name__ == "__main__":
    main()
