
"""HealthSync/PISI3 — HDBSCAN e outliers multivariados.

Execução na raiz do repositório:
    python modelos/clusterizacao/analise_hdbscan_outliers.py

Não altera os perfis do K-Means.
Requer scikit-learn >= 1.3.
"""

from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import chi2
from sklearn.cluster import HDBSCAN
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


PASTA = Path(__file__).resolve().parent
ORIGEM = PASTA / "healthsync_perfis_cardiometabolicos.csv"
SAIDA = PASTA / "analise_hdbscan_outliers"

FEATURES = [
    "idade",
    "imc",
    "pressao_sistolica",
    "pressao_diastolica",
    "colesterol_total",
]

ESTRATOS = {
    "E1": "HbA1c < 5.7",
    "E2": "5.7 <= HbA1c < 6.5",
    "E3": "HbA1c >= 6.5",
}

MIN_CLUSTER_SIZES = [15, 30, 50]
MIN_SAMPLES = [5, 10, 20]
SEMENTE = 42


def silhouette_sem_ruido(X, labels):
    """Calcula a silhueta somente dos pontos agrupados."""
    atribuidos = labels != -1
    grupos = np.unique(labels[atribuidos])
    n = int(atribuidos.sum())

    if len(grupos) < 2 or n <= len(grupos):
        return np.nan

    return float(
        silhouette_score(
            X[atribuidos],
            labels[atribuidos],
            sample_size=min(1000, n),
            random_state=SEMENTE,
        )
    )


def executar_hdbscan(X, codigo):
    """Testa diferentes configurações do HDBSCAN."""
    resultados = []
    rotulos = {}

    for tamanho in MIN_CLUSTER_SIZES:
        for vizinhos in MIN_SAMPLES:
            modelo = HDBSCAN(
                min_cluster_size=tamanho,
                min_samples=vizinhos,
                metric="euclidean",
                cluster_selection_method="eom",
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The default value of `copy`",
                    category=FutureWarning,
                )
                labels = modelo.fit_predict(X)

            n_grupos = len(set(labels) - {-1})
            pct_ruido = float(np.mean(labels == -1) * 100)
            sil = silhouette_sem_ruido(X, labels)

            chave = (tamanho, vizinhos)
            rotulos[chave] = labels

            resultados.append({
                "estrato": codigo,
                "min_cluster_size": tamanho,
                "min_samples": vizinhos,
                "n_clusters": n_grupos,
                "n_ruido": int((labels == -1).sum()),
                "ruido_pct": pct_ruido,
                "n_alocados": int((labels != -1).sum()),
                "silhouette_sem_ruido": sil,
            })

    tabela = pd.DataFrame(resultados)

    # Critério exploratório:
    # pelo menos dois grupos e no máximo 40% de ruído.
    elegiveis = tabela[
        (tabela["n_clusters"] >= 2)
        & (tabela["ruido_pct"] <= 40)
        & tabela["silhouette_sem_ruido"].notna()
    ]

    if len(elegiveis):
        linha = elegiveis.sort_values(
            ["silhouette_sem_ruido", "ruido_pct"],
            ascending=[False, True],
        ).iloc[0]

        criterio = (
            ">=2 grupos, <=40% de ruido; "
            "maior silhouette dos alocados"
        )
    else:
        # Se não houver configuração satisfatória,
        # registra a menor taxa de ruído encontrada.
        linha = tabela.sort_values(
            [
                "ruido_pct",
                "n_clusters",
                "min_cluster_size",
                "min_samples",
            ],
            ascending=[True, False, True, True],
        ).iloc[0]

        criterio = (
            "sem configuracao elegivel; "
            "menor percentual de ruido (exploratorio)"
        )

    chave = (
        int(linha["min_cluster_size"]),
        int(linha["min_samples"]),
    )

    return tabela, linha.to_dict(), rotulos[chave], criterio


def executar_outliers(X):
    """Compara Mahalanobis robusto e Isolation Forest."""

    # Mahalanobis robusto com estimativa de covariância.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
        )
        cov = MinCovDet(
            random_state=SEMENTE
        ).fit(X)

    dist2 = cov.mahalanobis(X)

    # Limite exploratório qui-quadrado a 97,5%.
    limite = float(
        chi2.ppf(0.975, df=X.shape[1])
    )

    out_maha = dist2 > limite

    # O parâmetro contamination determina
    # aproximadamente 3% de observações sinalizadas.
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=SEMENTE,
        n_jobs=-1,
    )

    out_iso = iso.fit_predict(X) == -1
    score_iso = iso.decision_function(X)

    return dist2, limite, out_maha, score_iso, out_iso


def figura_estrato(X, labels, maha, iso, codigo, pct_pca):
    """Visualiza HDBSCAN e outliers na projeção PCA."""

    xy = PCA(
        n_components=2,
        random_state=SEMENTE,
    ).fit_transform(X)

    fig, eixos = plt.subplots(
        1, 3, figsize=(16, 4.6),
        sharex=True, sharey=True,
    )

    grupos = sorted(set(labels) - {-1})

    # Primeiro gráfico: agrupamentos HDBSCAN.
    for grupo in grupos:
        mascara = labels == grupo

        eixos[0].scatter(
            xy[mascara, 0],
            xy[mascara, 1],
            s=9,
            alpha=0.55,
            label=f"Grupo {grupo}",
        )

    mascara_ruido = labels == -1

    if mascara_ruido.any():
        eixos[0].scatter(
            xy[mascara_ruido, 0],
            xy[mascara_ruido, 1],
            s=9,
            c="#969696",
            alpha=0.35,
            label="Ruido (-1)",
        )

    eixos[0].set_title(
        f"HDBSCAN: {len(grupos)} grupos; "
        f"ruido {mascara_ruido.mean():.0%}"
    )

    if len(grupos) <= 8:
        eixos[0].legend(fontsize=7, loc="best")

    # Segundo e terceiro gráficos: outliers.
    comparacoes = [
        (eixos[1], maha, "Mahalanobis robusto"),
        (eixos[2], iso, "Isolation Forest (3% definido)"),
    ]

    for eixo, selecionados, titulo in comparacoes:
        eixo.scatter(
            xy[~selecionados, 0],
            xy[~selecionados, 1],
            s=8,
            alpha=0.25,
            color="#0E766D",
        )

        eixo.scatter(
            xy[selecionados, 0],
            xy[selecionados, 1],
            s=13,
            alpha=0.85,
            color="#D94B57",
            label="Atipicos",
        )

        eixo.set_title(
            f"{titulo}: {selecionados.mean():.1%}"
        )
        eixo.legend(fontsize=8)

    for eixo in eixos:
        eixo.set_xlabel("PC1")
        eixo.grid(alpha=0.15)

    eixos[0].set_ylabel("PC2")

    fig.suptitle(
        f"{codigo} — PCA apenas para visualizacao "
        f"({pct_pca:.1%} em PC1+PC2)"
    )

    fig.tight_layout()

    fig.savefig(
        SAIDA / f"hdbscan_outliers_{codigo}.png",
        dpi=150,
    )
    plt.close(fig)


def main():
    if not ORIGEM.is_file():
        raise FileNotFoundError(
            f"Base nao encontrada: {ORIGEM}"
        )

    SAIDA.mkdir(parents=True, exist_ok=True)

    dados = pd.read_csv(
        ORIGEM, encoding="utf-8-sig"
    )

    exigidas = FEATURES + [
        "estrato_hba1c",
        "cluster",
        "perfil_id",
    ]

    faltantes = sorted(
        set(exigidas) - set(dados.columns)
    )

    if faltantes:
        raise ValueError(
            f"Colunas ausentes no CSV: {faltantes}"
        )

    grades = []
    resumo = []
    pessoas = []

    for codigo, nome in ESTRATOS.items():
        bloco = dados.loc[
            dados["estrato_hba1c"] == nome
        ].copy()

        if bloco.empty:
            raise ValueError(
                f"Sem dados no estrato {codigo}: {nome}"
            )

        if bloco[FEATURES].isna().any().any():
            raise ValueError(
                f"O estrato {codigo} possui "
                "valores ausentes nas 5 variaveis"
            )

        # Mesma estratégia de escala da clusterização.
        X = RobustScaler().fit_transform(
            bloco[FEATURES]
        )

        grade, melhor, labels, criterio = executar_hdbscan(
            X, codigo
        )
        grades.append(grade)

        dist2, limite, maha, score_iso, iso = (
            executar_outliers(X)
        )

        pca = PCA(
            n_components=2,
            random_state=SEMENTE,
        ).fit(X)

        var_pca = float(
            pca.explained_variance_ratio_.sum()
        )

        figura_estrato(
            X, labels, maha, iso, codigo, var_pca
        )

        resumo.append({
            "estrato": codigo,
            "n_participantes": len(bloco),
            "kmeans_grupos_existentes": int(
                bloco["cluster"].nunique()
            ),
            "hdbscan_grupos": int(
                melhor["n_clusters"]
            ),
            "hdbscan_min_cluster_size": int(
                melhor["min_cluster_size"]
            ),
            "hdbscan_min_samples": int(
                melhor["min_samples"]
            ),
            "hdbscan_ruido_pct": melhor["ruido_pct"],
            "hdbscan_silhouette_sem_ruido": melhor[
                "silhouette_sem_ruido"
            ],
            "criterio_selecao": criterio,
            "mahalanobis_limite_dist2": limite,
            "mahalanobis_atipicos_pct": float(
                maha.mean() * 100
            ),
            "isolation_atipicos_pct": float(
                iso.mean() * 100
            ),
            "concordancia_ambos_pct": float(
                (maha & iso).mean() * 100
            ),
            "pca_2_componentes_variancia_pct": (
                var_pca * 100
            ),
        })

        # Resultados individuais, sem modificar o CSV original.
        pessoas.append(pd.DataFrame({
            "linha_base_original": (
                bloco.index.to_numpy() + 2
            ),
            "estrato": codigo,
            "perfil_kmeans_existente": (
                bloco["perfil_id"].to_numpy()
            ),
            "grupo_hdbscan": labels,
            "ruido_hdbscan": labels == -1,
            "distancia_mahalanobis_quadrada": dist2,
            "atipico_mahalanobis": maha,
            "score_isolation_forest": score_iso,
            "atipico_isolation_forest": iso,
        }))

    # Salvar todos os resultados.
    pd.concat(
        grades, ignore_index=True
    ).to_csv(
        SAIDA / "grade_hdbscan.csv",
        index=False,
    )

    pd.DataFrame(resumo).to_csv(
        SAIDA / "resumo_por_estrato.csv",
        index=False,
    )

    pd.concat(
        pessoas, ignore_index=True
    ).to_csv(
        SAIDA / "classificacao_exploratoria.csv",
        index=False,
    )

    metodologia = {
        "base": ORIGEM.name,
        "semente": SEMENTE,
        "variaveis": FEATURES,
        "escalonamento": (
            "RobustScaler separado por estrato"
        ),
        "hdbscan_min_cluster_size": MIN_CLUSTER_SIZES,
        "hdbscan_min_samples": MIN_SAMPLES,
        "silhouette": (
            "somente pontos atribuidos; "
            "nao comparar diretamente a silhueta K-Means"
        ),
        "mahalanobis": (
            "MinCovDet; distancia quadrada > "
            "chi2.ppf(0.975, df=5)"
        ),
        "isolation_forest": (
            "contamination=0.03 "
            "(forcada por configuracao)"
        ),
        "aviso": (
            "Analises exploratorias; nenhuma observacao "
            "removida e K-Means preservado"
        ),
    }

    (SAIDA / "metodologia.json").write_text(
        json.dumps(
            metodologia,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    colunas = [
        "estrato",
        "n_participantes",
        "kmeans_grupos_existentes",
        "hdbscan_grupos",
        "hdbscan_ruido_pct",
        "mahalanobis_atipicos_pct",
        "isolation_atipicos_pct",
        "pca_2_componentes_variancia_pct",
    ]

    print("\nRESUMO HDBSCAN / OUTLIERS")
    print(
        pd.DataFrame(resumo)[colunas]
        .round(2)
        .to_string(index=False)
    )

    print(f"\nResultados salvos em: {SAIDA}")


if __name__ == "__main__":
    main()
