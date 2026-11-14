"""HealthSync L5 — Integrante 2: RandomOverSampler e SMOTENC.

Executar na raiz do repositorio:
    python -m classificacao_l5.comparacao_reamostragem_l5

Os dois metodos sao aplicados APENAS ao treino de cada fold.
Os 872 participantes reservados para teste nao sao utilizados.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as PipelineImb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.svm import SVC

from classificacao_l5.preparacao_l5 import (
    FEATURES_BASE, carregar_dados, preparar_dados_l5, dividir_treino_teste,
)
from classificacao_l5.comparacao_algoritmos_l5 import (
    METRICAS, NUMERO_FOLDS, PASTA_SAIDA, SEMENTE, VERSAO,
)

NUMERICAS = [
    "idade", "imc", "pressao_sistolica", "pressao_diastolica", "colesterol_total",
]


def criar_algoritmos():
    """Mesmas configuracoes dos algoritmos sem balanceamento."""
    return {
        "regressao_logistica": LogisticRegression(max_iter=2000, random_state=SEMENTE),
        "random_forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, class_weight=None,
            n_jobs=-1, random_state=SEMENTE,
        ),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=None),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, early_stopping=False, random_state=SEMENTE,
        ),
    }


def preparar_para_reamostragem():
    """Imputa dentro de cada fold; representa sexo como categoria inteira."""
    categorico = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("numericas", SimpleImputer(strategy="median"), NUMERICAS),
        ("sexo", categorico, ["sexo"]),
    ], sparse_threshold=0.0)


def preparar_depois_reamostragem():
    """Faz escala das numericas e one-hot do sexo apos a reamostragem."""
    return ColumnTransformer([
        ("numericas", RobustScaler(), [0, 1, 2, 3, 4]),
        ("sexo", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [5]),
    ], sparse_threshold=0.0)


def criar_pipeline(metodo, estimador):
    if metodo == "random_oversampler":
        amostrador = RandomOverSampler(random_state=SEMENTE)
    elif metodo == "smotenc":
        # A ultima coluna e sexo, portanto categorica. Nao interpolar 0 e 1!
        amostrador = SMOTENC(
            categorical_features=[5], k_neighbors=5, random_state=SEMENTE,
        )
    else:
        raise ValueError(f"Metodo de reamostragem desconhecido: {metodo}")

    return PipelineImb([
        ("pre_reamostragem", preparar_para_reamostragem()),
        ("reamostragem", amostrador),
        ("pos_reamostragem", preparar_depois_reamostragem()),
        ("modelo", estimador),
    ])


def executar_comparacao_reamostragem():
    if VERSAO != "A":
        raise ValueError("Este experimento utiliza a versao A.")

    df = carregar_dados()
    X_A, X_B, y, _ = preparar_dados_l5(df)
    splits = dividir_treino_teste(
        X_A, X_B, y, test_size=0.20, random_state=SEMENTE,
    )
    X_treino, y_treino = splits["X_train_A"], splits["y_train"]

    if list(X_treino.columns) != list(FEATURES_BASE):
        raise ValueError("As variaveis da versao A nao coincidem com a preparacao oficial.")
    if not X_treino.index.equals(y_treino.index):
        raise ValueError("Indices de X e y desalinhados.")
    if set(X_treino.index) & set(splits["X_test_A"].index):
        raise AssertionError("Ha indices presentes em treino e teste.")

    cv = StratifiedKFold(
        n_splits=NUMERO_FOLDS, shuffle=True, random_state=SEMENTE,
    )
    resumos, folds = [], []
    algoritmos = criar_algoritmos()
    metodos = ("random_oversampler", "smotenc")

    print("\n=== L5: REAMOSTRAGEM NO TREINAMENTO ===", flush=True)
    print(f"Treino: {len(y_treino)} | Teste reservado: {len(splits['y_test'])}", flush=True)
    print(f"Distribuicao original treino: {y_treino.value_counts().sort_index().to_dict()}")

    for metodo in metodos:
        for nome, estimador in algoritmos.items():
            print(f"\nValidando {nome} com {metodo}...", flush=True)
            modelo = criar_pipeline(metodo, estimador)
            resultado = cross_validate(
                modelo, X_treino, y_treino,
                scoring=METRICAS, cv=cv,
                n_jobs=1, error_score="raise", return_train_score=False,
            )
            linha = {
                "versao": VERSAO,
                "algoritmo": nome,
                "balanceamento": metodo,
                "n_treino": len(y_treino),
                "n_folds": NUMERO_FOLDS,
            }
            for metrica in METRICAS:
                valores = resultado[f"test_{metrica}"]
                linha[f"cv_{metrica}_media"] = float(np.mean(valores))
                linha[f"cv_{metrica}_desvio"] = float(np.std(valores, ddof=0))
                for i, valor in enumerate(valores, 1):
                    folds.append({
                        "versao": VERSAO, "algoritmo": nome,
                        "balanceamento": metodo, "fold": i,
                        "metrica": metrica, "valor": float(valor),
                    })
            resumos.append(linha)
            print(
                f" AUC={linha['cv_auc_roc_media']:.3f} | "
                f"AP={linha['cv_average_precision_media']:.3f} | "
                f"Precision={linha['cv_precisao_media']:.3f} | "
                f"Recall={linha['cv_recall_media']:.3f} | "
                f"F1={linha['cv_f1_media']:.3f}", flush=True,
            )

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    tabela = pd.DataFrame(resumos)
    tabela.to_csv(
        PASTA_SAIDA / "comparacao_reamostragem_l5.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(folds).to_csv(
        PASTA_SAIDA / "comparacao_reamostragem_folds_l5.csv",
        index=False, encoding="utf-8-sig",
    )

    metadados = {
        "descricao": "Comparacao entre RandomOverSampler e SMOTENC",
        "populacao": "diabetes_informado == 'Nao'",
        "versao": VERSAO, "features": list(FEATURES_BASE),
        "n_treino": len(y_treino),
        "n_teste_reservado": len(splits["y_test"]),
        "teste_utilizado": False,
        "folds": NUMERO_FOLDS, "random_state": SEMENTE,
        "metodos": list(metodos), "algoritmos": list(algoritmos),
        "observacao": (
            "Imputacao, reamostragem, codificacao e escalonamento ajustados "
            "apenas nos dados de treinamento de cada fold. "
            "O conjunto de validacao mantem a distribuicao original."
        ),
        "versao_imblearn": imblearn.__version__,
    }
    (PASTA_SAIDA / "metadados_reamostragem_l5.json").write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    anteriores = [
        PASTA_SAIDA / "comparacao_sem_balanceamento_l5.csv",
        PASTA_SAIDA / "comparacao_com_pesos_l5.csv",
    ]
    tabelas = [pd.read_csv(p) for p in anteriores if p.exists()]
    tabelas.append(tabela)
    resumo_geral = pd.concat(tabelas, ignore_index=True)
    colunas = [
        "algoritmo", "balanceamento", "cv_auc_roc_media",
        "cv_average_precision_media", "cv_precisao_media",
        "cv_recall_media", "cv_f1_media",
    ]
    resumo_geral = resumo_geral[colunas]
    resumo_geral.to_csv(
        PASTA_SAIDA / "comparacao_todos_balanceamentos_l5.csv",
        index=False, encoding="utf-8-sig",
    )
    print("\n=== COMPARACAO GERAL ===")
    print(resumo_geral.round(3).to_string(index=False))
    print(f"\nArquivos gerados: {PASTA_SAIDA}")
    print("Teste e modelos originais preservados.")
    return tabela


if __name__ == "__main__":
    executar_comparacao_reamostragem()
