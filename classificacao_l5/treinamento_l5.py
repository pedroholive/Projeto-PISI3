import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from classificacao_l5.preparacao_l5 import (
    FEATURES_BASE,
    FEATURES_AMPLIADAS,
    carregar_dados,
    preparar_dados_l5,
    dividir_treino_teste,
)


# =========================================================
# 1. CONFIGURAÇÕES
# =========================================================

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

PASTA_MODELOS = (
    RAIZ_PROJETO / "modelos" / "classificacao_L5"
)

PASTA_AVALIACAO = (
    RAIZ_PROJETO / "modelos" / "avaliacao_L5"
)

SEMENTE = 42
NUMERO_FOLDS = 5

METRICAS_CV = {
    "auc_roc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precisao": "precision",
    "acuracia": "accuracy",
    "acuracia_balanceada": "balanced_accuracy",
}


# =========================================================
# 2. PRÉ-PROCESSAMENTO
# =========================================================

def criar_preprocessador(versao):
    """Cria transformações ajustadas somente durante o fit."""

    if versao not in ("A", "B"):
        raise ValueError(
            "Versão inválida. Escolha 'A' ou 'B'."
        )

    colunas_numericas = [
        "idade",
        "imc",
        "pressao_sistolica",
        "pressao_diastolica",
        "colesterol_total",
    ]

    # Tratamento das variáveis numéricas.
    numerico = Pipeline([
        (
            "imputador",
            SimpleImputer(strategy="median"),
        ),
        (
            "escalador",
            RobustScaler(),
        ),
    ])

    # Tratamento da variável sexo.
    sexo = Pipeline([
        (
            "imputador",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "codificador",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    transformacoes = [
        (
            "numerico",
            numerico,
            colunas_numericas,
        ),
        (
            "sexo",
            sexo,
            ["sexo"],
        ),
    ]

    # Versão B: acrescenta hipertensão informada.
    if versao == "B":

        hipertensao = Pipeline([
            (
                "imputador",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "codificador",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ])

        transformacoes.append(
            (
                "hipertensao",
                hipertensao,
                ["hipertensao_informada"],
            )
        )

    return ColumnTransformer(
        transformers=transformacoes,
        remainder="drop",
    )


# =========================================================
# 3. CONSTRUÇÃO DOS PIPELINES
# =========================================================

def criar_pipeline(versao, algoritmo):
    """Cria o pipeline completo de cada algoritmo."""

    if algoritmo == "regressao_logistica":

        estimador = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEMENTE,
        )

    elif algoritmo == "random_forest":

        estimador = RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEMENTE,
            n_jobs=-1,
        )

    else:
        raise ValueError(
            f"Algoritmo desconhecido: {algoritmo}"
        )

    return Pipeline([
        (
            "preprocessador",
            criar_preprocessador(versao),
        ),
        (
            "modelo",
            estimador,
        ),
    ])


# =========================================================
# 4. TREINAMENTO E VALIDAÇÃO CRUZADA
# =========================================================

def treinar_modelos_l5():
    """Valida no treino e salva os quatro modelos ajustados."""

    # Carrega os dados originais.
    df = carregar_dados()

    # Prepara as duas versões.
    X_A, X_B, y, resumo = preparar_dados_l5(df)

    # Divide em treino e teste antes de ajustar qualquer
    # imputador, codificador ou scaler.
    splits = dividir_treino_teste(
        X_A,
        X_B,
        y,
        test_size=0.20,
        random_state=SEMENTE,
    )

    y_train = splits["y_train"]

    # Os mesmos folds serão utilizados nas quatro combinações.
    cv = StratifiedKFold(
        n_splits=NUMERO_FOLDS,
        shuffle=True,
        random_state=SEMENTE,
    )

    # Cria as pastas de saída.
    PASTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    PASTA_AVALIACAO.mkdir(
        parents=True,
        exist_ok=True,
    )

    resumo_cv = []
    resultados_folds = []
    artefatos = []

    # Executa as versões A e B.
    for versao in ("A", "B"):

        X_train = splits[f"X_train_{versao}"]

        features_esperadas = (
            FEATURES_BASE
            if versao == "A"
            else FEATURES_AMPLIADAS
        )

        # Confirma que nenhuma feature adicional foi incluída.
        if list(X_train.columns) != list(features_esperadas):
            raise ValueError(
                f"Features da versão {versao} "
                "diferentes das definidas na preparação."
            )

        # Compara Regressão Logística e Random Forest.
        for algoritmo in (
            "regressao_logistica",
            "random_forest",
        ):

            nome_execucao = (
                f"versao_{versao}_{algoritmo}"
            )

            print(
                f"\nValidando {nome_execucao} "
                "(somente treino)...",
                flush=True,
            )

            pipeline = criar_pipeline(
                versao,
                algoritmo,
            )

            # O pipeline completo é reajustado dentro
            # de cada fold. Não há vazamento do scaler.
            resultados = cross_validate(
                pipeline,
                X_train,
                y_train,
                scoring=METRICAS_CV,
                cv=cv,
                n_jobs=1,
                return_train_score=False,
                error_score="raise",
            )

            linha = {
                "versao": versao,
                "modelo": algoritmo,
            }

            # Calcula médias, desvios e valores por fold.
            for metrica in METRICAS_CV:

                valores = resultados[
                    f"test_{metrica}"
                ]

                linha[
                    f"cv_{metrica}_media"
                ] = float(valores.mean())

                linha[
                    f"cv_{metrica}_desvio"
                ] = float(valores.std(ddof=0))

                for indice_fold, valor in enumerate(
                    valores,
                    start=1,
                ):

                    resultados_folds.append({
                        "versao": versao,
                        "modelo": algoritmo,
                        "fold": indice_fold,
                        "metrica": metrica,
                        "valor": float(valor),
                    })

            resumo_cv.append(linha)

            # Treinamento definitivo somente no conjunto
            # de treinamento, após a validação cruzada.
            # O conjunto de teste permanece reservado.
            pipeline.fit(
                X_train,
                y_train,
            )

            arquivo = (
                PASTA_MODELOS
                / f"{nome_execucao}.joblib"
            )

            joblib.dump(
                pipeline,
                arquivo,
            )

            artefatos.append({
                "versao": versao,
                "modelo": algoritmo,
                "arquivo": arquivo.name,
                "features": list(X_train.columns),
            })

            print(
                "  AUC-ROC média CV: "
                f"{linha['cv_auc_roc_media']:.4f} "
                f"(± {linha['cv_auc_roc_desvio']:.4f})",
                flush=True,
            )

    # =====================================================
    # 5. EXPORTAÇÃO DAS MÉTRICAS
    # =====================================================

    tabela = pd.DataFrame(resumo_cv)

    tabela.to_csv(
        PASTA_AVALIACAO
        / "validacao_cruzada_l5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        resultados_folds
    ).to_csv(
        PASTA_AVALIACAO
        / "validacao_cruzada_folds_l5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =====================================================
    # 6. METADADOS DOS EXPERIMENTOS
    # =====================================================

    metadados = {
        "descricao": (
            "Triagem experimental da faixa de HbA1c; "
            "não é diagnóstico"
        ),
        "alvo": (
            "0: HbA1c < 5.7; 1: HbA1c >= 5.7"
        ),
        "filtro": (
            "diabetes_informado == 'Nao'"
        ),
        "n_participantes": int(len(y)),
        "n_treino": int(len(y_train)),
        "n_teste_reservado": int(
            len(splits["y_test"])
        ),
        "split_teste": 0.20,
        "random_state": SEMENTE,
        "cv_folds": NUMERO_FOLDS,
        "metricas": list(METRICAS_CV),
        "test_set_utilizado_na_avaliacao": False,
        "resumo_preparacao": {
            chave: int(valor)
            for chave, valor in resumo.items()
        },
        "artefatos": artefatos,
    }

    (
        PASTA_MODELOS
        / "metadados_l5.json"
    ).write_text(
        json.dumps(
            metadados,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # 7. RESUMO FINAL
    # =====================================================

    print(
        "\nTREINAMENTO L5 CONCLUÍDO — "
        "conjunto de teste preservado."
    )

    print(
        tabela[
            [
                "versao",
                "modelo",
                "cv_auc_roc_media",
                "cv_f1_media",
            ]
        ].to_string(index=False)
    )

    print(
        f"Modelos salvos em: {PASTA_MODELOS}"
    )

    print(
        "Resultados da validação cruzada: "
        f"{PASTA_AVALIACAO}"
    )

    return tabela


# =========================================================
# 8. EXECUÇÃO
# =========================================================

if __name__ == "__main__":
    treinar_modelos_l5()
