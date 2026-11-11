import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from classificacao_l5.preparacao_l5 import (
    FEATURES_BASE,
    FEATURES_AMPLIADAS,
    carregar_dados,
    preparar_dados_l5,
    dividir_treino_teste,
)

from classificacao_l5.treinamento_l5 import (
    criar_pipeline,
    NUMERO_FOLDS,
    SEMENTE,
)


# =========================================================
# 1. CONFIGURAÇÕES
# =========================================================

RAIZ = Path(__file__).resolve().parents[1]

MODELOS = RAIZ / "modelos" / "classificacao_L5"
AVALIACAO = RAIZ / "modelos" / "avaliacao_L5"

COMBINACOES = [
    (versao, algoritmo)
    for versao in ("A", "B")
    for algoritmo in ("regressao_logistica", "random_forest")
]

PROIBIDAS = {
    "hba1c",
    "hba1c_binaria",
    "perfil_id",
    "diabetes_informado",
    "id_participante",
    "glicose_jejum",
    "estrato_hba1c",
}


# =========================================================
# 2. FIXTURES
# =========================================================

@pytest.fixture(scope="module")
def dados():
    """Carrega e prepara a base uma única vez."""
    return preparar_dados_l5(carregar_dados())


@pytest.fixture(scope="module")
def splits(dados):
    """Reconstrói o mesmo split do treinamento."""
    x_a, x_b, y, _ = dados

    return dividir_treino_teste(
        x_a,
        x_b,
        y,
        test_size=0.20,
        random_state=42,
    )


# =========================================================
# 3. TESTES DA PREPARAÇÃO
# =========================================================

def test_filtro_e_limiar_com_exemplo_controlado():
    """Confere o corte 5,7 e o filtro por diabetes informado."""

    base = {
        "idade": 45,
        "imc": 25.0,
        "pressao_sistolica": 120,
        "pressao_diastolica": 80,
        "colesterol_total": 185,
        "sexo": "Feminino",
        "hipertensao_informada": 0,
        "diabetes_informado": "Nao",
        "perfil_id": "cluster_fake",
        "id_participante": 123,
    }

    dados_teste = pd.DataFrame([
        {
            **base,
            "hba1c": 5.69,
            "hipertensao_informada": np.nan,
        },
        {**base, "hba1c": 5.70},
        {**base, "hba1c": 6.10},
        {
            **base,
            "hba1c": 5.30,
            "diabetes_informado": "Sim",
        },
        {
            **base,
            "hba1c": 6.50,
            "diabetes_informado": "Borderline",
        },
        {**base, "hba1c": np.nan},
    ])

    x_a, x_b, y, resumo = preparar_dados_l5(
        dados_teste
    )

    assert y.tolist() == [0, 1, 1]
    assert len(x_a) == len(x_b) == 3

    assert (
        resumo["participantes_sem_diabetes_informado"]
        == 4
    )

    assert (
        resumo["participantes_com_alvo_valido"]
        == 3
    )

    assert (
        x_b["hipertensao_informada"].isna().sum()
        == 1
    )


def test_filtro_e_distribuicao_reais(dados):
    """Confere os números do dataset da equipe."""

    x_a, x_b, y, resumo = dados

    assert resumo["participantes_antes_filtro"] == 5283
    assert resumo["participantes_sem_diabetes_informado"] == 4360
    assert resumo["participantes_com_alvo_valido"] == 4360

    assert (y == 0).sum() == 3089
    assert (y == 1).sum() == 1271

    assert len(x_a) == len(x_b) == len(y) == 4360

    assert resumo["ausentes_hipertensao_informada"] == 2


def test_features_e_ausencia_de_vazamento(dados):
    """Confere as entradas permitidas nas versões A e B."""

    x_a, x_b, y, _ = dados

    assert FEATURES_BASE == [
        "idade",
        "imc",
        "pressao_sistolica",
        "pressao_diastolica",
        "colesterol_total",
        "sexo",
    ]

    assert FEATURES_AMPLIADAS == (
        FEATURES_BASE + ["hipertensao_informada"]
    )

    assert list(x_a.columns) == FEATURES_BASE
    assert list(x_b.columns) == FEATURES_AMPLIADAS

    assert not PROIBIDAS.intersection(x_a.columns)
    assert not PROIBIDAS.intersection(x_b.columns)

    assert x_a.index.equals(x_b.index)
    assert x_a.index.equals(y.index)


# =========================================================
# 4. TESTES DO SPLIT
# =========================================================

def test_split_identico_para_versoes_e_sem_sobreposicao(splits):
    """Confere treino/teste e divisão comum às duas versões."""

    treino = splits["y_train"].index
    teste = splits["y_test"].index

    assert len(treino) == 3488
    assert len(teste) == 872

    assert set(treino).isdisjoint(set(teste))

    for versao in ("A", "B"):

        assert splits[
            f"X_train_{versao}"
        ].index.equals(treino)

        assert splits[
            f"X_test_{versao}"
        ].index.equals(teste)

    assert (splits["y_train"] == 0).sum() == 2471
    assert (splits["y_train"] == 1).sum() == 1017

    assert (splits["y_test"] == 0).sum() == 618
    assert (splits["y_test"] == 1).sum() == 254


def test_split_reproduzivel(dados, splits):
    """Mesmo random_state deve gerar o mesmo split."""

    x_a, x_b, y, _ = dados

    repetido = dividir_treino_teste(
        x_a,
        x_b,
        y,
        test_size=0.20,
        random_state=42,
    )

    for nome in splits:
        assert splits[nome].index.equals(
            repetido[nome].index
        )


# =========================================================
# 5. TESTES DOS PIPELINES
# =========================================================

@pytest.mark.parametrize(
    "versao,algoritmo",
    COMBINACOES,
)
def test_pipeline_tem_preprocessamento_interno(
    versao,
    algoritmo,
):
    """Confere que as transformações nascem sem ajuste."""

    pipeline = criar_pipeline(
        versao,
        algoritmo,
    )

    assert isinstance(pipeline, Pipeline)

    assert list(pipeline.named_steps) == [
        "preprocessador",
        "modelo",
    ]

    pre = pipeline.named_steps["preprocessador"]

    assert isinstance(pre, ColumnTransformer)
    assert pre.remainder == "drop"

    transformadores = {
        nome: etapa
        for nome, etapa, _ in pre.transformers
    }

    numerico = transformadores["numerico"]

    assert isinstance(
        numerico.named_steps["imputador"],
        SimpleImputer,
    )

    escalador = numerico.named_steps["escalador"]

    assert isinstance(escalador, RobustScaler)

    # O scaler não pode estar ajustado antecipadamente.
    assert not hasattr(escalador, "center_")

    assert not hasattr(
        numerico.named_steps["imputador"],
        "statistics_",
    )

    sexo = transformadores["sexo"]

    assert isinstance(
        sexo.named_steps["codificador"],
        OneHotEncoder,
    )

    assert not hasattr(
        sexo.named_steps["codificador"],
        "categories_",
    )

    assert (
        "hipertensao" in transformadores
    ) == (versao == "B")

    if versao == "B":
        assert isinstance(
            transformadores["hipertensao"]
            .named_steps["imputador"],
            SimpleImputer,
        )


def test_cv_estratificada_utiliza_somente_treino(splits):
    """Confere que nenhum fold contém participantes do teste."""

    y_train = splits["y_train"]

    cv = StratifiedKFold(
        n_splits=NUMERO_FOLDS,
        shuffle=True,
        random_state=SEMENTE,
    )

    vistos = []

    for pos_treino, pos_validacao in cv.split(
        splits["X_train_A"],
        y_train,
    ):

        idx_tr = set(
            y_train.iloc[pos_treino].index
        )

        idx_val = set(
            y_train.iloc[pos_validacao].index
        )

        assert idx_tr.isdisjoint(idx_val)

        assert idx_val.isdisjoint(
            set(splits["y_test"].index)
        )

        assert set(
            y_train.iloc[pos_validacao].unique()
        ) == {0, 1}

        vistos.extend(idx_val)

    assert (
        len(vistos)
        == len(set(vistos))
        == len(y_train)
    )


# =========================================================
# 6. TESTES DOS MODELOS SALVOS
# =========================================================

def test_metadados_e_arquivos_de_modelos(splits):
    """Confere se todos os modelos foram exportados."""

    arquivo = MODELOS / "metadados_l5.json"

    assert arquivo.is_file(), (
        f"Metadados ausentes: {arquivo}"
    )

    meta = json.loads(
        arquivo.read_text(encoding="utf-8")
    )

    assert meta["random_state"] == 42
    assert meta["cv_folds"] == 5

    assert meta["n_treino"] == len(
        splits["y_train"]
    )

    assert meta["n_teste_reservado"] == len(
        splits["y_test"]
    )

    assert meta["test_set_utilizado_na_avaliacao"] is False

    artefatos = {
        (x["versao"], x["modelo"])
        for x in meta["artefatos"]
    }

    assert artefatos == set(COMBINACOES)

    for versao, modelo in COMBINACOES:

        caminho = (
            MODELOS
            / f"versao_{versao}_{modelo}.joblib"
        )

        assert caminho.is_file()


@pytest.mark.parametrize(
    "versao,algoritmo",
    COMBINACOES,
)
def test_modelos_salvos_preveem_sem_retreinar(
    versao,
    algoritmo,
    splits,
):
    """Carrega modelos da equipe e testa previsões."""

    caminho = (
        MODELOS
        / f"versao_{versao}_{algoritmo}.joblib"
    )

    assert caminho.is_file(), (
        f"Modelo ausente: {caminho}"
    )

    # Carregar apenas arquivos joblib confiáveis,
    # produzidos pela própria equipe.
    modelo = joblib.load(caminho)

    X = splits[f"X_test_{versao}"].head(3)

    assert list(modelo.feature_names_in_) == list(
        X.columns
    )

    assert list(modelo.classes_) == [0, 1]

    probabilidades = modelo.predict_proba(X)

    assert probabilidades.shape == (3, 2)

    assert np.isfinite(
        probabilidades
    ).all()

    assert np.all(
        (probabilidades >= 0)
        & (probabilidades <= 1)
    )

    assert np.allclose(
        probabilidades.sum(axis=1),
        1,
    )


# =========================================================
# 7. TESTES DAS MÉTRICAS EXPORTADAS
# =========================================================

def test_metricas_final_exportadas_e_coerentes(splits):
    """Confere as quatro avaliações e suas matrizes."""

    caminho = (
        AVALIACAO
        / "metricas_teste_l5.csv"
    )

    assert caminho.is_file(), (
        f"Métricas não encontradas: {caminho}"
    )

    tabela = pd.read_csv(caminho)

    assert len(tabela) == 4

    assert set(
        zip(tabela["versao"], tabela["modelo"])
    ) == set(COMBINACOES)

    colunas = [
        "auc_roc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensibilidade",
        "especificidade",
        "f1",
    ]

    for coluna in colunas:

        assert pd.to_numeric(
            tabela[coluna],
            errors="raise",
        ).between(0, 1).all()

    for _, linha in tabela.iterrows():

        tn, fp, fn, tp = (
            int(linha[c])
            for c in ("tn", "fp", "fn", "tp")
        )

        assert min(tn, fp, fn, tp) >= 0

        assert tn + fp == int(
            (splits["y_test"] == 0).sum()
        )

        assert fn + tp == int(
            (splits["y_test"] == 1).sum()
        )

        total = tn + fp + fn + tp

        assert total == int(
            linha["participantes_teste"]
        )

        assert abs(
            (tn + tp) / total
            - linha["accuracy"]
        ) < 1e-9

        assert abs(
            tp / (tp + fn)
            - linha["recall_sensibilidade"]
        ) < 1e-9

        assert abs(
            tn / (tn + fp)
            - linha["especificidade"]
        ) < 1e-9


def test_validacao_cruzada_exportada():
    """Confere o resumo e os folds da validação cruzada."""

    caminho = (
        AVALIACAO
        / "validacao_cruzada_l5.csv"
    )

    assert caminho.is_file()

    tabela = pd.read_csv(caminho)

    assert len(tabela) == 4

    assert set(
        zip(tabela["versao"], tabela["modelo"])
    ) == set(COMBINACOES)

    for nome in (
        "auc_roc",
        "f1",
        "recall",
        "precisao",
        "acuracia",
    ):

        assert tabela[
            f"cv_{nome}_media"
        ].between(0, 1).all()

    folds = pd.read_csv(
        AVALIACAO
        / "validacao_cruzada_folds_l5.csv"
    )

    for versao, algoritmo in COMBINACOES:

        sub = folds[
            (folds["versao"] == versao)
            & (folds["modelo"] == algoritmo)
        ]

        assert set(sub["fold"]) == set(
            range(1, NUMERO_FOLDS + 1)
        )


# =========================================================
# 8. TESTES DOS ARTEFATOS VISUAIS
# =========================================================

@pytest.mark.parametrize(
    "versao,algoritmo",
    COMBINACOES,
)
def test_matriz_e_importancia_exportadas(
    versao,
    algoritmo,
    splits,
):
    """Confere matrizes e importâncias de cada modelo."""

    nome = f"versao_{versao}_{algoritmo}"

    pasta_m = (
        AVALIACAO
        / "matrizes_confusao"
    )

    pasta_i = (
        AVALIACAO
        / "importancia_variaveis"
    )

    matriz_csv = pasta_m / f"matriz_{nome}.csv"

    importancia_csv = (
        pasta_i
        / f"importancia_{nome}.csv"
    )

    assert matriz_csv.is_file()
    assert (
        pasta_m / f"matriz_{nome}.png"
    ).is_file()

    assert importancia_csv.is_file()
    assert (
        pasta_i / f"importancia_{nome}.png"
    ).is_file()

    matriz = pd.read_csv(
        matriz_csv,
        index_col=0,
    )

    assert matriz.shape == (2, 2)

    assert int(
        matriz.to_numpy().sum()
    ) == len(splits["y_test"])

    imp = pd.read_csv(
        importancia_csv
    )

    assert set(
        imp["variavel"]
    ) == set(
        splits[f"X_test_{versao}"].columns
    )

    # A importância por permutação pode ser negativa.
    assert np.isfinite(
        imp["queda_auc_media"]
    ).all()


def test_curvas_roc_e_importancias_consolidadas():
    """Confere as curvas ROC e o resumo das importâncias."""

    assert (
        AVALIACAO / "curvas_roc_l5.png"
    ).is_file()

    pontos = pd.read_csv(
        AVALIACAO / "curvas_roc_pontos_l5.csv"
    )

    assert set(
        zip(pontos["versao"], pontos["modelo"])
    ) == set(COMBINACOES)

    assert pontos[
        "taxa_falsos_positivos"
    ].between(0, 1).all()

    assert pontos[
        "sensibilidade"
    ].between(0, 1).all()

    imp = pd.read_csv(
        AVALIACAO
        / "importancia_variaveis_l5.csv"
    )

    assert set(imp["modelo"]) == {
        f"versao_{versao}_{algoritmo}"
        for versao, algoritmo in COMBINACOES
    }

    assert not PROIBIDAS.intersection(
        set(imp["variavel"])
    )
