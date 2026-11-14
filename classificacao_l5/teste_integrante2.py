"""Testes dos experimentos de classificacao do Integrante 2 (L5).

Executar a partir da raiz do Projeto-PISI3:
    python -m pytest classificacao_l5/teste_integrante2.py -v

Nao reexecuta os experimentos completos e nao modifica arquivos do projeto.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as PipelineImb
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from classificacao_l5.preparacao_l5 import (
    FEATURES_BASE,
    carregar_dados,
    dividir_treino_teste,
    preparar_dados_l5,
)
from classificacao_l5.comparacao_algoritmos_l5 import (
    PASTA_SAIDA,
    criar_algoritmos as criar_sem_pesos,
    criar_pipeline_comparacao,
)
from classificacao_l5.comparacao_pesos_l5 import criar_algoritmos_com_pesos
from classificacao_l5.comparacao_reamostragem_l5 import (
    criar_pipeline as criar_pipeline_reamostragem,
)
from classificacao_l5.otimizacao_limiar_l5 import (
    avaliar_limiar,
    escolher_limiar,
    limiar_padrao,
    pontuacoes,
    tipo_score,
)
from classificacao_l5.otimizacao_conjunta_l5 import RAIZ, PASTA_MODELOS


@pytest.fixture(scope="module")
def divisao_real():
    """Usa a preparacao oficial sem treinar modelos ou avaliar o holdout."""
    base = carregar_dados()
    X_A, X_B, y, _ = preparar_dados_l5(base)
    return dividir_treino_teste(X_A, X_B, y, test_size=0.20, random_state=42)


@pytest.fixture
def dados_sinteticos():
    """Dados inventados: nenhuma observacao do teste real e utilizada."""
    rng = np.random.default_rng(42)
    n = 72
    y = np.array([0] * 48 + [1] * 24, dtype=int)
    ordem = rng.permutation(n)
    y = y[ordem]
    X = pd.DataFrame({
        "idade": rng.integers(20, 80, size=n),
        "imc": rng.uniform(19, 42, size=n),
        "pressao_sistolica": rng.uniform(100, 165, size=n),
        "pressao_diastolica": rng.uniform(60, 100, size=n),
        "colesterol_total": rng.uniform(130, 275, size=n),
        "sexo": rng.choice(["Homem", "Mulher"], size=n),
    })
    return X[FEATURES_BASE], pd.Series(y)


def test_divisao_oficial_sem_sobreposicao(divisao_real):
    d = divisao_real
    assert len(d["X_train_A"]) == len(d["y_train"])
    assert len(d["X_test_A"]) == len(d["y_test"])
    assert len(d["X_train_A"]) + len(d["X_test_A"]) == 4360
    assert len(d["X_train_A"]) == 3488
    assert len(d["X_test_A"]) == 872
    assert set(d["X_train_A"].index).isdisjoint(d["X_test_A"].index)
    assert d["X_train_A"].index.equals(d["X_train_B"].index)
    assert d["X_test_A"].index.equals(d["X_test_B"].index)
    assert list(d["X_train_A"].columns) == list(FEATURES_BASE)
    assert d["y_train"].nunique() == 2
    assert abs(d["y_train"].mean() - d["y_test"].mean()) < 0.01


def test_features_nao_incluem_resposta_ou_identificador(divisao_real):
    proibidas = {
        "hba1c", "hba1c_binaria", "diabetes_informado", "perfil_id",
        "cluster", "estrato_hba1c", "glicose_jejum", "id_participante",
    }
    assert proibidas.isdisjoint(divisao_real["X_train_A"].columns)


def test_algoritmos_sem_balanceamento():
    modelos = criar_sem_pesos()
    assert set(modelos) == {
        "dummy", "regressao_logistica", "random_forest",
        "svm_rbf", "hist_gradient_boosting",
    }
    assert isinstance(modelos["dummy"], DummyClassifier)
    for nome in ("regressao_logistica", "random_forest", "svm_rbf", "hist_gradient_boosting"):
        assert modelos[nome].get_params().get("class_weight") is None
        pipeline = criar_pipeline_comparacao(clone(modelos[nome]))
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["preprocessador", "modelo"]


def test_algoritmos_com_pesos():
    modelos = criar_algoritmos_com_pesos()
    assert set(modelos) == {
        "regressao_logistica", "random_forest", "svm_rbf",
        "hist_gradient_boosting",
    }
    assert modelos["regressao_logistica"].class_weight == "balanced"
    assert modelos["random_forest"].class_weight == "balanced_subsample"
    assert modelos["svm_rbf"].class_weight == "balanced"
    assert modelos["hist_gradient_boosting"].class_weight == "balanced"


@pytest.mark.parametrize("metodo,classe", [
    ("random_oversampler", RandomOverSampler),
    ("smotenc", SMOTENC),
])
def test_reamostragem_esta_dentro_do_pipeline(metodo, classe):
    pipeline = criar_pipeline_reamostragem(metodo, LogisticRegression(max_iter=1000))
    assert isinstance(pipeline, PipelineImb)
    assert list(pipeline.named_steps) == [
        "pre_reamostragem", "reamostragem", "pos_reamostragem", "modelo"
    ]
    assert isinstance(pipeline.named_steps["reamostragem"], classe)
    if metodo == "smotenc":
        assert list(pipeline.named_steps["reamostragem"].categorical_features) == [5]


@pytest.mark.parametrize("metodo", ["random_oversampler", "smotenc"])
def test_reamostragem_apenas_no_fit_de_cada_fold(metodo, dados_sinteticos):
    X, y = dados_sinteticos
    pipeline = criar_pipeline_reamostragem(metodo, LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # cross_val_predict treina um clone em cada fold e devolve uma predicao
    # para cada observacao ORIGINAL, sem reamostrar o fold de validacao.
    predicoes = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    assert len(predicoes) == len(X)
    assert set(np.unique(predicoes)).issubset({0, 1})

    # Em uma amostra inteiramente sintetica, confirma que o balanceador
    # atua durante o fit e nao altera a cardinalidade das predicoes.
    pipeline.fit(X, y)
    dados_imputados = pipeline.named_steps["pre_reamostragem"].transform(X)
    sampler = clone(pipeline.named_steps["reamostragem"])
    X_res, y_res = sampler.fit_resample(dados_imputados, y)
    assert len(y_res) > len(y)
    assert np.bincount(np.asarray(y_res, dtype=int))[0] == np.bincount(np.asarray(y_res, dtype=int))[1]
    assert len(pipeline.predict(X.head(8))) == 8
    if metodo == "smotenc":
        # A categoria sexo nao pode virar 0.27, 0.83 etc.
        categorias_originais = set(np.unique(dados_imputados[:, 5]))
        assert set(np.unique(X_res[:, 5])).issubset(categorias_originais)


def test_tipo_de_score_e_limiar_padrao():
    logistica = criar_pipeline_comparacao(LogisticRegression())
    from sklearn.svm import SVC
    svm = criar_pipeline_comparacao(SVC(probability=False))
    assert tipo_score(logistica) == "probabilidade"
    assert tipo_score(svm) == "margem_svm"
    assert limiar_padrao("probabilidade") == 0.5
    assert limiar_padrao("margem_svm") == 0.0


def test_metricas_de_limiar_usam_classe_positiva():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.35, 0.55, 0.85])
    m = avaliar_limiar(y, scores, 0.5)
    assert (m["vp"], m["fp"], m["vn"], m["fn"]) == (2, 0, 3, 1)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(0.8)


def test_limiar_escolhido_respeita_meta_na_base_interna():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.4, 0.35, 0.55, 0.85])
    limiar, candidatos = escolher_limiar(y, scores, "probabilidade", recall_minimo=0.65)
    assert len(candidatos) >= 2
    assert avaliar_limiar(y, scores, limiar)["recall"] >= 0.65
    # A meta interna nao garante o mesmo recall em dados externos.


def test_exportacao_compara_os_balanceamentos():
    arquivo = PASTA_SAIDA / "comparacao_todos_balanceamentos_l5.csv"
    assert arquivo.exists(), f"Execute os 3 experimentos antes dos testes: {arquivo}"
    df = pd.read_csv(arquivo)
    assert {"algoritmo", "balanceamento", "cv_precisao_media", "cv_recall_media", "cv_f1_media", "cv_auc_roc_media"}.issubset(df.columns)
    assert {"nenhum", "pesos_de_classes", "random_oversampler", "smotenc"}.issubset(set(df["balanceamento"]))
    assert len(df) == 17  # Dummy + 4 sem pesos + 4 com pesos + 8 reamostrados
    for coluna in ("cv_precisao_media", "cv_recall_media", "cv_f1_media", "cv_auc_roc_media"):
        assert df[coluna].notna().all()
        assert df[coluna].between(0, 1).all()


def test_candidatos_e_limiares_exportados(divisao_real):
    arquivo = PASTA_SAIDA / "otimizacao_conjunta_candidatos_l5.csv"
    assert arquivo.exists(), f"Execute a otimizacao conjunta antes dos testes: {arquivo}"
    df = pd.read_csv(arquivo)
    assert set(df["modelo"]) == {"regressao_logistica_pesos", "svm_rbf_pesos"}
    assert {"modelo", "arquivo_joblib", "tipo_score", "limiar_candidato"}.issubset(df.columns)

    # Carrega os modelos prontos: nao reexecuta GridSearchCV nem .fit().
    X_amostra = divisao_real["X_train_A"].head(12)
    for _, linha in df.iterrows():
        caminho = RAIZ / Path(str(linha["arquivo_joblib"]))
        assert caminho.is_file(), f"Modelo candidato ausente: {caminho}"
        assert caminho.resolve().is_relative_to(PASTA_MODELOS.resolve())
        modelo = joblib.load(caminho)  # Apenas artefatos confiaveis, gerados pelo projeto.
        assert list(X_amostra.columns) == list(FEATURES_BASE)
        assert tipo_score(modelo) == linha["tipo_score"]
        scores = pontuacoes(modelo, X_amostra)
        assert scores.shape == (len(X_amostra),)
        assert np.isfinite(scores).all()
        limiar = float(linha["limiar_candidato"])
        assert np.isfinite(limiar)
        predicoes = (scores >= limiar).astype(int)
        assert len(predicoes) == len(X_amostra)
        assert set(predicoes).issubset({0, 1})
