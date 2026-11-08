
import pytest

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from classificacao.preparacao_dados import (
    ESTRATOS,
    carregar_dados,
    preparar_estratos,
    dividir_treino_teste,
)

from classificacao.treinamento import (
    criar_pipeline,
    treinar_modelos,
    salvar_modelos,
)


@pytest.fixture
def modelos_treinados():

    # Prepara os dados reais.
    df = carregar_dados()
    estratos = preparar_estratos(df)
    splits = dividir_treino_teste(estratos)

    # Treina os três classificadores.
    modelos = treinar_modelos(splits)

    return modelos, splits


def test_criacao_pipeline():

    # Cria um pipeline.
    modelo = criar_pipeline()

    # Verifica o tipo do pipeline.
    assert isinstance(modelo, Pipeline)

    # Verifica o scaler.
    assert isinstance(
        modelo.named_steps["scaler"],
        RobustScaler
    )

    # Verifica o classificador.
    assert isinstance(
        modelo.named_steps["classificador"],
        RandomForestClassifier
    )


def test_quantidade_modelos(modelos_treinados):

    modelos, _ = modelos_treinados

    assert len(modelos) == 3


def test_modelos_treinados(modelos_treinados):

    modelos, _ = modelos_treinados

    for estrato in ESTRATOS:

        assert hasattr(modelos[estrato], "classes_")


def test_quantidade_classes(modelos_treinados):

    modelos, _ = modelos_treinados

    esperados = [2, 6, 2]

    for estrato, quantidade in zip(ESTRATOS, esperados):

        assert len(modelos[estrato].classes_) == quantidade


def test_predicoes(modelos_treinados):

    modelos, splits = modelos_treinados

    for estrato in ESTRATOS:

        # Recupera o conjunto de teste.
        X_test = splits[estrato]["X_test"]

        # Realiza previsões.
        previsoes = modelos[estrato].predict(X_test)

        # Confere a quantidade de previsões.
        assert len(previsoes) == len(X_test)

        # Confere se as previsões pertencem às classes aprendidas.
        assert set(previsoes).issubset(
            set(modelos[estrato].classes_)
        )


def test_salvar_modelos(modelos_treinados, tmp_path):

    modelos, _ = modelos_treinados

    # Salva os modelos em diretório temporário.
    diretorio = salvar_modelos(modelos, tmp_path)

    # Confere se os três arquivos foram criados.
    assert len(list(diretorio.glob("*.joblib"))) == 3
