
import pytest

from classificacao.preparacao_dados import (
    carregar_dados,
    preparar_estratos,
    dividir_treino_teste,
    FEATURES,
    ESTRATOS,
)


@pytest.fixture
def dados_preparados():

    # Carrega o dataset real do HealthSync.
    df = carregar_dados()

    # Separa os três estratos.
    estratos = preparar_estratos(df)

    # Retorna os conjuntos para reutilização nos testes.
    return df, estratos


def test_quantidade_participantes(dados_preparados):

    # Recupera o DataFrame.
    df, _ = dados_preparados

    # Confirma o tamanho da base utilizada no projeto.
    assert len(df) == 5202


def test_quantidade_estratos(dados_preparados):

    # Recupera os estratos.
    _, estratos = dados_preparados

    # Confirma a existência dos três estratos.
    assert len(estratos) == 3


def test_quantidade_clusters(dados_preparados):

    # Recupera os estratos.
    _, estratos = dados_preparados

    # Quantidades esperadas de perfis.
    esperados = [2, 6, 2]

    # Confere os perfis em cada grupo.
    for estrato, esperado in zip(ESTRATOS, esperados):

        assert estratos[estrato]["perfil_id"].nunique() == esperado


def test_divisao_treino_teste(dados_preparados):

    # Recupera os estratos.
    _, estratos = dados_preparados

    # Executa a divisão estratificada.
    splits = dividir_treino_teste(estratos)

    # Confere os tamanhos esperados.
    assert len(splits[ESTRATOS[0]]["X_train"]) == 2521
    assert len(splits[ESTRATOS[0]]["X_test"]) == 631

    assert len(splits[ESTRATOS[1]]["X_train"]) == 1120
    assert len(splits[ESTRATOS[1]]["X_test"]) == 281

    assert len(splits[ESTRATOS[2]]["X_train"]) == 519
    assert len(splits[ESTRATOS[2]]["X_test"]) == 130


def test_features_corretas(dados_preparados):

    # Recupera os estratos preparados.
    _, estratos = dados_preparados

    # Executa o split.
    splits = dividir_treino_teste(estratos)

    # Verifica se todos os modelos receberão as cinco features.
    for estrato in ESTRATOS:

        assert list(splits[estrato]["X_train"].columns) == FEATURES


def test_sem_vazamento_entre_treino_e_teste(dados_preparados):

    # Recupera os estratos.
    _, estratos = dados_preparados

    # Divide os dados.
    splits = dividir_treino_teste(estratos)

    # Verifica se os índices são disjuntos.
    for estrato in ESTRATOS:

        treino = splits[estrato]["X_train"]
        teste = splits[estrato]["X_test"]

        assert set(treino.index).isdisjoint(set(teste.index))


def test_todas_classes_presentes(dados_preparados):

    # Recupera os estratos.
    _, estratos = dados_preparados

    # Executa a divisão.
    splits = dividir_treino_teste(estratos)

    # Confirma que nenhuma classe foi perdida.
    for estrato in ESTRATOS:

        classes_originais = set(
            estratos[estrato]["perfil_id"]
        )

        classes_treino = set(
            splits[estrato]["y_train"]
        )

        classes_teste = set(
            splits[estrato]["y_test"]
        )

        assert classes_treino == classes_originais
        assert classes_teste == classes_originais
