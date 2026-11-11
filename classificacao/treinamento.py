
"""
HealthSync — Treinamento para Atribuição de Subperfil.

O Random Forest aprende a reproduzir o perfil_id,
rótulo estatístico previamente gerado pelo K-Means.

O objetivo deste pipeline não é prever diabetes,
hipertensão ou outro desfecho clínico independente.

Os modelos são treinados separadamente por estrato
de HbA1c, preservando o conjunto de variáveis,
o RobustScaler e a divisão treino/teste.
"""


from pathlib import Path

import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

from classificacao.preparacao_dados import (
    ESTRATOS,
    FEATURES,
    carregar_dados,
    preparar_estratos,
    dividir_treino_teste,
)


# Diretório onde os modelos treinados serão armazenados.
DIRETORIO_MODELOS = (
    Path(__file__).resolve().parents[1]
    / "modelos"
    / "modelosclassificacao"
)


def criar_pipeline(random_state=42):
    """
    Cria um pipeline com RobustScaler e Random Forest.
    """

    # Define as etapas sequenciais do modelo.
    pipeline = Pipeline([
        
        # Escalonamento robusto das variáveis.
        ("scaler", RobustScaler()),

        # Random Forest de atribuição de subperfil.
        ("classificador", RandomForestClassifier(

            # Quantidade de árvores da floresta.
            n_estimators=300,

            # Controla a aleatoriedade.
            random_state=random_state,

            # Distribui o processamento entre os núcleos.
            n_jobs=-1,

            # Compensa parcialmente o desbalanceamento.
            class_weight="balanced_subsample",

            # Critério para avaliar as divisões das árvores.
            criterion="gini",

            # Limita o número de folhas para reduzir overfitting.
            min_samples_leaf=2,
        ))
    ])

    # Retorna o pipeline ainda não treinado.
    return pipeline


def treinar_modelos(splits, random_state=42):
    """
    Treina um modelo de atribuição independente por estrato.
    """

    # Dicionário para armazenar os modelos treinados.
    modelos = {}

    # Percorre os três estratos.
    for estrato in ESTRATOS:

        print(f"\nTreinando modelo: {estrato}")

        # Recupera os dados de treinamento.
        dados = splits[estrato]

        X_train = dados["X_train"]
        y_train = dados["y_train"]

        # Cria um pipeline independente.
        pipeline = criar_pipeline(random_state)

        # Ajusta o scaler e treina o classificador.
        pipeline.fit(X_train, y_train)

        # Armazena o modelo treinado.
        modelos[estrato] = pipeline

        # Apresenta as informações do treinamento.
        print("Modelo treinado com sucesso!")
        print(f"Amostras: {len(X_train)}")
        print(f"Classes aprendidas: {list(pipeline.classes_)}")

    # Retorna os três modelos.
    return modelos


def salvar_modelos(modelos, diretorio=DIRETORIO_MODELOS):
    """
    Salva os modelos treinados em arquivos Joblib.
    """

    # Cria o diretório caso ainda não exista.
    diretorio = Path(diretorio)
    diretorio.mkdir(parents=True, exist_ok=True)

    # Mapeia os estratos para nomes de arquivos.
    nomes = {
        ESTRATOS[0]: "modelo_estrato_1.joblib",
        ESTRATOS[1]: "modelo_estrato_2.joblib",
        ESTRATOS[2]: "modelo_estrato_3.joblib",
    }

    # Salva cada pipeline individualmente.
    for estrato, modelo in modelos.items():

        caminho = diretorio / nomes[estrato]

        # Serializa o pipeline completo.
        joblib.dump(modelo, caminho)

        print(f"Modelo salvo: {caminho}")

    return diretorio


def executar_treinamento():
    """
    Executa o processo completo de preparação e treinamento.
    """

    # Carrega o dataset original.
    df = carregar_dados()

    # Separa os três estratos.
    dados_estratos = preparar_estratos(df)

    # Realiza a divisão de treino e teste.
    splits = dividir_treino_teste(dados_estratos)

    # Treina os três classificadores.
    modelos = treinar_modelos(splits)

    # Salva os modelos no projeto.
    salvar_modelos(modelos)

    print("\nTREINAMENTO CONCLUÍDO COM SUCESSO!")

    return modelos, splits


# Executa somente quando o arquivo é chamado diretamente.
if __name__ == "__main__":
    executar_treinamento()
