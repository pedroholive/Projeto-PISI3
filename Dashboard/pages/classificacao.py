"""HealthSync — L8: atribuição de subperfil cardiometabólico.

O alvo perfil_id foi gerado pelo K-Means a partir dos indicadores.
O Random Forest aprende a reproduzir essa partição estatística;
as métricas NÃO demonstram previsão de doença ou validação clínica.
Esta página apenas apresenta resultados já exportados pela equipe.
"""

import base64
import json
import math
import re
import unicodedata
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html


ROOT = Path(__file__).resolve().parents[2]
PASTA_AVALIACAO = ROOT / "modelos" / "avaliacao_classificacao"
ARQUIVO_CSV = PASTA_AVALIACAO / "metricas_classificacao.csv"
PASTA_JSON = ROOT / "modelos" / "modelosclassificacao" / "resultados"

ESTRATOS = {
    "estrato_1": ("HbA1c < 5.7", "HbA1c < 5,7%", 1),
    "estrato_2": (
        "5.7 <= HbA1c < 6.5",
        "HbA1c de 5,7% a menos de 6,5%",
        2,
    ),
    "estrato_3": ("HbA1c >= 6.5", "HbA1c ≥ 6,5%", 3),
}

MODELOS = {
    "random_forest": "Random Forest (atribuição de subperfil)",
    "regressao_logistica": "Regressão Logística (baseline)",
    "kmeans": "K-Means (baseline)",
}

METRICAS = {
    "accuracy": "Acurácia",
    "precision_macro": "Precisão macro",
    "recall_macro": "Recall macro",
    "f1_macro": "F1-score macro",
    "f1_weighted": "F1-score ponderado",
}

ALIAS_MODELOS = {
    "random_forest": ("random_forest", "randomforest", "rf"),
    "regressao_logistica": (
        "regressao_logistica", "regressaologistica",
        "logistic_regression", "logisticregression",
        "logreg", "reg_log", "lr",
    ),
    "kmeans": ("kmeans", "k_means", "k_mean"),
}

ALIAS_METRICAS = {
    "accuracy": ("accuracy", "acuracia", "acc"),
    "precision_macro": ("precision_macro", "precisao_macro", "precision"),
    "recall_macro": ("recall_macro", "revocacao_macro", "recall"),
    "f1_macro": (
        "f1_macro", "f1_score_macro", "f1score_macro", "f1_score", "f1"
    ),
    "f1_weighted": ("f1_weighted", "f1_score_weighted", "f1_ponderado"),
}

VARIAVEIS = {
    "idade": "Idade",
    "imc": "IMC",
    "pressao_sistolica": "Pressão sistólica",
    "pressao_diastolica": "Pressão diastólica",
    "colesterol_total": "Colesterol total",
}


dash.register_page(
    __name__, path="/classificacao", name="Atribuição de Subperfil", order=7
)


# =========================================================
# LEITURA DAS MÉTRICAS EXPORTADAS
# =========================================================

def normalizar(valor):
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower().replace("≥", ">=").replace("≤", "<=")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", texto)).strip("_")


def identificar_coluna(coluna):
    """Aceita 'accuracy', 'rf_accuracy', 'accuracy_logreg' etc."""
    nome = normalizar(coluna)
    if nome.startswith("baseline_"):
        nome = nome[len("baseline_"):]
    if nome.endswith("_baseline"):
        nome = nome[:-len("_baseline")]
    nome = nome.replace("_baseline_", "_")

    for modelo, aliases in ALIAS_MODELOS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if nome.startswith(alias + "_"):
                nome_metrica = nome[len(alias) + 1:]
            elif nome.endswith("_" + alias):
                nome_metrica = nome[:-(len(alias) + 1)]
            else:
                continue
            for metrica, nomes in ALIAS_METRICAS.items():
                if nome_metrica in nomes:
                    return modelo, metrica

    # Colunas originais sem sufixo de modelo pertencem ao Random Forest.
    for metrica, nomes in ALIAS_METRICAS.items():
        if nome in nomes:
            return "random_forest", metrica
    return None


def converter_numero(valor):
    if pd.isna(valor) or isinstance(valor, bool):
        return None
    try:
        numero = float(str(valor).strip().replace(",", "."))
    except (ValueError, TypeError):
        return None
    return numero if math.isfinite(numero) and 0 <= numero <= 1 else None


def identificar_estrato(valor):
    referencia = normalizar(valor)
    for chave, (nome_csv, _, indice) in ESTRATOS.items():
        if referencia in {
            normalizar(nome_csv), normalizar(chave), str(indice), f"e{indice}"
        }:
            return chave
    return None


def carregar_metricas():
    """Releitura a cada seleção; não recalcula nem treina modelos."""
    if not ARQUIVO_CSV.is_file():
        return {}, "Arquivo metricas_classificacao.csv não encontrado."
    try:
        tabela = pd.read_csv(
            ARQUIVO_CSV, sep=None, engine="python", encoding="utf-8-sig"
        )
    except (OSError, ValueError, pd.errors.ParserError) as erro:
        return {}, f"Não foi possível abrir o CSV: {erro}"

    tabela.columns = [str(c).strip() for c in tabela.columns]
    coluna_estrato = next(
        (c for c in tabela if normalizar(c) == "estrato"), None
    )
    if coluna_estrato is None:
        return {}, "O CSV precisa conter a coluna 'estrato'."

    colunas = {
        coluna: identificar_coluna(coluna)
        for coluna in tabela.columns if coluna != coluna_estrato
    }
    colunas = {c: par for c, par in colunas.items() if par is not None}

    resultados = {}
    for _, linha in tabela.iterrows():
        estrato = identificar_estrato(linha[coluna_estrato])
        if estrato is None:
            continue
        por_modelo = resultados.setdefault(estrato, {})
        for coluna, (modelo, metrica) in colunas.items():
            valor = converter_numero(linha[coluna])
            if valor is not None:
                por_modelo.setdefault(modelo, {})[metrica] = valor

    if not resultados:
        return {}, "Nenhum estrato reconhecido no CSV."
    return resultados, None


def opcoes_modelos(estrato, resultados):
    """Oferece apenas modelos com alguma métrica naquele estrato."""
    disponiveis = resultados.get(estrato, {})
    return [
        {"label": nome, "value": chave}
        for chave, nome in MODELOS.items()
        if disponiveis.get(chave)
    ]


# =========================================================
# CARTÕES, TABELA E GRÁFICOS
# =========================================================

def porcentagem(valor):
    return "—" if valor is None else f"{valor * 100:.2f}%".replace(".", ",")


def aviso(mensagem):
    return html.Div(mensagem, className="cf-empty")


def cartao(titulo, conteudo, nota=None):
    return html.Div(
        [
            html.H3(titulo, className="cf-card-title"),
            conteudo,
            html.P(nota, className="cf-card-note") if nota else None,
        ],
        className="cf-card",
    )


def indicador(titulo, valor, nota):
    return html.Div(
        [
            html.P(titulo, className="cf-kpi-label"),
            html.Strong(porcentagem(valor), className="cf-kpi-value"),
            html.P(nota, className="cf-kpi-note"),
        ],
        className="cf-kpi",
    )


def criar_tabela(resultados):
    linhas = []
    for estrato, (_, nome_estrato, _) in ESTRATOS.items():
        for modelo, nome_modelo in MODELOS.items():
            valores = resultados.get(estrato, {}).get(modelo, {})
            if not valores:
                continue
            celulas = [nome_estrato, nome_modelo] + [
                porcentagem(valores.get(metrica)) for metrica in METRICAS
            ]
            linhas.append(html.Tr([html.Td(texto) for texto in celulas]))

    if not linhas:
        return aviso("Ainda não existem métricas para comparar.")

    cabecalhos = ["Estrato", "Modelo", *METRICAS.values()]
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(c) for c in cabecalhos])),
                html.Tbody(linhas),
            ],
            className="cf-table",
        ),
        className="cf-table-wrap",
    )


def criar_matriz(estrato):
    """Exibe a matriz PNG do Random Forest, se exportada."""
    indice = ESTRATOS[estrato][2]
    arquivo = PASTA_AVALIACAO / f"matriz_estrato_{indice}.png"
    if not arquivo.is_file():
        return aviso("A matriz de confusão ainda não foi exportada.")
    try:
        imagem = base64.b64encode(arquivo.read_bytes()).decode("ascii")
    except OSError:
        return aviso("Não foi possível ler a matriz de confusão.")
    return html.Div(
        [
            html.Img(
                src=f"data:image/png;base64,{imagem}",
                alt=f"Matriz de confusão — {ESTRATOS[estrato][1]}",
                style={"width": "100%", "height": "auto", "objectFit": "contain"},
            ),
            html.P(
                "Matriz de confusão do Random Forest exportada pela avaliação.",
                className="cf-card-note",
            ),
        ],
        className="cf-confusion-container",
    )


def criar_importancias(estrato):
    """Mostra importâncias apenas quando exportadas pelo colega."""
    arquivo = PASTA_JSON / f"{estrato}.json"
    if not arquivo.is_file():
        return aviso("Importância das variáveis ainda não exportada.")
    try:
        conteudo = json.loads(arquivo.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return aviso("Não foi possível ler o arquivo de importâncias.")

    dados = (
        conteudo.get("importancia_variaveis", {})
        if isinstance(conteudo, dict) else {}
    )
    if not isinstance(dados, dict):
        return aviso("Importâncias não disponíveis neste arquivo.")

    pares = []
    for coluna, valor in dados.items():
        if coluna in VARIAVEIS:
            numero = converter_numero(valor)
            if numero is not None:
                pares.append((VARIAVEIS[coluna], numero))
    if not pares:
        return aviso("Importâncias não disponíveis neste arquivo.")

    pares.sort(key=lambda item: item[1])
    figura = go.Figure(
        go.Bar(
            x=[numero for _, numero in pares],
            y=[nome for nome, _ in pares],
            orientation="h",
            marker_color="#0E766D",
        )
    )
    figura.update_layout(
        height=400,
        margin=dict(l=140, r=30, t=25, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Importância relativa",
        font=dict(family="Arial, sans-serif", color="#52615E"),
    )
    return html.Div(
        [
            dcc.Graph(figure=figura, config={"displayModeBar": False}),
            html.P(
                "Importância do Random Forest, conforme os valores exportados. "
                "Não indica causalidade.",
                className="cf-feature-note",
            ),
        ],
        className="cf-feature-container",
    )


# =========================================================
# LAYOUT: A PÁGINA CONTINUA NA ROTA /classificacao
# =========================================================

resultados_iniciais, _ = carregar_metricas()
opcoes_iniciais = opcoes_modelos("estrato_1", resultados_iniciais)
modelo_inicial = (
    "random_forest"
    if any(opcao["value"] == "random_forest" for opcao in opcoes_iniciais)
    else (opcoes_iniciais[0]["value"] if opcoes_iniciais else None)
)

layout = html.Div(
    [
        html.Div(
            [
                html.P("MACHINE LEARNING · NHANES 2021–2023", className="cf-eyebrow"),
                html.H1("Atribuição de subperfil cardiometabólico", className="cf-heading"),
                html.P(
                    "Avaliação da concordância entre os perfis gerados por K-Means "
                    "e os rótulos atribuídos por classificadores em cada estrato "
                    "de HbA1c. Não se trata de diagnóstico ou previsão de doença.",
                    className="cf-intro",
                ),
            ]
        ),
        html.Div(
            [
                html.Strong("Como interpretar as métricas: "),
                "O perfil_id foi gerado pelo K-Means a partir dos indicadores "
                "dos participantes. O F1 macro elevado do Random Forest "
                "(aproximadamente 0,92 a 0,97) mede a capacidade de reproduzir "
                "essa partição estatística, não de prever um desfecho novo. "
                "A Regressão Logística serve como linha de base, quando disponível.",
            ],
            className="cf-info",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Estrato glicêmico", htmlFor="cf-estrato",
                            className="cf-filter-label",
                        ),
                        dcc.Dropdown(
                            id="cf-estrato", value="estrato_1", clearable=False,
                            options=[
                                {"label": dados[1], "value": chave}
                                for chave, dados in ESTRATOS.items()
                            ],
                        ),
                    ],
                    className="cf-filter-group",
                ),
                html.Div(
                    [
                        html.Label(
                            "Modelo", htmlFor="cf-modelo", className="cf-filter-label"
                        ),
                        dcc.Dropdown(
                            id="cf-modelo", value=modelo_inicial,
                            options=opcoes_iniciais, clearable=False,
                            placeholder="Sem modelos avaliados",
                        ),
                    ],
                    className="cf-filter-group",
                ),
            ],
            className="cf-filters",
        ),
        html.Div(id="cf-kpis", className="cf-kpis"),
        cartao(
            "Comparação da atribuição de subperfis por estrato",
            html.Div(id="cf-comparacao"),
            "Valores exportados pela equipe. Traços (—) indicam métricas "
            "que não foram disponibilizadas para aquele modelo.",
        ),
        html.Div(
            [
                cartao("Matriz de confusão", html.Div(id="cf-matriz")),
                cartao("Importância das variáveis", html.Div(id="cf-importancia")),
            ],
            className="cf-grid",
        ),
        html.Div(
            "Limitação de uso no aplicativo: a HbA1c é necessária para "
            "selecionar um dos três estratos antes da atribuição do subperfil. "
            "Sem esse exame, o pipeline atual não atribui subperfil. Dentro "
            "do estrato, as cinco entradas são idade, IMC, pressão sistólica, "
            "pressão diastólica e colesterol total.",
            className="cf-info",
        ),
        html.P(
            "Fonte: modelos/avaliacao_classificacao/metricas_classificacao.csv "
            "e demais arquivos exportados pela avaliação.",
            className="cf-source",
        ),
    ],
    className="cf-page",
)


# =========================================================
# CALLBACKS
# =========================================================

@callback(
    Output("cf-modelo", "options"),
    Output("cf-modelo", "value"),
    Input("cf-estrato", "value"),
    State("cf-modelo", "value"),
)
def atualizar_opcoes_modelos(estrato, modelo_atual):
    if estrato not in ESTRATOS:
        estrato = "estrato_1"
    resultados, _ = carregar_metricas()
    opcoes = opcoes_modelos(estrato, resultados)
    valores = {opcao["value"] for opcao in opcoes}
    if modelo_atual in valores:
        escolhido = modelo_atual
    elif "random_forest" in valores:
        escolhido = "random_forest"
    else:
        escolhido = opcoes[0]["value"] if opcoes else None
    return opcoes, escolhido


@callback(
    Output("cf-kpis", "children"),
    Output("cf-comparacao", "children"),
    Output("cf-matriz", "children"),
    Output("cf-importancia", "children"),
    Input("cf-estrato", "value"),
    Input("cf-modelo", "value"),
)
def atualizar_classificacao(estrato, modelo):
    if estrato not in ESTRATOS:
        estrato = "estrato_1"

    resultados, erro = carregar_metricas()
    tabela = aviso(erro) if erro else criar_tabela(resultados)
    disponiveis = resultados.get(estrato, {})

    if modelo not in disponiveis:
        mensagem = (
            "Nenhum resultado de avaliação disponível para este estrato."
            if not disponiveis else
            "Selecione um dos modelos com métricas disponíveis."
        )
        return [aviso(mensagem)], tabela, aviso(mensagem), aviso(mensagem)

    valores = disponiveis[modelo]
    # Nos cartões, não renderiza métricas ausentes da baseline.
    kpis = [
        indicador(
            titulo, valores[metrica],
            f"{MODELOS[modelo]} · {ESTRATOS[estrato][1]}",
        )
        for metrica, titulo in METRICAS.items()
        if metrica in valores
    ]

    if modelo == "random_forest":
        matriz = criar_matriz(estrato)
        importancias = criar_importancias(estrato)
    else:
        matriz = aviso(
            "A avaliação ainda não disponibilizou matriz de confusão "
            "para esta baseline."
        )
        importancias = aviso(
            "A avaliação ainda não disponibilizou importância das variáveis "
            "para esta baseline."
        )

    return kpis, tabela, matriz, importancias
