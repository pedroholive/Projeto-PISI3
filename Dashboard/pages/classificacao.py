
"""
HealthSync — L8: Classificação supervisionada.

Apresenta os resultados dos classificadores Random Forest
treinados por estrato glicêmico no projeto PISI3.

Esta página não realiza treinamento e não utiliza resultados
fictícios. Os números são carregados de arquivos JSON
exportados pela etapa de avaliação.
"""

import json
import math
from pathlib import Path

import dash
from dash import Input, Output, callback, dcc, html
import plotly.graph_objects as go


# =========================================================
# 1. CAMINHOS E CONFIGURAÇÕES
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTADOS = (
    ROOT / "modelos" / "modelosclassificacao" / "resultados"
)

ESTRATOS = {
    "estrato_1": {
        "nome": "HbA1c < 5,7%",
        "arquivo": "estrato_1.json",
    },
    "estrato_2": {
        "nome": "HbA1c de 5,7% a menos de 6,5%",
        "arquivo": "estrato_2.json",
    },
    "estrato_3": {
        "nome": "HbA1c ≥ 6,5%",
        "arquivo": "estrato_3.json",
    },
}

VARIAVEIS = {
    "idade": "Idade",
    "imc": "IMC",
    "pressao_sistolica": "Pressão sistólica",
    "pressao_diastolica": "Pressão diastólica",
    "colesterol_total": "Colesterol total",
}

METRICAS = {
    "acuracia": "Acurácia",
    "precisao": "Precisão",
    "recall": "Recall",
    "f1_score": "F1-score",
}

VERDE = "#0E766D"
VERDE_ESCURO = "#173F3A"
CINZA = "#52615E"


dash.register_page(
    __name__,
    path="/classificacao",
    name="Classificação",
    order=7,
)


# =========================================================
# 2. LEITURA E VALIDAÇÃO DOS RESULTADOS
# =========================================================

def numero_valido(valor):
    """Verifica se o valor é numérico, finito e não booleano."""
    return (
        isinstance(valor, (int, float))
        and not isinstance(valor, bool)
        and math.isfinite(valor)
    )


def carregar_resultado(estrato):
    """
    Carrega os resultados exportados de um estrato.

    Retorna None caso o arquivo ainda não exista.
    """
    if estrato not in ESTRATOS:
        return None

    caminho = RESULTADOS / ESTRATOS[estrato]["arquivo"]

    if not caminho.is_file():
        return None

    try:
        with caminho.open("r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        if not isinstance(dados, dict):
            return None

        return dados

    except (OSError, ValueError, TypeError):
        return None


def obter_metricas(dados):
    """Lê apenas métricas válidas entre zero e um."""
    if not isinstance(dados, dict):
        return {}

    origem = dados.get("metricas", {})

    if not isinstance(origem, dict):
        return {}

    return {
        chave: float(valor)
        for chave, valor in origem.items()
        if chave in METRICAS
        and numero_valido(valor)
        and 0 <= valor <= 1
    }


def obter_matriz(dados):
    """Valida classes e matriz de confusão."""
    if not isinstance(dados, dict):
        return None

    classes = dados.get("classes")
    matriz = dados.get("matriz_confusao")

    if not isinstance(classes, list) or not classes:
        return None

    if not all(isinstance(c, (str, int)) for c in classes):
        return None

    if len(set(map(str, classes))) != len(classes):
        return None

    if not isinstance(matriz, list):
        return None

    n = len(classes)

    if len(matriz) != n:
        return None

    for linha in matriz:
        if not isinstance(linha, list) or len(linha) != n:
            return None

        if not all(
            isinstance(v, int)
            and not isinstance(v, bool)
            and v >= 0
            for v in linha
        ):
            return None

    return [str(c) for c in classes], matriz


def obter_importancias(dados):
    """Lê importâncias não negativas e finitas."""
    if not isinstance(dados, dict):
        return {}

    origem = dados.get("importancia_variaveis", {})

    if not isinstance(origem, dict):
        return {}

    return {
        chave: float(valor)
        for chave, valor in origem.items()
        if chave in VARIAVEIS
        and numero_valido(valor)
        and valor >= 0
    }


# =========================================================
# 3. COMPONENTES VISUAIS
# =========================================================

def cartao(titulo, conteudo, nota=None, extra=None):
    """Cria um cartão seguindo o CSS da classificação."""
    classes = "cf-card"

    if extra:
        classes += f" {extra}"

    return html.Div(
        [
            html.H3(titulo, className="cf-card-title"),
            conteudo,
            (
                html.P(nota, className="cf-card-note")
                if nota else None
            ),
        ],
        className=classes,
    )


def indicador(titulo, valor, nota):
    """Cria um cartão de métrica."""
    return html.Div(
        [
            html.P(titulo, className="cf-kpi-label"),
            html.Strong(valor, className="cf-kpi-value"),
            html.P(nota, className="cf-kpi-note"),
        ],
        className="cf-kpi",
    )


def aviso(texto):
    """Mensagem exibida quando faltam resultados."""
    return html.Div(texto, className="cf-empty")


def formatar_metrica(valor):
    """Converte métricas da escala 0–1 para porcentagem."""
    if valor is None:
        return "—"

    return f"{valor * 100:.2f}%".replace(".", ",")


def estilizar(figura, altura=400):
    """Aplica o padrão visual dos gráficos HealthSync."""
    figura.update_layout(
        font=dict(
            family="Arial, sans-serif",
            color=CINZA,
            size=12,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=25, t=35, b=55),
        height=altura,
        hoverlabel=dict(bgcolor="white"),
    )

    figura.update_xaxes(
        showgrid=False,
        zeroline=False,
        automargin=True,
    )

    figura.update_yaxes(
        gridcolor="#E8EFED",
        zeroline=False,
        automargin=True,
    )

    return figura


# =========================================================
# 4. COMPARAÇÃO DAS MÉTRICAS
# =========================================================

def criar_tabela_comparacao():
    """Compara os resultados disponíveis dos três estratos."""
    cabecalhos = [
        "Estrato",
        "Modelo",
        "Acurácia",
        "Precisão",
        "Recall",
        "F1-score",
    ]

    linhas = []
    disponiveis = 0

    for chave, configuracao in ESTRATOS.items():
        dados = carregar_resultado(chave)
        metricas = obter_metricas(dados)

        if metricas:
            disponiveis += 1

        valores = [
            configuracao["nome"],
            "Random Forest",
        ]

        valores.extend(
            formatar_metrica(metricas.get(chave_metrica))
            for chave_metrica in METRICAS
        )

        linhas.append(
            html.Tr([html.Td(valor) for valor in valores])
        )

    tabela = html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [html.Th(titulo) for titulo in cabecalhos]
                    )
                ),
                html.Tbody(linhas),
            ],
            className="cf-table",
        ),
        className="cf-table-wrap",
    )

    if disponiveis == 0:
        return html.Div(
            [
                tabela,
                aviso(
                    "As métricas ainda não foram exportadas. "
                    "A tabela será preenchida após a avaliação "
                    "dos modelos."
                ),
            ]
        )

    return tabela


# =========================================================
# 5. MATRIZ DE CONFUSÃO
# =========================================================

def criar_matriz_confusao(dados):
    """Constrói a matriz com os valores reais exportados."""
    resultado = obter_matriz(dados)

    if resultado is None:
        return aviso(
            "A matriz de confusão ainda não está disponível "
            "para este estrato."
        )

    classes, matriz = resultado

    figura = go.Figure(
        data=go.Heatmap(
            z=matriz,
            x=classes,
            y=classes,
            colorscale=[
                [0, "#F0FAF6"],
                [0.5, "#74C7AA"],
                [1, "#087E75"],
            ],
            text=matriz,
            texttemplate="%{text}",
            textfont=dict(size=14),
            hovertemplate=(
                "Classe real: %{y}<br>"
                "Classe prevista: %{x}<br>"
                "Quantidade: %{z}<extra></extra>"
            ),
            colorbar=dict(title="Registros"),
            zmin=0,
        )
    )

    figura.update_xaxes(
        title="Classe prevista",
        type="category",
        side="bottom",
    )

    figura.update_yaxes(
        title="Classe real",
        type="category",
        autorange="reversed",
    )

    estilizar(figura, max(400, 75 * len(classes)))

    return html.Div(
        [
            dcc.Graph(
                figure=figura,
                config={"displayModeBar": False},
                className="cf-confusion-graph",
            ),
            html.Div(
                "As linhas representam as classes reais e as "
                "colunas as classes previstas. Os valores da "
                "diagonal correspondem às classificações corretas.",
                className="cf-confusion-note",
            ),
        ],
        className="cf-confusion-container",
    )


# =========================================================
# 6. IMPORTÂNCIA DAS VARIÁVEIS
# =========================================================

def criar_grafico_importancia(dados):
    """Exibe a importância das variáveis do Random Forest."""
    importancias = obter_importancias(dados)

    if not importancias:
        return aviso(
            "A importância das variáveis ainda não foi "
            "exportada para este estrato."
        )

    pares = sorted(
        importancias.items(),
        key=lambda item: item[1],
    )

    nomes = [
        VARIAVEIS[chave]
        for chave, _ in pares
    ]

    valores = [
        valor
        for _, valor in pares
    ]

    figura = go.Figure(
        go.Bar(
            x=valores,
            y=nomes,
            orientation="h",
            marker_color=VERDE,
            text=[
                f"{valor:.3f}"
                for valor in valores
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>"
                "Importância: %{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    figura.update_xaxes(
        title="Importância relativa (impureza)",
        rangemode="tozero",
    )

    figura.update_layout(
        margin=dict(l=140, r=65, t=35, b=55),
    )

    estilizar(figura, 410)

    figura.update_layout(
        margin=dict(l=140, r=65, t=35, b=55)
    )

    return html.Div(
        [
            dcc.Graph(
                figure=figura,
                config={"displayModeBar": False},
                className="cf-feature-graph",
            ),
            html.P(
                "Quando exportadas de feature_importances_, "
                "as barras representam a importância baseada "
                "na redução de impureza do Random Forest. "
                "Esses valores não indicam causalidade.",
                className="cf-feature-note",
            ),
        ],
        className="cf-feature-container",
    )


# =========================================================
# 7. LAYOUT PRINCIPAL
# =========================================================

layout = html.Div(
    [
        # Cabeçalho
        html.Div(
            [
                html.P(
                    "MACHINE LEARNING · NHANES 2021–2023",
                    className="cf-eyebrow",
                ),
                html.H1(
                    "Classificação supervisionada",
                    className="cf-heading",
                ),
                html.P(
                    "Explore os resultados dos classificadores "
                    "Random Forest utilizados para reproduzir "
                    "os perfis cardiometabólicos identificados "
                    "pela clusterização. Os resultados não "
                    "representam diagnósticos clínicos.",
                    className="cf-intro",
                ),
            ]
        ),

        # Seleção de estrato
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Estrato glicêmico",
                            htmlFor="cf-estrato",
                            className="cf-filter-label",
                        ),
                        dcc.Dropdown(
                            id="cf-estrato",
                            options=[
                                {
                                    "label": configuracao["nome"],
                                    "value": chave,
                                }
                                for chave, configuracao
                                in ESTRATOS.items()
                            ],
                            value="estrato_1",
                            clearable=False,
                        ),
                    ],
                    className="cf-filter-group",
                ),
                html.Div(
                    [
                        html.Label(
                            "Algoritmo",
                            className="cf-filter-label",
                        ),
                        html.Div(
                            "Random Forest",
                            className="cf-info",
                        ),
                    ],
                    className="cf-filter-group",
                ),
            ],
            className="cf-filters",
        ),

        # Indicadores do modelo selecionado
        html.Div(
            id="cf-kpis",
            className="cf-kpis",
        ),

        # Comparação dos resultados
        cartao(
            "Comparação dos modelos por estrato",
            html.Div(id="cf-comparacao"),
            "Comparação descritiva dos classificadores "
            "Random Forest treinados em estratos distintos. "
            "Diferenças de desempenho não indicam, isoladamente, "
            "superioridade entre populações ou tarefas.",
        ),

        # Gráficos
        html.Div(
            [
                cartao(
                    "Matriz de confusão",
                    html.Div(id="cf-matriz"),
                    "Distribuição das classes reais e previstas "
                    "no conjunto de teste do estrato selecionado.",
                ),
                cartao(
                    "Importância das variáveis",
                    html.Div(id="cf-importancia"),
                    "Contribuição das variáveis ao modelo "
                    "selecionado, conforme a avaliação exportada.",
                ),
            ],
            className="cf-grid",
        ),

        # Observação metodológica
        html.Div(
            "Os modelos utilizam idade, IMC, pressão sistólica, "
            "pressão diastólica e colesterol total para "
            "classificar perfis estatísticos definidos na "
            "clusterização. Os resultados descrevem o "
            "desempenho experimental dos modelos e não devem "
            "ser interpretados como previsão de doenças ou "
            "validação clínica.",
            className="cf-info",
        ),

        # Fonte
        html.P(
            "Fonte: experimentos de classificação do projeto "
            "HealthSync (NHANES 2021–2023). "
            "Resultados carregados de "
            "modelos/modelosclassificacao/resultados/.",
            className="cf-source",
        ),
    ],
    className="cf-page",
)


# =========================================================
# 8. ATUALIZAÇÃO DA PÁGINA
# =========================================================

@callback(
    Output("cf-kpis", "children"),
    Output("cf-comparacao", "children"),
    Output("cf-matriz", "children"),
    Output("cf-importancia", "children"),
    Input("cf-estrato", "value"),
)
def atualizar_classificacao(estrato):
    """Atualiza os resultados ao selecionar o estrato."""
    if estrato not in ESTRATOS:
        estrato = "estrato_1"

    dados = carregar_resultado(estrato)
    metricas = obter_metricas(dados)

    # Quatro indicadores principais.
    kpis = [
        indicador(
            titulo,
            formatar_metrica(metricas.get(chave)),
            (
                "Resultado do conjunto de teste"
                if chave in metricas
                else "Aguardando avaliação"
            ),
        )
        for chave, titulo in METRICAS.items()
    ]

    # Tabela comparativa dos três estratos.
    comparacao = criar_tabela_comparacao()

    # Matriz e importância do estrato escolhido.
    matriz = criar_matriz_confusao(dados)
    importancia = criar_grafico_importancia(dados)

    return kpis, comparacao, matriz, importancia
