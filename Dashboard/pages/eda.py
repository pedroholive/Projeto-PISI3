from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html


dash.register_page(
    __name__,
    path="/eda",
    name="Análise Exploratória",
    order=1
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "healthsync_nhanes_2021_2023_core.csv"
)

df = pd.read_csv(DATA_PATH)


def filtrar_dados(sexo):
    if sexo == "Todos":
        return df.copy()

    return df[df["sexo"] == sexo].copy()


def formatar_inteiro(valor):
    return f"{int(valor):,}".replace(",", ".")


def formatar_decimal(valor, unidade=""):
    valor_formatado = f"{valor:.1f}".replace(".", ",")

    if unidade:
        return f"{valor_formatado} {unidade}"

    return valor_formatado


def criar_fig_hba1c(dados):
    fig = px.histogram(
        dados.dropna(subset=["hba1c"]),
        x="hba1c",
        nbins=30,
        labels={"hba1c": "HbA1c (%)"}
    )

    fig.update_layout(
        title=None,
        xaxis_title="HbA1c (%)",
        yaxis_title="Número de participantes",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )

    fig.update_traces(marker_color="#0E766D")

    return fig


# =========================================================
# ESTILO DOS GRÁFICOS
# =========================================================

COR_PRINCIPAL = "#0E766D"
COR_SECUNDARIA = "#34D399"
COR_TEXTO = "#52615E"
COR_TITULO = "#263330"
COR_GRID = "#E8EFED"
COR_BORDA = "#DCE8E4"


def aplicar_estilo_grafico(
    fig,
    x_title=None,
    y_title=None,
    showlegend=False
):
    """
    Aplica o padrão visual do HealthSync aos gráficos Plotly.
    """

    fig.update_layout(
        title=None,

        font=dict(
            family="Arial, Helvetica, sans-serif",
            size=12,
            color=COR_TEXTO
        ),

        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",

        margin=dict(
            l=55,
            r=25,
            t=25,
            b=55
        ),

        showlegend=showlegend,

        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=COR_BORDA,
            font=dict(
                color=COR_TITULO,
                size=12
            )
        )
    )

    fig.update_xaxes(
        title_text=x_title,

        title_font=dict(
            size=12,
            color=COR_TEXTO
        ),

        tickfont=dict(
            size=11,
            color=COR_TEXTO
        ),

        showgrid=False,

        showline=True,
        linecolor=COR_BORDA,
        linewidth=1,

        ticks="outside",
        tickcolor=COR_BORDA,

        zeroline=False,

        automargin=True
    )

    fig.update_yaxes(
        title_text=y_title,

        title_font=dict(
            size=12,
            color=COR_TEXTO
        ),

        tickfont=dict(
            size=11,
            color=COR_TEXTO
        ),

        showgrid=True,
        gridcolor=COR_GRID,
        gridwidth=1,

        showline=False,

        zeroline=False,

        automargin=True
    )

    return fig


# =========================================================
# HbA1c
# =========================================================

def criar_fig_hba1c(dados):
    dados_grafico = dados.dropna(
        subset=["hba1c"]
    )

    fig = px.histogram(
        dados_grafico,
        x="hba1c",
        nbins=30,
        labels={
            "hba1c": "HbA1c (%)"
        }
    )

    fig.update_traces(
        marker=dict(
            color=COR_PRINCIPAL,
            line=dict(
                color="#FFFFFF",
                width=0.6
            )
        ),

        opacity=0.90,

        hovertemplate=(
            "<b>HbA1c</b>: %{x:.1f}%"
            "<br><b>Participantes</b>: %{y}"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="HbA1c (%)",
        y_title="Número de participantes"
    )

    return fig


# =========================================================
# HbA1c POR DIABETES INFORMADO
# =========================================================

def criar_fig_hba1c_diabetes(dados):
    dados_grafico = dados.dropna(
        subset=[
            "hba1c",
            "diabetes_informado"
        ]
    ).copy()

    fig = px.box(
        dados_grafico,

        x="diabetes_informado",
        y="hba1c",

        category_orders={
            "diabetes_informado": [
                "Nao",
                "Borderline",
                "Sim"
            ]
        },

        labels={
            "diabetes_informado": "Diabetes informado",
            "hba1c": "HbA1c (%)"
        }
    )

    fig.update_traces(
        fillcolor="rgba(14, 118, 109, 0.16)",

        line=dict(
            color=COR_PRINCIPAL,
            width=1.6
        ),

        marker=dict(
            color=COR_PRINCIPAL,
            size=4,
            opacity=0.45
        ),

        hovertemplate=(
            "<b>HbA1c</b>: %{y:.1f}%"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Diabetes informado",
        y_title="HbA1c (%)"
    )

    return fig


# =========================================================
# PRESSÃO SISTÓLICA
# =========================================================

def criar_fig_pressao_sistolica(dados):
    dados_grafico = dados.dropna(
        subset=["pressao_sistolica"]
    )

    fig = px.histogram(
        dados_grafico,

        x="pressao_sistolica",

        nbins=30,

        labels={
            "pressao_sistolica":
                "Pressão sistólica (mmHg)"
        }
    )

    fig.update_traces(
        marker=dict(
            color=COR_PRINCIPAL,

            line=dict(
                color="#FFFFFF",
                width=0.6
            )
        ),

        opacity=0.90,

        hovertemplate=(
            "<b>Pressão sistólica</b>: %{x:.1f} mmHg"
            "<br><b>Participantes</b>: %{y}"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Pressão sistólica (mmHg)",
        y_title="Número de participantes"
    )

    return fig


# =========================================================
# PRESSÃO POR HIPERTENSÃO INFORMADA
# =========================================================

def criar_fig_pressao_hipertensao(dados):
    dados_grafico = dados.dropna(
        subset=[
            "pressao_sistolica",
            "hipertensao_informada"
        ]
    ).copy()

    dados_grafico["hipertensao_label"] = (
        dados_grafico["hipertensao_informada"]
        .map({
            0.0: "Não",
            1.0: "Sim"
        })
    )

    fig = px.box(
        dados_grafico,

        x="hipertensao_label",
        y="pressao_sistolica",

        category_orders={
            "hipertensao_label": [
                "Não",
                "Sim"
            ]
        },

        labels={
            "hipertensao_label":
                "Hipertensão informada",

            "pressao_sistolica":
                "Pressão sistólica (mmHg)"
        }
    )

    fig.update_traces(
        fillcolor="rgba(14, 118, 109, 0.16)",

        line=dict(
            color=COR_PRINCIPAL,
            width=1.6
        ),

        marker=dict(
            color=COR_PRINCIPAL,
            size=4,
            opacity=0.45
        ),

        hovertemplate=(
            "<b>Pressão sistólica</b>: %{y:.1f} mmHg"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Hipertensão informada",
        y_title="Pressão sistólica (mmHg)"
    )

    return fig


# =========================================================
# COLESTEROL
# =========================================================

def criar_fig_colesterol(dados):
    dados_grafico = dados.dropna(
        subset=["colesterol_total"]
    )

    fig = px.histogram(
        dados_grafico,

        x="colesterol_total",

        nbins=30,

        labels={
            "colesterol_total":
                "Colesterol total (mg/dL)"
        }
    )

    fig.update_traces(
        marker=dict(
            color=COR_PRINCIPAL,

            line=dict(
                color="#FFFFFF",
                width=0.6
            )
        ),

        opacity=0.90,

        hovertemplate=(
            "<b>Colesterol total</b>: %{x:.1f} mg/dL"
            "<br><b>Participantes</b>: %{y}"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Colesterol total (mg/dL)",
        y_title="Número de participantes"
    )

    return fig


# =========================================================
# COLESTEROL POR HISTÓRICO INFORMADO
# =========================================================

def criar_fig_colesterol_informado(dados):
    dados_grafico = dados.dropna(
        subset=[
            "colesterol_total",
            "colesterol_alto_informado"
        ]
    ).copy()

    dados_grafico["colesterol_alto_label"] = (
        dados_grafico["colesterol_alto_informado"]
        .map({
            0.0: "Não",
            1.0: "Sim"
        })
    )

    fig = px.violin(
        dados_grafico,

        x="colesterol_alto_label",
        y="colesterol_total",

        box=True,
        points=False,

        category_orders={
            "colesterol_alto_label": [
                "Não",
                "Sim"
            ]
        },

        labels={
            "colesterol_alto_label":
                "Colesterol alto informado",

            "colesterol_total":
                "Colesterol total (mg/dL)"
        }
    )

    fig.update_traces(
        fillcolor="rgba(14, 118, 109, 0.18)",

        line=dict(
            color=COR_PRINCIPAL,
            width=1.6
        ),

        opacity=0.95,

        hovertemplate=(
            "<b>Colesterol total</b>: %{y:.1f} mg/dL"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Colesterol alto informado",
        y_title="Colesterol total (mg/dL)"
    )

    return fig


# =========================================================
# CORRELAÇÃO
# =========================================================

def criar_fig_correlacao(dados):
    colunas_correlacao = [
        "imc",
        "hba1c",
        "pressao_sistolica",
        "pressao_diastolica",
        "colesterol_total"
    ]

    matriz_correlacao = (
        dados[colunas_correlacao]
        .corr()
        .round(2)
    )

    nomes_correlacao = [
        "IMC",
        "HbA1c",
        "Pressão sistólica",
        "Pressão diastólica",
        "Colesterol total"
    ]

    fig = px.imshow(
        matriz_correlacao,

        x=nomes_correlacao,
        y=nomes_correlacao,

        text_auto=".2f",

        aspect="auto",

        color_continuous_scale=[
            [0.00, "#8C6A7C"],
            [0.25, "#C5AEB8"],
            [0.50, "#F5F7F6"],
            [0.75, "#8BC9BA"],
            [1.00, "#0E766D"]
        ],

        zmin=-1,
        zmax=1,

        labels={
            "color": "Correlação"
        }
    )

    fig.update_layout(
        title=None,

        font=dict(
            family="Arial, Helvetica, sans-serif",
            size=12,
            color=COR_TEXTO
        ),

        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",

        margin=dict(
            l=40,
            r=40,
            t=30,
            b=50
        ),

        coloraxis_colorbar=dict(
            title=dict(
                text="Correlação",
                font=dict(
                    size=11,
                    color=COR_TEXTO
                )
            ),

            tickfont=dict(
                size=10,
                color=COR_TEXTO
            ),

            thickness=12,

            len=0.75,

            outlinewidth=0
        )
    )

    fig.update_xaxes(
        title=None,

        tickfont=dict(
            size=11,
            color=COR_TEXTO
        ),

        side="bottom",

        showgrid=False,

        zeroline=False,

        automargin=True
    )

    fig.update_yaxes(
        title=None,

        tickfont=dict(
            size=11,
            color=COR_TEXTO
        ),

        showgrid=False,

        zeroline=False,

        automargin=True
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b> × <b>%{x}</b>"
            "<br>Correlação: %{z:.2f}"
            "<extra></extra>"
        )
    )

    return fig


# LAYOUT

layout = html.Div(
    [
        # CABEÇALHO
        
        html.Section(
            [
                html.Div(
                    [
                        # Conteúdo do cabeçalho
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            className="eda-header-dot"
                                        ),

                                        html.Span(
                                            "ANÁLISE EXPLORATÓRIA"
                                        ),
                                    ],
                                    className="eda-eyebrow"
                                ),

                                html.H1(
                                    "Explore os indicadores "
                                    "relacionados à saúde cardiometabólica.",
                                    className="eda-title"
                                ),

                                html.P(
                                    "Explore a distribuição e as relações "
                                    "entre indicadores como HbA1c, pressão "
                                    "arterial, IMC e colesterol nos dados "
                                    "do NHANES 2021–2023.",
                                    className="eda-subtitle"
                                ),
                            ],
                            className="eda-header-content"
                        ),

                        # Filtro
                        html.Div(
                            [
                                html.Label(
                                    "Filtrar por sexo",
                                    htmlFor="filtro-sexo",
                                    className="eda-filter-label"
                                ),

                                dcc.Dropdown(
                                    id="filtro-sexo",
                                    options=[
                                        {
                                            "label": "Todos",
                                            "value": "Todos"
                                        },
                                        {
                                            "label": "Mulher",
                                            "value": "Mulher"
                                        },
                                        {
                                            "label": "Homem",
                                            "value": "Homem"
                                        },
                                    ],
                                    value="Todos",
                                    clearable=False,
                                    searchable=False,
                                    className="eda-dropdown"
                                ),
                            ],
                            className="eda-header-filter"
                        ),
                    ],
                    className="eda-header-inner"
                )
            ],
            className="eda-header"
        ),


        # =====================================================
        # VISÃO GERAL
        # =====================================================
        html.Section(
            [
                html.Div(
                    [
                        # Cabeçalho da seção
                        html.Div(
                            [
                                html.P(
                                    "VISÃO GERAL",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Visão geral da amostra",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Resumo dos principais indicadores "
                                    "considerando o filtro selecionado.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        # Cards
                        html.Div(
                            [
                                # Participantes
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "AMOSTRA",
                                                    className="eda-card-tag"
                                                ),

                                                html.P(
                                                    "Participantes",
                                                    className="eda-card-label"
                                                ),
                                            ]
                                        ),

                                        html.H3(
                                            id="card-participantes",
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

                                # Idade
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "IDADE",
                                                    className="eda-card-tag"
                                                ),

                                                html.P(
                                                    "Idade média",
                                                    className="eda-card-label"
                                                ),
                                            ]
                                        ),

                                        html.H3(
                                            id="card-idade-media",
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

                                # IMC
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "IMC",
                                                    className="eda-card-tag"
                                                ),

                                                html.P(
                                                    "IMC médio",
                                                    className="eda-card-label"
                                                ),
                                            ]
                                        ),

                                        html.H3(
                                            id="card-imc-medio",
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

                                # Pressão sistólica
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "PAS",
                                                    className="eda-card-tag"
                                                ),

                                                html.P(
                                                    "Pressão sistólica média",
                                                    className="eda-card-label"
                                                ),
                                            ]
                                        ),

                                        html.H3(
                                            id="card-pressao-sistolica",
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

                                # Pressão diastólica
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "PAD",
                                                    className="eda-card-tag"
                                                ),

                                                html.P(
                                                    "Pressão diastólica média",
                                                    className="eda-card-label"
                                                ),
                                            ]
                                        ),

                                        html.H3(
                                            id="card-pressao-diastolica",
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

                                # Colesterol
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "COL",
                                                    className="eda-card-tag"
                                                ),

                                                html.P(
                                                    "Colesterol total médio",
                                                    className="eda-card-label"
                                                ),
                                            ]
                                        ),

                                        html.H3(
                                            id="card-colesterol-medio",
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),
                            ],
                            className="eda-summary-grid"
                        ),
                    ],
                    className="eda-content-container"
                )
            ],
            className="eda-overview-section"
        ),


        # =====================================================
        # PERFIL GLICÊMICO
        # =====================================================
        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "GLICEMIA",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Perfil glicêmico",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "A HbA1c é utilizada nesta análise como um dos indicadores do perfil glicêmico da amostra. "
                                    "Sua distribuição permite observar a variabilidade entre os participantes e "
                                    "comparar os valores medidos com as informações de diabetes relatadas no questionário. ",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                # Distribuição HbA1c
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Distribuição de HbA1c",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição dos níveis "
                                                    "de hemoglobina glicada "
                                                    "entre os participantes.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id="grafico-hba1c",
                                            config={
                                                "displayModeBar": False
                                            },
                                            className="eda-graph"
                                        ),
                                    ],
                                    className="eda-chart-card"
                                ),

                                # HbA1c por diabetes informado
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "HbA1c por diabetes "
                                                    "informado",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Comparação da "
                                                    "distribuição de HbA1c "
                                                    "entre participantes que "
                                                    "informaram diferentes "
                                                    "condições relacionadas "
                                                    "ao diabetes.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-hba1c-diabetes"
                                            ),
                                            config={
                                                "displayModeBar": False
                                            },
                                            className="eda-graph"
                                        ),
                                    ],
                                    className="eda-chart-card"
                                ),
                            ],
                            className="eda-charts-grid"
                        ),
                    ],
                    className="eda-content-container"
                )
            ],
            className="eda-analysis-section"
        ),


        # =====================================================
        # PRESSÃO ARTERIAL
        # =====================================================
        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "PRESSÃO ARTERIAL",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Perfil da pressão arterial",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "As medidas de pressão arterial contribuem para a caracterização do perfil cardiovascular da amostra. "
                                    "Nesta seção, a pressão sistólica é explorada tanto em sua distribuição geral quanto em "
                                    "comparação com o histórico de hipertensão informado pelos participantes. ",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                # Distribuição da pressão
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Distribuição da pressão "
                                                    "sistólica",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição das medidas "
                                                    "de pressão arterial "
                                                    "sistólica entre os "
                                                    "participantes.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id="grafico-pressao-sistolica",
                                            config={
                                                "displayModeBar": False
                                            },
                                            className="eda-graph"
                                        ),
                                    ],
                                    className="eda-chart-card"
                                ),

                                # Pressão por hipertensão
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Pressão sistólica por "
                                                    "hipertensão informada",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Comparação da pressão "
                                                    "sistólica entre "
                                                    "participantes com e sem "
                                                    "hipertensão informada.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-pressao-hipertensao"
                                            ),
                                            config={
                                                "displayModeBar": False
                                            },
                                            className="eda-graph"
                                        ),
                                    ],
                                    className="eda-chart-card"
                                ),
                            ],
                            className="eda-charts-grid"
                        ),
                    ],
                    className="eda-content-container"
                )
            ],
            className="eda-analysis-section eda-section-alt"
        ),


        # =====================================================
        # PERFIL LIPÍDICO
        # =====================================================
        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "COLESTEROL",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Perfil lipídico",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "O colesterol total compõe o conjunto de indicadores analisados para caracterizar aspectos do perfil lipídico da amostra. "
                                    "As visualizações permitem observar sua distribuição e comparar os valores "
                                    "medidos entre participantes com e sem histórico informado de colesterol alto.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                # Distribuição colesterol
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Distribuição do "
                                                    "colesterol total",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição das medidas "
                                                    "de colesterol total entre "
                                                    "os participantes.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id="grafico-colesterol",
                                            config={
                                                "displayModeBar": False
                                            },
                                            className="eda-graph"
                                        ),
                                    ],
                                    className="eda-chart-card"
                                ),

                                # Colesterol informado
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Colesterol total por "
                                                    "histórico informado",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Comparação do colesterol "
                                                    "total medido entre "
                                                    "participantes que "
                                                    "informaram ou não "
                                                    "histórico de colesterol "
                                                    "alto.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-colesterol-informado"
                                            ),
                                            config={
                                                "displayModeBar": False
                                            },
                                            className="eda-graph"
                                        ),
                                    ],
                                    className="eda-chart-card"
                                ),
                            ],
                            className="eda-charts-grid"
                        ),
                    ],
                    className="eda-content-container"
                )
            ],
            className="eda-analysis-section"
        ),


        # =====================================================
        # CORRELAÇÕES
        # =====================================================
        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "RELAÇÕES ENTRE INDICADORES",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Relações cardiometabólicas",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "A análise de correlação permite explorar como alguns dos indicadores selecionados variam conjuntamente na amostra. "
                                    "A matriz apresenta a intensidade e a direção das relações lineares entre IMC, HbA1c, pressão arterial e colesterol total. "
                                    "Esses resultados são exploratórios e não estabelecem relações de causa e efeito.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3(
                                            "Correlação entre indicadores",
                                            className="eda-chart-title"
                                        ),

                                        html.P(
                                            "Valores próximos de 1 ou -1 indicam relações "
                                            "lineares mais fortes, enquanto valores próximos "
                                            "de 0 indicam relações lineares mais fracas.",
                                            className="eda-chart-description"
                                        ),
                                    ],
                                    className="eda-chart-header"
                                ),

                                dcc.Graph(
                                    id="grafico-correlacao",
                                    config={
                                        "displayModeBar": False
                                    },
                                    className="eda-graph"
                                ),

                                html.Div(
                                    [
                                        html.Span(
                                            "i",
                                            className="eda-info-icon"
                                        ),

                                        html.P(
                                            "A correlação descreve a "
                                            "associação linear entre os "
                                            "indicadores e não estabelece "
                                            "uma relação de causalidade."
                                        ),
                                    ],
                                    className="eda-correlation-note"
                                ),
                            ],
                            className=(
                                "eda-chart-card "
                                "eda-correlation-card"
                            )
                        ),
                    ],
                    className="eda-content-container"
                )
            ],
            className="eda-analysis-section eda-section-alt"
        ),
    ],
    className="eda-page"
)


@callback(
    Output("card-participantes", "children"),
    Output("card-idade-media", "children"),
    Output("card-imc-medio", "children"),
    Output("card-pressao-sistolica", "children"),
    Output("card-pressao-diastolica", "children"),
    Output("card-colesterol-medio", "children"),
    Output("grafico-hba1c", "figure"),
    Output("grafico-hba1c-diabetes", "figure"),
    Output("grafico-pressao-sistolica", "figure"),
    Output("grafico-pressao-hipertensao", "figure"),
    Output("grafico-colesterol", "figure"),
    Output("grafico-colesterol-informado", "figure"),
    Output("grafico-correlacao", "figure"),
    Input("filtro-sexo", "value")
)
def atualizar_dashboard(sexo):
    dados = filtrar_dados(sexo)

    total_participantes = dados["id_participante"].nunique()
    idade_media = dados["idade"].mean()
    imc_medio = dados["imc"].mean()

    pressao_sistolica_media = (
        dados["pressao_sistolica"].mean()
    )

    pressao_diastolica_media = (
        dados["pressao_diastolica"].mean()
    )

    colesterol_total_medio = (
        dados["colesterol_total"].mean()
    )

    card_participantes = formatar_inteiro(
        total_participantes
    )

    card_idade_media = formatar_decimal(
        idade_media,
        "anos"
    )

    card_imc_medio = formatar_decimal(
        imc_medio,
        "kg/m²"
    )

    card_pressao_sistolica = formatar_decimal(
        pressao_sistolica_media,
        "mmHg"
    )

    card_pressao_diastolica = formatar_decimal(
        pressao_diastolica_media,
        "mmHg"
    )

    card_colesterol_medio = formatar_decimal(
        colesterol_total_medio,
        "mg/dL"
    )

    fig_hba1c = criar_fig_hba1c(dados)

    fig_hba1c_diabetes = criar_fig_hba1c_diabetes(
        dados
    )

    fig_pressao_sistolica = criar_fig_pressao_sistolica(
        dados
    )

    fig_pressao_hipertensao = criar_fig_pressao_hipertensao(
        dados
    )

    fig_colesterol = criar_fig_colesterol(dados)

    fig_colesterol_informado = (
        criar_fig_colesterol_informado(dados)
    )

    fig_correlacao = criar_fig_correlacao(dados)

    return (
        card_participantes,
        card_idade_media,
        card_imc_medio,
        card_pressao_sistolica,
        card_pressao_diastolica,
        card_colesterol_medio,
        fig_hba1c,
        fig_hba1c_diabetes,
        fig_pressao_sistolica,
        fig_pressao_hipertensao,
        fig_colesterol,
        fig_colesterol_informado,
        fig_correlacao
    )