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


def criar_fig_hba1c_diabetes(dados):
    dados_grafico = dados.dropna(
        subset=["hba1c", "diabetes_informado"]
    )

    fig = px.box(
        dados_grafico,
        x="diabetes_informado",
        y="hba1c",
        category_orders={
            "diabetes_informado": ["Nao", "Borderline", "Sim"]
        },
        labels={
            "diabetes_informado": "Diabetes informado",
            "hba1c": "HbA1c (%)"
        }
    )

    fig.update_layout(
        title=None,
        xaxis_title="Diabetes informado",
        yaxis_title="HbA1c (%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )

    fig.update_traces(
        marker_color="#0E766D",
        line_color="#0E766D"
    )

    return fig


def criar_fig_pressao_sistolica(dados):
    fig = px.histogram(
        dados.dropna(subset=["pressao_sistolica"]),
        x="pressao_sistolica",
        nbins=30,
        labels={
            "pressao_sistolica": "Pressão sistólica (mmHg)"
        }
    )

    fig.update_layout(
        title=None,
        xaxis_title="Pressão sistólica (mmHg)",
        yaxis_title="Número de participantes",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )

    fig.update_traces(marker_color="#0E766D")

    return fig


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
            "hipertensao_label": ["Não", "Sim"]
        },
        labels={
            "hipertensao_label": "Hipertensão informada",
            "pressao_sistolica": "Pressão sistólica (mmHg)"
        }
    )

    fig.update_layout(
        title=None,
        xaxis_title="Hipertensão informada",
        yaxis_title="Pressão sistólica (mmHg)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )

    fig.update_traces(
        marker_color="#0E766D",
        line_color="#0E766D"
    )

    return fig


def criar_fig_colesterol(dados):
    fig = px.histogram(
        dados.dropna(subset=["colesterol_total"]),
        x="colesterol_total",
        nbins=30,
        labels={
            "colesterol_total": "Colesterol total (mg/dL)"
        }
    )

    fig.update_layout(
        title=None,
        xaxis_title="Colesterol total (mg/dL)",
        yaxis_title="Número de participantes",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )

    fig.update_traces(marker_color="#0E766D")

    return fig


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
            "colesterol_alto_label": ["Não", "Sim"]
        },
        labels={
            "colesterol_alto_label": "Colesterol alto informado",
            "colesterol_total": "Colesterol total (mg/dL)"
        }
    )

    fig.update_layout(
        title=None,
        xaxis_title="Colesterol alto informado",
        yaxis_title="Colesterol total (mg/dL)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )

    fig.update_traces(
        fillcolor="rgba(14, 118, 109, 0.35)",
        line_color="#0E766D"
    )

    return fig


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
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels={"color": "Correlação"}
    )

    fig.update_layout(
        title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
        coloraxis_colorbar=dict(title="Correlação")
    )

    return fig


layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Análise Exploratória dos Dados",
                    className="eda-title"
                ),
                html.P(
                    "Explore os principais indicadores "
                    "cardiometabólicos da amostra.",
                    className="eda-subtitle"
                ),
            ],
            className="eda-header"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Sexo",
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
                    className="eda-filter-item"
                ),
            ],
            className="eda-filter-card"
        ),

        html.H2(
            "Visão geral da amostra",
            className="eda-section-title"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Participantes",
                            className="eda-card-label"
                        ),
                        html.H3(
                            id="card-participantes",
                            className="eda-card-value"
                        ),
                    ],
                    className="eda-card"
                ),

                html.Div(
                    [
                        html.P(
                            "Idade média",
                            className="eda-card-label"
                        ),
                        html.H3(
                            id="card-idade-media",
                            className="eda-card-value"
                        ),
                    ],
                    className="eda-card"
                ),

                html.Div(
                    [
                        html.P(
                            "IMC médio",
                            className="eda-card-label"
                        ),
                        html.H3(
                            id="card-imc-medio",
                            className="eda-card-value"
                        ),
                    ],
                    className="eda-card"
                ),
            ],
            className="eda-cards"
        ),

        html.H2(
            "Perfil glicêmico",
            className="eda-section-title"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "Distribuição de HbA1c",
                            className="eda-chart-title"
                        ),
                        html.P(
                            "Distribuição dos níveis de hemoglobina "
                            "glicada entre os participantes.",
                            className="eda-chart-description"
                        ),
                        dcc.Graph(
                            id="grafico-hba1c",
                            config={"displayModeBar": False},
                            className="eda-graph"
                        ),
                    ],
                    className="eda-chart-card"
                ),

                html.Div(
                    [
                        html.H3(
                            "HbA1c por diabetes informado",
                            className="eda-chart-title"
                        ),
                        html.P(
                            "Comparação da distribuição de HbA1c "
                            "entre participantes que informaram "
                            "diferentes condições relacionadas "
                            "ao diabetes.",
                            className="eda-chart-description"
                        ),
                        dcc.Graph(
                            id="grafico-hba1c-diabetes",
                            config={"displayModeBar": False},
                            className="eda-graph"
                        ),
                    ],
                    className="eda-chart-card"
                ),
            ],
            className="eda-charts-grid"
        ),

        html.H2(
            "Pressão arterial",
            className="eda-section-title"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Pressão sistólica média",
                            className="eda-card-label"
                        ),
                        html.H3(
                            id="card-pressao-sistolica",
                            className="eda-card-value"
                        ),
                    ],
                    className="eda-card"
                ),

                html.Div(
                    [
                        html.P(
                            "Pressão diastólica média",
                            className="eda-card-label"
                        ),
                        html.H3(
                            id="card-pressao-diastolica",
                            className="eda-card-value"
                        ),
                    ],
                    className="eda-card"
                ),
            ],
            className="eda-pressure-cards"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "Distribuição da pressão sistólica",
                            className="eda-chart-title"
                        ),
                        html.P(
                            "Distribuição das medidas de pressão "
                            "arterial sistólica entre os participantes.",
                            className="eda-chart-description"
                        ),
                        dcc.Graph(
                            id="grafico-pressao-sistolica",
                            config={"displayModeBar": False},
                            className="eda-graph"
                        ),
                    ],
                    className="eda-chart-card"
                ),

                html.Div(
                    [
                        html.H3(
                            "Pressão sistólica por hipertensão informada",
                            className="eda-chart-title"
                        ),
                        html.P(
                            "Comparação da pressão sistólica entre "
                            "participantes com e sem hipertensão "
                            "informada.",
                            className="eda-chart-description"
                        ),
                        dcc.Graph(
                            id="grafico-pressao-hipertensao",
                            config={"displayModeBar": False},
                            className="eda-graph"
                        ),
                    ],
                    className="eda-chart-card"
                ),
            ],
            className="eda-charts-grid"
        ),

        html.H2(
            "Perfil lipídico",
            className="eda-section-title"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Colesterol total médio",
                            className="eda-card-label"
                        ),
                        html.H3(
                            id="card-colesterol-medio",
                            className="eda-card-value"
                        ),
                    ],
                    className="eda-card"
                ),
            ],
            className="eda-lipid-cards"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "Distribuição do colesterol total",
                            className="eda-chart-title"
                        ),
                        html.P(
                            "Distribuição das medidas de colesterol "
                            "total entre os participantes.",
                            className="eda-chart-description"
                        ),
                        dcc.Graph(
                            id="grafico-colesterol",
                            config={"displayModeBar": False},
                            className="eda-graph"
                        ),
                    ],
                    className="eda-chart-card"
                ),

                html.Div(
                    [
                        html.H3(
                            "Colesterol total por histórico informado",
                            className="eda-chart-title"
                        ),
                        html.P(
                            "Comparação do colesterol total medido "
                            "entre participantes que informaram ou não "
                            "histórico de colesterol alto.",
                            className="eda-chart-description"
                        ),
                        dcc.Graph(
                            id="grafico-colesterol-informado",
                            config={"displayModeBar": False},
                            className="eda-graph"
                        ),
                    ],
                    className="eda-chart-card"
                ),
            ],
            className="eda-charts-grid"
        ),

        html.H2(
            "Relações cardiometabólicas",
            className="eda-section-title"
        ),

        html.Div(
            [
                html.H3(
                    "Correlação entre indicadores",
                    className="eda-chart-title"
                ),
                html.P(
                    "Matriz de correlação entre IMC, HbA1c, "
                    "pressão arterial e colesterol total. "
                    "Valores próximos de 1 ou -1 indicam relações "
                    "lineares mais fortes, enquanto valores próximos "
                    "de 0 indicam relações lineares mais fracas.",
                    className="eda-chart-description"
                ),
                dcc.Graph(
                    id="grafico-correlacao",
                    config={"displayModeBar": False},
                    className="eda-graph"
                ),
            ],
            className="eda-chart-card eda-correlation-card"
        ),
    ],
    className="eda-container"
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