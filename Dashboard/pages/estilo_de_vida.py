from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "healthsync_nhanes_2021_2023_core.csv"
)

df = pd.read_csv(DATA_PATH)


dash.register_page(
    __name__,
    path="/estilo-de-vida",
    name="Estilo de Vida",
    order=3
)


COR_PRINCIPAL = "#0E766D"
COR_TEXTO = "#52615E"
COR_GRID = "#E8EFED"
COR_BORDA = "#DCE8E4"


def aplicar_estilo_grafico(
    fig,
    x_title=None,
    y_title=None,
    showlegend=False
):
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
        showlegend=showlegend
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


def criar_fig_sem_dados():
    fig = px.scatter()

    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(
            visible=False
        ),
        yaxis=dict(
            visible=False
        ),
        annotations=[
            dict(
                text=(
                    "Nenhum participante corresponde "
                    "aos filtros selecionados."
                ),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(
                    size=14,
                    color=COR_TEXTO
                )
            )
        ]
    )

    return fig


def formatar_minutos(valor):
    if pd.isna(valor):
        return "—"

    return f"{valor:,.0f}".replace(",", ".")


def criar_fig_atividade_moderada(dados):
    dados_grafico = dados.dropna(
        subset=["atv_moderada_min_semana"]
    )

    if dados_grafico.empty:
        return criar_fig_sem_dados()

    fig = px.histogram(
        dados_grafico,
        x="atv_moderada_min_semana",
        nbins=30,
        labels={
            "atv_moderada_min_semana":
                "Atividade moderada (min/semana)"
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
            "<b>Atividade moderada</b>: "
            "%{x:.0f} min/semana"
            "<br><b>Participantes</b>: %{y}"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Atividade moderada (min/semana)",
        y_title="Número de participantes"
    )

    return fig


def criar_fig_atividade_vigorosa(dados):
    dados_grafico = dados.dropna(
        subset=["atv_vigorosa_min_semana"]
    )

    if dados_grafico.empty:
        return criar_fig_sem_dados()

    fig = px.histogram(
        dados_grafico,
        x="atv_vigorosa_min_semana",
        nbins=30,
        labels={
            "atv_vigorosa_min_semana":
                "Atividade vigorosa (min/semana)"
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
            "<b>Atividade vigorosa</b>: "
            "%{x:.0f} min/semana"
            "<br><b>Participantes</b>: %{y}"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Atividade vigorosa (min/semana)",
        y_title="Número de participantes"
    )

    return fig


def criar_fig_sedentarismo(dados):
    dados_grafico = dados.dropna(
        subset=["minutos_sedentarios_dia"]
    )

    if dados_grafico.empty:
        return criar_fig_sem_dados()

    fig = px.histogram(
        dados_grafico,
        x="minutos_sedentarios_dia",
        nbins=30,
        labels={
            "minutos_sedentarios_dia":
                "Tempo sedentário (min/dia)"
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
            "<b>Tempo sedentário</b>: "
            "%{x:.0f} min/dia"
            "<br><b>Participantes</b>: %{y}"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Tempo sedentário (min/dia)",
        y_title="Número de participantes"
    )

    return fig


def criar_fig_atividade_sexo(dados):
    dados_grafico = dados.dropna(
        subset=[
            "atv_moderada_min_semana",
            "sexo"
        ]
    ).copy()

    if dados_grafico.empty:
        return criar_fig_sem_dados()

    fig = px.box(
        dados_grafico,
        x="sexo",
        y="atv_moderada_min_semana",
        category_orders={
            "sexo": [
                "Mulher",
                "Homem"
            ]
        },
        labels={
            "sexo": "Sexo",
            "atv_moderada_min_semana":
                "Atividade moderada (min/semana)"
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
            opacity=0.35
        ),
        hovertemplate=(
            "<b>Atividade moderada</b>: "
            "%{y:.0f} min/semana"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Sexo",
        y_title="Atividade moderada (min/semana)"
    )

    return fig


def criar_fig_sedentarismo_idade(dados):
    dados_grafico = dados.dropna(
        subset=[
            "idade",
            "minutos_sedentarios_dia"
        ]
    ).copy()

    if dados_grafico.empty:
        return criar_fig_sem_dados()

    dados_grafico["faixa_etaria"] = pd.cut(
        dados_grafico["idade"],
        bins=[
            19,
            39,
            59,
            float("inf")
        ],
        labels=[
            "20–39 anos",
            "40–59 anos",
            "60+ anos"
        ]
    )

    fig = px.box(
        dados_grafico,
        x="faixa_etaria",
        y="minutos_sedentarios_dia",
        category_orders={
            "faixa_etaria": [
                "20–39 anos",
                "40–59 anos",
                "60+ anos"
            ]
        },
        labels={
            "faixa_etaria": "Faixa etária",
            "minutos_sedentarios_dia":
                "Tempo sedentário (min/dia)"
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
            opacity=0.35
        ),
        hovertemplate=(
            "<b>Tempo sedentário</b>: "
            "%{y:.0f} min/dia"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Faixa etária",
        y_title="Tempo sedentário (min/dia)"
    )

    return fig


def filtrar_dados(
    dados,
    sexo,
    faixa_etaria
):
    dados_filtrados = dados.copy()

    if sexo != "Todos":
        dados_filtrados = dados_filtrados[
            dados_filtrados["sexo"] == sexo
        ]

    if faixa_etaria == "20-39":
        dados_filtrados = dados_filtrados[
            dados_filtrados["idade"].between(
                20,
                39
            )
        ]

    elif faixa_etaria == "40-59":
        dados_filtrados = dados_filtrados[
            dados_filtrados["idade"].between(
                40,
                59
            )
        ]

    elif faixa_etaria == "60+":
        dados_filtrados = dados_filtrados[
            dados_filtrados["idade"] >= 60
        ]

    return dados_filtrados


def criar_filtro(
    titulo,
    id_componente,
    opcoes
):
    return html.Div(
        [
            html.P(
                titulo,
                style={
                    "fontSize": "12px",
                    "fontWeight": "700",
                    "letterSpacing": "0.06em",
                    "textTransform": "uppercase",
                    "color": COR_TEXTO,
                    "margin": "0 0 10px 0"
                }
            ),

            dcc.RadioItems(
                id=id_componente,
                options=opcoes,
                value="Todos",
                inline=True,
                labelStyle={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "marginRight": "20px",
                    "marginBottom": "8px",
                    "cursor": "pointer",
                    "fontSize": "14px",
                    "fontWeight": "500",
                    "color": COR_TEXTO
                },
                inputStyle={
                    "marginRight": "7px",
                    "accentColor": COR_PRINCIPAL
                }
            ),
        ],
        style={
            "padding": "16px 0",
            "borderBottom": "1px solid #E8EFED"
        }
    )


layout = html.Div(
    [
        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            className="eda-header-dot"
                                        ),

                                        html.Span(
                                            "ESTILO DE VIDA"
                                        ),
                                    ],
                                    className="eda-eyebrow"
                                ),

                                html.H1(
                                    "Estilo de Vida",
                                    className="eda-title"
                                ),

                                html.P(
                                    "Explore características relacionadas "
                                    "à atividade física e ao comportamento "
                                    "sedentário dos participantes.",
                                    className="eda-subtitle"
                                ),
                            ],
                            className="eda-header-content"
                        ),
                    ],
                    className="eda-header-inner"
                ),
            ],
            className="eda-header"
        ),

        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "EXPLORE OS DADOS",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Explore a amostra",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Escolha características dos "
                                    "participantes para observar como "
                                    "os indicadores de atividade física "
                                    "e comportamento sedentário mudam "
                                    "entre diferentes grupos.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                criar_filtro(
                                    "Quem você quer analisar?",
                                    "filtro-estilo-sexo",
                                    [
                                        {
                                            "label": "Todos",
                                            "value": "Todos"
                                        },
                                        {
                                            "label": "Mulheres",
                                            "value": "Mulher"
                                        },
                                        {
                                            "label": "Homens",
                                            "value": "Homem"
                                        }
                                    ]
                                ),

                                criar_filtro(
                                    "Qual faixa etária?",
                                    "filtro-estilo-idade",
                                    [
                                        {
                                            "label": "Todos",
                                            "value": "Todos"
                                        },
                                        {
                                            "label": "20–39 anos",
                                            "value": "20-39"
                                        },
                                        {
                                            "label": "40–59 anos",
                                            "value": "40-59"
                                        },
                                        {
                                            "label": "60+ anos",
                                            "value": "60+"
                                        }
                                    ]
                                ),

                                html.Div(
                                    [
                                        html.Span(
                                            id="contador-estilo",
                                            style={
                                                "fontSize": "22px",
                                                "fontWeight": "700",
                                                "color": COR_PRINCIPAL
                                            }
                                        ),

                                        html.P(
                                            "Os indicadores e gráficos "
                                            "abaixo representam o grupo "
                                            "selecionado.",
                                            style={
                                                "margin": "4px 0 0 0",
                                                "fontSize": "13px",
                                                "color": COR_TEXTO
                                            }
                                        ),
                                    ],
                                    style={
                                        "marginTop": "20px",
                                        "padding": "18px 20px",
                                        "background": "#F1F8F6",
                                        "borderRadius": "12px",
                                        "border": (
                                            "1px solid #DCE8E4"
                                        )
                                    }
                                ),
                            ],
                            style={
                                "background": "#FFFFFF",
                                "border": "1px solid #DCE8E4",
                                "borderRadius": "16px",
                                "padding": "8px 24px 24px 24px"
                            }
                        ),
                    ],
                    className="eda-content-container"
                ),
            ],
            className="eda-analysis-section"
        ),

        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "RESUMO",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Indicadores de estilo de vida",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Resumo dos tempos de atividade "
                                    "física e comportamento sedentário "
                                    "do grupo selecionado.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "ATIVIDADE MODERADA",
                                            className="eda-card-tag"
                                        ),

                                        html.P(
                                            "Mediana semanal",
                                            className="eda-card-label"
                                        ),

                                        html.H3(
                                            id=(
                                                "card-estilo-"
                                                "atividade-moderada"
                                            ),
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            "ATIVIDADE VIGOROSA",
                                            className="eda-card-tag"
                                        ),

                                        html.P(
                                            "Mediana semanal",
                                            className="eda-card-label"
                                        ),

                                        html.H3(
                                            id=(
                                                "card-estilo-"
                                                "atividade-vigorosa"
                                            ),
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            "SEDENTARISMO",
                                            className="eda-card-tag"
                                        ),

                                        html.P(
                                            "Mediana diária",
                                            className="eda-card-label"
                                        ),

                                        html.H3(
                                            id=(
                                                "card-estilo-"
                                                "sedentarismo"
                                            ),
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
                ),
            ],
            className="eda-analysis-section"
        ),

        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "ATIVIDADE FÍSICA",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Níveis de atividade física",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Distribuição do tempo semanal "
                                    "dedicado a atividades físicas "
                                    "moderadas e vigorosas.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Atividade moderada",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição dos "
                                                    "minutos semanais de "
                                                    "atividade física "
                                                    "moderada.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-estilo-"
                                                "atividade-moderada"
                                            ),
                                            config={
                                                "displayModeBar": False
                                            },
                                            className="eda-graph"
                                        ),
                                    ],
                                    className="eda-chart-card"
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Atividade vigorosa",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição dos "
                                                    "minutos semanais de "
                                                    "atividade física "
                                                    "vigorosa.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-estilo-"
                                                "atividade-vigorosa"
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
                ),
            ],
            className=(
                "eda-analysis-section eda-section-alt"
            )
        ),

        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "COMPORTAMENTO SEDENTÁRIO",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Tempo sedentário diário",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Distribuição do tempo diário "
                                    "de comportamento sedentário "
                                    "registrado para os participantes.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Minutos sedentários "
                                                    "por dia",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição do tempo "
                                                    "sedentário diário "
                                                    "na amostra.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-estilo-"
                                                "sedentarismo"
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
                ),
            ],
            className="eda-analysis-section"
        ),

        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "COMPARAÇÕES",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Estilo de vida entre grupos",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Comparações descritivas de "
                                    "atividade física e comportamento "
                                    "sedentário entre diferentes "
                                    "grupos da amostra.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Atividade moderada "
                                                    "por sexo",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Comparação da "
                                                    "distribuição dos "
                                                    "minutos semanais de "
                                                    "atividade moderada "
                                                    "entre mulheres e "
                                                    "homens.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-estilo-"
                                                "atividade-sexo"
                                            ),
                                            config={
                                                "displayModeBar": False
                                            },
                                            className="eda-graph"
                                        ),
                                    ],
                                    className="eda-chart-card"
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Tempo sedentário "
                                                    "por faixa etária",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Comparação do tempo "
                                                    "sedentário diário "
                                                    "entre diferentes "
                                                    "faixas de idade.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-estilo-"
                                                "sedentarismo-idade"
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
                ),
            ],
            className=(
                "eda-analysis-section eda-section-alt"
            )
        ),
    ],
    className="eda-page"
)


@callback(
    Output(
        "card-estilo-atividade-moderada",
        "children"
    ),
    Output(
        "card-estilo-atividade-vigorosa",
        "children"
    ),
    Output(
        "card-estilo-sedentarismo",
        "children"
    ),
    Output(
        "grafico-estilo-atividade-moderada",
        "figure"
    ),
    Output(
        "grafico-estilo-atividade-vigorosa",
        "figure"
    ),
    Output(
        "grafico-estilo-sedentarismo",
        "figure"
    ),
    Output(
        "grafico-estilo-atividade-sexo",
        "figure"
    ),
    Output(
        "grafico-estilo-sedentarismo-idade",
        "figure"
    ),
    Output(
        "contador-estilo",
        "children"
    ),
    Input(
        "filtro-estilo-sexo",
        "value"
    ),
    Input(
        "filtro-estilo-idade",
        "value"
    )
)
def atualizar_estilo_de_vida(
    sexo,
    faixa_etaria
):
    dados_filtrados = filtrar_dados(
        df,
        sexo,
        faixa_etaria
    )

    mediana_moderada = (
        dados_filtrados[
            "atv_moderada_min_semana"
        ].median()
    )

    mediana_vigorosa = (
        dados_filtrados[
            "atv_vigorosa_min_semana"
        ].median()
    )

    mediana_sedentarismo = (
        dados_filtrados[
            "minutos_sedentarios_dia"
        ].median()
    )

    card_moderada = (
        f"{formatar_minutos(mediana_moderada)} min"
    )

    card_vigorosa = (
        f"{formatar_minutos(mediana_vigorosa)} min"
    )

    card_sedentarismo = (
        f"{formatar_minutos(mediana_sedentarismo)} min"
    )

    quantidade = dados_filtrados[
        "id_participante"
    ].nunique()

    if quantidade == 0:
        contador = "Nenhum participante encontrado"
    elif quantidade == 1:
        contador = "1 participante encontrado"
    else:
        contador = (
            f"{quantidade:,}"
            .replace(",", ".")
            + " participantes encontrados"
        )

    return (
        card_moderada,
        card_vigorosa,
        card_sedentarismo,
        criar_fig_atividade_moderada(
            dados_filtrados
        ),
        criar_fig_atividade_vigorosa(
            dados_filtrados
        ),
        criar_fig_sedentarismo(
            dados_filtrados
        ),
        criar_fig_atividade_sexo(
            dados_filtrados
        ),
        criar_fig_sedentarismo_idade(
            dados_filtrados
        ),
        contador
    )