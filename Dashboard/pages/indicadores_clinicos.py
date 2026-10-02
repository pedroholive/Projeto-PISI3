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
    path="/indicadores-clinicos",
    name="Indicadores Clínicos",
    order=2
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
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
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


def criar_fig_hba1c(dados):
    dados_grafico = dados.dropna(
        subset=["hba1c"]
    )

    if dados_grafico.empty:
        return criar_fig_sem_dados()

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


def criar_fig_hba1c_diabetes(dados):
    dados_grafico = dados.dropna(
        subset=[
            "hba1c",
            "diabetes_informado"
        ]
    ).copy()

    if dados_grafico.empty:
        return criar_fig_sem_dados()

    dados_grafico["diabetes_label"] = (
        dados_grafico["diabetes_informado"]
        .replace({
            "Nao": "Não"
        })
    )

    fig = px.box(
        dados_grafico,
        x="diabetes_label",
        y="hba1c",
        category_orders={
            "diabetes_label": [
                "Não",
                "Borderline",
                "Sim"
            ]
        },
        labels={
            "diabetes_label":
                "Diabetes informado",
            "hba1c":
                "HbA1c (%)"
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


def criar_fig_pressao_sistolica(dados):
    dados_grafico = dados.dropna(
        subset=["pressao_sistolica"]
    )

    if dados_grafico.empty:
        return criar_fig_sem_dados()

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
            "<b>Pressão sistólica</b>: "
            "%{x:.1f} mmHg"
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


def criar_fig_pressao_hipertensao(dados):
    dados_grafico = dados.dropna(
        subset=[
            "pressao_sistolica",
            "hipertensao_informada"
        ]
    ).copy()

    if dados_grafico.empty:
        return criar_fig_sem_dados()

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
            "<b>Pressão sistólica</b>: "
            "%{y:.1f} mmHg"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Hipertensão informada",
        y_title="Pressão sistólica (mmHg)"
    )

    return fig


def criar_fig_colesterol(dados):
    dados_grafico = dados.dropna(
        subset=["colesterol_total"]
    )

    if dados_grafico.empty:
        return criar_fig_sem_dados()

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
            "<b>Colesterol total</b>: "
            "%{x:.1f} mg/dL"
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


def criar_fig_colesterol_informado(dados):
    dados_grafico = dados.dropna(
        subset=[
            "colesterol_total",
            "colesterol_alto_informado"
        ]
    ).copy()

    if dados_grafico.empty:
        return criar_fig_sem_dados()

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
            "<b>Colesterol total</b>: "
            "%{y:.1f} mg/dL"
            "<extra></extra>"
        )
    )

    aplicar_estilo_grafico(
        fig,
        x_title="Colesterol alto informado",
        y_title="Colesterol total (mg/dL)"
    )

    return fig


def filtrar_dados(
    dados,
    sexo,
    diabetes,
    hipertensao,
    colesterol
):
    dados_filtrados = dados.copy()

    if sexo != "Todos":
        dados_filtrados = dados_filtrados[
            dados_filtrados["sexo"] == sexo
        ]

    if diabetes != "Todos":
        dados_filtrados = dados_filtrados[
            dados_filtrados[
                "diabetes_informado"
            ] == diabetes
        ]

    if hipertensao != "Todos":
        dados_filtrados = dados_filtrados[
            dados_filtrados[
                "hipertensao_informada"
            ] == hipertensao
        ]

    if colesterol != "Todos":
        dados_filtrados = dados_filtrados[
            dados_filtrados[
                "colesterol_alto_informado"
            ] == colesterol
        ]

    return dados_filtrados


def criar_filtro(
    titulo,
    id_componente,
    opcoes,
    valor="Todos"
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
                value=valor,
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
                                            "INDICADORES CLÍNICOS"
                                        ),
                                    ],
                                    className="eda-eyebrow"
                                ),

                                html.H1(
                                    "Indicadores Clínicos",
                                    className="eda-title"
                                ),

                                html.P(
                                    "Explore indicadores relacionados "
                                    "ao perfil glicêmico, à pressão "
                                    "arterial e ao perfil lipídico "
                                    "dos participantes.",
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
                                    "Escolha as características do grupo "
                                    "que você deseja analisar. Os filtros "
                                    "podem ser combinados e todos os "
                                    "gráficos serão atualizados "
                                    "automaticamente.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                criar_filtro(
                                    "Quem você quer analisar?",
                                    "filtro-clinico-sexo",
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
                                    "Qual condição de diabetes?",
                                    "filtro-clinico-diabetes",
                                    [
                                        {
                                            "label": "Todos",
                                            "value": "Todos"
                                        },
                                        {
                                            "label": "Não",
                                            "value": "Nao"
                                        },
                                        {
                                            "label": "Borderline",
                                            "value": "Borderline"
                                        },
                                        {
                                            "label": "Sim",
                                            "value": "Sim"
                                        }
                                    ]
                                ),

                                criar_filtro(
                                    "Hipertensão informada?",
                                    "filtro-clinico-hipertensao",
                                    [
                                        {
                                            "label": "Todos",
                                            "value": "Todos"
                                        },
                                        {
                                            "label": "Não",
                                            "value": 0.0
                                        },
                                        {
                                            "label": "Sim",
                                            "value": 1.0
                                        }
                                    ]
                                ),

                                criar_filtro(
                                    "Colesterol alto informado?",
                                    "filtro-clinico-colesterol",
                                    [
                                        {
                                            "label": "Todos",
                                            "value": "Todos"
                                        },
                                        {
                                            "label": "Não",
                                            "value": 0.0
                                        },
                                        {
                                            "label": "Sim",
                                            "value": 1.0
                                        }
                                    ]
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    id=(
                                                        "contador-"
                                                        "filtros-clinicos"
                                                    ),
                                                    style={
                                                        "fontSize": "22px",
                                                        "fontWeight": "700",
                                                        "color": (
                                                            COR_PRINCIPAL
                                                        )
                                                    }
                                                ),

                                                html.P(
                                                    "Os gráficos abaixo "
                                                    "representam o grupo "
                                                    "selecionado.",
                                                    style={
                                                        "margin": "4px 0 0 0",
                                                        "fontSize": "13px",
                                                        "color": COR_TEXTO
                                                    }
                                                ),
                                            ]
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
                                    "PERFIL GLICÊMICO",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Indicadores de glicemia",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Análise da distribuição da "
                                    "hemoglobina glicada e sua relação "
                                    "com a condição de diabetes "
                                    "informada pelos participantes.",
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
                                                    "Distribuição de HbA1c",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição dos "
                                                    "valores de hemoglobina "
                                                    "glicada na amostra.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id="grafico-clinico-hba1c",
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
                                                    "HbA1c por diabetes "
                                                    "informado",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Comparação da "
                                                    "distribuição de HbA1c "
                                                    "entre os grupos de "
                                                    "diabetes informado.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-clinico-"
                                                "hba1c-diabetes"
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
                                    "PRESSÃO ARTERIAL",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Indicadores de pressão arterial",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Análise da distribuição da pressão "
                                    "arterial sistólica e sua relação "
                                    "com o histórico de hipertensão "
                                    "informado pelos participantes.",
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
                                                    "Distribuição da pressão "
                                                    "sistólica",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição dos valores "
                                                    "de pressão arterial "
                                                    "sistólica observados "
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
                                                "grafico-clinico-"
                                                "pressao-sistolica"
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
                                                    "Pressão sistólica por "
                                                    "hipertensão informada",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Comparação da pressão "
                                                    "sistólica entre "
                                                    "participantes que "
                                                    "informaram ou não "
                                                    "hipertensão.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-clinico-"
                                                "pressao-hipertensao"
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
                                    "PERFIL LIPÍDICO",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Indicadores de colesterol",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Análise da distribuição do "
                                    "colesterol total e sua relação "
                                    "com o histórico de colesterol alto "
                                    "informado pelos participantes.",
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
                                                    "Distribuição do "
                                                    "colesterol total",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição dos "
                                                    "valores de colesterol "
                                                    "total observados "
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
                                                "grafico-clinico-"
                                                "colesterol"
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
                                                    "Colesterol total por "
                                                    "colesterol alto "
                                                    "informado",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Comparação da "
                                                    "distribuição de "
                                                    "colesterol total entre "
                                                    "participantes que "
                                                    "informaram ou não "
                                                    "colesterol alto.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-clinico-"
                                                "colesterol-informado"
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
        "grafico-clinico-hba1c",
        "figure"
    ),
    Output(
        "grafico-clinico-hba1c-diabetes",
        "figure"
    ),
    Output(
        "grafico-clinico-pressao-sistolica",
        "figure"
    ),
    Output(
        "grafico-clinico-pressao-hipertensao",
        "figure"
    ),
    Output(
        "grafico-clinico-colesterol",
        "figure"
    ),
    Output(
        "grafico-clinico-colesterol-informado",
        "figure"
    ),
    Output(
        "contador-filtros-clinicos",
        "children"
    ),
    Input(
        "filtro-clinico-sexo",
        "value"
    ),
    Input(
        "filtro-clinico-diabetes",
        "value"
    ),
    Input(
        "filtro-clinico-hipertensao",
        "value"
    ),
    Input(
        "filtro-clinico-colesterol",
        "value"
    )
)
def atualizar_indicadores_clinicos(
    sexo,
    diabetes,
    hipertensao,
    colesterol
):
    dados_filtrados = filtrar_dados(
        df,
        sexo,
        diabetes,
        hipertensao,
        colesterol
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
        criar_fig_hba1c(
            dados_filtrados
        ),
        criar_fig_hba1c_diabetes(
            dados_filtrados
        ),
        criar_fig_pressao_sistolica(
            dados_filtrados
        ),
        criar_fig_pressao_hipertensao(
            dados_filtrados
        ),
        criar_fig_colesterol(
            dados_filtrados
        ),
        criar_fig_colesterol_informado(
            dados_filtrados
        ),
        contador
    )