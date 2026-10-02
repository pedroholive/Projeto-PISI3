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
    path="/relacoes-cardiometabolicas",
    name="Relações Cardiometabólicas",
    order=4
)


COR_PRINCIPAL = "#0E766D"
COR_MEDIA = "#34A89A"
COR_CLARA = "#9ADBCF"
COR_TEXTO = "#52615E"
COR_GRID = "#E8EFED"
COR_BORDA = "#DCE8E4"


VARIAVEIS_CORRELACAO = {
    "idade": "Idade",
    "imc": "IMC",
    "hba1c": "HbA1c",
    "glicose_jejum": "Glicose jejum",
    "pressao_sistolica": "Pressão sistólica",
    "pressao_diastolica": "Pressão diastólica",
    "colesterol_total": "Colesterol total",
    "atv_moderada_min_semana": "Ativ. moderada",
    "atv_vigorosa_min_semana": "Ativ. vigorosa",
    "minutos_sedentarios_dia": "Tempo sedentário"
}


def aplicar_estilo_scatter(
    fig,
    x_title,
    y_title
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
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
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
        showgrid=True,
        gridcolor=COR_GRID,
        gridwidth=1,
        showline=True,
        linecolor=COR_BORDA,
        linewidth=1,
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
        showline=True,
        linecolor=COR_BORDA,
        linewidth=1,
        zeroline=False,
        automargin=True
    )

    fig.update_traces(
        marker=dict(
            size=7,
            opacity=0.62,
            line=dict(
                width=0.4,
                color="rgba(255,255,255,0.55)"
            )
        )
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


def criar_heatmap(dados):
    if dados.empty:
        return criar_fig_sem_dados()

    colunas = list(
        VARIAVEIS_CORRELACAO.keys()
    )

    dados_correlacao = (
        dados[colunas]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    correlacao = dados_correlacao.corr()

    if correlacao.isna().all().all():
        return criar_fig_sem_dados()

    correlacao = correlacao.rename(
        index=VARIAVEIS_CORRELACAO,
        columns=VARIAVEIS_CORRELACAO
    )

    fig = px.imshow(
        correlacao,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        color_continuous_scale=[
            [0.0, "#D7ECE8"],
            [0.5, "#FFFFFF"],
            [1.0, COR_PRINCIPAL]
        ],
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
            l=25,
            r=25,
            t=25,
            b=25
        ),
        coloraxis_colorbar=dict(
            title="Correlação",
            thickness=14,
            len=0.75
        )
    )

    fig.update_xaxes(
        side="bottom",
        tickangle=-35,
        tickfont=dict(
            size=11,
            color=COR_TEXTO
        ),
        automargin=True
    )

    fig.update_yaxes(
        tickfont=dict(
            size=11,
            color=COR_TEXTO
        ),
        automargin=True
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b>"
            "<br>%{x}"
            "<br>Correlação: %{z:.2f}"
            "<extra></extra>"
        )
    )

    return fig


def criar_scatter_imc_pressao(dados):
    dados_grafico = dados.dropna(
        subset=[
            "imc",
            "pressao_sistolica",
            "hipertensao_informada"
        ]
    ).copy()

    if dados_grafico.empty:
        return criar_fig_sem_dados()

    dados_grafico[
        "Hipertensão informada"
    ] = (
        dados_grafico[
            "hipertensao_informada"
        ]
        .map({
            0.0: "Não",
            1.0: "Sim"
        })
    )

    fig = px.scatter(
        dados_grafico,
        x="imc",
        y="pressao_sistolica",
        color="Hipertensão informada",
        color_discrete_map={
            "Não": COR_CLARA,
            "Sim": COR_PRINCIPAL
        },
        category_orders={
            "Hipertensão informada": [
                "Não",
                "Sim"
            ]
        },
        labels={
            "imc": "IMC",
            "pressao_sistolica":
                "Pressão sistólica (mmHg)"
        },
        hover_data={
            "imc": ":.1f",
            "pressao_sistolica": ":.1f",
            "Hipertensão informada": True
        }
    )

    aplicar_estilo_scatter(
        fig,
        x_title="IMC",
        y_title="Pressão sistólica (mmHg)"
    )

    return fig


def criar_scatter_imc_hba1c(dados):
    dados_grafico = dados.dropna(
        subset=[
            "imc",
            "hba1c",
            "diabetes_informado"
        ]
    ).copy()

    if dados_grafico.empty:
        return criar_fig_sem_dados()

    dados_grafico[
        "Diabetes informado"
    ] = (
        dados_grafico[
            "diabetes_informado"
        ]
        .replace({
            "Nao": "Não"
        })
    )

    fig = px.scatter(
        dados_grafico,
        x="imc",
        y="hba1c",
        color="Diabetes informado",
        category_orders={
            "Diabetes informado": [
                "Não",
                "Borderline",
                "Sim"
            ]
        },
        color_discrete_map={
            "Não": COR_CLARA,
            "Borderline": COR_MEDIA,
            "Sim": COR_PRINCIPAL
        },
        labels={
            "imc": "IMC",
            "hba1c": "HbA1c (%)"
        },
        hover_data={
            "imc": ":.1f",
            "hba1c": ":.1f",
            "Diabetes informado": True
        }
    )

    aplicar_estilo_scatter(
        fig,
        x_title="IMC",
        y_title="HbA1c (%)"
    )

    return fig


def filtrar_dados(
    dados,
    sexo,
    faixa_etaria,
    diabetes,
    hipertensao
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
                                            "RELAÇÕES CARDIOMETABÓLICAS"
                                        ),
                                    ],
                                    className="eda-eyebrow"
                                ),

                                html.H1(
                                    "Relações Cardiometabólicas",
                                    className="eda-title"
                                ),

                                html.P(
                                    "Explore associações entre medidas "
                                    "demográficas, clínicas e de estilo "
                                    "de vida presentes na amostra.",
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
                                    "Explore as relações",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Escolha características dos "
                                    "participantes para investigar "
                                    "como as relações entre os "
                                    "indicadores mudam em diferentes "
                                    "grupos da amostra.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                criar_filtro(
                                    "Quem você quer analisar?",
                                    "filtro-relacoes-sexo",
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
                                    "filtro-relacoes-idade",
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

                                criar_filtro(
                                    "Qual condição de diabetes?",
                                    "filtro-relacoes-diabetes",
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
                                    "filtro-relacoes-hipertensao",
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
                                        html.Span(
                                            id="contador-relacoes",
                                            style={
                                                "fontSize": "22px",
                                                "fontWeight": "700",
                                                "color": COR_PRINCIPAL
                                            }
                                        ),

                                        html.P(
                                            "O mapa de correlações e os "
                                            "gráficos abaixo representam "
                                            "o grupo selecionado.",
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
                                    "CORRELAÇÕES",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Mapa de correlações",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "O mapa apresenta a intensidade e "
                                    "a direção das correlações lineares "
                                    "entre indicadores cardiometabólicos, "
                                    "idade e medidas relacionadas ao "
                                    "estilo de vida.",
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
                                                    "Relações entre "
                                                    "indicadores",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Valores próximos de 1 "
                                                    "indicam associação "
                                                    "linear positiva, "
                                                    "valores próximos de -1 "
                                                    "indicam associação "
                                                    "linear negativa e "
                                                    "valores próximos de 0 "
                                                    "indicam associação "
                                                    "linear fraca.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id="grafico-relacoes-heatmap",
                                            config={
                                                "displayModeBar": False
                                            },
                                            style={
                                                "height": "720px"
                                            },
                                            className="eda-graph"
                                        ),
                                    ],
                                    className=(
                                        "eda-chart-card "
                                        "eda-correlation-card"
                                    )
                                ),
                            ]
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
                                    "RELAÇÕES ESPECÍFICAS",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Exploração entre indicadores",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Os gráficos de dispersão permitem "
                                    "observar como duas medidas variam "
                                    "entre os participantes e como "
                                    "condições informadas se distribuem "
                                    "nessas relações.",
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
                                                    "IMC e pressão "
                                                    "sistólica",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Relação entre IMC e "
                                                    "pressão sistólica, "
                                                    "diferenciando os "
                                                    "participantes pelo "
                                                    "histórico de "
                                                    "hipertensão informado.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-relacoes-"
                                                "imc-pressao"
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
                                                    "IMC e HbA1c",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Relação entre IMC e "
                                                    "hemoglobina glicada, "
                                                    "diferenciando os "
                                                    "participantes pela "
                                                    "condição de diabetes "
                                                    "informada.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            id=(
                                                "grafico-relacoes-"
                                                "imc-hba1c"
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
    ],
    className="eda-page"
)


@callback(
    Output(
        "grafico-relacoes-heatmap",
        "figure"
    ),
    Output(
        "grafico-relacoes-imc-pressao",
        "figure"
    ),
    Output(
        "grafico-relacoes-imc-hba1c",
        "figure"
    ),
    Output(
        "contador-relacoes",
        "children"
    ),
    Input(
        "filtro-relacoes-sexo",
        "value"
    ),
    Input(
        "filtro-relacoes-idade",
        "value"
    ),
    Input(
        "filtro-relacoes-diabetes",
        "value"
    ),
    Input(
        "filtro-relacoes-hipertensao",
        "value"
    )
)
def atualizar_relacoes(
    sexo,
    faixa_etaria,
    diabetes,
    hipertensao
):
    dados_filtrados = filtrar_dados(
        df,
        sexo,
        faixa_etaria,
        diabetes,
        hipertensao
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
        criar_heatmap(
            dados_filtrados
        ),
        criar_scatter_imc_pressao(
            dados_filtrados
        ),
        criar_scatter_imc_hba1c(
            dados_filtrados
        ),
        contador
    )