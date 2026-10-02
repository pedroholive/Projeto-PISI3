from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dash_table, dcc, html


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "healthsync_nhanes_2021_2023_core.csv"
)

df = pd.read_csv(DATA_PATH)


def formatar_inteiro(valor):
    return f"{int(valor):,}".replace(",", ".")


def classificar_tipo(coluna):
    if coluna == "id_participante":
        return "Identificador"

    if coluna in [
        "sexo",
        "diabetes_informado",
        "hipertensao_informada",
        "hipertensao_2consultas",
        "colesterol_alto_informado",
        "medicamento_pressao",
    ]:
        return "Categórica"

    return "Numérica"


dominios = {
    "id_participante": "Identificação",
    "peso_estatistico_mec": "Desenho amostral",
    "peso_estatistico_entrevista": "Desenho amostral",
    "psu": "Desenho amostral",
    "estrato_amostral": "Desenho amostral",
    "idade": "Demográfico",
    "sexo": "Demográfico",
    "imc": "Antropométrico",
    "pressao_sistolica": "Pressão arterial",
    "pressao_diastolica": "Pressão arterial",
    "leituras_validas_sistolica": "Pressão arterial",
    "leituras_validas_diastolica": "Pressão arterial",
    "hba1c": "Perfil glicêmico",
    "glicose_jejum": "Perfil glicêmico",
    "colesterol_total": "Perfil lipídico",
    "diabetes_informado": "Perfil glicêmico",
    "hipertensao_informada": "Histórico / Pressão",
    "hipertensao_2consultas": "Histórico / Pressão",
    "colesterol_alto_informado": "Perfil lipídico",
    "medicamento_pressao": "Histórico / Pressão",
    "atv_moderada_min_semana": "Estilo de vida",
    "atv_vigorosa_min_semana": "Estilo de vida",
    "minutos_sedentarios_dia": "Estilo de vida",
}


descricoes = {
    "id_participante":
        "Identificador único do participante.",

    "peso_estatistico_mec":
        "Peso estatístico associado aos exames do participante.",

    "peso_estatistico_entrevista":
        "Peso estatístico associado à entrevista do participante.",

    "psu":
        "Unidade primária de amostragem do desenho amostral.",

    "estrato_amostral":
        "Estrato utilizado no desenho amostral.",

    "idade":
        "Idade do participante em anos.",

    "sexo":
        "Sexo registrado para o participante.",

    "imc":
        "Índice de Massa Corporal do participante.",

    "pressao_sistolica":
        "Medida de pressão arterial sistólica.",

    "pressao_diastolica":
        "Medida de pressão arterial diastólica.",

    "leituras_validas_sistolica":
        "Quantidade de leituras válidas utilizadas para a pressão sistólica.",

    "leituras_validas_diastolica":
        "Quantidade de leituras válidas utilizadas para a pressão diastólica.",

    "hba1c":
        "Medida de hemoglobina glicada (HbA1c).",

    "glicose_jejum":
        "Medida de glicose em jejum.",

    "colesterol_total":
        "Medida de colesterol total.",

    "diabetes_informado":
        "Condição relacionada ao diabetes informada no questionário.",

    "hipertensao_informada":
        "Histórico de hipertensão informado no questionário.",

    "hipertensao_2consultas":
        "Informação relacionada ao histórico de hipertensão em duas ou mais consultas.",

    "colesterol_alto_informado":
        "Histórico de colesterol alto informado no questionário.",

    "medicamento_pressao":
        "Informação relacionada ao uso de medicamento para pressão arterial.",

    "atv_moderada_min_semana":
        "Minutos semanais de atividade física moderada.",

    "atv_vigorosa_min_semana":
        "Minutos semanais de atividade física vigorosa.",

    "minutos_sedentarios_dia":
        "Minutos de comportamento sedentário por dia.",
}


def criar_dicionario_dados():
    registros = []

    for coluna in df.columns:
        registros.append(
            {
                "Variável": coluna,
                "Tipo": classificar_tipo(coluna),
                "Domínio": dominios.get(coluna, "Outros"),
                "Válidos": formatar_inteiro(df[coluna].count()),
                "Ausentes": formatar_inteiro(
                    df[coluna].isna().sum()
                ),
                "Descrição": descricoes.get(
                    coluna,
                    "Descrição não cadastrada."
                ),
            }
        )

    return pd.DataFrame(registros)


dicionario_df = criar_dicionario_dados()


total_participantes = df["id_participante"].nunique()
total_variaveis = df.shape[1]
idade_media = df["idade"].mean()
imc_medio = df["imc"].mean()


contagem_sexo = (
    df["sexo"]
    .value_counts()
    .rename_axis("sexo")
    .reset_index(name="participantes")
)


fig_sexo = px.bar(
    contagem_sexo,
    x="sexo",
    y="participantes",
    labels={
        "sexo": "Sexo",
        "participantes": "Número de participantes"
    }
)

fig_sexo.update_traces(
    marker=dict(
        color="#0E766D",
        line=dict(
            color="#FFFFFF",
            width=0.6
        )
    ),
    hovertemplate=(
        "<b>%{x}</b>"
        "<br>Participantes: %{y}"
        "<extra></extra>"
    )
)

fig_sexo.update_layout(
    title=None,
    font=dict(
        family="Arial, Helvetica, sans-serif",
        size=12,
        color="#52615E"
    ),
    plot_bgcolor="rgba(0, 0, 0, 0)",
    paper_bgcolor="rgba(0, 0, 0, 0)",
    margin=dict(
        l=55,
        r=25,
        t=25,
        b=55
    ),
    showlegend=False
)

fig_sexo.update_xaxes(
    title_text="Sexo",
    showgrid=False,
    showline=True,
    linecolor="#DCE8E4",
    linewidth=1,
    ticks="outside",
    tickcolor="#DCE8E4",
    zeroline=False,
    automargin=True
)

fig_sexo.update_yaxes(
    title_text="Número de participantes",
    showgrid=True,
    gridcolor="#E8EFED",
    gridwidth=1,
    showline=False,
    zeroline=False,
    automargin=True
)


fig_idade = px.histogram(
    df.dropna(subset=["idade"]),
    x="idade",
    nbins=20,
    labels={
        "idade": "Idade (anos)"
    }
)

fig_idade.update_traces(
    marker=dict(
        color="#0E766D",
        line=dict(
            color="#FFFFFF",
            width=0.6
        )
    ),
    opacity=0.90,
    hovertemplate=(
        "<b>Idade</b>: %{x} anos"
        "<br><b>Participantes</b>: %{y}"
        "<extra></extra>"
    )
)

fig_idade.update_layout(
    title=None,
    font=dict(
        family="Arial, Helvetica, sans-serif",
        size=12,
        color="#52615E"
    ),
    plot_bgcolor="rgba(0, 0, 0, 0)",
    paper_bgcolor="rgba(0, 0, 0, 0)",
    margin=dict(
        l=55,
        r=25,
        t=25,
        b=55
    ),
    showlegend=False
)

fig_idade.update_xaxes(
    title_text="Idade (anos)",
    showgrid=False,
    showline=True,
    linecolor="#DCE8E4",
    linewidth=1,
    ticks="outside",
    tickcolor="#DCE8E4",
    zeroline=False,
    automargin=True
)

fig_idade.update_yaxes(
    title_text="Número de participantes",
    showgrid=True,
    gridcolor="#E8EFED",
    gridwidth=1,
    showline=False,
    zeroline=False,
    automargin=True
)


dash.register_page(
    __name__,
    path="/visao-geral",
    name="Visão Geral",
    order=1
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
                                            "VISÃO GERAL"
                                        ),
                                    ],
                                    className="eda-eyebrow"
                                ),

                                html.H1(
                                    "Visão Geral dos Dados",
                                    className="eda-title"
                                ),

                                html.P(
                                    "Conheça a estrutura dos dados e as "
                                    "principais características da amostra "
                                    "utilizada nas análises do HealthSync.",
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
                                    "RESUMO DA AMOSTRA",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Principais características",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Resumo geral da estrutura dos dados "
                                    "e das principais características "
                                    "da amostra analisada.",
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
                                            formatar_inteiro(
                                                total_participantes
                                            ),
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "DADOS",
                                                    className="eda-card-tag"
                                                ),

                                                html.P(
                                                    "Variáveis",
                                                    className="eda-card-label"
                                                ),
                                            ]
                                        ),

                                        html.H3(
                                            str(total_variaveis),
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

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
                                            (
                                                f"{idade_media:.2f}"
                                                .replace(".", ",")
                                                + " anos"
                                            ),
                                            className="eda-card-value"
                                        ),
                                    ],
                                    className="eda-card"
                                ),

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
                                            (
                                                f"{imc_medio:.2f}"
                                                .replace(".", ",")
                                                + " kg/m²"
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
            className="eda-overview-section"
        ),

        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "CARACTERIZAÇÃO DA AMOSTRA",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Perfil dos participantes",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Visualizações que ajudam a compreender "
                                    "a composição da amostra segundo "
                                    "características demográficas.",
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
                                                    "Distribuição por sexo",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Número de participantes "
                                                    "segundo o sexo registrado "
                                                    "na base de dados.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            figure=fig_sexo,
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
                                                    "Distribuição de idade",
                                                    className="eda-chart-title"
                                                ),

                                                html.P(
                                                    "Distribuição das idades "
                                                    "dos participantes da "
                                                    "amostra analisada.",
                                                    className=(
                                                        "eda-chart-description"
                                                    )
                                                ),
                                            ],
                                            className="eda-chart-header"
                                        ),

                                        dcc.Graph(
                                            figure=fig_idade,
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
            className="eda-analysis-section eda-section-alt"
        ),

        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "ESTRUTURA DOS DADOS",
                                    className="eda-section-eyebrow"
                                ),

                                html.H2(
                                    "Dicionário dos Dados",
                                    className="eda-section-title"
                                ),

                                html.P(
                                    "Consulte as variáveis disponíveis, "
                                    "seus domínios e a quantidade de "
                                    "registros válidos e ausentes.",
                                    className="eda-section-description"
                                ),
                            ],
                            className="eda-section-header"
                        ),

                        html.Div(
                            [
                                html.Label(
                                    "Filtrar por domínio",
                                    htmlFor="filtro-dominio",
                                    className="eda-filter-label"
                                ),

                                dcc.Dropdown(
                                    id="filtro-dominio",
                                    options=[
                                        {
                                            "label": "Todos os domínios",
                                            "value": "Todos"
                                        }
                                    ] + [
                                        {
                                            "label": dominio,
                                            "value": dominio
                                        }
                                        for dominio in sorted(
                                            dicionario_df[
                                                "Domínio"
                                            ].unique()
                                        )
                                    ],
                                    value="Todos",
                                    clearable=False,
                                    searchable=False,
                                    className="eda-dropdown"
                                ),
                            ],
                            style={
                                "maxWidth": "360px",
                                "marginBottom": "24px"
                            }
                        ),

                        html.Div(
                            [
                                dash_table.DataTable(
                                    id="tabela-dicionario",

                                    columns=[
                                        {
                                            "name": "Variável",
                                            "id": "Variável"
                                        },
                                        {
                                            "name": "Tipo",
                                            "id": "Tipo"
                                        },
                                        {
                                            "name": "Domínio",
                                            "id": "Domínio"
                                        },
                                        {
                                            "name": "Válidos",
                                            "id": "Válidos"
                                        },
                                        {
                                            "name": "Ausentes",
                                            "id": "Ausentes"
                                        },
                                        {
                                            "name": "Descrição",
                                            "id": "Descrição"
                                        },
                                    ],

                                    data=dicionario_df.to_dict(
                                        "records"
                                    ),

                                    sort_action="native",

                                    style_table={
                                        "overflowX": "auto",
                                        "width": "100%"
                                    },

                                    style_header={
                                        "backgroundColor": "#E8F5F1",
                                        "color": "#0E766D",
                                        "fontWeight": "700",
                                        "fontSize": "14px",
                                        "border": (
                                            "1px solid #DCE8E4"
                                        ),
                                        "padding": "16px",
                                        "textAlign": "left"
                                    },

                                    style_cell={
                                        "backgroundColor": "#FFFFFF",
                                        "color": "#52615E",
                                        "border": (
                                            "1px solid #E8EFED"
                                        ),
                                        "padding": "16px",
                                        "fontFamily": (
                                            "Arial, Helvetica, "
                                            "sans-serif"
                                        ),
                                        "fontSize": "14px",
                                        "lineHeight": "1.5",
                                        "textAlign": "left",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                        "verticalAlign": "middle"
                                    },

                                    style_cell_conditional=[
                                        {
                                            "if": {
                                                "column_id": "Variável"
                                            },
                                            "fontWeight": "600",
                                            "color": "#263330",
                                            "minWidth": "210px",
                                            "width": "210px"
                                        },
                                        {
                                            "if": {
                                                "column_id": "Tipo"
                                            },
                                            "minWidth": "120px",
                                            "width": "120px"
                                        },
                                        {
                                            "if": {
                                                "column_id": "Domínio"
                                            },
                                            "minWidth": "170px",
                                            "width": "170px"
                                        },
                                        {
                                            "if": {
                                                "column_id": "Válidos"
                                            },
                                            "textAlign": "center",
                                            "minWidth": "100px",
                                            "width": "100px"
                                        },
                                        {
                                            "if": {
                                                "column_id": "Ausentes"
                                            },
                                            "textAlign": "center",
                                            "minWidth": "100px",
                                            "width": "100px"
                                        },
                                        {
                                            "if": {
                                                "column_id": "Descrição"
                                            },
                                            "minWidth": "350px"
                                        },
                                    ],
                                )
                            ],
                            className="eda-chart-card"
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
    Output("tabela-dicionario", "data"),
    Input("filtro-dominio", "value")
)
def atualizar_dicionario(dominio):
    if dominio == "Todos":
        dados_tabela = dicionario_df
    else:
        dados_tabela = dicionario_df[
            dicionario_df["Domínio"] == dominio
        ]

    return dados_tabela.to_dict("records")