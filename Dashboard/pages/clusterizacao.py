
"""Exploração descritiva dos perfis de clusterização NHANES já exportados.

Não refaz treinamento nem atribui diagnóstico ou risco clínico aos clusters.
"""
from pathlib import Path

import dash
from dash import Input, Output, callback, dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "modelos" / "clusterizacao"
DADOS = pd.read_csv(BASE / "healthsync_perfis_cardiometabolicos.csv")
METRICAS = pd.read_csv(BASE / "metricas_kmeans_por_estrato.csv")

ESTRATOS = {
    "HbA1c < 5.7": ("E1", "HbA1c < 5,7%"),
    "HbA1c 5.7-6.4": ("E2", "HbA1c 5,7%–6,4%"),
    "HbA1c >= 6.5": ("E3", "HbA1c ≥ 6,5%"),
}

VARIAVEIS = {
    "idade": "Idade (anos)",
    "imc": "IMC (kg/m²)",
    "pressao_sistolica": "Pressão sistólica (mmHg)",
    "pressao_diastolica": "Pressão diastólica (mmHg)",
    "colesterol_total": "Colesterol total (mg/dL)",
    "atv_moderada_min_semana": "Atividade moderada (min/sem)",
    "atv_vigorosa_min_semana": "Atividade vigorosa (min/sem)",
    "minutos_sedentarios_dia": "Sedentarismo (min/dia)",
}

COLORS = [
    "#087e75",
    "#d49b46",
    "#5e84b3",
    "#9471b0",
    "#d36f63",
    "#759a5b",
]

# Confere se os dados respeitam os estratos exportados.
assert set(DADOS["estrato_glicemico"].unique()) == set(ESTRATOS)

dash.register_page(
    __name__,
    path="/clusterizacao",
    name="Clusterização",
    order=6,
)


def estilizar(fig, altura=380):
    fig.update_layout(
        font=dict(
            family="Arial, sans-serif",
            color="#52615e",
            size=12,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=42, r=24, t=30, b=40),
        height=altura,
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        automargin=True,
    )

    fig.update_yaxes(
        gridcolor="#e8efed",
        zeroline=False,
        automargin=True,
    )

    return fig


def cartao(titulo, conteudo, nota=None):
    return html.Div(
        [
            html.H3(titulo, className="cl-card-title"),
            conteudo,
            html.P(nota, className="cl-card-note") if nota else None,
        ],
        className="cl-card",
    )


def numero(titulo, valor, subtitulo):
    return html.Div(
        [
            html.P(titulo, className="cl-kpi-label"),
            html.Strong(valor, className="cl-kpi-value"),
            html.P(subtitulo, className="cl-kpi-note"),
        ],
        className="cl-kpi",
    )


layout = html.Div(
    [
        # Cabeçalho
        html.Div(
            [
                html.P(
                    "MACHINE LEARNING · NHANES 2021–2023",
                    className="cl-eyebrow",
                ),
                html.H1(
                    "Perfis cardiometabólicos",
                    className="cl-heading",
                ),
                html.P(
                    "Explore os agrupamentos K-Means dentro de cada "
                    "estrato de HbA1c. Os perfis descrevem padrões "
                    "observados na amostra, não diagnósticos individuais.",
                    className="cl-intro",
                ),
            ]
        ),

        # Filtros
        html.Div(
            [
                html.Label(
                    "Estrato glicêmico",
                    htmlFor="cl-estrato",
                    className="cl-filter-label",
                ),
                dcc.Dropdown(
                    id="cl-estrato",
                    options=[
                        {"label": nome, "value": chave}
                        for chave, (_, nome) in ESTRATOS.items()
                    ],
                    value=list(ESTRATOS)[0],
                    clearable=False,
                ),

                html.Label(
                    "Indicador para comparar",
                    htmlFor="cl-variavel",
                    className="cl-filter-label",
                ),
                dcc.Dropdown(
                    id="cl-variavel",
                    options=[
                        {"label": v, "value": k}
                        for k, v in VARIAVEIS.items()
                    ],
                    value="imc",
                    clearable=False,
                ),
            ],
            className="cl-filters",
        ),

        # Indicadores principais
        html.Div(
            id="cl-kpis",
            className="cl-kpis",
        ),

        # Primeira linha de gráficos
        html.Div(
            [
                cartao(
                    "Participantes por perfil",
                    dcc.Graph(
                        id="cl-tamanhos",
                        config={"displayModeBar": False},
                    ),
                    "Contagens da amostra analisada (não ponderadas).",
                ),

                cartao(
                    "Distribuição do indicador",
                    dcc.Graph(
                        id="cl-boxplot",
                        config={"displayModeBar": False},
                    ),
                    "Boxplots mostram a distribuição individual "
                    "e possíveis valores extremos.",
                ),
            ],
            className="cl-grid",
        ),

        # Segunda linha de gráficos
        html.Div(
            [
                cartao(
                    "Comparação multivariada dos perfis",
                    dcc.Graph(
                        id="cl-radar",
                        config={"displayModeBar": False},
                    ),
                    "Cada eixo representa o percentil da mediana "
                    "do cluster na distribuição do estrato (0–100). "
                    "Valores maiores não representam necessariamente "
                    "maior risco.",
                ),

                cartao(
                    "Silhouette por número de clusters (k)",
                    dcc.Graph(
                        id="cl-silhouette",
                        config={"displayModeBar": False},
                    ),
                    "Métricas exportadas da avaliação K-Means. "
                    "A linha marca o k usado na segmentação.",
                ),
            ],
            className="cl-grid",
        ),

        # Nova seção: caracterização dos clusters
        cartao(
            "Caracterização dos perfis",
            html.Div(id="cl-caracterizacao"),
            "Síntese automatizada das medianas amostrais; "
            "não corresponde a diagnóstico ou classificação de risco.",
        ),

        # Tabela de resumo
        cartao(
            "Resumo descritivo dos perfis",
            html.Div(id="cl-resumo"),
            "Medianas da amostra sem ponderação; as comparações "
            "não estabelecem causalidade ou significância estatística.",
        ),

        # Fonte dos dados
        html.P(
            "Fonte: modelos/clusterizacao/"
            "healthsync_perfis_cardiometabolicos.csv "
            "e metricas_kmeans_por_estrato.csv. "
            "Resultados provenientes do notebook "
            "clusterizacao_healthsync_final.ipynb.",
            className="cl-source",
        ),
    ],
    className="cl-page",
)


@callback(
    Output("cl-kpis", "children"),
    Output("cl-tamanhos", "figure"),
    Output("cl-boxplot", "figure"),
    Output("cl-radar", "figure"),
    Output("cl-silhouette", "figure"),
    Output("cl-resumo", "children"),
    Output("cl-caracterizacao", "children"),
    Input("cl-estrato", "value"),
    Input("cl-variavel", "value"),
)
def atualizar(estrato, variavel):

    if estrato not in ESTRATOS:
        estrato = list(ESTRATOS)[0]

    if variavel not in VARIAVEIS:
        variavel = "imc"

    dados = DADOS.loc[
        DADOS["estrato_glicemico"] == estrato
    ].copy()

    dados["perfil"] = dados["perfil_id"].str.replace(
        "_", " · ", regex=False
    )

    codigo, nome = ESTRATOS[estrato]

    ordem_ids = sorted(
        dados["perfil_id"].unique(),
        key=lambda x: int(x.split("C")[-1]),
    )

    ordem = [
        x.replace("_", " · ")
        for x in ordem_ids
    ]

    metricas = METRICAS.loc[
        METRICAS["estrato"] == codigo
    ].sort_values("k")

    n_clusters = len(ordem)

    linha_k = metricas.loc[
        metricas["k"] == n_clusters
    ]

    sil = (
        linha_k["silhouette"].iloc[0]
        if not linha_k.empty
        else None
    )

    # KPIs
    kpis = [
        numero(
            "Participantes",
            f"{len(dados):,}".replace(",", "."),
            nome,
        ),
        numero(
            "Perfis K-Means",
            str(n_clusters),
            "Clusters neste estrato",
        ),
        numero(
            "Silhouette",
            f"{sil:.3f}" if sil is not None else "—",
            "Para o k utilizado",
        ),
    ]

    # Gráfico: participantes por cluster
    quantidades = (
        dados.groupby("perfil")
        .size()
        .reindex(ordem)
        .reset_index(name="n")
    )

    fig_tam = px.bar(
        quantidades,
        x="perfil",
        y="n",
        text="n",
        color="perfil",
        color_discrete_sequence=COLORS,
        category_orders={"perfil": ordem},
    )

    fig_tam.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig_tam.update_layout(showlegend=False)

    fig_tam.update_yaxes(
        title="Participantes",
        rangemode="tozero",
    )

    fig_tam.update_xaxes(title="Perfil")

    estilizar(fig_tam)

    # Gráfico: distribuição do indicador
    fig_box = px.box(
        dados,
        x="perfil",
        y=variavel,
        color="perfil",
        points="outliers",
        category_orders={"perfil": ordem},
        color_discrete_sequence=COLORS,
    )

    fig_box.update_layout(showlegend=False)
    fig_box.update_xaxes(title="Perfil")
    fig_box.update_yaxes(title=VARIAVEIS[variavel])

    estilizar(fig_box)

    # Medianas para caracterização e radar
    medianas = (
        dados.groupby("perfil_id")[list(VARIAVEIS)]
        .median()
        .reindex(ordem_ids)
    )

    validas = [
        col for col in VARIAVEIS
        if dados[col].nunique() > 1
    ]

    # Gráfico de radar
    fig_rad = go.Figure()

    for i, (perfil_id, linha) in enumerate(medianas.iterrows()):

        valores = [
            float(100 * (dados[col] <= linha[col]).mean())
            for col in validas
        ]

        rotulos = [
            VARIAVEIS[col].split(" (")[0]
            for col in validas
        ]

        fig_rad.add_trace(
            go.Scatterpolar(
                r=valores + valores[:1],
                theta=rotulos + rotulos[:1],
                name=perfil_id,
                mode="lines+markers",
                fill="toself",
                opacity=0.65,
                line=dict(
                    color=COLORS[i % len(COLORS)],
                    width=2,
                ),
            )
        )

    fig_rad.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#e8efed",
            ),
            angularaxis=dict(
                gridcolor="#e8efed"
            ),
        ),
        legend=dict(
            orientation="h",
            y=-0.15,
        ),
    )

    estilizar(fig_rad, 470)

    # Gráfico: Silhouette por k
    fig_sil = px.line(
        metricas,
        x="k",
        y="silhouette",
        markers=True,
    )

    fig_sil.update_traces(
        line_color="#087e75",
        marker=dict(size=8),
    )

    fig_sil.add_vline(
        x=n_clusters,
        line_dash="dash",
        line_color="#d49b46",
        annotation_text=f"k aplicado = {n_clusters}",
    )

    fig_sil.update_xaxes(
        title="Número de clusters (k)",
        dtick=1,
    )

    fig_sil.update_yaxes(title="Silhouette")

    estilizar(fig_sil)

    # Tabela de medianas
    resumo = (
        dados.groupby("perfil_id")[list(VARIAVEIS)]
        .median()
        .reindex(ordem_ids)
    )

    exibicao = resumo[
        [
            "idade",
            "imc",
            "pressao_sistolica",
            "colesterol_total",
            "minutos_sedentarios_dia",
        ]
    ]

    cabecalhos = [
        "Perfil",
        "N",
        "Idade (anos)",
        "IMC",
        "PAS (mmHg)",
        "Colesterol (mg/dL)",
        "Sedentário (min/dia)",
    ]

    linhas = [
        html.Tr([
            html.Th(c)
            for c in cabecalhos
        ])
    ]

    for perfil_id, linha in exibicao.iterrows():

        n = int(
            (dados["perfil_id"] == perfil_id).sum()
        )

        linhas.append(
            html.Tr(
                [
                    html.Td(perfil_id),
                    html.Td(
                        f"{n:,}".replace(",", ".")
                    ),
                ]
                + [
                    html.Td(f"{x:.1f}")
                    for x in linha
                ]
            )
        )

    tabela = html.Div(
        html.Table(
            [
                html.Thead(linhas[0]),
                html.Tbody(linhas[1:]),
            ],
            className="cl-table",
        ),
        className="cl-table-wrap",
    )

    # --------------------------------------------------
    # CARACTERIZAÇÃO DOS PERFIS CARDIOMETABÓLICOS
    # --------------------------------------------------
    # Compara medianas dos clusters dentro do mesmo
    # estrato, sem atribuir diagnóstico ou risco.
    # --------------------------------------------------

    variaveis_destaque = [
        "idade",
        "imc",
        "pressao_sistolica",
        "colesterol_total",
        "minutos_sedentarios_dia",
    ]

    resumos = []

    for perfil_id in ordem_ids:

        n = int(
            (dados["perfil_id"] == perfil_id).sum()
        )

        proporcao = 100 * n / len(dados)

        atributos = []

        for coluna in variaveis_destaque:

            valores = medianas[coluna].dropna()

            if (
                len(valores) < 2
                or valores.max() == valores.min()
            ):
                continue

            valor = medianas.loc[
                perfil_id, coluna
            ]

            # Destaca apenas extremos entre clusters.
            if valor == valores.max():
                atributos.append(
                    (coluna, "Maior mediana")
                )

            elif valor == valores.min():
                atributos.append(
                    (coluna, "Menor mediana")
                )

        if not atributos:
            descricao = (
                "Sem extremos entre os "
                "indicadores selecionados."
            )
        else:
            descricao = None

        resumos.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                perfil_id,
                                className="cl-profile-name",
                            ),
                            html.Span(
                                f"{n:,}".replace(",", ".")
                                + (
                                    f" participantes · "
                                    f"{proporcao:.1f}% do estrato"
                                ).replace(".", ","),
                                className="cl-profile-count",
                            ),
                        ],
                        className="cl-profile-head",
                    ),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        tipo,
                                        className="cl-profile-tag",
                                    ),

                                    html.Span(
                                        VARIAVEIS[coluna],
                                        className="cl-profile-indicator",
                                    ),

                                    html.Strong(
                                        f"{medianas.loc[perfil_id, coluna]:.1f}"
                                        .replace(".", ","),
                                        className="cl-profile-value",
                                    ),
                                ],
                                className="cl-profile-feature",
                            )
                            for coluna, tipo in atributos
                        ],
                        className="cl-profile-features",
                    )
                    if atributos
                    else html.P(
                        descricao,
                        className="cl-profile-empty",
                    ),
                ],
                className="cl-profile-card",
            )
        )

    caracterizacao = html.Div(
        [
            html.P(
                "Destaques relativos apenas aos clusters "
                "do estrato selecionado. Uma mediana mais "
                "alta ou mais baixa não indica, por si só, "
                "melhor ou pior saúde.",
                className="cl-profile-explainer",
            ),

            html.Div(
                resumos,
                className="cl-profile-grid",
            ),
        ]
    )

    return (
        kpis,
        fig_tam,
        fig_box,
        fig_rad,
        fig_sil,
        tabela,
        caracterizacao,
    )
