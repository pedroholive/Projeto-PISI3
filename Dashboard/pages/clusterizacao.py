
"""Exploração descritiva dos perfis de clusterização NHANES já exportados.

Não refaz treinamento nem atribui diagnóstico ou risco clínico aos clusters.
"""
from pathlib import Path
import base64

import dash
from dash import Input, Output, callback, dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "modelos" / "clusterizacao"
AUDITORIA = BASE / "analise_hdbscan_outliers"
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

# Configurações selecionadas do DBSCAN no notebook original.
# Fonte: modelos/clusterizacao/clusterizacao_healthsync_final.ipynb,
# seção 5.3. O menor grupo foi 4, 4 e 5, respectivamente.
# Esses resultados NÃO são recalculados pela página.
DBSCAN_NOTEBOOK = {
    "E1": {"grupos": 2, "ruido": 81, "menor_grupo": 4},
    "E2": {"grupos": 2, "ruido": 42, "menor_grupo": 4},
    "E3": {"grupos": 2, "ruido": 70, "menor_grupo": 5},
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

        # Auditoria exploratória de agrupamentos e outliers.
        html.Div(
            [
                html.P(
                    "COMPARAÇÃO DE ALGORITMOS · ANÁLISE EXPLORATÓRIA",
                    className="cl-eyebrow",
                ),
                html.H2(
                    "HDBSCAN e outliers multivariados",
                    className="cl-audit-title",
                ),
                html.P(
                    "Compare o K-Means utilizado na segmentação com "
                    "o DBSCAN documentado no notebook e o HDBSCAN "
                    "avaliado em diferentes configurações. "
                    "Os outliers são observações estatisticamente "
                    "atípicas, não diagnósticos clínicos.",
                    className="cl-intro",
                ),
            ],
            className="cl-audit-heading",
        ),
        html.Div(id="cl-auditoria-resultados"),

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


# =========================================================
# Auditoria complementar: HDBSCAN, DBSCAN e outliers
# =========================================================

def percentual(valor):
    """Formata uma porcentagem com uma casa decimal."""
    return f"{float(valor):.1f}%".replace(".", ",")


def aviso_auditoria(mensagem):
    return html.Div(
        [
            html.Strong("Resultados complementares indisponíveis"),
            html.P(mensagem),
            html.P(
                "Gere-os com: python "
                "modelos/clusterizacao/analise_hdbscan_outliers.py"
            ),
        ],
        className="cl-audit-warning",
    )


def imagem_auditoria(codigo):
    """Exibe o PNG gerado pelo script, sem mover arquivos para assets/."""
    caminho = AUDITORIA / f"hdbscan_outliers_{codigo}.png"
    if not caminho.is_file():
        return html.P(
            "Gráfico ainda não gerado para este estrato.",
            className="cl-audit-warning",
        )

    imagem = base64.b64encode(caminho.read_bytes()).decode("ascii")
    return html.Img(
        src=f"data:image/png;base64,{imagem}",
        alt=(
            f"Estrato {codigo}: projeção PCA dos grupos HDBSCAN "
            "e pontos sinalizados por Mahalanobis e Isolation Forest"
        ),
        className="cl-audit-image",
    )


def tabela_auditoria(cabecalhos, linhas):
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(c) for c in cabecalhos])),
                html.Tbody([
                    html.Tr([html.Td(valor) for valor in linha])
                    for linha in linhas
                ]),
            ],
            className="cl-table",
        ),
        className="cl-table-wrap",
    )


@callback(
    Output("cl-auditoria-resultados", "children"),
    Input("cl-estrato", "value"),
)
def atualizar_auditoria(estrato):
    """Lê os CSVs existentes; não executa treinamento no dashboard."""
    if estrato not in ESTRATOS:
        estrato = list(ESTRATOS)[0]

    codigo, _ = ESTRATOS[estrato]
    arquivo_resumo = AUDITORIA / "resumo_por_estrato.csv"
    arquivo_grade = AUDITORIA / "grade_hdbscan.csv"

    if not arquivo_resumo.is_file() or not arquivo_grade.is_file():
        return aviso_auditoria(
            "Ainda não encontrei resumo_por_estrato.csv e "
            "grade_hdbscan.csv na pasta de análise HDBSCAN."
        )

    try:
        resumo = pd.read_csv(arquivo_resumo)
        grade = pd.read_csv(arquivo_grade)
        colunas_resumo = {
            "estrato", "n_participantes", "kmeans_grupos_existentes",
            "hdbscan_grupos", "hdbscan_ruido_pct", "hdbscan_min_cluster_size",
            "hdbscan_min_samples", "hdbscan_silhouette_sem_ruido",
            "mahalanobis_atipicos_pct", "isolation_atipicos_pct",
            "concordancia_ambos_pct", "pca_2_componentes_variancia_pct",
        }
        colunas_grade = {
            "estrato", "min_cluster_size", "min_samples", "ruido_pct"
        }
        if not colunas_resumo.issubset(resumo.columns):
            raise ValueError("faltam colunas no resumo")
        if not colunas_grade.issubset(grade.columns):
            raise ValueError("faltam colunas na grade")

        resultado = resumo.loc[resumo["estrato"] == codigo]
        grade = grade.loc[grade["estrato"] == codigo].copy()
        if len(resultado) != 1 or grade.empty:
            raise ValueError("estrato não encontrado ou duplicado")
        resultado = resultado.iloc[0]

        total = len(DADOS.loc[DADOS["estrato_glicemico"] == estrato])
        if int(resultado["n_participantes"]) != total:
            return aviso_auditoria(
                "A base atual e os resultados do HDBSCAN têm "
                "quantidades diferentes de participantes. "
                "Reexecute a análise para evitar comparações incorretas."
            )

        kmeans_grupos = int(resultado["kmeans_grupos_existentes"])
        hdbscan_grupos = int(resultado["hdbscan_grupos"])
        ruido_hdbscan = float(resultado["hdbscan_ruido_pct"])
        ruido_dbscan = DBSCAN_NOTEBOOK[codigo]["ruido"] / total * 100

        # O menor grupo HDBSCAN é obtido a partir dos rótulos exportados.
        menor_hdbscan = "—"
        individuais = AUDITORIA / "classificacao_exploratoria.csv"
        if individuais.is_file():
            individuais_df = pd.read_csv(individuais)
            sub = individuais_df.loc[
                (individuais_df["estrato"] == codigo)
                & (individuais_df["grupo_hdbscan"] != -1)
            ]
            if not sub.empty:
                menor_hdbscan = str(
                    int(sub["grupo_hdbscan"].value_counts().min())
                )

        tabela_comparativa = tabela_auditoria(
            ["Algoritmo", "Grupos", "Ruído", "Cobertura", "Menor grupo"],
            [
                ["K-Means (principal)", str(kmeans_grupos), "0,0%", "100,0%",
                 str(int(DADOS.loc[DADOS["estrato_glicemico"] == estrato]
                         .groupby("perfil_id").size().min()))],
                ["DBSCAN (notebook)",
                 str(DBSCAN_NOTEBOOK[codigo]["grupos"]),
                 percentual(ruido_dbscan), percentual(100 - ruido_dbscan),
                 str(DBSCAN_NOTEBOOK[codigo]["menor_grupo"])],
                ["HDBSCAN (exploratório)", str(hdbscan_grupos),
                 percentual(ruido_hdbscan),
                 percentual(100 - ruido_hdbscan), menor_hdbscan],
            ],
        )

        grade["min_samples"] = grade["min_samples"].astype(str)
        fig = px.line(
            grade,
            x="min_cluster_size",
            y="ruido_pct",
            color="min_samples",
            markers=True,
            labels={
                "min_cluster_size": "Tamanho mínimo do grupo",
                "ruido_pct": "Observações classificadas como ruído (%)",
                "min_samples": "min_samples",
            },
            color_discrete_sequence=COLORS,
        )
        fig.update_yaxes(range=[0, 105])
        estilizar(fig, 340)
        fig.update_layout(
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=45, r=15, t=20, b=68),
        )

        sil = resultado["hdbscan_silhouette_sem_ruido"]
        sil_texto = "não calculável" if pd.isna(sil) else f"{sil:.3f}"
        k = int(resultado["hdbscan_min_cluster_size"])
        amostras = int(resultado["hdbscan_min_samples"])

        return html.Div(
            [
                html.Div(
                    [
                        numero("Grupos HDBSCAN", str(hdbscan_grupos),
                               "Na configuração selecionada"),
                        numero("Ruído HDBSCAN", percentual(ruido_hdbscan),
                               "Participantes sem grupo atribuído"),
                        numero("Variância PCA (2 eixos)",
                               percentual(resultado["pca_2_componentes_variancia_pct"]),
                               "Apenas visualização, sem reduzir o treino"),
                    ],
                    className="cl-kpis",
                ),
                cartao(
                    "Comparação dos algoritmos",
                    tabela_comparativa,
                    "DBSCAN: resultados fixados no notebook original. "
                    "K-Means e HDBSCAN: dados do estrato atual. "
                    "O ruído do HDBSCAN pode mudar com os parâmetros.",
                ),
                cartao(
                    f"HDBSCAN e detecção de outliers · {codigo}",
                    imagem_auditoria(codigo),
                    "Projeções PCA 2D para visualização. "
                    "Os outliers não foram excluídos do dataset.",
                ),
                html.Div(
                    [
                        cartao(
                            "Outliers multivariados",
                            html.Div(
                                [
                                    numero("Mahalanobis robusto",
                                           percentual(resultado["mahalanobis_atipicos_pct"]),
                                           "Limite exploratório qui-quadrado (97,5%)"),
                                    numero("Isolation Forest",
                                           percentual(resultado["isolation_atipicos_pct"]),
                                           "Contaminação definida em 3%"),
                                    numero("Ambos os métodos",
                                           percentual(resultado["concordancia_ambos_pct"]),
                                           "Interseção das sinalizações"),
                                ],
                                className="cl-audit-outlier-grid",
                            ),
                            "Métodos com critérios diferentes; "
                            "não representam diagnósticos de saúde.",
                        ),
                        cartao(
                            "Sensibilidade aos parâmetros HDBSCAN",
                            dcc.Graph(
                                figure=fig,
                                config={"displayModeBar": False},
                            ),
                            f"Configuração exibida: min_cluster_size={k}, "
                            f"min_samples={amostras}; silhouette dos "
                            f"agrupados: {sil_texto}. "
                            "A silhouette exclui o ruído e não pode ser "
                            "comparada diretamente à do K-Means.",
                        ),
                    ],
                    className="cl-grid",
                ),
                html.P(
                    "Fontes: analise_hdbscan_outliers/resumo_por_estrato.csv, "
                    "grade_hdbscan.csv e gráficos por estrato; "
                    "DBSCAN: notebook clusterizacao_healthsync_final.ipynb, "
                    "seção 5.3. Análises exploratórias com RobustScaler "
                    "por estrato e cinco variáveis cardiometabólicas. "
                    "A escolha de parâmetros do HDBSCAN não comprovou "
                    "superioridade universal de qualquer método.",
                    className="cl-source",
                ),
            ],
            className="cl-audit-content",
        )

    except (ValueError, KeyError, TypeError, OSError) as erro:
        return aviso_auditoria(
            "Não foi possível ler os resultados HDBSCAN: "
            f"{erro}. Confira os arquivos gerados."
        )
