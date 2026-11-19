"""HealthSync — L5: triagem de HbA1c (< 5,7 contra >= 5,7).

Diferente da atribuição de subperfil, aqui o alvo (HbA1c) é um exame de
laboratório que NÃO entra como variável de entrada. A página apenas apresenta
os resultados exportados por classificacao/classificacao_binaria.py.
"""

import base64
import json
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html


ROOT = Path(__file__).resolve().parents[2]
PASTA = ROOT / "modelos" / "classificacao_binaria"
ARQUIVO_METRICAS = PASTA / "metricas_binaria.csv"
ARQUIVO_IMPORTANCIA = PASTA / "importancia_variaveis.csv"
ARQUIVO_ROC = PASTA / "curvas_roc.png"
PASTA_SHAP = ROOT / "modelos" / "explicabilidade_shap"
PASTA_JUSTICA = ROOT / "modelos" / "avaliacao_justica"

VISOES_SHAP = {
    "shap_global.png": "Importância global",
    "shap_resumo.png": "Distribuição dos efeitos",
    "shap_individual.png": "Explicação individual",
}

POPULACOES = {
    "todos": "Toda a base",
    "sem_diagnostico": "Sem diagnóstico de diabetes (triagem)",
}

COR_MODELOS = {
    "Regressão Logística": "#0E766D",
    "Random Forest": "#E58A3A",
}

NOMES_VARIAVEIS = {
    "idade": "Idade",
    "sexo_homem": "Sexo masculino",
    "imc": "IMC",
    "pressao_sistolica": "Pressão sistólica",
    "pressao_diastolica": "Pressão diastólica",
    "colesterol_total": "Colesterol total",
    "atv_moderada_min_semana": "Atividade moderada (min/sem)",
    "atv_vigorosa_min_semana": "Atividade vigorosa (min/sem)",
    "minutos_sedentarios_dia": "Tempo sedentário (min/dia)",
    "hipertensao_informada": "Hipertensão informada",
    "colesterol_alto_informado": "Colesterol alto informado",
    "glicose_jejum": "Glicose em jejum",
    "cintura_cm": "Circunferência abdominal",
    "hdl": "Colesterol HDL",
}


dash.register_page(
    __name__, path="/triagem", name="Triagem de HbA1c", order=8
)


# =========================================================
# LEITURA DOS RESULTADOS EXPORTADOS
# =========================================================

def carregar(arquivo):
    try:
        return pd.read_csv(arquivo)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


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
            html.Strong(valor, className="cf-kpi-value"),
            html.P(nota, className="cf-kpi-note"),
        ],
        className="cf-kpi",
    )


def numero(valor, casas=3):
    return f"{valor:.{casas}f}".replace(".", ",")


def pct(valor):
    return f"{valor * 100:.1f}%".replace(".", ",")


# =========================================================
# COMPONENTES
# =========================================================

def criar_kpis(df):
    def auc(experimento, modelo="Regressão Logística"):
        linha = df[(df.experimento == experimento) & (df.modelo == modelo)]
        return None if linha.empty else linha.iloc[0]

    itens = [
        ("E1 essencial", "Variáveis que o app coleta: idade, sexo, IMC, "
                          "pressões e colesterol."),
        ("E7 + cintura e HDL", "Essencial + estilo de vida + informadas + "
                                "cintura e HDL (conjunto do simulador)."),
        ("E5 jejum sem glicose", "Mesmas pessoas do experimento com glicose "
                                  "(comparação justa)."),
        ("E6 jejum com glicose", "Essencial + glicose em jejum (exige exame)."),
    ]
    cartoes = []
    for experimento, nota in itens:
        linha = auc(experimento)
        if linha is None:
            continue
        cartoes.append(indicador(
            f"AUC · {experimento}",
            f"{numero(linha.cv_auc_media)} ± {numero(linha.cv_auc_dp)}",
            f"Regressão Logística, validação cruzada 5×3. {nota}",
        ))
    return cartoes


def criar_grafico_auc(df):
    ordem = list(dict.fromkeys(df.experimento))
    fig = go.Figure()
    for modelo, cor in COR_MODELOS.items():
        sub = df[df.modelo == modelo].set_index("experimento").reindex(ordem)
        fig.add_bar(
            name=modelo, x=ordem, y=sub.cv_auc_media,
            error_y=dict(type="data", array=sub.cv_auc_dp, visible=True),
            marker_color=cor,
            hovertemplate="%{x}<br>AUC %{y:.3f}<extra>" + modelo + "</extra>",
        )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#64736E")
    fig.update_layout(
        barmode="group", height=380, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="AUC (validação cruzada)", range=[0.4, 0.9]),
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def criar_importancia(df_imp, populacao, experimento, modelo):
    sub = df_imp[
        (df_imp.populacao == populacao)
        & (df_imp.experimento == experimento)
        & (df_imp.modelo == modelo)
    ].sort_values("queda_auc_media")
    if sub.empty:
        return aviso("Importância das variáveis não disponível.")
    fig = go.Figure(go.Bar(
        x=sub.queda_auc_media, y=[NOMES_VARIAVEIS.get(v, v) for v in sub.variavel],
        orientation="h", marker_color=COR_MODELOS[modelo],
        error_x=dict(type="data", array=sub.queda_auc_dp, visible=True),
        hovertemplate="%{y}<br>queda de AUC %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(280, 44 * len(sub) + 60),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Queda da AUC ao embaralhar a variável"),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def criar_tabela(df):
    colunas = [
        ("experimento", "Experimento"), ("modelo", "Modelo"), ("n", "N"),
        ("cv_auc_media", "AUC (CV)"), ("teste_auc", "AUC (teste)"),
        ("teste_recall_pos", "Sensibilidade"),
        ("teste_especificidade", "Especificidade"),
        ("teste_f1_macro", "F1 macro"),
    ]
    linhas = []
    for _, r in df.iterrows():
        linhas.append(html.Tr([
            html.Td(r.experimento), html.Td(r.modelo), html.Td(int(r.n)),
            html.Td(numero(r.cv_auc_media)), html.Td(numero(r.teste_auc)),
            html.Td(pct(r.teste_recall_pos)),
            html.Td(pct(r.teste_especificidade)),
            html.Td(numero(r.teste_f1_macro)),
        ]))
    return html.Div(
        html.Table(
            [html.Thead(html.Tr([html.Th(t) for _, t in colunas])),
             html.Tbody(linhas)],
            className="cf-table",
        ),
        className="cf-table-wrap",
    )


def criar_roc():
    if not ARQUIVO_ROC.is_file():
        return aviso("As curvas ROC ainda não foram exportadas.")
    imagem = base64.b64encode(ARQUIVO_ROC.read_bytes()).decode("ascii")
    return html.Img(
        src=f"data:image/png;base64,{imagem}",
        alt="Curvas ROC do Random Forest por experimento",
        style={"width": "100%", "height": "auto", "objectFit": "contain"},
    )


# =========================================================
# AUDITORIA E7 — SHAP E JUSTIÇA
# =========================================================

def pasta_populacao(pasta_base, populacao):
    """Impede que um valor não previsto seja usado como caminho de arquivo."""
    return pasta_base / populacao if populacao in POPULACOES else None


def imagem_png(caminho, texto_alternativo):
    if not caminho.is_file():
        return aviso(
            "Gráfico ainda não gerado. Execute o script correspondente "
            "para esta população."
        )
    try:
        codificada = base64.b64encode(caminho.read_bytes()).decode("ascii")
    except OSError:
        return aviso("Não foi possível abrir esta imagem.")
    return html.Img(
        src=f"data:image/png;base64,{codificada}",
        alt=texto_alternativo,
        className="tr-auditoria-imagem",
    )


def ler_json(caminho):
    try:
        with caminho.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def formatar_percentual(valor):
    return "—" if pd.isna(valor) else pct(valor)


def criar_ranking_shap(populacao):
    pasta = pasta_populacao(PASTA_SHAP, populacao)
    if pasta is None:
        return aviso("População inválida.")
    dados = carregar(pasta / "importancia_shap.csv")
    if dados is None or dados.empty:
        return aviso(
            "Execute: python -m classificacao.explicabilidade_shap "
            + ("--populacao todos" if populacao == "todos" else "")
        )
    if not {"variavel", "shap_abs_medio"}.issubset(dados.columns):
        return aviso("O CSV do SHAP não contém as colunas esperadas.")

    resumo = ler_json(pasta / "resumo.json")
    topo = dados.sort_values("shap_abs_medio", ascending=False).head(8)
    linhas = [
        html.Tr([
            html.Td(NOMES_VARIAVEIS.get(r.variavel, r.variavel)),
            html.Td(numero(r.shap_abs_medio, 4)),
        ])
        for _, r in topo.iterrows()
    ]
    informacoes = []
    if resumo.get("n_explicados") is not None:
        informacoes.append(
            html.P(f"Observações explicadas: {resumo['n_explicados']}",
                   className="tr-auditoria-destaque")
        )
    if resumo.get("teste_auc") is not None:
        informacoes.append(
            html.P(f"AUC da Random Forest no teste: {numero(resumo['teste_auc'])}",
                   className="tr-auditoria-destaque")
        )
    return html.Div([
        *informacoes,
        html.Div(html.Table([
            html.Thead(html.Tr([html.Th("Variável"), html.Th("|SHAP| médio")])),
            html.Tbody(linhas),
        ], className="cf-table"), className="cf-table-wrap"),
        html.P(
            "Valores médios absolutos na probabilidade prevista; "
            "importância não demonstra causalidade nem vazamento.",
            className="cf-card-note",
        ),
    ])


def criar_grafico_justica(recorte):
    ordem = ("Mulher", "Homem") if recorte.eixo.iloc[0] == "sexo" else (
        "< 40", "40–59", "60+"
    )
    recorte = recorte.set_index("grupo").reindex(
        [v for v in ordem if v in recorte.grupo.to_list()]
    ).reset_index()
    if recorte.empty:
        return aviso("Nenhum grupo disponível para o filtro selecionado.")

    rotulos = [f"{r.grupo} (n={int(r.n)})" for _, r in recorte.iterrows()]
    fig = go.Figure()
    for nome, coluna, inferior, superior, cor in (
        ("Sensibilidade", "recall_sensibilidade", "recall_ic95_inf",
         "recall_ic95_sup", "#0E766D"),
        ("Falsos positivos", "taxa_falsos_positivos", "fpr_ic95_inf",
         "fpr_ic95_sup", "#E58A3A"),
    ):
        y = recorte[coluna].fillna(0)
        fig.add_bar(
            name=nome, x=rotulos, y=y,
            marker_color=cor,
            error_y=dict(
                type="data", visible=True,
                array=(recorte[superior] - recorte[coluna]).fillna(0),
                arrayminus=(recorte[coluna] - recorte[inferior]).fillna(0),
            ),
            customdata=recorte[["n_positivos_reais", "n_negativos_reais"]],
            hovertemplate=(
                "%{x}<br>Taxa: %{y:.1%}<br>Positivos reais: %{customdata[0]}"
                "<br>Negativos reais: %{customdata[1]}<extra>" + nome + "</extra>"
            ),
        )
    fig.update_layout(
        barmode="group", height=370,
        margin=dict(l=10, r=10, t=12, b=30),
        yaxis=dict(title="Proporção", tickformat=".0%", range=[0, 1.05]),
        xaxis=dict(title="Subgrupos (tamanho no teste)"),
        legend=dict(orientation="h", y=1.14),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def criar_tabela_justica(recorte):
    linhas = []
    for _, r in recorte.iterrows():
        linhas.append(html.Tr([
            html.Td(str(r.grupo)),
            html.Td(int(r.n)),
            html.Td(int(r.n_positivos_reais)),
            html.Td(formatar_percentual(r.recall_sensibilidade)),
            html.Td(formatar_percentual(r.precisao_positiva)),
            html.Td(formatar_percentual(r.especificidade)),
            html.Td(formatar_percentual(r.taxa_falsos_positivos)),
            html.Td("—" if pd.isna(r.auc) else numero(r.auc)),
        ]))
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(c) for c in [
            "Grupo", "N", "Positivos", "Sensibilidade", "Precisão",
            "Especificidade", "Falsos positivos", "AUC",
        ]])),
        html.Tbody(linhas),
    ], className="cf-table"), className="cf-table-wrap")


def resumo_justica(recorte):
    if recorte.empty:
        return ""
    s = recorte["recall_sensibilidade"].dropna()
    diferenca = float((s.max() - s.min()) * 100) if len(s) > 1 else None
    notas = []
    if diferenca is not None:
        notas.append(f"Diferença máxima de sensibilidade: {numero(diferenca, 1)} pontos percentuais.")
    pequenos = recorte[recorte.n_positivos_reais < 30]
    if not pequenos.empty:
        nomes = ", ".join(str(x) for x in pequenos.grupo)
        notas.append(
            f"Atenção à incerteza: o(s) grupo(s) {nomes} "
            "tem/têm menos de 30 positivos reais no teste."
        )
    return html.Div([
        html.P(nota, className="tr-auditoria-destaque") for nota in notas
    ])


# =========================================================
# LAYOUT
# =========================================================

df_inicial = carregar(ARQUIVO_METRICAS)
EXPERIMENTOS = (
    list(dict.fromkeys(df_inicial.experimento)) if df_inicial is not None else []
)

layout = html.Div(
    [
        html.Div(
            [
                html.P("MACHINE LEARNING · NHANES 2021–2023", className="cf-eyebrow"),
                html.H1("Triagem de HbA1c ≥ 5,7%", className="cf-heading"),
                html.P(
                    "Classificação binária: a pessoa tem HbA1c abaixo ou acima "
                    "de 5,7%? A HbA1c é o alvo e não entra como variável. "
                    "É um apoio à triagem, não um diagnóstico.",
                    className="cf-intro",
                ),
            ]
        ),
        html.Div(
            [
                html.Strong("Como interpretar: "),
                "Ao contrário da atribuição de subperfil, o alvo aqui não é "
                "derivado das entradas, então a AUC mede capacidade real de "
                "separar os dois grupos. Os resultados são exibidos abaixo "
                "por experimento e população. A glicemia em jejum exige "
                "exame laboratorial.",
            ],
            className="cf-info",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("População", htmlFor="tr-populacao",
                                   className="cf-filter-label"),
                        dcc.Dropdown(
                            id="tr-populacao", value="sem_diagnostico", clearable=False,
                            options=[{"label": n, "value": k}
                                     for k, n in POPULACOES.items()],
                        ),
                    ],
                    className="cf-filter-group",
                ),
                html.Div(
                    [
                        html.Label("Experimento (importância das variáveis)",
                                   htmlFor="tr-experimento",
                                   className="cf-filter-label"),
                        dcc.Dropdown(
                            id="tr-experimento", clearable=False,
                            value=EXPERIMENTOS[0] if EXPERIMENTOS else None,
                            options=[{"label": e, "value": e} for e in EXPERIMENTOS],
                        ),
                    ],
                    className="cf-filter-group",
                ),
                html.Div(
                    [
                        html.Label("Modelo (importância)", htmlFor="tr-modelo",
                                   className="cf-filter-label"),
                        dcc.Dropdown(
                            id="tr-modelo", value="Regressão Logística",
                            clearable=False,
                            options=[{"label": m, "value": m} for m in COR_MODELOS],
                        ),
                    ],
                    className="cf-filter-group",
                ),
            ],
            className="cf-filters",
        ),
        html.Div(id="tr-kpis", className="cf-kpis"),
        cartao(
            "AUC por experimento e modelo",
            html.Div(id="tr-auc"),
            "Barras: média da validação cruzada repetida (5 dobras × 3); "
            "barras de erro: desvio-padrão. O modelo Dummy (sempre a "
            "prevalência) tem AUC 0,50.",
        ),
        html.Div(
            [
                cartao("Importância das variáveis", html.Div(id="tr-importancia"),
                       "Queda da AUC no conjunto de teste ao embaralhar cada variável."),
                cartao("Curvas ROC", criar_roc(),
                       "Random Forest no conjunto de teste (20%), por população."),
            ],
            className="cf-grid",
        ),
        cartao(
            "Métricas detalhadas",
            html.Div(id="tr-tabela"),
            "Sensibilidade e especificidade com limiar 0,5 e pesos de classe "
            "nos modelos supervisionados. O limiar ideal depende do custo de falso positivo e "
            "falso negativo e não foi otimizado.",
        ),
        html.Div(
            [
                html.P("AUDITORIA DOS MODELOS · EXPERIMENTO E7",
                       className="cf-eyebrow"),
                html.H2("Explicabilidade e justiça", className="tr-auditoria-titulo"),
                html.P(
                    "Resultados complementares do E7 (+ cintura e HDL), "
                    "sincronizados com a população selecionada acima. "
                    "O SHAP explica a Random Forest; a justiça compara "
                    "Random Forest e Regressão Logística no teste reservado.",
                    className="cf-intro",
                ),
            ],
            className="tr-auditoria-cabecalho",
        ),
        html.Div(
            [
                cartao(
                    "SHAP · explicações do E7",
                    html.Div([
                        dcc.RadioItems(
                            id="tr-shap-visao",
                            options=[{"label": rotulo, "value": arquivo}
                                     for arquivo, rotulo in VISOES_SHAP.items()],
                            value="shap_global.png",
                            className="tr-shap-abas",
                            labelStyle={"display": "inline-flex", "alignItems": "center"},
                        ),
                        html.Div(id="tr-shap-grafico", className="tr-shap-figura"),
                    ]),
                    "SHAP explica contribuições à probabilidade prevista de "
                    "HbA1c ≥ 5,7%; não representa causalidade nem diagnóstico.",
                ),
                cartao(
                    "Variáveis de maior influência",
                    html.Div(id="tr-shap-ranking"),
                    "Ranking na amostra de teste usada pelo script SHAP; "
                    "pode diferir da importância por permutação acima.",
                ),
            ],
            className="cf-grid",
        ),
        cartao(
            "Justiça · desempenho por subgrupo",
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Modelo", htmlFor="tr-justica-modelo",
                                   className="cf-filter-label"),
                        dcc.Dropdown(
                            id="tr-justica-modelo", value="Random Forest",
                            clearable=False,
                            options=[{"label": modelo, "value": modelo}
                                     for modelo in COR_MODELOS],
                        ),
                    ], className="cf-filter-group"),
                    html.Div([
                        html.Label("Comparar por", htmlFor="tr-justica-eixo",
                                   className="cf-filter-label"),
                        dcc.Dropdown(
                            id="tr-justica-eixo", value="faixa_etaria",
                            clearable=False,
                            options=[
                                {"label": "Faixa etária", "value": "faixa_etaria"},
                                {"label": "Sexo", "value": "sexo"},
                            ],
                        ),
                    ], className="cf-filter-group"),
                ], className="cf-filters tr-auditoria-filtros"),
                html.Div(id="tr-justica-conteudo"),
            ]),
            "Sensibilidade e taxa de falsos positivos com limiar 0,5. "
            "Barras de erro: intervalos de Wilson aproximados (95%). "
            "Os grupos têm tamanhos e prevalências diferentes.",
        ),
        html.Div(
            "Justiça: análise exploratória e não ponderada. "
            "As diferenças observadas não provam discriminação nem ausência dela. "
            "Os intervalos não incorporam o plano amostral complexo do NHANES. "
            "Os resultados não devem orientar decisões clínicas individuais.",
            className="cf-info",
        ),
        html.Div(
            "Limitações: base NHANES 2021–2023 dos EUA, sem uso dos pesos "
            "amostrais e sem validação externa. Os resultados valem para este "
            "conjunto e não substituem exame laboratorial nem avaliação "
            "médica. A Regressão Logística tem desempenho equivalente ao do "
            "Random Forest.",
            className="cf-info",
        ),
        html.P(
            "Fontes: modelos/classificacao_binaria/, "
            "modelos/explicabilidade_shap/ e modelos/avaliacao_justica/. "
            "Resultados gerados pelos scripts em classificacao/.",
            className="cf-source",
        ),
    ],
    className="cf-page",
)


# =========================================================
# CALLBACKS
# =========================================================

@callback(
    Output("tr-kpis", "children"),
    Output("tr-auc", "children"),
    Output("tr-tabela", "children"),
    Input("tr-populacao", "value"),
)
def atualizar_resultados(populacao):
    df = carregar(ARQUIVO_METRICAS)
    if df is None:
        msg = aviso("Execute classificacao/classificacao_binaria.py para gerar "
                    "metricas_binaria.csv.")
        return [], msg, msg
    sub = df[df.populacao == populacao]
    if sub.empty:
        msg = aviso("Sem resultados para esta população.")
        return [], msg, msg
    sem_dummy = sub[sub.modelo != "Dummy (prevalência)"]
    return criar_kpis(sem_dummy), criar_grafico_auc(sem_dummy), criar_tabela(sub)


@callback(
    Output("tr-importancia", "children"),
    Input("tr-populacao", "value"),
    Input("tr-experimento", "value"),
    Input("tr-modelo", "value"),
)
def atualizar_importancia(populacao, experimento, modelo):
    df_imp = carregar(ARQUIVO_IMPORTANCIA)
    if df_imp is None:
        return aviso("importancia_variaveis.csv não encontrado.")
    return criar_importancia(df_imp, populacao, experimento, modelo)


@callback(
    Output("tr-shap-grafico", "children"),
    Output("tr-shap-ranking", "children"),
    Input("tr-populacao", "value"),
    Input("tr-shap-visao", "value"),
)
def atualizar_shap(populacao, visao):
    pasta = pasta_populacao(PASTA_SHAP, populacao)
    if pasta is None or visao not in VISOES_SHAP:
        msg = aviso("Seleção inválida.")
        return msg, msg
    grafico = imagem_png(pasta / visao, VISOES_SHAP[visao])
    return grafico, criar_ranking_shap(populacao)


@callback(
    Output("tr-justica-conteudo", "children"),
    Input("tr-populacao", "value"),
    Input("tr-justica-modelo", "value"),
    Input("tr-justica-eixo", "value"),
)
def atualizar_justica(populacao, modelo, eixo):
    pasta = pasta_populacao(PASTA_JUSTICA, populacao)
    if pasta is None or modelo not in COR_MODELOS or eixo not in ("sexo", "faixa_etaria"):
        return aviso("Seleção inválida.")
    tabela = carregar(pasta / "justica_por_grupos.csv")
    if tabela is None:
        comando = "python -m classificacao.avaliacao_justica"
        if populacao == "todos":
            comando += " --populacao todos"
        return aviso(f"Resultados não disponíveis. Execute: {comando}")
    necessarias = {
        "modelo", "eixo", "grupo", "n", "n_positivos_reais",
        "n_negativos_reais", "recall_sensibilidade", "recall_ic95_inf",
        "recall_ic95_sup", "taxa_falsos_positivos", "fpr_ic95_inf",
        "fpr_ic95_sup", "precisao_positiva", "especificidade", "auc",
    }
    if not necessarias.issubset(tabela.columns):
        return aviso("O CSV de justiça não contém as colunas esperadas.")
    recorte = tabela[(tabela.modelo == modelo) & (tabela.eixo == eixo)].copy()
    if recorte.empty:
        return aviso("Nenhum resultado de justiça para esta seleção.")
    return html.Div([
        criar_grafico_justica(recorte),
        resumo_justica(recorte),
        criar_tabela_justica(recorte),
    ], className="tr-justica-resultados")
