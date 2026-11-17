"""HealthSync — L5: triagem de HbA1c (< 5,7 contra >= 5,7).

Diferente da atribuição de subperfil, aqui o alvo (HbA1c) é um exame de
laboratório que NÃO entra como variável de entrada. A página apenas apresenta
os resultados exportados por classificacao/classificacao_binaria.py.
"""

import base64
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
        ("E4 ampliada", "Essencial + estilo de vida + hipertensão e "
                         "colesterol alto informados."),
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
                "separar os dois grupos. Com as variáveis que o app coleta o "
                "desempenho é moderado (AUC ≈ 0,75). A glicemia em jejum "
                "melhora bastante, mas exige exame laboratorial.",
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
                            id="tr-populacao", value="todos", clearable=False,
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
            "Sensibilidade e especificidade com limiar 0,5 e classes balanceadas "
            "no treino. O limiar ideal depende do custo de falso positivo e "
            "falso negativo e não foi otimizado.",
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
            "Fonte: modelos/classificacao_binaria/ (gerado por "
            "classificacao/classificacao_binaria.py).",
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
