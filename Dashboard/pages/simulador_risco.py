"""HealthSync — Simulador de triagem de HbA1c.

O usuário preenche variáveis clínicas e vê a probabilidade estimada de
HbA1c >= 5,7%. É um APOIO à triagem (quem vale a pena encaminhar para exame),
não um diagnóstico. Os modelos são treinados na primeira utilização, a partir
de dataset/healthsync_classificacao_v2.csv, e ficam em cache.
"""

import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dcc, html
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
ARQUIVO = ROOT / "dataset" / "healthsync_classificacao_v2.csv"
LIMIAR_HBA1C = 5.7
SEMENTE = 42

dash.register_page(
    __name__, path="/simulador", name="Simulador de Triagem", order=9
)

# =========================================================
# VARIÁVEIS
# =========================================================

NUMERICAS = {
    "idade": ("Idade", "anos", 18, 90),
    "imc": ("IMC", "kg/m²", 12, 70),
    "pressao_sistolica": ("Pressão sistólica", "mmHg", 70, 230),
    "pressao_diastolica": ("Pressão diastólica", "mmHg", 30, 140),
    "colesterol_total": ("Colesterol total", "mg/dL", 80, 450),
    "hdl": ("HDL", "mg/dL", 15, 150),
    "cintura_cm": ("Circunferência da cintura", "cm", 50, 200),
    "atv_moderada_min_semana": ("Atividade moderada", "min/semana", 0, 3000),
    "atv_vigorosa_min_semana": ("Atividade vigorosa", "min/semana", 0, 3000),
    "minutos_sedentarios_dia": ("Tempo sedentário", "min/dia", 0, 1000),
    "razao_renda_pobreza": ("Razão renda/pobreza", "0 a 5", 0, 5),
}

CATEGORICAS = {
    "sexo": ("Sexo", [("Homem", "Homem"), ("Mulher", "Mulher")]),
    "hipertensao_informada": (
        "Pressão alta informada por médico",
        [(0.0, "Não"), (1.0, "Sim")],
    ),
    "colesterol_alto_informado": (
        "Colesterol alto informado por médico",
        [(0.0, "Não"), (1.0, "Sim")],
    ),
    "raca_etnia": (
        "Raça/etnia (categorias do NHANES)",
        [(1.0, "Mexicano-americano"), (2.0, "Outro hispânico"),
         (3.0, "Branco não hispânico"), (4.0, "Negro não hispânico"),
         (6.0, "Asiático não hispânico"), (7.0, "Outra/multirracial")],
    ),
    "escolaridade": (
        "Escolaridade",
        [(1.0, "Menos de 9 anos"), (2.0, "9 a 11 anos"),
         (3.0, "Ensino médio"), (4.0, "Superior incompleto/técnico"),
         (5.0, "Superior completo ou mais")],
    ),
}

VERSOES = {
    "clinica": {
        "nome": "Clínica (13 variáveis)",
        "num": ["idade", "imc", "pressao_sistolica", "pressao_diastolica",
                "colesterol_total", "hdl", "cintura_cm",
                "atv_moderada_min_semana", "atv_vigorosa_min_semana",
                "minutos_sedentarios_dia"],
        "cat": ["sexo", "hipertensao_informada", "colesterol_alto_informado"],
    },
    "ampliada": {
        "nome": "Ampliada (16 variáveis, inclui demográficas)",
        "num": ["idade", "imc", "pressao_sistolica", "pressao_diastolica",
                "colesterol_total", "hdl", "cintura_cm",
                "atv_moderada_min_semana", "atv_vigorosa_min_semana",
                "minutos_sedentarios_dia", "razao_renda_pobreza"],
        "cat": ["sexo", "hipertensao_informada", "colesterol_alto_informado",
                "raca_etnia", "escolaridade"],
    },
}

MODELOS = {
    "lr": "Regressão Logística",
    "gb": "Gradient Boosting",
}


def rotulo(nome):
    if nome in NUMERICAS:
        return NUMERICAS[nome][0]
    return CATEGORICAS[nome][0]


# =========================================================
# TREINO (com cache)
# =========================================================

@lru_cache(maxsize=1)
def carregar():
    d = pd.read_csv(ARQUIVO, encoding="utf-8-sig")
    return d, (d["hba1c"] >= LIMIAR_HBA1C).astype(int)


def criar_pipeline(modelo, num, cat):
    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")),
                        ("s", RobustScaler())]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    if modelo == "lr":
        clf = LogisticRegression(max_iter=3000)
    else:
        clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            l2_regularization=1.0, random_state=SEMENTE)
    return Pipeline([("t", pre), ("c", clf)])


def preparar_X(d, num, cat):
    X = d[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype(object)
    return X


@lru_cache(maxsize=8)
def treinar(modelo, versao):
    d, y = carregar()
    cfg = VERSOES[versao]
    num, cat = cfg["num"], cfg["cat"]
    X = preparar_X(d, num, cat)

    # Desempenho e limiares por validação cruzada
    oof = cross_val_predict(
        criar_pipeline(modelo, num, cat), X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEMENTE),
        method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    pr, rc, th = precision_recall_curve(y, oof)

    def limiar_para(sens):
        i = np.where(rc[:-1] >= sens)[0][-1]
        return float(th[i]), float(pr[i])

    lim_alerta, prec_alerta = limiar_para(0.80)
    lim_alto, prec_alto = limiar_para(0.50)

    # Importância global (permutação em um conjunto de teste)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMENTE)
    m_teste = criar_pipeline(modelo, num, cat).fit(Xtr, ytr)
    imp = permutation_importance(m_teste, Xte, yte, scoring="roc_auc",
                                 n_repeats=5, random_state=SEMENTE)
    importancia = pd.Series(imp.importances_mean, index=num + cat)
    importancia = importancia.sort_values(ascending=False)

    # Modelo final em todos os dados
    final = criar_pipeline(modelo, num, cat).fit(X, y)
    medianas = d[num].median().to_dict()

    return {
        "modelo": final, "num": num, "cat": cat, "auc": auc,
        "lim_alerta": lim_alerta, "prec_alerta": prec_alerta,
        "lim_alto": lim_alto, "prec_alto": prec_alto,
        "importancia": importancia, "medianas": medianas,
        "X_treino": X,
    }


def contribuicoes_lr(info, linha):
    """Contribuição de cada variável (coef x desvio em relação à média)."""
    pre = info["modelo"].named_steps["t"]
    clf = info["modelo"].named_steps["c"]
    nomes = pre.get_feature_names_out()
    z = pre.transform(linha)
    z = z.toarray() if hasattr(z, "toarray") else np.asarray(z)
    zt = pre.transform(info["X_treino"])
    zt = zt.toarray() if hasattr(zt, "toarray") else np.asarray(zt)
    contrib = clf.coef_[0] * (z[0] - zt.mean(axis=0))
    por_var = {}
    for nome, valor in zip(nomes, contrib):
        base = nome.split("__", 1)[1]
        original = base
        if nome.startswith("c__"):
            original = next(c for c in info["cat"] if base.startswith(c + "_"))
        por_var[original] = por_var.get(original, 0.0) + float(valor)
    return pd.Series(por_var).sort_values(key=np.abs, ascending=True)


# =========================================================
# COMPONENTES
# =========================================================

def campo(nome, info):
    if nome in NUMERICAS:
        titulo, unidade, mn, mx = NUMERICAS[nome]
        valor = info["medianas"].get(nome)
        entrada = dcc.Input(
            id={"tipo": "campo", "nome": nome}, type="number",
            min=mn, max=mx, value=round(float(valor), 1) if valor is not None else None,
            debounce=False, className="sim-input",
            placeholder=f"{unidade}",
        )
        sub = f"{unidade} · vazio usa a mediana"
    else:
        titulo, opcoes = CATEGORICAS[nome]
        d, _ = carregar()
        moda = d[nome].mode().iloc[0]
        entrada = dcc.Dropdown(
            id={"tipo": "campo", "nome": nome}, clearable=False,
            options=[{"label": r, "value": v} for v, r in opcoes],
            value=moda,
        )
        sub = "selecione"
    return html.Div(
        [html.Label(titulo, className="sim-label"), entrada,
         html.Span(sub, className="sim-sub")],
        className="sim-campo",
    )


def grafico_barras(serie, titulo_x, cor, altura=None):
    fig = go.Figure(go.Bar(
        x=serie.values, y=[rotulo(v) for v in serie.index], orientation="h",
        marker_color=cor,
        hovertemplate="%{y}<br>%{x:.3f}<extra></extra>"))
    fig.update_layout(
        height=altura or max(260, 34 * len(serie) + 70),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title=titulo_x), plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def cartao(titulo, conteudo, nota=None):
    return html.Div(
        [html.H3(titulo, className="cf-card-title"), conteudo,
         html.P(nota, className="cf-card-note") if nota else None],
        className="cf-card")


# =========================================================
# LAYOUT
# =========================================================

layout = html.Div(
    [
        html.Div(
            [
                html.P("MACHINE LEARNING · NHANES 2021–2023", className="cf-eyebrow"),
                html.H1("Simulador de triagem de HbA1c", className="cf-heading"),
                html.P(
                    "Preencha as variáveis e veja a probabilidade estimada de "
                    "HbA1c ≥ 5,7%. É um apoio à decisão de quem vale a pena "
                    "encaminhar para exame, não um diagnóstico.",
                    className="cf-intro"),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [html.Label("Modelo", className="cf-filter-label"),
                     dcc.Dropdown(
                         id="sim-modelo", value="lr", clearable=False,
                         options=[{"label": n, "value": k}
                                  for k, n in MODELOS.items()])],
                    className="cf-filter-group"),
                html.Div(
                    [html.Label("Conjunto de variáveis", className="cf-filter-label"),
                     dcc.Dropdown(
                         id="sim-versao", value="clinica", clearable=False,
                         options=[{"label": v["nome"], "value": k}
                                  for k, v in VERSOES.items()])],
                    className="cf-filter-group"),
            ],
            className="cf-filters"),
        dcc.Loading(html.Div(id="sim-formulario"), type="circle"),
        html.Button("Calcular risco", id="sim-calcular", n_clicks=0,
                    className="sim-botao"),
        html.Div(id="sim-resultado"),
        html.Div(id="sim-desempenho"),
        html.Div(id="sim-importancia"),
        dcc.Store(id="sim-historico", data=[]),
        cartao("Histórico desta sessão", html.Div(id="sim-historico-lista")),
        html.Div(
            "Limitações: modelo treinado no NHANES 2021–2023 (EUA), sem pesos "
            "amostrais e sem validação externa. A probabilidade reflete a "
            "prevalência dessa base (39,5% com HbA1c ≥ 5,7). Raça/etnia, "
            "escolaridade e renda entram apenas na versão ampliada, como "
            "análise epidemiológica. Este simulador não substitui exame "
            "laboratorial nem avaliação médica.",
            className="cf-info"),
        html.P("Fonte: dataset/healthsync_classificacao_v2.csv", className="cf-source"),
    ],
    className="cf-page",
)


# =========================================================
# CALLBACKS
# =========================================================

@callback(
    Output("sim-formulario", "children"),
    Output("sim-desempenho", "children"),
    Output("sim-importancia", "children"),
    Input("sim-modelo", "value"),
    Input("sim-versao", "value"),
)
def montar(modelo, versao):
    if not ARQUIVO.is_file():
        aviso = html.Div(
            "Arquivo dataset/healthsync_classificacao_v2.csv não encontrado.",
            className="cf-empty")
        return aviso, None, None
    info = treinar(modelo, versao)
    ordem = list(info["importancia"].index)
    form = html.Div(
        [html.P("Os campos estão ordenados pela importância global no modelo "
                "selecionado. Campos numéricos vazios usam a mediana da base.",
                className="cf-card-note"),
         html.Div([campo(n, info) for n in ordem], className="sim-grid")],
        className="cf-card")
    desempenho = cartao(
        "Desempenho do modelo (validação cruzada)",
        html.Div(
            [
                html.Div([html.P("AUC", className="cf-kpi-label"),
                          html.Strong(f"{info['auc']:.3f}".replace(".", ","),
                                      className="cf-kpi-value")], className="cf-kpi"),
                html.Div([html.P("Limiar de atenção", className="cf-kpi-label"),
                          html.Strong(f"{info['lim_alerta']*100:.0f}%",
                                      className="cf-kpi-value"),
                          html.P(f"sensibilidade 80%, precisão "
                                 f"{info['prec_alerta']*100:.0f}%",
                                 className="cf-kpi-note")], className="cf-kpi"),
                html.Div([html.P("Limiar de risco alto", className="cf-kpi-label"),
                          html.Strong(f"{info['lim_alto']*100:.0f}%",
                                      className="cf-kpi-value"),
                          html.P(f"sensibilidade 50%, precisão "
                                 f"{info['prec_alto']*100:.0f}%",
                                 className="cf-kpi-note")], className="cf-kpi"),
            ],
            className="cf-kpis"),
        "Os limiares vêm da validação cruzada: o de atenção prioriza capturar "
        "a maioria dos casos (sensibilidade), aceitando falsos alarmes.")
    imp = cartao(
        "Importância global das variáveis",
        grafico_barras(info["importancia"].sort_values(),
                       "Queda da AUC ao embaralhar a variável", "#0E766D"))
    return form, desempenho, imp


@callback(
    Output("sim-resultado", "children"),
    Output("sim-historico", "data"),
    Input("sim-calcular", "n_clicks"),
    State("sim-modelo", "value"),
    State("sim-versao", "value"),
    State({"tipo": "campo", "nome": ALL}, "value"),
    State({"tipo": "campo", "nome": ALL}, "id"),
    State("sim-historico", "data"),
    prevent_initial_call=True,
)
def calcular(_, modelo, versao, valores, ids, historico):
    info = treinar(modelo, versao)
    valores_por_nome = {i["nome"]: v for i, v in zip(ids, valores)}
    linha = {}
    for n in info["num"]:
        v = valores_por_nome.get(n)
        linha[n] = np.nan if v is None else float(v)
    for n in info["cat"]:
        linha[n] = valores_por_nome.get(n)
    X = pd.DataFrame([linha])[info["num"] + info["cat"]]
    for c in info["cat"]:
        X[c] = X[c].astype(object)

    p = float(info["modelo"].predict_proba(X)[0, 1])
    if p >= info["lim_alto"]:
        faixa, cor, msg = ("Risco alto", "#B23A48",
                           "Recomenda-se exame de HbA1c.")
    elif p >= info["lim_alerta"]:
        faixa, cor, msg = ("Atenção", "#B7791F",
                           "Vale considerar um exame de HbA1c.")
    else:
        faixa, cor, msg = ("Risco menor", "#0E766D",
                           "Não exclui a necessidade de exame se houver sintomas "
                           "ou outros fatores de risco.")

    partes = [
        html.Div(
            [html.Strong(f"{p*100:.1f}%".replace(".", ","),
                         style={"fontSize": "44px", "color": cor}),
             html.P("probabilidade estimada de HbA1c ≥ 5,7%",
                    className="cf-kpi-label"),
             html.Div(html.Div(style={"width": f"{min(p,1)*100:.1f}%",
                                      "background": cor, "height": "10px",
                                      "borderRadius": "6px"}),
                      style={"background": "#E8EFED", "borderRadius": "6px"}),
             html.P(f"{faixa}: {msg}", style={"fontWeight": 600, "marginTop": "14px"}),
             html.P(f"Modelo: {MODELOS[modelo]} · {VERSOES[versao]['nome']}",
                    className="cf-card-note"),
             html.P("Esta estimativa é um apoio à triagem e não substitui o "
                    "diagnóstico de um profissional de saúde.",
                    className="cf-card-note")],
            style={"textAlign": "center"}),
    ]
    if modelo == "lr":
        contrib = contribuicoes_lr(info, X)
        top = contrib.tail(8)  # as 8 maiores contribuições (em módulo)
        partes.append(html.H4("O que mais pesou neste resultado",
                              className="cf-card-title",
                              style={"marginTop": "22px"}))
        partes.append(grafico_barras(
            top, "Contribuição (log-odds; positivo = aumenta o risco)", cor, 320))
    else:
        partes.append(html.P(
            "A explicação individual está disponível para a Regressão Logística.",
            className="cf-card-note"))

    novo = {"p": p, "faixa": faixa, "modelo": MODELOS[modelo],
            "versao": VERSOES[versao]["nome"],
            "hora": datetime.now().strftime("%d/%m/%Y às %H:%M")}
    return html.Div(partes, className="cf-card"), ([novo] + (historico or []))[:10]


@callback(Output("sim-historico-lista", "children"),
          Input("sim-historico", "data"))
def mostrar_historico(historico):
    if not historico:
        return html.Div("Nenhuma avaliação nesta sessão.", className="cf-empty")
    return html.Div([
        html.Div(
            [html.Strong(f"{h['p']*100:.1f}% · {h['faixa']}".replace(".", ",")),
             html.Div(f"{h['modelo']} · {h['versao']}", className="cf-card-note"),
             html.Div(h["hora"], className="cf-card-note")],
            style={"padding": "10px 0", "borderBottom": "1px solid #E8EFED"})
        for h in historico])
