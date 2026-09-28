import dash
from dash import html

dash.register_page(
    __name__,
    path="/eda",
    name="Análise Exploratória",
    order=1
)

layout = html.Div(
    [
        html.H1("Análise Exploratória dos Dados"),

        html.P(
            "Os gráficos interativos da EDA "
            "serão adicionados nesta página."
        ),
    ]
)