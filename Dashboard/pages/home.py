import dash
from dash import html

dash.register_page(
    __name__,
    path="/",
    name="Início",
    order=0
)

layout = html.Div(
    [
        html.H1("Bem-vindo ao HealthSync"),

        html.P(
            "Dashboard de análise de indicadores "
            "cardiometabólicos."
        ),

        html.H2("Sobre o projeto"),

        html.P(
            "O HealthSync é um projeto de acompanhamento "
            "de glicemia, pressão arterial e outros "
            "indicadores relacionados à saúde."
        ),
    ]
)