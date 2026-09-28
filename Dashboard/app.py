import dash
from dash import Dash, html, dcc

app = Dash(
    __name__,
    use_pages=True,
    title="HealthSync | Dashboard"
)

app.layout = html.Div(
    [
        html.Aside(
            [
                html.H2("HealthSync"),
                html.P("Dashboard Cardiometabólico"),

                html.Hr(),

                html.Nav(
                    [
                        dcc.Link(
                            page["name"],
                            href=page["relative_path"],
                            className="nav-link"
                        )
                        for page in dash.page_registry.values()
                    ]
                ),
            ],
            className="sidebar"
        ),

        html.Main(
            dash.page_container,
            className="main-content"
        ),
    ],
    className="app-container"
)

if __name__ == "__main__":
    app.run(debug=True)