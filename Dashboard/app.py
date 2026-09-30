import dash
from dash import Dash, html, dcc, Input, Output

app = Dash(
    __name__,
    use_pages=True,
    title="HealthSync | Dashboard"
)

app.layout = html.Div(
    [
        # URL atual
        dcc.Location(
            id="url",
            refresh=False
        ),

        html.Div(
            [
                # =====================================================
                # SIDEBAR
                # =====================================================
                html.Aside(
                    [
                        # Marca
                        html.Div(
                            [
                                html.Div(
                                    "H",
                                    className="sidebar-logo"
                                ),

                                html.Div(
                                    [
                                        html.H2(
                                            "HealthSync",
                                            className="sidebar-brand-name"
                                        ),

                                        html.P(
                                            "Health Analytics",
                                            className="sidebar-brand-subtitle"
                                        ),
                                    ],
                                    className="sidebar-brand-text"
                                ),
                            ],
                            className="sidebar-brand"
                        ),

                        # Navegação
                        html.Div(
                            [
                                html.P(
                                    "NAVEGAÇÃO",
                                    className="sidebar-section-title"
                                ),

                                html.Nav(
                                    [
                                        dcc.Link(
                                            [
                                                html.Span(
                                                    (
                                                        "⌂"
                                                        if page["path"] == "/"
                                                        else "▥"
                                                    ),
                                                    className="nav-icon"
                                                ),

                                                html.Span(
                                                    page["name"],
                                                    className="nav-text"
                                                ),
                                            ],
                                            href=page["relative_path"],
                                            id={
                                                "type": "nav-link",
                                                "path": page["path"]
                                            },
                                            className="nav-link"
                                        )
                                        for page in dash.page_registry.values()
                                    ],
                                    className="sidebar-nav"
                                ),
                            ],
                            className="sidebar-navigation"
                        ),

                        # Rodapé
                        html.Div(
                            [
                                html.Div(
                                    className="sidebar-footer-line"
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            className="sidebar-dataset-dot"
                                        ),

                                        html.Div(
                                            [
                                                html.Strong(
                                                    "NHANES 2021–2023"
                                                ),

                                                html.P(
                                                    "Dados populacionais"
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="sidebar-dataset"
                                ),
                            ],
                            className="sidebar-footer"
                        ),
                    ],
                    className="sidebar"
                ),

                # =====================================================
                # CONTEÚDO
                # =====================================================
                html.Main(
                    dash.page_container,
                    className="main-content"
                ),
            ],
            className="app-container"
        ),
    ]
)


# =========================================================
# PÁGINA ATIVA NA SIDEBAR
# =========================================================

@app.callback(
    Output(
        {"type": "nav-link", "path": dash.ALL},
        "className"
    ),
    Input("url", "pathname")
)
def atualizar_link_ativo(pathname):

    classes = []

    for page in dash.page_registry.values():

        if pathname == page["path"]:
            classes.append("nav-link nav-link-active")
        else:
            classes.append("nav-link")

    return classes


if __name__ == "__main__":
    app.run(debug=True)