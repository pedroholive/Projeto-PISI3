import dash
from dash import html, dcc

dash.register_page(
    __name__,
    path="/",
    name="Início",
    order=0
)


layout = html.Div(
    [
        # =====================================================
        # HERO
        # =====================================================
        html.Section(
            [
                html.Div(
                    [
                        # Lado esquerdo
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            className="hero-badge-dot"
                                        ),
                                        html.Span(
                                            "ANÁLISE CARDIOMETABÓLICA"
                                        ),
                                    ],
                                    className="hero-badge",
                                ),

                                html.H1(
                                    [
                                        "Dados que ajudam a compreender ",
                                        html.Span(
                                            "a saúde cardiometabólica.",
                                            className="hero-highlight",
                                        ),
                                    ],
                                    className="hero-title",
                                ),

                                html.P(
                                    "Explore indicadores relacionados à "
                                    "glicemia, pressão arterial, IMC, "
                                    "colesterol e atividade física por meio "
                                    "de análises baseadas nos dados do "
                                    "NHANES 2021–2023.",
                                    className="hero-description",
                                ),

                                html.Div(
                                    [
                                        dcc.Link(
                                            "Explorar análises",
                                            href="/eda",
                                            className="hero-button-primary",
                                        ),

                                        html.A(
                                            "Sobre os dados",
                                            href="#sobre-dados",
                                            className="hero-button-secondary",
                                        ),
                                    ],
                                    className="hero-actions",
                                ),
                            ],
                            className="hero-content",
                        ),

                        # Lado direito
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.P(
                                                            "VISÃO GERAL",
                                                            className="preview-label",
                                                        ),
                                                        html.H3(
                                                            "Indicadores "
                                                            "cardiometabólicos",
                                                            className="preview-title",
                                                        ),
                                                    ]
                                                ),

                                                html.Span(
                                                    "NHANES",
                                                    className="preview-badge",
                                                ),
                                            ],
                                            className="preview-header",
                                        ),

                                        # Indicadores visuais
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "HbA1c",
                                                            className="metric-name",
                                                        ),
                                                        html.Div(
                                                            "%",
                                                            className="metric-symbol",
                                                        ),
                                                        html.Div(
                                                            className="metric-line",
                                                        ),
                                                    ],
                                                    className="preview-metric",
                                                ),

                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Pressão",
                                                            className="metric-name",
                                                        ),
                                                        html.Div(
                                                            "mmHg",
                                                            className="metric-symbol",
                                                        ),
                                                        html.Div(
                                                            className="metric-line",
                                                        ),
                                                    ],
                                                    className="preview-metric",
                                                ),

                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "IMC",
                                                            className="metric-name",
                                                        ),
                                                        html.Div(
                                                            "kg/m²",
                                                            className="metric-symbol",
                                                        ),
                                                        html.Div(
                                                            className="metric-line",
                                                        ),
                                                    ],
                                                    className="preview-metric",
                                                ),
                                            ],
                                            className="preview-metrics",
                                        ),

                                        # Gráfico apenas decorativo
                                        html.Div(
                                            [
                                                html.Div(
                                                    className="chart-bar bar-1"
                                                ),
                                                html.Div(
                                                    className="chart-bar bar-2"
                                                ),
                                                html.Div(
                                                    className="chart-bar bar-3"
                                                ),
                                                html.Div(
                                                    className="chart-bar bar-4"
                                                ),
                                                html.Div(
                                                    className="chart-bar bar-5"
                                                ),
                                                html.Div(
                                                    className="chart-bar bar-6"
                                                ),
                                                html.Div(
                                                    className="chart-bar bar-7"
                                                ),
                                                html.Div(
                                                    className="chart-bar bar-8"
                                                ),
                                            ],
                                            className="preview-chart",
                                        ),

                                        html.Div(
                                            [
                                                html.Span(
                                                    className="preview-status-dot"
                                                ),
                                                html.Span(
                                                    "Exploração de dados de saúde populacional"
                                                ),
                                            ],
                                            className="preview-footer",
                                        ),
                                    ],
                                    className="hero-preview-card",
                                ),
                            ],
                            className="hero-visual",
                        ),
                    ],
                    className="hero-inner",
                )
            ],
            className="home-hero",
        ),

        # =====================================================
        # INDICADORES ANALISADOS
        # =====================================================
        html.Section(
            [
                html.Div(
                    [
                        html.P(
                            "EXPLORAÇÃO DOS DADOS",
                            className="section-eyebrow",
                        ),

                        html.H2(
                            "Principais indicadores analisados",
                            className="section-title",
                        ),

                        html.P(
                            "O dashboard reúne diferentes indicadores "
                            "para apoiar a exploração de características "
                            "glicêmicas e cardiometabólicas da amostra.",
                            className="section-description",
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "HbA1c",
                                            className="indicator-icon",
                                        ),
                                        html.H3("Glicemia"),
                                        html.P(
                                            "Exploração da hemoglobina "
                                            "glicada e sua distribuição "
                                            "na amostra."
                                        ),
                                    ],
                                    className="indicator-card",
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            "SYS",
                                            className="indicator-icon",
                                        ),
                                        html.H3("Pressão arterial"),
                                        html.P(
                                            "Análise das medidas de pressão "
                                            "arterial e de suas distribuições."
                                        ),
                                    ],
                                    className="indicator-card",
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            "IMC",
                                            className="indicator-icon",
                                        ),
                                        html.H3("Índice de massa corporal"),
                                        html.P(
                                            "Exploração do IMC como um dos "
                                            "indicadores do perfil "
                                            "cardiometabólico."
                                        ),
                                    ],
                                    className="indicator-card",
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            "COL",
                                            className="indicator-icon",
                                        ),
                                        html.H3("Colesterol"),
                                        html.P(
                                            "Visualização dos dados de "
                                            "colesterol disponíveis "
                                            "na amostra."
                                        ),
                                    ],
                                    className="indicator-card",
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            "ATV",
                                            className="indicator-icon",
                                        ),
                                        html.H3("Atividade física"),
                                        html.P(
                                            "Exploração de informações "
                                            "relacionadas à atividade física "
                                            "e ao comportamento sedentário."
                                        ),
                                    ],
                                    className="indicator-card",
                                ),
                            ],
                            className="indicators-grid",
                        ),
                    ],
                    className="section-container",
                )
            ],
            className="indicators-section",
        ),

        # =====================================================
        # SOBRE OS DADOS
        # =====================================================
        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "SOBRE OS DADOS",
                                    className="section-eyebrow",
                                ),

                                html.H2(
                                    "Dados de saúde populacional",
                                    className="section-title",
                                ),

                                html.P(
                                    "As análises do HealthSync utilizam "
                                    "dados do National Health and Nutrition "
                                    "Examination Survey (NHANES) 2021–2023, "
                                    "reunindo informações utilizadas na "
                                    "exploração de indicadores "
                                    "cardiometabólicos.",
                                    className="section-description",
                                ),
                            ],
                            className="data-content",
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "Fonte",
                                            className="data-card-label",
                                        ),
                                        html.Strong("NHANES"),
                                    ],
                                    className="data-info-card",
                                ),

                                html.Div(
                                    [
                                        html.Span(
                                            "Período",
                                            className="data-card-label",
                                        ),
                                        html.Strong("2021–2023"),
                                    ],
                                    className="data-info-card",
                                ),

                                html.Div(
                                    [
                                        html.Span(
                                            "Foco",
                                            className="data-card-label",
                                        ),
                                        html.Strong(
                                            "Saúde cardiometabólica"
                                        ),
                                    ],
                                    className="data-info-card",
                                ),
                            ],
                            className="data-cards",
                        ),
                    ],
                    className="data-inner",
                )
            ],
            id="sobre-dados",
            className="data-section",
        ),

        # =====================================================
        # MACHINE LEARNING
        # =====================================================
        html.Section(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "MACHINE LEARNING",
                                    className="ml-eyebrow",
                                ),

                                html.H2(
                                    "Exploração de perfis "
                                    "cardiometabólicos",
                                    className="ml-title",
                                ),

                                html.P(
                                    "O HealthSync também utiliza técnicas "
                                    "de aprendizado de máquina para explorar "
                                    "padrões e diferentes perfis presentes "
                                    "nos dados.",
                                    className="ml-description",
                                ),
                            ],
                            className="ml-content",
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(
                                            "01",
                                            className="ml-step-number",
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "Preparação"
                                                ),
                                                html.P(
                                                    "Tratamento e seleção "
                                                    "dos dados."
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="ml-step",
                                ),

                                html.Div(
                                    [
                                        html.Span(
                                            "02",
                                            className="ml-step-number",
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "Análise"
                                                ),
                                                html.P(
                                                    "Aplicação de técnicas "
                                                    "de aprendizado "
                                                    "de máquina."
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="ml-step",
                                ),

                                html.Div(
                                    [
                                        html.Span(
                                            "03",
                                            className="ml-step-number",
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "Perfis"
                                                ),
                                                html.P(
                                                    "Exploração e "
                                                    "caracterização dos "
                                                    "padrões encontrados."
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="ml-step",
                                ),
                            ],
                            className="ml-steps",
                        ),
                    ],
                    className="ml-inner",
                )
            ],
            className="ml-section",
        ),
    ],
    className="home-page",
)