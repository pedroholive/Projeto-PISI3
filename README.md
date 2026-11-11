<div align="center">

# 🩺 HealthSync — Acompanhamento Cardiometabólico

**Dos dados de saúde à análise: exploração de indicadores glicêmicos, pressão arterial, perfil cardiometabólico e desenvolvimento de modelos de Machine Learning.**

`Python` · `Pandas` · `Scikit-Learn` · `Matplotlib` · `Seaborn` 

<a href="#-visão-geral">Visão geral</a> ·
<a href="#️-o-healthsync">HealthSync</a> ·
<a href="#️-o-dataset">Dataset</a> ·
<a href="#-análise-exploratória-de-dados">EDA</a> ·
<a href="#-machine-learning">Machine Learning</a>

</div>

---

## 📌 Visão geral

O **HealthSync** é um projeto acadêmico voltado à análise e ao acompanhamento de indicadores relacionados à saúde cardiometabólica, com foco principalmente em **glicemia** e **pressão arterial**.

O projeto integra análise de dados, Machine Learning e desenvolvimento de uma aplicação para investigar três questões principais:

1. **O que os dados revelam?** — análise exploratória de indicadores demográficos, clínicos e comportamentais relacionados ao perfil cardiometabólico.

2. **Como diferentes indicadores se relacionam?** — investigação das relações entre glicemia, pressão arterial, IMC, colesterol, atividade física, sedentarismo e histórico informado de saúde.

3. **Como esses dados podem apoiar o HealthSync?** — investigação de
perfis cardiometabólicos por meio de aprendizado de máquina,
incluindo a clusterização e a atribuição de subperfis estatísticos,
com possibilidade de integração futura ao aplicativo.

Atualmente, o projeto possui as etapas de **preparação dos dados, Análise Exploratória de Dados (EDA), clusterização e atribuição de subperfil** desenvolvidas. A etapa de **dashboard está em desenvolvimento**.

> O HealthSync possui finalidade acadêmica e de apoio ao acompanhamento de informações de saúde. O sistema não substitui diagnóstico, avaliação ou orientação realizada por profissionais de saúde.

### NHANES × HealthSync

Os dados utilizados nesta etapa do projeto não correspondem aos registros dos usuários do aplicativo.

O **NHANES** fornece dados de diferentes participantes e é utilizado para análise exploratória e desenvolvimento dos modelos de Machine Learning.

O **HealthSync**, por outro lado, permitirá o registro sucessivo de informações do próprio usuário, possibilitando o acompanhamento individual das medidas ao longo do tempo.

**NHANES → análise e desenvolvimento dos modelos**

**HealthSync → registro, visualização e acompanhamento dos indicadores do usuário**

## 🗂️ O Dataset

O projeto utiliza dados do **National Health and Nutrition Examination Survey (NHANES), ciclo agosto de 2021 a agosto de 2023**.

Após a integração e preparação dos diferentes componentes da pesquisa, foi construída uma base analítica voltada aos indicadores utilizados pelo HealthSync.

<div align="center">

| 👥 Participantes | 🧮 Variáveis | 🩺 Domínio | 📅 Ciclo |
|:---:|:---:|:---:|:---:|
| **5.283** | **23** | Saúde cardiometabólica | **2021–2023** |

</div>

As variáveis estão organizadas em diferentes dimensões:

**1. Perfil demográfico e corporal**  
`idade` · `sexo` · `imc`

**2. Perfil glicêmico**  
`hba1c` · `glicose_jejum` · `diabetes_informado`

**3. Pressão arterial**  
`pressao_sistolica` · `pressao_diastolica` · `hipertensao_informada`

**4. Perfil lipídico**  
`colesterol_total` · `colesterol_alto_informado`

**5. Estilo de vida**  
`atv_moderada_min_semana` · `atv_vigorosa_min_semana` · `minutos_sedentarios_dia`

### Bases analíticas

Foram organizados dois conjuntos principais:

- **CORE — 5.283 participantes:** base principal utilizada na análise exploratória;
- **FASTING — 2.955 participantes:** subamostra utilizada nas análises específicas de glicose em jejum.

A separação permite analisar a glicose em jejum sem reduzir desnecessariamente a amostra utilizada nas demais investigações.

## 🔍 Análise Exploratória de Dados

A pasta [`analises/`](analises/) reúne as investigações realizadas para compreender a estrutura, qualidade e comportamento dos dados antes da etapa de Machine Learning.

| Análise | Foco |
|---|---|
| **Perfil da amostra** | Idade, sexo e distribuição do IMC |
| **Perfil glicêmico** | HbA1c, glicose em jejum e histórico informado de diabetes |
| **Pressão arterial** | Pressão sistólica, diastólica e histórico informado de hipertensão |
| **Perfil lipídico** | Colesterol total e histórico informado de colesterol elevado |
| **Atividade física e sedentarismo** | Atividade moderada, vigorosa e comportamento sedentário |
| **Relações cardiometabólicas** | Relações entre IMC, HbA1c, pressão arterial e colesterol |
| **Histórico × medidas objetivas** | Comparação entre informações relatadas e medidas obtidas nos exames |

### 💡 Principais achados da EDA

- **Qualidade da base:** não foram identificados participantes duplicados ou registros completamente repetidos na base CORE.

- **Perfil glicêmico:** a HbA1c apresentou média de **5,78%** e mediana de **5,50%**, com distribuição assimétrica à direita.

- **Histórico de diabetes:** participantes que informaram diabetes apresentaram valores de HbA1c mais elevados em comparação aos que não informaram a condição, embora exista sobreposição entre os grupos.

- **Pressão arterial:** a pressão sistólica apresentou média de **122,67 mmHg** e a diastólica de **74,75 mmHg**.

- **Sistólica × diastólica:** as duas medidas apresentaram associação linear positiva (**r = 0,612**).

- **Perfil lipídico:** o colesterol total apresentou média de **188,52 mg/dL** e mediana de **186 mg/dL**.

- **Atividade física:** **54,4%** dos participantes com informação disponível apresentaram zero minutos semanais de atividade vigorosa.

- **Sedentarismo:** a mediana do tempo sedentário correspondeu a aproximadamente **5 horas por dia**.

- **Relações cardiometabólicas:** entre indicadores de diferentes dimensões, as correlações lineares observadas foram relativamente pequenas, mostrando que diferentes variáveis podem fornecer informações complementares sobre os participantes.

> Os resultados representam associações descritivas na amostra analisada. Correlação não implica causalidade e as medidas analisadas não devem ser interpretadas isoladamente como diagnósticos.


## 🤖 Machine Learning — Perfis e Atribuição de Subperfil

O projeto utiliza técnicas de aprendizado de máquina para investigar
agrupamentos estatísticos de participantes e a possibilidade de
reproduzir esses agrupamentos por meio de modelos supervisionados.

### Clusterização — K-Means

A clusterização utiliza o algoritmo K-Means para identificar
subperfis cardiometabólicos dentro de três estratos de hemoglobina
glicada (HbA1c).

Os agrupamentos são representados pela variável `perfil_id`,
que identifica o subperfil atribuído a cada participante.

Esses subperfis representam agrupamentos estatísticos e não
correspondem necessariamente a categorias clínicas ou níveis
de risco validados.

### Atribuição de Subperfil — Random Forest

O Random Forest foi utilizado para reproduzir os identificadores
de subperfil previamente gerados pela clusterização.

As cinco variáveis de entrada do modelo são:

- Idade;
- IMC;
- Pressão arterial sistólica;
- Pressão arterial diastólica;
- Colesterol total.

O treinamento utiliza três modelos independentes, um para
cada estrato de HbA1c.

O pipeline mantém o RobustScaler e a separação estratificada
entre treinamento e teste, com 80% e 20% dos registros,
respectivamente.

### Interpretação dos resultados

Uma limitação importante foi identificada na definição do alvo
da classificação supervisionada.

A variável `perfil_id` foi construída anteriormente pelo K-Means,
a partir de características dos próprios participantes.

Consequentemente, o Random Forest utiliza indicadores relacionados
àqueles que deram origem ao alvo que procura reproduzir.

Os valores elevados de F1-score macro, aproximadamente entre
0,92 e 0,97, demonstram que o modelo consegue reproduzir
os subperfis estatísticos atribuídos pelo K-Means no conjunto
de teste.

Esses resultados não comprovam capacidade de prever um novo
desfecho clínico, diagnosticar diabetes ou hipertensão, nem
estimar risco cardiometabólico individual.

Portanto, neste pipeline, o modelo deve ser apresentado como
um mecanismo de atribuição de subperfil, e não como um
classificador de doenças.

### Comparação com linhas de base

A avaliação dos modelos inclui uma comparação com a
Regressão Logística, utilizada como linha de base.

A comparação considera os mesmos rótulos de subperfil e
a mesma divisão entre treinamento e teste.

Os resultados permitem verificar como diferentes algoritmos
conseguem reproduzir a partição estatística estabelecida
pela clusterização.

O desempenho elevado de uma linha de base reforça a
necessidade de interpretar as métricas como concordância
com os agrupamentos existentes, não como validação clínica.

### Limitação relacionada à HbA1c

A inferência de subperfil exige a disponibilidade de HbA1c
para identificar qual dos três modelos deve ser utilizado.

A HbA1c não integra as cinco variáveis de entrada do
Random Forest dentro do estrato, mas é necessária
para selecionar o modelo correspondente.

Essa exigência representa uma limitação para a futura
integração com o aplicativo HealthSync, pois nem todos
os pacientes terão resultados recentes desse exame.

O pipeline atual não possui uma alternativa validada
para realizar a atribuição de subperfil sem HbA1c.

### Aplicação futura

Os modelos treinados permanecem disponíveis para utilização
em uma funcionalidade de atribuição de subperfil.

Entretanto, essa funcionalidade deve ser apresentada
como uma classificação de agrupamentos estatísticos,
sem atribuição automática de diagnósticos ou riscos clínicos.


## 🧠 L5 — Classificação Preditiva da Faixa de HbA1c

### 🎯 Objetivo

Esta etapa do **HealthSync** investiga a possibilidade de estimar a faixa de hemoglobina glicada (HbA1c) de participantes sem diagnóstico de diabetes autorrelatado, utilizando indicadores clínicos e demográficos.

Diferentemente do modelo anterior de **Atribuição de Subperfil**, que reproduz rótulos gerados pelo K-Means, a L5 utiliza um alvo binário construído a partir de uma medida laboratorial.

**Definição das classes:**

- **Classe 0:** HbA1c < 5,7%.
- **Classe 1:** HbA1c ≥ 5,7%.

> [!IMPORTANT]
> A classificação é experimental e não representa diagnóstico médico nem um sistema de triagem clinicamente validado.

---

### 📊 Dataset e preparação

**Fonte:** NHANES 2021–2023.

**Arquivo utilizado:** `dataset/healthsync_nhanes_2021_2023_core.csv`

Foram selecionados apenas participantes com `diabetes_informado == "Nao"`. As categorias `Sim` e `Borderline` não foram incluídas.

| Indicador | Quantidade |
|---|---:|
| Participantes iniciais | 5.283 |
| Participantes após o filtro | 4.360 |
| Classe 0 — HbA1c < 5,7% | 3.089 |
| Classe 1 — HbA1c ≥ 5,7% | 1.271 |
| Conjunto de treinamento | 3.488 |
| Conjunto de teste | 872 |

A divisão foi estratificada, reservando:

- **80%** dos registros para treinamento;
- **20%** para avaliação final;
- `random_state=42` para garantir a reprodutibilidade.

---

### 🔬 Variáveis de entrada

Foram desenvolvidas duas versões experimentais para avaliar o efeito da inclusão de informações adicionais.

#### Versão A — Variáveis básicas

Utiliza seis variáveis:

1. Idade;
2. Índice de massa corporal (IMC);
3. Pressão arterial sistólica;
4. Pressão arterial diastólica;
5. Colesterol total;
6. Sexo.

#### Versão B — Variáveis ampliadas

Utiliza todas as variáveis da **Versão A**, acrescentando:

7. Hipertensão informada (`hipertensao_informada`).

> [!NOTE]
> A variável `diabetes_informado` foi utilizada exclusivamente para filtrar os participantes. Ela não foi utilizada como entrada dos classificadores.

Para prevenir vazamento de dados (*data leakage*), foram excluídas das features a HbA1c contínua, `perfil_id`, identificadores dos participantes e variáveis que pudessem revelar diretamente o alvo.

---

### ⚙️ Treinamento e prevenção de vazamento de dados

Foram utilizados dois algoritmos supervisionados:

- **Regressão Logística**
- **Random Forest**

Cada algoritmo foi treinado nas versões A e B, totalizando **quatro modelos**.

O pré-processamento foi implementado com pipelines do `scikit-learn`, utilizando:

- `SimpleImputer` para tratamento de valores ausentes;
- `OneHotEncoder` para codificação das variáveis categóricas;
- `RobustScaler` para escalonamento das variáveis numéricas;
- `ColumnTransformer` para organizar as transformações.

A validação cruzada utilizou `StratifiedKFold`, com **5 folds**, exclusivamente no conjunto de treinamento.

Os transformadores foram ajustados dentro de cada fold, evitando que informações das amostras de validação fossem utilizadas durante o ajuste do pré-processamento.

---

### 📈 Resultados da validação cruzada

Os resultados médios obtidos durante a validação cruzada foram:

| Versão | Modelo | AUC-ROC média | F1 médio |
|---|---|---:|---:|
| A | Regressão Logística | 0,7150 | 0,5291 |
| A | Random Forest | 0,6872 | 0,3806 |
| B | Regressão Logística | **0,7172** | **0,5305** |
| B | Random Forest | 0,6863 | 0,3867 |

A **Regressão Logística da Versão B** apresentou a maior AUC-ROC média na validação cruzada.

Entretanto, a diferença em relação à Versão A foi pequena, não sendo suficiente, isoladamente, para afirmar uma melhoria relevante.

---

### 📉 Resultados no conjunto de teste

Os modelos foram avaliados nos **872 registros reservados**, que não participaram do ajuste dos pipelines.

| Versão | Modelo | AUC-ROC | Sensibilidade | Especificidade | F1-score |
|---|---|---:|---:|---:|---:|
| A | Regressão Logística | **0,7257** | **72,83%** | 61,49% | **0,5465** |
| A | Random Forest | 0,7071 | 35,43% | **85,28%** | 0,4138 |
| B | Regressão Logística | 0,7249 | 70,87% | 62,46% | 0,5405 |
| B | Random Forest | 0,7134 | 37,40% | 84,47% | 0,4270 |

**Principais observações:**

- A Regressão Logística apresentou maior sensibilidade na identificação da classe 1 (HbA1c ≥ 5,7%).
- O Random Forest apresentou maior especificidade, porém menor sensibilidade.
- A inclusão da variável `hipertensao_informada` não produziu uma melhoria consistente em todas as métricas.
- A seleção de modelos deve considerar a validação cruzada realizada no treinamento, evitando decisões baseadas exclusivamente no conjunto de teste.

---

### 📁 Estrutura dos arquivos

A implementação está organizada em uma pasta independente para preservar os modelos anteriores do projeto.

```text
Projeto-PISI3/
│
├── classificacao_l5/
│   ├── preparacao_l5.py
│   ├── treinamento_l5.py
│   ├── avaliacao_l5.py
│   └── teste_l5.py
│
├── dataset/
│   └── healthsync_nhanes_2021_2023_core.csv
│
└── modelos/
    ├── classificacao_L5/
    │   ├── versao_A_regressao_logistica.joblib
    │   ├── versao_A_random_forest.joblib
    │   ├── versao_B_regressao_logistica.joblib
    │   ├── versao_B_random_forest.joblib
    │   └── metadados_l5.json
    │
    └── avaliacao_L5/
        ├── validacao_cruzada_l5.csv
        ├── validacao_cruzada_folds_l5.csv
        ├── metricas_teste_l5.csv
        ├── curvas_roc_l5.png
        ├── curvas_roc_pontos_l5.csv
        ├── importancia_variaveis_l5.csv
        ├── matrizes_confusao/
        └── importancia_variaveis/
```

---

### ▶️ Como executar

Execute os comandos abaixo a partir da raiz do projeto, com o ambiente Python configurado e as dependências instaladas.

**1. Preparar os dados:**

```bash
python -m classificacao_l5.preparacao_l5
```

**2. Treinar os modelos:**

```bash
python -m classificacao_l5.treinamento_l5
```

**3. Avaliar os modelos:**

```bash
python -m classificacao_l5.avaliacao_l5
```

**4. Executar os testes automatizados:**

```bash
python -m pytest classificacao_l5/teste_l5.py -v
```

---

### ✅ Testes automatizados

A implementação possui uma suíte de testes com `pytest`, responsável por verificar:

- Filtragem correta dos participantes;
- Construção do alvo binário;
- Exclusão de variáveis proibidas;
- Divisão estratificada e reproduzível;
- Ausência de sobreposição entre treino e teste;
- Organização do pré-processamento dentro dos pipelines;
- Validação cruzada utilizando apenas o treinamento;
- Carregamento dos quatro modelos;
- Consistência das métricas exportadas;
- Presença das matrizes de confusão, curvas ROC e importâncias.

**Resultado da execução:**

```text
=================== 22 passed in 16.24s ===================
```

Todos os 22 testes automatizados foram aprovados, sem falhas.

---

### ⚠️ Limitações metodológicas

Os resultados constituem evidência exploratória de capacidade preditiva moderada para a faixa de HbA1c.

Entre as limitações identificadas estão:

- Utilização de dados observacionais, sem demonstração de causalidade;
- Ausência de diabetes definida por informação autorrelatada;
- Desbalanceamento entre as classes;
- Ausência de validação externa independente;
- Ausência de avaliação clínica prospectiva;
- Necessidade de considerar o desenho amostral complexo do NHANES para inferências populacionais;
- Importâncias por permutação calculadas no conjunto de teste apenas para interpretação exploratória, sem reutilização para selecionar variáveis e reavaliar o mesmo teste.

> [!WARNING]
> Os modelos não são instrumentos clinicamente validados para diagnosticar diabetes, determinar risco individual ou orientar decisões médicas.

A possível integração ao aplicativo **HealthSync** permanece como uma etapa futura, condicionada a validações adicionais.

## 🖥️ Dashboard

O projeto possui um dashboard desenvolvido com Dash,
destinado à apresentação das análises e dos resultados
dos modelos de aprendizado de máquina.

A página **Atribuição de Subperfil** apresenta:

- Seleção dos três estratos glicêmicos;
- Resultados do Random Forest;
- Comparação com linhas de base disponíveis;
- Acurácia, precisão, recall e F1-score;
- Matrizes de confusão exportadas pela avaliação;
- Área de importância das variáveis, quando disponível;
- Observações metodológicas sobre as limitações dos modelos.

As métricas são carregadas a partir do arquivo:

`modelos/avaliacao_classificacao/metricas_classificacao.csv`

A página preserva a rota `/classificacao` por compatibilidade.

Os resultados apresentados são experimentais e não devem
ser interpretados como diagnósticos individuais.


## 🧱 Stack utilizada

<div align="center">

`Python` · `Pandas` · `NumPy` · `Scikit-Learn` · `Matplotlib` · `Seaborn` · `Plotly Dash` · `Jupyter Notebook` 

</div>

## 📊 Status do projeto

| Etapa | Status |
|---|:---:|
| Seleção do dataset | ✅ Concluído |
| Preparação e integração dos dados | ✅ Concluído |
| Análise Exploratória de Dados | ✅ Concluído |
| Definição dos targets | ✅ Concluído |
| Preparação para Machine Learning | ✅ Concluído |
| Clusterização (K-Means) | ✅ Concluído |
| Atribuição de subperfil (Random Forest) | ✅ Concluído |
| Avaliação com linhas de base | ✅ Concluído |
| Modelos de estimativa de risco | ⏳ Pendente |
| Dashboard | 🚧 Em desenvolvimento |
| Integração com o HealthSync | ⏳ Pendente |

