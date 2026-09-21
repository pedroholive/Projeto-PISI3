# 🩺 HealthSync

O **HealthSync** é um sistema desenvolvido com o objetivo de apoiar o acompanhamento de indicadores relacionados à saúde cardiometabólica, com foco no registro e visualização da **glicemia** e da **pressão arterial**.

Além do acompanhamento dos indicadores, o projeto utiliza técnicas de **Análise de Dados** e **Machine Learning** para investigar padrões e características relacionadas ao perfil cardiometabólico dos usuários.

> O HealthSync possui caráter acadêmico e de apoio ao acompanhamento de informações de saúde, não substituindo avaliação, diagnóstico ou orientação realizada por profissionais de saúde.

---

## 🎯 Objetivo

O projeto busca integrar acompanhamento de indicadores de saúde e análise de dados em uma única aplicação.

Entre os principais objetivos estão:

- permitir o registro de glicemia e pressão arterial;
- acompanhar a evolução dessas medidas ao longo do tempo;
- apresentar informações de forma visual e organizada;
- considerar características clínicas, demográficas e comportamentais do usuário;
- utilizar Machine Learning para apoiar a identificação de padrões e estimativas relacionadas ao risco de diabetes e hipertensão.

---

## 📱 Funcionalidades previstas

O HealthSync está sendo desenvolvido para oferecer funcionalidades como:

- registro de glicemia;
- registro de pressão arterial sistólica e diastólica;
- histórico das medições do usuário;
- visualização da evolução dos indicadores;
- cadastro de informações relacionadas ao perfil e estilo de vida;
- análise do perfil cardiometabólico;
- estimativa de risco relacionada a diabetes;
- estimativa de risco relacionada a hipertensão.

As funcionalidades relacionadas a Machine Learning ainda estão em desenvolvimento e dependem da definição, treinamento e validação dos modelos.

---

## 📊 Dados utilizados

Para a etapa de análise de dados e desenvolvimento dos modelos, o projeto utiliza dados do **National Health and Nutrition Examination Survey (NHANES), ciclo agosto de 2021 a agosto de 2023**.

Foram selecionadas informações relacionadas a:

- idade;
- sexo;
- índice de massa corporal (IMC);
- pressão arterial sistólica;
- pressão arterial diastólica;
- hemoglobina glicada (HbA1c);
- glicose em jejum;
- colesterol total;
- atividade física moderada;
- atividade física vigorosa;
- comportamento sedentário;
- histórico informado de diabetes;
- histórico informado de hipertensão;
- histórico informado de colesterol elevado.

### Bases analíticas

Após a preparação dos dados, foram organizados dois conjuntos principais:

**CORE**

Base principal utilizada nas análises exploratórias, contendo **5.283 participantes adultos** com as principais medidas clínicas selecionadas.

**FASTING**

Subamostra contendo **2.955 participantes** com informação disponível de glicose em jejum.

A separação permite utilizar a glicose em jejum quando necessário sem reduzir desnecessariamente a amostra das demais análises.

---

## 🔎 Análise Exploratória de Dados

Antes da construção dos modelos de Machine Learning, foi realizada uma Análise Exploratória de Dados (EDA) para compreender a estrutura, qualidade e comportamento das variáveis.

A análise foi organizada nos seguintes eixos:

### Perfil da amostra

Análise de características como:

- idade;
- sexo;
- IMC.

### Perfil glicêmico

Análise de:

- distribuição da HbA1c;
- valores extremos;
- glicose em jejum;
- histórico informado de diabetes;
- relação entre HbA1c e diabetes informado.

### Pressão arterial

Análise de:

- pressão sistólica;
- pressão diastólica;
- número de leituras válidas;
- relação entre sistólica e diastólica;
- histórico informado de hipertensão;
- relação entre idade e pressão sistólica.

### Perfil lipídico

Análise de:

- colesterol total;
- valores extremos;
- histórico informado de colesterol elevado;
- relação entre colesterol medido e histórico informado.

### Atividade física e sedentarismo

Análise de:

- atividade física moderada;
- atividade física vigorosa;
- tempo sedentário;
- valores extremos;
- relação entre atividade física e comportamento sedentário.

### Relações cardiometabólicas

Foram investigadas associações entre diferentes indicadores, incluindo:

- IMC × HbA1c;
- IMC × pressão arterial;
- HbA1c × pressão arterial;
- colesterol × outros indicadores cardiometabólicos.

As relações observadas na EDA são interpretadas como **associações descritivas** e não como relações de causa e efeito.

---

## 🩺 Histórico informado × medidas objetivas

Uma etapa específica da análise compara informações relatadas pelos participantes com medidas obtidas durante os exames do NHANES:

- diabetes informado × HbA1c;
- hipertensão informada × pressão arterial;
- colesterol alto informado × colesterol total.

Essa análise é importante porque histórico de saúde e medida atual representam informações diferentes e não devem ser considerados automaticamente equivalentes.

Essa distinção também será considerada na definição das variáveis utilizadas pelos modelos do HealthSync.

---

## 🤖 Machine Learning

A etapa de Machine Learning está em desenvolvimento.

A proposta é estudar inicialmente dois problemas separados:

- estimativa relacionada ao risco de diabetes;
- estimativa relacionada ao risco de hipertensão.

Antes do treinamento dos modelos serão definidas as variáveis-alvo e as características preditoras utilizadas em cada problema.

Um cuidado importante será evitar **data leakage**. Caso uma medida clínica seja utilizada diretamente para construir uma variável-alvo, essa mesma informação não deverá ser utilizada de forma circular como preditora do alvo.

Os modelos serão avaliados antes de sua integração ao HealthSync.

---

## 🔄 NHANES × HealthSync

É importante diferenciar o papel dos dados utilizados no desenvolvimento do projeto.

O **NHANES** fornece dados transversais de diferentes participantes e é utilizado neste projeto para análise exploratória e desenvolvimento dos modelos de Machine Learning.

O **HealthSync**, por outro lado, será responsável pelo registro de informações dos próprios usuários. Medidas como glicemia e pressão arterial poderão ser armazenadas ao longo do tempo, possibilitando o acompanhamento individual da evolução desses indicadores.

Portanto:

NHANES → análise de dados e desenvolvimento dos modelos.

HealthSync → registro, visualização e acompanhamento dos dados do usuário.

---

## 🛠️ Tecnologias

O projeto utiliza ou prevê a utilização das seguintes tecnologias:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Git
- GitHub

---

## 📂 Estrutura do projeto

A organização do repositório é dividida entre etapas de preparação dos dados, análises exploratórias, desenvolvimento dos modelos e aplicação.

```text
Projeto-PISI3/
│
├── analises/
│   └── notebooks de análise exploratória
│
├── dados/
│   └── bases utilizadas pelo projeto
│
└── README.md
