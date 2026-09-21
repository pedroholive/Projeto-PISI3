<div align="center">

<img src="CAMINHO_DA_LOGO" alt="HealthSync" width="130" />

# 🩺 HealthSync — Acompanhamento Cardiometabólico

**Dos dados de saúde à análise: exploração de indicadores glicêmicos, pressão arterial, perfil cardiometabólico e desenvolvimento de modelos de Machine Learning.**

`Python` · `Pandas` · `Matplotlib` · `Seaborn` 

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

3. **Como esses dados podem apoiar o HealthSync?** — preparação de modelos de Machine Learning voltados à estimativa de risco relacionada a diabetes e hipertensão e futura integração dessas análises ao sistema.

Atualmente, o projeto possui a etapa de **preparação dos dados e Análise Exploratória de Dados (EDA)** desenvolvida. As etapas de **Machine Learning e dashboard estão em desenvolvimento**.

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

## 🤖 Machine Learning

> 🚧 **Etapa em desenvolvimento**

A etapa de Machine Learning será desenvolvida a partir dos resultados obtidos durante a análise exploratória.

Inicialmente, serão investigados dois problemas separados:

### 🩸 Diabetes

Desenvolvimento de um modelo relacionado à estimativa de risco de diabetes a partir de características clínicas, demográficas e comportamentais disponíveis.

### ❤️ Hipertensão

Desenvolvimento de um modelo relacionado à estimativa de risco de hipertensão considerando características relevantes disponíveis na base.

### Cuidados metodológicos

Antes do treinamento serão definidos:

- as variáveis-alvo de cada problema;
- as características utilizadas como preditoras;
- o tratamento de valores ausentes e extremos;
- as estratégias de transformação e normalização;
- a divisão entre treino e teste;
- as métricas utilizadas na avaliação dos modelos.

Um dos principais cuidados será evitar **data leakage**.

Caso uma medida clínica seja utilizada diretamente para construir uma variável-alvo, essa mesma informação não será utilizada de forma circular como variável preditora do alvo.

Os algoritmos e modelos finais ainda não foram definidos e serão selecionados a partir de experimentação e avaliação.

## 🖥️ Dashboard

> 🚧 **Em desenvolvimento**

O projeto prevê o desenvolvimento de um dashboard utilizando **Dash**, destinado à visualização dos principais indicadores e resultados das análises.

A estrutura e as funcionalidades serão documentadas nesta seção conforme a implementação avançar.

## 🧱 Stack utilizada

<div align="center">

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Jupyter Notebook` 

</div>

## 📊 Status do projeto

| Etapa | Status |
|---|:---:|
| Seleção do dataset | ✅ Concluído |
| Preparação e integração dos dados | ✅ Concluído |
| Análise Exploratória de Dados | ✅ Concluído |
| Definição dos targets | 🔄 Em desenvolvimento |
| Preparação para Machine Learning | ⏳ Próxima etapa |
| Treinamento dos modelos | ⏳ Pendente |
| Avaliação dos modelos | ⏳ Pendente |
| Dashboard | 🚧 Em desenvolvimento |
| Integração com o HealthSync | ⏳ Pendente |
