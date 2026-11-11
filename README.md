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
