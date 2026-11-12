# Relatório de Preparação e Qualidade dos Dados — L5

**Projeto:** HealthSync — Projeto-PISI3  
**Responsável:** Integrante 1 — Preparação e Tratamento dos Dados  
**Dataset:** NHANES 2021–2023

## 1. Objetivo

Verificar a qualidade dos dados utilizados no modelo de classificação binária de HbA1c e documentar os procedimentos de seleção, tratamento e divisão dos participantes.

A classificação considera duas classes:

- Classe 0: HbA1c abaixo de 5,7%.
- Classe 1: HbA1c igual ou superior a 5,7%.

O modelo busca investigar a possibilidade de identificar HbA1c elevada por meio de variáveis clínicas e antropométricas, sem utilizar o resultado da HbA1c como variável preditora.

## 2. Seleção dos participantes

A base original contém 5.283 participantes.

Após selecionar os indivíduos com `diabetes_informado == "Nao"` e verificar a disponibilidade da variável-alvo, permaneceram 4.360 participantes.

A distribuição das classes foi:

| Classe | Participantes | Percentual |
|---|---:|---:|
| Classe 0 | 3.089 | 70,85% |
| Classe 1 | 1.271 | 29,15% |

Foi identificado um desbalanceamento entre as classes, aspecto que deverá ser considerado durante o treinamento e a avaliação dos modelos.

## 3. Variáveis utilizadas

**Versão A — Essencial:**

- Idade
- IMC
- Pressão sistólica
- Pressão diastólica
- Colesterol total
- Sexo

**Versão B — Ampliada:**

Contém todas as variáveis da versão A, acrescentando `hipertensao_informada`.

## 4. Análise de valores ausentes

Não foram identificados valores ausentes nas variáveis da versão A.

Na versão B, a variável `hipertensao_informada` apresentou dois valores ausentes, correspondentes a aproximadamente 0,046% da população elegível.

O pipeline atual utiliza imputação pela categoria mais frequente para essa variável. Os valores de preenchimento são aprendidos nos dados de treinamento.

## 5. Investigação de valores extremos

O método do intervalo interquartil (IQR) identificou as seguintes quantidades de observações extremas na população elegível:

| Variável | Valores extremos identificados |
|---|---:|
| Idade | 0 |
| IMC | 120 |
| Pressão sistólica | 119 |
| Pressão diastólica | 65 |
| Colesterol total | 59 |

Essas contagens são por variável e podem representar participantes repetidos.

Na análise específica do conjunto de treinamento, não foram encontrados valores iguais ou inferiores a zero nas cinco variáveis numéricas examinadas.

Os valores extremos foram preservados, pois sua identificação estatística não comprova erro de medição. Eventuais tratamentos adicionais deverão ser justificados e definidos utilizando apenas os dados de treinamento.

## 6. Consistência da pressão arterial

Foram analisados os 3.488 participantes do treinamento.

Não foram encontrados registros com pressão sistólica menor ou igual à pressão diastólica.

Essa verificação indica consistência lógica para a relação examinada, sem comprovar a precisão individual das medidas.

## 7. Pré-processamento

A revisão do pipeline existente identificou os seguintes procedimentos:

- `SimpleImputer(strategy="median")` para variáveis numéricas.
- `RobustScaler` para escalonamento das variáveis numéricas.
- `SimpleImputer(strategy="most_frequent")` para variáveis categóricas.
- `OneHotEncoder` para codificação de sexo e hipertensão informada.
- `ColumnTransformer` para organizar as transformações.

Esses procedimentos estão integrados ao pipeline de treinamento, permitindo que os parâmetros das transformações sejam aprendidos somente durante o ajuste do modelo.

## 8. Divisão dos dados

Foi utilizada uma divisão estratificada de 80% para treinamento e 20% para teste:

| Conjunto | Participantes |
|---|---:|
| Treinamento | 3.488 |
| Teste | 872 |

Não houve sobreposição de índices entre os conjuntos.

A validação cruzada do treinamento utiliza cinco folds estratificados, com as transformações ajustadas dentro de cada fold.

## 9. Conclusão

A auditoria inicial identificou uma base com baixa quantidade de valores ausentes nas variáveis selecionadas, sem IDs duplicados na população analisada e sem inconsistências na relação entre as pressões sistólica e diastólica no treinamento.

Os valores extremos foram mantidos por não haver evidência suficiente para considerá-los inválidos.

A estrutura de preparação e pré-processamento existente é adequada como ponto de partida para os experimentos de classificação, balanceamento e comparação de algoritmos.

Como continuidade, recomenda-se validar a execução dos testes automatizados e documentar eventuais modificações realizadas durante os novos experimentos.