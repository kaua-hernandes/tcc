# Fundamentação Matemática e Modelagem Estatística: Analytics Criminal SP

- **Aluno:** Kauã Hernandes
- **Projeto de TCC:** Analytics Criminal SP (Inteligência Geoespacial e Modelagem Preditiva de Crimes)
- **Módulo:** `src/models/predictor.py` (`CriminalPredictor`)
- **Branch:** `feat/TCC`
- **Data:** Setembro de 2026

---

## 1. Introdução e Justificativa Teórica

Modelar a criminalidade através de uma regressão linear simples univariada ($y = ax + b$) é uma abordagem ingênua que desconsidera dois fatores fundamentais da dinâmica criminal no mundo real:

1. **Sazonalidade Cíclica (Efeito Calendário):** O volume de ocorrências criminais (como roubos, furtos e lesões) não varia de forma estritamente linear ao longo dos meses; ele sofre oscilações periódicas induzidas por sazonalidades de calendário (meses de férias escolares, pagamentos de décimo terceiro salário, festividades e datas comemorativas).
2. **Incerteza Amostral e Variância Residual:** Nenhuma previsão pontual em segurança pública é determinística. Para a tomada de decisões táticas e alocação de viaturas do policiamento preventivo, é imprescindível estimar **Intervalos de Confiança (Confidence Intervals - CI)** probabilísticos que delimitem o melhor e o pior cenário de risco (limites inferior e superior).

Para superar essas limitações, foi desenvolvido um **modelo matemático multivariado de séries temporais baseado em Decomposição Harmônica de Fourier com Regularização Ridge ($L_2$) e Propagação de Incerteza**.

---

## 2. Estrutura Matemática do Modelo

A série temporal agregada de ocorrências $y_t$ para um determinado bairro e tipologia criminal no instante temporal $t$ é expressa pela seguinte formulação:

$$y_t = \beta_0 + \beta_1 t + \sum_{k=1}^{K} \left[ \alpha_k \cos\left(\frac{2\pi k \cdot m_t}{T}\right) + \gamma_k \sin\left(\frac{2\pi k \cdot m_t}{T}\right) \right] + \epsilon_t$$

Onde:
- **$y_t \in \mathbb{R}_{\ge 0}$:** Volume total de crimes observados no período $t$.
- **$t \in \{0, 1, 2, \dots, n-1\}$:** Índice cronológico contínuo (`PERIODO_IDX`), representando a **Tendência Secular** (Secular Trend) da criminalidade no bairro.
- **$\beta_0 \in \mathbb{R}$:** Nível médio basal da criminalidade (intercepto da reta).
- **$\beta_1 \in \mathbb{R}$:** Coeficiente angular de tendência. Se $\beta_1 > 0$, indica tendência temporal de **aumento sistemático** do crime; se $\beta_1 < 0$, indica **queda sustentada**.
- **$m_t \in \{1, 2, \dots, 12\}$:** Mês correspondente da ocorrência (Janeiro = 1, Dezembro = 12).
- **$T = 12$:** Período fundamental do ciclo harmônico (ano civil de 12 meses).
- **$K = 1$:** Ordem da decomposição harmônica de Fourier para frequências anuais.
- **$\alpha_k, \gamma_k \in \mathbb{R}$:** Coeficientes de amplitude e fase harmônica das ondas trigonométricas de Fourier.
- **$\epsilon_t \sim \mathcal{N}(0, \sigma^2)$:** Ruído branco gaussiano (resíduos com média zero e variância $\sigma^2$).

### Vetor de Atributos (Matriz de Projeto $X$):
Para cada período temporal observado $i$, o vetor de entrada $\mathbf{x}_i \in \mathbb{R}^3$ é definido como:

$$\mathbf{x}_i = \begin{bmatrix} t_i & \sin\left(\frac{2\pi m_i}{12}\right) & \cos\left(\frac{2\pi m_i}{12}\right) \end{bmatrix}$$

E o vetor completo de parâmetros a ser estimado é:

$$\boldsymbol{\beta} = \begin{bmatrix} \beta_1 \\ \gamma_1 \\ \alpha_1 \end{bmatrix}$$

---

## 3. Estimação de Parâmetros via Regularização Ridge ($L_2$ Tikhonov)

Séries temporais a nível de bairros frequentemente apresentam tamanhos amostrais reduzidos ($n < 24$ meses). Nesses cenários, o método clássico de Mínimos Quadrados Ordinários (OLS) tende a sofrer de **alta variância (overfitting)** e instabilidade por multicolinearidade.

Para mitigar esse problema e garantir estabilidade numérica, adota-se a **Regularização Ridge (Penalidade $L_2$)**, cuja função de custo penalizada $\mathcal{L}(\boldsymbol{\beta})$ é dada por:

$$\mathcal{L}(\boldsymbol{\beta}) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - \beta_0 - \mathbf{x}_i \boldsymbol{\beta} \right)^2 + \frac{\lambda}{2} \|\boldsymbol{\beta}\|_2^2$$

Onde:
- **$\lambda = 0.5$:** Hiperparâmetro de regularização que penaliza magnitudes excessivas nos coeficientes.
- **$\|\boldsymbol{\beta}\|_2^2 = \sum_{j} \beta_j^2$:** Norma euclidiana ao quadrado do vetor de coeficientes.

A solução analítica da equação normal regularizada é obtida fechando o gradiente em zero ($\nabla_{\boldsymbol{\beta}} \mathcal{L} = 0$):

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \left( \mathbf{X}^T \mathbf{X} + \lambda \mathbf{I} \right)^{-1} \mathbf{X}^T \mathbf{y}$$

Como $\left( \mathbf{X}^T \mathbf{X} + \lambda \mathbf{I} \right)$ é estritamente positiva definida para qualquer $\lambda > 0$, a matriz é **sempre inversível**, eliminando singularidades mesmo com poucos dados históricos.

---

## 4. Estimativa de Variância Residual

Após o ajuste dos parâmetros pelo estimador Ridge, os valores previstos para o conjunto histórico são $\hat{y}_i = \hat{\beta}_0 + \mathbf{x}_i \hat{\boldsymbol{\beta}}$.

A variância residual amostral não-viesada $\hat{\sigma}^2$ é calculada corrigindo os graus de liberdade pelo número de parâmetros $p = 3$:

$$\hat{\sigma}^2 = \frac{1}{n - p} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2$$

O desvio padrão dos resíduos é dado por:

$$\hat{\sigma} = \sqrt{\hat{\sigma}^2}$$

---

## 5. Propagação de Incerteza e Intervalos de Confiança (95% CI)

Ao realizar uma projeção para um horizonte futuro $h \in \{1, 2, \dots, H\}$ à frente do último registro ($t_{future} = t_{last} + h$), a previsão pontual é:

$$\hat{y}_{future} = \max\left( 0, \; \hat{\beta}_0 + \hat{\beta}_1 t_{future} + \hat{\alpha} \cos\left(\frac{2\pi m_{future}}{12}\right) + \hat{\gamma} \sin\left(\frac{2\pi m_{future}}{12}\right) \right)$$

### Erro Padrão da Previsão $\text{SE}(h)$:
A incerteza de uma projeção futura não é constante; **ela se expande à medida que projetamos períodos mais distantes no tempo**. Pela teoria da amostragem em regressão linear, o erro padrão da previsão pontual para o período $t_{future}$ é dado pela equação de propagação de erro:

$$\text{SE}(h) = \hat{\sigma} \sqrt{1 + \frac{1}{n} + \frac{\left( t_{future} - \bar{t} \right)^2}{\sum_{i=1}^{n} \left( t_i - \bar{t} \right)^2}}$$

Onde:
- **$\bar{t} = \frac{1}{n} \sum_{i=1}^{n} t_i$:** Ponto médio cronológico da série de treino.
- **$\frac{1}{n}$:** Incerteza associada à estimação do intercepto médio.
- **$\frac{\left( t_{future} - \bar{t} \right)^2}{\sum \left( t_i - \bar{t} \right)^2}$:** Penalização pelo afastamento temporal em relação ao centro da massa de dados.
- O termo **$+1$** sob o radical reflete a variabilidade intrínseca de uma nova observação pontual futura.

### Intervalo de Confiança Bilateral a 95%:
Sob a premissa assintótica de normalidade dos resíduos, para um nível de confiança $\alpha = 0.05$, o valor crítico da distribuição normal padrão é $Z_{1 - \alpha/2} = 1.96$.

Os limites superior e inferior do intervalo de confiança são:

$$\text{IC}_{\text{Inferior}} (95\%) = \max\left(0, \; \hat{y}_{future} - 1.96 \cdot \text{SE}(h)\right)$$

$$\text{IC}_{\text{Superior}} (95\%) = \hat{y}_{future} + 1.96 \cdot \text{SE}(h)$$

A função $\max(0, \cdot)$ assegura a consistência física e penal de que **não existem quantidades negativas de crimes**.

---

## 6. Métricas Formais de Validação Estatística

Para quantificar matematicamente a fidelidade do modelo, o módulo calcula e expõe cinco métricas:

### 1. Coeficiente de Determinação ($R^2$):
Mede a proporção da variância total das ocorrências que é explicada pelo modelo:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

### 2. Raiz do Erro Quadrático Médio (RMSE - Root Mean Squared Error):
Penaliza erros de grande magnitude de forma quadrática:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### 3. Erro Médio Absoluto (MAE - Mean Absolute Error):
Fornece a discrepância média em unidades absolutas de boletins de ocorrência:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 4. Erro Percentual Médio Absoluto (MAPE - Mean Absolute Percentage Error):
Mede o desvio percentual médio relativo ao volume real:

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### 5. Critério de Informação de Akaike (AIC - Akaike Information Criterion):
Avalia a qualidade relativa do modelo estatístico penalizando o número de parâmetros $p$ para evitar complexidade desnecessária:

$$\text{AIC} = n \cdot \ln\left( \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \right) + 2p$$

---

## 7. Comparativo: Regressão Simples vs. Modelo Matemático Desenvolvido

| Característica | Regressão Linear Simples ($y = ax+b$) | Modelo Proposto (Ridge + Fourier + IC) |
| :--- | :---: | :---: |
| **Componente de Tendência Secular** | Sim | Sim ($\beta_1 t$) |
| **Modelagem de Sazonalidade Anual** | Não | **Sim (Base Harmônica de Fourier)** |
| **Estabilidade contra Overfitting** | Baixa | **Alta (Penalidade Ridge $L_2$)** |
| **Estimativa de Risco e Incerteza** | Apenas pontual | **Sim (Intervalo de Confiança 95% $\text{SE}(h)$)** |
| **Garantia de Valores Não-Negativos** | Não (pode prever crimes negativos) | **Sim ($\max(0, \hat{y})$)** |
| **Métricas Formais Expostas** | Apenas $R^2$ básico | **$R^2$, RMSE, MAE, MAPE, AIC, $\hat{\sigma}$** |

---

## 8. Conclusão para a Banca Examinadora

A formulação matemática implementada no `CriminalPredictor` fornece robustez científica ao TCC:
1. Atende aos rigores da estatística matemática aplicada a séries temporais.
2. Permite aos gestores de segurança pública visualizar não apenas uma previsão única, mas a **faixa estatística de dispersão esperada** a 95% de probabilidade.
3. Demonstra domínio de álgebra linear, análise harmônica (Fourier) e otimização convexa regularizada no contexto da Ciência de Dados.
