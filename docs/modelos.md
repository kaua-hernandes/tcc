# Modelagem Preditiva e Estatística Matemática

Resumo da formulação matemática implementada no módulo `src/models/predictor.py` (`CriminalPredictor`).

---

## 1. Estrutura Matemática do Modelo

Diferente de regressões lineares simples ($y = ax + b$), o modelo implementado captura simultaneamente a **tendência temporal contínua** e a **sazonalidade anual de 12 meses** via decomposição harmônica de Fourier:

$$y_t = \beta_0 + \beta_1 t + \alpha \cos\left(\frac{2\pi m_t}{12}\right) + \gamma \sin\left(\frac{2\pi m_t}{12}\right) + \epsilon_t$$

- **$t$:** Índice cronológico contínuo (`PERIODO_IDX`), representando a tendência secular (alta ou queda ao longo dos anos).
- **$\sin, \cos$:** Ondas harmônicas trigonométricas que modelam a sazonalidade cíclica dos meses do ano ($m_t \in \{1, \dots, 12\}$).
- **$\beta_0, \beta_1, \alpha, \gamma$:** Coeficientes estimados via **Regularização Ridge ($L_2$)**.

---

## 2. Regularização Ridge ($L_2$ de Tikhonov)

Para séries a nível de bairros com poucos meses de histórico ($n < 24$), a regressão comum sofre de overfitting. O estimador Ridge penaliza pesos excessivos:

$$\min_{\boldsymbol{\beta}} \left( \frac{1}{2n} \sum_{i=1}^n \left( y_i - \beta_0 - \mathbf{x}_i \boldsymbol{\beta} \right)^2 + \frac{\lambda}{2} \|\boldsymbol{\beta}\|_2^2 \right)$$

Com $\lambda = 0.5$, a matriz de covariância $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})$ torna-se estritamente positiva definida e inversível, garantindo estabilidade numérica absoluta.

---

## 3. Propagação de Incerteza e Intervalo de Confiança a 95%

A previsão em segurança pública não é determinística. O erro padrão da projeção futura $\text{SE}(h)$ se expande conforme avançamos no tempo:

$$\text{SE}(h) = \hat{\sigma} \sqrt{1 + \frac{1}{n} + \frac{(t_{future} - \bar{t})^2}{\sum_{i=1}^n (t_i - \bar{t})^2}}$$

Intervalo de Confiança a 95% ($Z = 1.96$):

$$\text{IC}_{\text{Inferior}} = \max\left(0, \; \hat{y} - 1.96 \cdot \text{SE}(h)\right)$$
$$\text{IC}_{\text{Superior}} = \hat{y} + 1.96 \cdot \text{SE}(h)$$

*(A função $\max(0, \cdot)$ assegura que o modelo nunca preveja volumes negativos de crimes).*

---

## 4. Métricas de Validação Exportadas

O preditor calcula e expõe 5 métricas formais para a banca examinadora:
- **$R^2$:** Coeficiente de Determinação (proporção da variância explicada).
- **RMSE:** Raiz do Erro Quadrático Médio.
- **MAE:** Erro Médio Absoluto de boletins.
- **MAPE:** Erro Percentual Médio Absoluto.
- **AIC:** Critério de Informação de Akaike (qualidade relativa penalizando parâmetros).

*(Para deduções completas e fundamentação teórica detalhada, consulte [`math.MD`](../math.MD)).*
