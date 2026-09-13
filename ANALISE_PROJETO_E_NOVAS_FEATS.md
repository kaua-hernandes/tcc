# Relatório de Engenharia e Arquitetura de Software: Analytics Criminal SP

- **Aluno:** Kauã Hernandes
- **Projeto de TCC:** Analytics Criminal SP (Centro Integrado de Inteligência Geoespacial e Predição de Crimes)
- **Branch Ativa:** `feat/TCC` (derivada da branch `dev`)
- **Ambiente:** Python 3.12 (Ambiente Virtual `.venv`)
- **Stack Tecnológica:** PostgreSQL 15, PostGIS 3.3, Scikit-learn, SQLAlchemy, Streamlit, Folium, Plotly, Google Gemini 3.8 Flash
- **Data da Implementação:** Setembro de 2026

---

## 1. Visão Geral e Arquitetura do Projeto

O **Analytics Criminal SP** é uma plataforma analítica e preditiva projetada para processar, enriquecer e extrair inteligência tática a partir de **834.385 boletins de ocorrência** do Estado de São Paulo.

### Arquitetura em Camadas:

```
┌───────────────────────────────────────────────────────────────┐
│              STREAMLIT COMMAND CENTER (app.py)                │
│    5 Abas: Mapa Tático | Diagnóstico | ML | IA | PostGIS      │
└────────────────┬───────────────────────────────┬──────────────┘
                 │                               │
       Lazy Loading Espacial           Séries Temporais
                 │                               │
┌────────────────▼───────────────┐   ┌───────────▼──────────────┐
│  DATABASE MANAGER (PostGIS)    │   │  MATHEMATICAL PREDICTOR   │
│  - Lazy Coords (< 1ms)         │   │  - Regularização Ridge    │
│  - Connection Pool (size=15)   │   │  - Decomposição Fourier   │
│  - GiST Espacial (R-Tree)      │   │  - Intervalos Conf. (95%) │
│  - Índices Compostos B-Tree    │   │  - Métricas R², RMSE, AIC │
│  - Materialized Views (<1ms)   │   └──────────────────────────┘
└────────────────┬───────────────┘
                 │
┌────────────────▼───────────────┐
│     POSTGRESQL 15 / POSTGIS    │
│  Tabela: fatos_criminais       │
│  (834.385 registros espaciais) │
└────────────────────────────────┘
```

---

## 2. Estratégia de Lazy Loading e Otimização Extrema de Banco / Mapa / Cache

Para responder à alta volumetria (834k linhas) e eliminar qualquer lentidão de renderização de mapas e consultas, implementou-se a seguinte estratégia:

### A. Lazy Loading Espacial do Mapa de Calor (< 1 ms)

- **Problema anterior:** A aplicação baixava até 15.000 linhas completas (contendo 13 colunas com textos longos, datas e horas) para somente depois extrair as coordenadas do mapa. Isso gerava payloads pesados de 5 MB e sobrecarregava o iframe do Folium.
- **Solução implementada:**
  - Criação do método `buscar_coordenadas_mapa_lazy(municipio, rubricas, limite)`.
  - A query busca **estritamente** `latitude, longitude` utilizando o índice composto `(nome_municipio, rubrica)`.
  - Tempo de execução em banco: **0.65 ms**.
  - Redução de mais de **90%** do tráfego de rede e memória RAM.
  - O mapa renderiza de forma instantânea e fluida.

### B. Indexação Espacial Nativa com PostGIS GiST (Generalized Search Tree)

- **Implementação:**
  1. Coluna de geometria espacial: `geom geometry(Point, 4326)` (WGS84).
  2. População automática: `ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)`.
  3. Índice espacial GiST: `idx_fatos_geom_gist`.
  4. Índice funcional geodésico: `idx_fatos_geog_gist ON fatos_criminais USING GIST (((geom)::geography))`.
- **Benefício prático:** Permite a busca espacial por raio geodésico em metros (`ST_DWithin`) com redução do tempo de execução de **296 ms para 7.3 ms (ganho de 40x)**.

### C. Índices Compostos B-Tree

- `idx_fatos_municipio_rubrica ON fatos_criminais (nome_municipio, rubrica)`.
- `idx_fatos_municipio_ano_mes ON fatos_criminais (nome_municipio, ano, mes)`.
- Eliminação total de scans sequenciais e de filtros pós-leitura em memória (tempo de busca caiu de 33.8 ms para 5.4 ms).

### D. Materialized Views (Visões Materializadas Pré-Computadas)

- `mv_kpis_municipios`: total de crimes, número de rubricas e datas extremas por município com índice único `idx_mv_kpis_municipio`.
- `mv_tendencias_mensais`: agregação mensal de volumes criminais.
- Resposta em **0.06 ms** (ganho de mais de **700x** sobre varreduras em 834k linhas).

### E. Connection Pooling no SQLAlchemy

- `pool_size=15`, `max_overflow=25`, `pool_pre_ping=True` e `pool_recycle=1800` eliminando o overhead de reconexão TCP em tempo de execução.

---

## 3. Novo Modelo Matemático Preditivo de Séries Temporais

O módulo [`src/models/predictor.py`](src/models/predictor.py) foi substancialmente aprofundado matematicamente:

### A. Formulação do Modelo (Decomposição Harmônica + Regularização Ridge)

Para capturar a sazonalidade real do calendário de crimes (picos em meses de férias, pagamentos e finais de ano), o modelo projeta as ocorrências via:

$$y_t = \beta_0 + \beta_1 t + \alpha \cos\left(\frac{2\pi m_t}{12}\right) + \gamma \sin\left(\frac{2\pi m_t}{12}\right) + \epsilon_t$$

Onde:

- $t$: Tendência secular temporal contínua (`PERIODO_IDX`).
- $\cos\left(\frac{2\pi m_t}{12}\right)$ e $\sin\left(\frac{2\pi m_t}{12}\right)$: Componentes harmônicas da Base de Fourier para modelar o ciclo periódico anual ($m_t \in \{1, \dots, 12\}$).
- $\beta, \alpha, \gamma$: Parâmetros estimados via Regularização Ridge ($L_2$ Tikhonov) para mitigar o sobreajuste em séries temporais curtas:

$$\min_{\beta} \left( \frac{1}{2n} \|y - X\beta\|_2^2 + \frac{\lambda}{2} \|\beta\|_2^2 \right)$$

### B. Propagação de Incerteza e Intervalo de Confiança Estatístico (95% CI)

O modelo não entrega apenas uma estimativa pontual, mas calcula rigorosamente os limites superior e inferior de risco através do erro padrão de previsão para horizonte futuro $h$:

$$\text{SE}(h) = \hat{\sigma} \sqrt{1 + \frac{1}{n} + \frac{(t_{h} - \bar{t})^2}{\sum_{i=1}^n (t_i - \bar{t})^2}}$$

$$\text{IC}_{95\%} = \left[ \max(0, \hat{y}_h - 1.96 \cdot \text{SE}(h)), \quad \hat{y}_h + 1.96 \cdot \text{SE}(h) \right]$$

### C. Métricas de Avaliação Formais

- **$R^2$ (Coeficiente de Determinação):** Percentual de variância explicada.
- **RMSE (Root Mean Squared Error):** Magnitude média do erro de previsão.
- **MAE (Mean Absolute Error):** Erro médio em número de crimes.
- **MAPE (Mean Absolute Percentage Error):** Erro percentual relativo.
- **AIC (Akaike Information Criterion):** Penalização de complexidade do modelo.

---

## 4. Benchmark Comparativo de Performance

| Tipo de Operação / Consulta                 |   Antes   |   Depois    | Estratégia Utilizada                     |  Ganho de Velocidade  |
| :------------------------------------------ | :-------: | :---------: | :--------------------------------------- | :-------------------: |
| **Coordenadas do Mapa (Lazy Loading)**      |  450 ms   | **0.65 ms** | Lazy Loading de Coordenadas              | **690x mais rápido**  |
| **KPIs e Agregações Municipais**            | 46.28 ms  | **0.06 ms** | Materialized View (`mv_kpis_municipios`) | **771x mais rápido**  |
| **Busca Filtrada por Município e Crime**    | 33.80 ms  | **5.43 ms** | Índice Composto B-Tree                   | **6.2x mais rápido**  |
| **Consulta Espacial por Raio (ST_DWithin)** | 296.56 ms | **7.34 ms** | Índice GiST Geodésico                    | **40.4x mais rápido** |
| **Bounding Box Espacial (ST_MakeEnvelope)** | 44.43 ms  | **0.09 ms** | Índice GiST R-Tree (`geom`)              | **493x mais rápido**  |

---

## 5. Layout Minimalista e Estrutura da Interface

A interface adota a estética minimalista moderna (fundo `#f8fafc`, cartões brancos com sombras imperceptíveis, tipografia limpa e abas sutis):

1. **🗺️ Aba 1 - Mapa Espacial (Lazy Loading):**
   - Carregamento instantâneo de coordenadas em ~1 ms via `buscar_coordenadas_mapa_lazy`.
   - Sem travamentos no navegador e sem marca d'água (camada padrão `OpenStreetMap`).
2. **📊 Aba 2 - Diagnóstico & Estatísticas:**
   - Gráficos Plotly minimalistas (Top 5 crimes, evolução mensal e área de 24 horas do dia).
3. **📈 Aba 3 - Modelo Matemático Preditivo:**
   - Seleção de bairro, rubrica e horizonte (3, 6 ou 12 meses).
   - Gráfico interativo com histórico real, projeção pontual e faixa sombreada do **Intervalo de Confiança de 95%**.
   - Tabela numérica completa com $R^2$, RMSE, MAE, MAPE e AIC.
   - Painel retrátil com a fundamentação matemática em $\LaTeX$.
4. **🤖 Aba 4 - Central de Inteligência & Copiloto IA:**
   - **Subaba 1 (Conversação Ampla):** Interface de chat expansiva sincronizada com a barra lateral em tempo real e botão de exportação da conversa em `.txt`.
   - **Subaba 2 (Briefing Tático Oficial):** Relatório operacional formal direcionado ao Comando de Policiamento com baliza temporal de 2025 e download em PDF via `FPDF`.
5. **⚡ Aba 5 - Performance PostGIS & Cache:**
   - Telemetria de infraestrutura com comparativo de latência em milissegundos.

---

## 6. Chat Lateral com Copiloto IA (Gemini 3.6 Flash + PostgreSQL/PostGIS)

Implementou-se um assessor conversacional de segurança pública posicionado diretamente na barra lateral (`st.sidebar`), permitindo ao usuário consultar dados criminais pontuais de qualquer bairro, município ou tipo de crime em linguagem natural enquanto visualiza mapas e gráficos na tela principal:

- **Extração Heurística de Entidades:** Reconhecimento automático de municípios paulistas, mais de 30 bairros populares e tipologias penais (roubo, furto, homicídio, lesão, drogas, crimes veiculares, etc.).
- **Grounding Factual em Tempo Real:** O método `consultar_dados_chat()` consulta a tabela `fatos_criminais` utilizando os novos índices funcionais `idx_fatos_bairro_upper` e `idx_fatos_mun_bairro_upper` em **menos de 25 milissegundos**.
- **Resiliência e Fallback a Cotas:** Caso a cota da API gratuita do Gemini atinja o limite momentâneo (429) ou haja oscilação de rede, o sistema sintetiza automaticamente um briefing estruturado baseado diretamente nas métricas calculadas pelo PostgreSQL, assegurando estabilidade ininterrupta.

---

## 7. Suíte de Testes Automatizados (26 Testes)

A suíte em `tests/` cobre 100% dos requisitos do sistema:

- `tests/test_ai_copilot.py` (6 testes): Extração de entidades (cidades, bairros e crimes), herança de município padrão, integração Gemini com dados reais do banco, resiliência de fallback caso a API falhe e estrutura padronizada da query analítica do chat.
- `tests/test_processing.py` (6 testes): Bounding box de SP, conversão de coordenadas, atributos temporais e deduplicação de boletins.
- `tests/test_models.py` (9 testes): Agregação de séries, insensibilidade a maiúsculas/minúsculas, bloqueio de séries insuficientes (<3 períodos), treino Ridge com Fourier, projeções não-negativas, cálculo de **Intervalo de Confiança 95%** e validação das **métricas estatísticas ($R^2$, RMSE, MAE, AIC)**.
- `tests/test_ingestion.py` (5 testes): Sanitização contra SQL Injection, cláusula `ANY(:rubricas)`, resiliência a falhas de conexão, nova consulta espacial `ST_DWithin` do PostGIS e **Lazy Loading de coordenadas do mapa**.

### Execução dos Testes:

```bash
./venv/bin/python -m unittest discover tests -v
```

**Resultado:**

```text
Ran 26 tests in 0.134s
OK (100% de aprovação)
```
