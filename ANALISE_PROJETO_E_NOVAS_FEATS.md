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
        Consultas Otimizadas           Séries Históricas
                 │                               │
┌────────────────▼───────────────┐   ┌───────────▼──────────────┐
│  DATABASE MANAGER (PostGIS)    │   │  SCIKIT-LEARN PREDICTOR   │
│  - Connection Pool (size=15)   │   │  - Regressão Linear       │
│  - GiST Espacial (R-Tree)      │   │  - Projeção Temporal      │
│  - Índices Compostos B-Tree    │   │  - Validação Não-Negativa │
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

## 2. Novos Conceitos de Banco de Dados Implementados (PostgreSQL / PostGIS)

Para otimizar o sistema e transformar o repositório em uma base geoespacial de alta performance, foram introduzidos os seguintes conceitos avançados:

### A. Indexação Espacial Nativa com PostGIS GiST (Generalized Search Tree)
- **Problema anterior:** As coordenadas eram armazenadas apenas como números decimais (`double precision`), exigindo filtros numéricos planos e inviabilizando consultas geodésicas precisas em metros.
- **Solução implementada:**
  1. Criação da coluna de geometria espacial: `geom geometry(Point, 4326)` seguindo a projeção esferoidal WGS84.
  2. População automática via função PostGIS: `ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)`.
  3. Criação de índice espacial **GiST (R-Tree)**: `idx_fatos_geom_gist`.
  4. Criação de índice funcional geodésico: `idx_fatos_geog_gist ON fatos_criminais USING GIST (((geom)::geography))`.
- **Benefício prático:** Permite a nova funcionalidade `buscar_por_raio_espacial()`, calculando ocorrências num raio em metros (`ST_DWithin`) com redução do tempo de execução de **296 ms para 7.3 ms (ganho de 40x)**.

### B. Índices Compostos B-Tree para Consultas Analíticas
- **Problema anterior:** A query principal do painel (`WHERE nome_municipio = :mun AND rubrica = ANY(:rubricas)`) utilizava índices de coluna única, forçando o PostgreSQL a fazer leituras sequenciais parciais e descartar milhares de linhas na memória (`Rows Removed by Filter: 3510`).
- **Solução implementada:**
  - Criação do índice composto: `CREATE INDEX idx_fatos_municipio_rubrica ON fatos_criminais (nome_municipio, rubrica);`
  - Criação do índice temporal composto: `CREATE INDEX idx_fatos_municipio_ano_mes ON fatos_criminais (nome_municipio, ano, mes);`
- **Benefício prático:** Eliminação total de linhas descartadas por filtro em memória. A busca analítica para municípios como Campinas caiu de **33.8 ms para 5.4 ms (ganho de 6.2x)**.

### C. Materialized Views (Visões Materializadas Pré-Computadas)
- **Problema anterior:** O cálculo de KPIs municipais (total de crimes por cidade, amplitude de datas e diversidade penal) exigia varrer centenas de milhares de linhas a cada renderização.
- **Solução implementada:**
  - `mv_kpis_municipios`: Visão materializada contendo os agregados pré-calculados de cada município com índice único `idx_mv_kpis_municipio`.
  - `mv_tendencias_mensais`: Agregação de volume criminal por mês e rubrica.
- **Benefício prático:** O tempo de resposta para KPIs municipais caiu de **46.3 ms para 0.06 ms (ganho de mais de 700x)**.

### D. Connection Pooling no SQLAlchemy
- **Implementação:**
  ```python
  self.engine = create_engine(
      url,
      pool_size=15,
      max_overflow=25,
      pool_timeout=30,
      pool_recycle=1800,
      pool_pre_ping=True
  )
  ```
- **Benefício prático:** Manutenção de conexões ativas reutilizáveis, eliminando o overhead de handshake TCP e autenticação a cada requisição de múltiplos analistas no Streamlit.

### E. Manutenção de Catálogo com VACUUM ANALYZE
- Executou-se `VACUUM ANALYZE fatos_criminais` e nas visões materializadas, reorganizando páginas de disco e gerando histogramas estatísticos atualizados no `pg_statistic` para que o otimizador de consultas (Cost-Based Optimizer) tome as decisões de execução mais rápidas.

---

## 3. Benchmark Comparativo de Performance

| Tipo de Operação / Consulta | Antes da Otimização | Após a Otimização | Estratégia Utilizada | Ganho de Velocidade |
| :--- | :---: | :---: | :--- | :---: |
| **Busca Filtrada por Município e Crime** | 33.80 ms | **5.43 ms** | Índice Composto B-Tree | **6.2x mais rápido** |
| **KPIs e Agregações Municipais** | 46.28 ms | **0.06 ms** | Materialized View (`mv_kpis_municipios`) | **771x mais rápido** |
| **Consulta Espacial por Raio (ST_DWithin)** | 296.56 ms | **7.34 ms** | Índice GiST Geodésico | **40.4x mais rápido** |
| **Bounding Box Espacial (ST_MakeEnvelope)** | 44.43 ms | **0.09 ms** | Índice GiST R-Tree (`geom`) | **493x mais rápido** |

---

## 4. Novas Features e Modernização Visual da Interface (Streamlit)

O arquivo `src/visualization/app.py` foi reestruturado com identidade visual de **Centro de Comando e Controle (CICC)**:

### 🎨 Design e Estilo Visual
- **Paleta Tática Dark:** Fundo `#0b0f19` (Deep Slate) com cartões em Glassmorphism, bordas translúcidas e acentos luminosos (`#38bdf8` e `#3b82f6`).
- **Header de Telemetria Dinâmico:** Badges em tempo real exibindo status do GiST, status das Materialized Views, latência da última query (em ms) e volumetria global do banco.
- **Cards com Indicadores:** Município, Volume Filtrado, Horário de Maior Incidência e Bairro com Maior Mancha Criminal.

### 📑 Organização em 5 Abas Estratégicas
1. **🗺️ Aba 1 - Mapa Tático Espacial:**
   - Mapa de calor em Folium com gradiente cromático (Índigo a Vermelho Puro).
   - Seletor de amostragem dinâmica (500 a 5.000 pontos) e troca de camada de satélite/rua.
2. **📊 Aba 2 - Diagnóstico & Estatísticas:**
   - Gráfico de barras horizontal com Top 5 tipologias penais em Plotly.
   - Gráfico de linha da evolução temporal mensal de 2025.
   - Gráfico de área da distribuição de crimes pelas 24 horas do dia.
3. **📈 Aba 3 - Projeção Preditiva (Scikit-Learn):**
   - Treinamento e ajuste da Regressão Linear sobre a série temporal histórica do bairro selecionado.
   - Gráfico comparativo: Histórico Real vs. Projeção Preditiva para os próximos trimestres.
   - Indicador automático de taxa de crescimento/queda esperada.
4. **🤖 Aba 4 - Copiloto Tático IA:**
   - Integração com Gemini 3.8 Flash para gerar diagnósticos estruturados de segurança pública.
   - Botão para exportação do relatório tático em formato **PDF** em tempo de execução via FPDF.
5. **⚡ Aba 5 - Central de Performance PostGIS:**
   - Painel de telemetria mostrando tamanho da tabela (MB), tamanho dos índices e contadores do banco.
   - Gráfico comparativo de latência de queries.
   - Relação dos 7 índices ativos no banco.

---

## 5. Suíte de Testes Automatizados (18 Testes)

A suíte em `tests/` cobre todas as camadas do sistema:
- `tests/test_processing.py` (6 testes): Bounding box do Estado de SP, conversão de coordenadas texto com vírgula, datas, horas e deduplicação de boletins.
- `tests/test_models.py` (7 testes): Agregação de séries temporais, case-insensitivity, bloqueio de séries insuficientes (<3 períodos), treino de regressão e garantia de predições não-negativas.
- `tests/test_ingestion.py` (5 testes): Sanitização de parâmetros contra SQL Injection, cláusula `ANY(:rubricas)`, resiliência a falhas de conexão, nova consulta espacial `ST_DWithin` do PostGIS e fallbacks da Materialized View.

### Comando para Execução dos Testes:
```bash
./venv/bin/python -m unittest discover tests -v
```

**Resultado:**
```text
Ran 18 tests in 0.060s
OK
```

---

## 6. Como Executar a Aplicação Localmente

1. **Ativar o Ambiente Virtual:**
   ```bash
   source venv/bin/activate
   ```
2. **Garantir que o Banco PostgreSQL/PostGIS está Ativo:**
   ```bash
   docker-compose up -d
   ```
3. **Executar a Aplicação Streamlit:**
   ```bash
   streamlit run src/visualization/app.py
   ```
