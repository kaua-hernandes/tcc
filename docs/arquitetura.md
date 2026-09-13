# Arquitetura do Sistema e Pipeline de Dados

Visão geral da engenharia de software, pipeline ETL e testes automatizados do projeto **Analytics Criminal SP**.

---

## 1. Diagrama de Arquitetura

```
┌──────────────────────────────────────────────────────────────┐
│             STREAMLIT COMMAND CENTER (app.py)                │
│  - Barra Lateral: Filtros Recolhíveis + Chat com Copiloto IA │
│  - 5 Abas: Mapa Espacial | Diagnóstico | Predição | IA | PostGIS
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
     Lazy Coords (< 1ms)              Predição Ridge + Fourier
               │                               │
┌──────────────▼───────────────┐   ┌───────────▼───────────────┐
│   DATABASE MANAGER (PostGIS) │   │  CRIMINAL PREDICTOR (ML)  │
│   - Connection Pool (15 conn)│   │  - Sazonalidade Fourier   │
│   - Índices GiST (Espacial)  │   │  - Intervalos 95% Conf.   │
│   - Materialized Views       │   │  - Métricas R², RMSE, AIC │
└──────────────┬───────────────┘   └───────────────────────────┘
               │                               ▲
               │                               │
┌──────────────▼───────────────┐               │
│   POSTGRESQL 15 / POSTGIS    │───────────────┘
│   Tabela: fatos_criminais    │
│   (834.385 boletins reais)   │
└──────────────▲───────────────┘
               │
┌──────────────┴───────────────┐
│   PIPELINE ETL (Processing)  │
│   - Bounding Box Estado SP   │
│   - Deduplicação por NUM_BO  │
│   - Engenharia de Atributos  │
└──────────────────────────────┘
```

---

## 2. Pipeline de Limpeza e Transformação (ETL)

O módulo `src/processing/clean_processor.py` executa o tratamento dos dados brutos:

1. **Delimitação Geográfica Rigorosa (Bounding Box de SP):**
   - Latitude: $[-25.35, -19.78]$
   - Longitude: $[-53.11, -44.16]$
   - Descarta anomalias, valores fora do estado de SP e coordenadas nulas.
2. **Conversão de Formatos:**
   - Trata coordenadas textuais com vírgula (ex: `'-23,5505'` $\to$ `-23.5505`).
3. **Engenharia de Atributos Temporais:**
   - Extração automática de `ANO`, `MES`, `DIA`, `DIA_SEMANA` ($0$ a $6$) e `HORA`.
4. **Deduplicação Determinística:**
   - Elimina registros duplicados baseados no número único do boletim (`NUM_BO`).

---

## 3. Suíte de Testes Automatizados (26 Testes)

Testes unitários construídos sob a metodologia TDD (Test-Driven Development) para validação da disciplina de TVS II:

- `tests/test_processing.py` (6 testes): Validação do pipeline ETL, filtros geográficos e deduplicação.
- `tests/test_ingestion.py` (5 testes): Prevenção de SQL Injection, busca por raio `ST_DWithin` e Lazy Loading.
- `tests/test_models.py` (9 testes): Treinamento Ridge com Fourier, não-negatividade e cálculo do Intervalo de Confiança a 95%.
- `tests/test_ai_copilot.py` (6 testes): Extração de entidades, fallback resiliente e integração PostGIS + Gemini.

Para rodar todos os testes:

```bash
./venv/bin/python -m unittest discover tests -v
```

_(Tempo médio de execução: ~0.13s com 100% de aprovação)._
