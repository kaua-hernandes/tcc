# 🛡️ Analytics Criminal SP: Centro Integrado de Inteligência Geoespacial e Predição de Crimes

Plataforma analítica e preditiva desenvolvida para processar, enriquecer e extrair inteligência tática a partir de mais de **834.000 registros criminais** do Estado de São Paulo.

- **Autor:** Kauã Hernandes
- **Curso:** Trabalho de Conclusão de Curso (TCC) / Ciência de Dados e Engenharia de Software
- **Branch Ativa:** `feat/TCC`
- **Ambiente:** Python 3.12 (`.venv`) | Docker PostGIS

---

## 🌟 Principais Recursos do Sistema

1. **🗺️ Mapa Espacial com Lazy Loading (< 1 ms):**
   - Carregamento ultraleve de coordenadas geográficas via índices espaciais, eliminando travamentos de renderização.
   - Camadas de mapa gratuitas e sem marcas d'água (`OpenStreetMap`).
2. **📈 Modelo Matemático Preditivo (Ridge + Decomposição Harmônica de Fourier):**
   - Captura tendências seculares e sazonalidades anuais de 12 meses via ondas de Fourier.
   - Regularização Ridge ($L_2$) para estabilidade numérica e prevenção de overfitting em pequenos históricos.
   - Cálculo estocástico de **Intervalos de Confiança a 95%** com propagação de incerteza $\text{SE}(h)$ e garantia de não-negatividade.
   - Métricas formais de validação: $R^2$, RMSE, MAE, MAPE e AIC.
3. **💬 Chat Lateral com Copiloto IA (Gemini 3.6 Flash + PostGIS Grounding):**
   - Assessor de inteligência policial integrado diretamente na barra lateral (`st.sidebar`), permitindo bater papo e consultar dados enquanto navega pelo mapa.
   - Consulta estatísticas reais de bairros, cidades e crimes em menos de 25 ms no PostgreSQL.
   - Mecanismo de fallback resiliente para cotas da API gratuita (HTTP 429).
4. **⚡ Engenharia de Banco de Dados de Alta Performance:**
   - PostgreSQL 15 com extensão espacial PostGIS 3.3.
   - Índices espaciais GiST (R-Tree) e buscas por raio geodésico em metros (`ST_DWithin`).
   - Índices compostos e funcionais `UPPER` para buscas textuais ultrarrápidas.
   - Visões Materializadas (`mv_kpis_municipios` e `mv_tendencias_mensais`) com resposta sub-milissegundo (< 0.1 ms).
   - Connection Pooling via SQLAlchemy (`pool_size=15`).
5. **📑 Central de Relatórios Oficiais:**
   - Geração e download de Briefing Tático Oficial direcionado ao Comando de Policiamento em PDF (`FPDF`).
   - Download da transcrição da conversa com a IA em `.txt`.

---

## 📚 Documentação Técnica (`docs/`)

A documentação detalhada do sistema está modularizada na pasta [`docs/`](docs/):

| Documento                                                                  | Assunto                                                                  |
| :------------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| [**`docs/pgsql.md`**](docs/pgsql.md)                                       | Estrutura de tabelas, índices GiST, B-Tree, Materialized Views e Pooling |
| [**`docs/cache.md`**](docs/cache.md)                                       | Estratégias de Cache no Streamlit e Lazy Loading espacial do mapa        |
| [**`docs/modelos.md`**](docs/modelos.md)                                   | Formulação matemática do preditor, equações Ridge e Intervalos de 95%    |
| [**`docs/copilot_ia.md`**](docs/copilot_ia.md)                             | Chat Lateral, Grounding no PostGIS, extração de entidades e fallback     |
| [**`docs/arquitetura.md`**](docs/arquitetura.md)                           | Diagrama de arquitetura, pipeline ETL (`clean_processor`) e testes       |
| [**`math.MD`**](math.MD)                                                   | Fundamentação teórica, álgebra linear e deduções completas               |
| [**`RT-TDS-2026-018_v0.4(1).docx`**](<RT-TDS-2026-018_v0.4(1).docx>)       | Relatório Técnico Oficial do TCC (UniSalesiano) 100% adaptado            |
| [**`ANALISE_PROJETO_E_NOVAS_FEATS.md`**](ANALISE_PROJETO_E_NOVAS_FEATS.md) | Relatório de arquitetura e catálogo das novas funcionalidades            |
| [**`test.MD`**](test.MD)                                                   | Relatório de Teste e Validação de Software (TVS II)                      |

---

## 📁 Estrutura de Pastas do Projeto

```text
├── config/                  # Configurações do projeto e paths
├── data/                    # Diretório para armazenamento dos dados
├── docs/                    # Documentação técnica modularizada
│   ├── arquitetura.md       # Arquitetura do sistema e pipeline ETL
│   ├── cache.md             # Estratégias de cache e lazy loading
│   ├── copilot_ia.md        # Chat lateral com Copiloto IA (Gemini + PostGIS)
│   ├── modelos.md           # Modelagem preditiva com Fourier e Ridge
│   └── pgsql.md             # Engenharia de banco PostgreSQL/PostGIS
├── logs/                    # Arquivos de log de execução
├── src/
│   ├── ingestion/           # Acesso ao PostgreSQL, queries e otimizações PostGIS
│   ├── models/              # Preditor estatístico (Ridge+Fourier) e Copiloto IA
│   ├── processing/          # Pipeline ETL, limpeza e engenharia de atributos
│   └── visualization/       # Dashboard interativo em Streamlit (app.py)
├── tests/                   # Suíte de testes unitários automatizados
├── docker-compose.yml       # Orquestração do container PostgreSQL / PostGIS
├── math.MD                  # Dedução matemática das equações do modelo preditivo
├── README.md                # Apresentação do projeto e instruções
├── requirements.txt         # Dependências do projeto Python
└── test.MD                  # Relatório de testes para disciplina de TVS II
```

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

- Python 3.12+
- Docker e Docker Compose instalados

### 2. Iniciar o Banco de Dados PostGIS

```bash
docker-compose up -d
```

### 3. Configurar o Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configurar as Variáveis de Ambiente (`.env`)

Certifique-se de que o arquivo `.env` contenha as credenciais do banco e a chave da API Gemini:

```env
DB_USER=postgres
DB_PASS=tcc_password_2026
DB_HOST=localhost
DB_PORT=5432
DB_NAME=criminal_db
GEMINI_API_KEY=sua_chave_gemini_aqui
```

### 5. Iniciar o Dashboard Streamlit

```bash
./venv/bin/streamlit run src/visualization/app.py
```

O painel estará disponível no seu navegador em `http://localhost:8501`.

---

## 🧪 Execução dos Testes Automatizados

A suíte conta com **26 testes unitários** automatizados cobrindo pipeline de limpeza, consultas PostGIS, modelo estatístico e Copiloto IA:

```bash
./venv/bin/python -m unittest discover tests -v
```

**Resultado:**

```text
Ran 26 tests in 0.134s
OK (100% de aprovação)
```
