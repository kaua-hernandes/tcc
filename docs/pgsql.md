# PostgreSQL 15 & PostGIS 3.3: Engenharia de Banco de Dados

Resumo técnico das estratégias de banco de dados implementadas sobre a base de **834.385 ocorrências criminais** da tabela `fatos_criminais`.

---

## 1. Estrutura da Tabela `fatos_criminais`

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `num_bo` | `VARCHAR` | Identificador único do boletim de ocorrência |
| `data_ocorrencia_bo` | `TIMESTAMP` | Data e hora do registro |
| `ano`, `mes`, `dia` | `INTEGER` | Atributos temporais discretizados |
| `dia_semana`, `hora` | `INTEGER` | Dia da semana (0=Segunda, 6=Domingo) e hora do crime |
| `rubrica` | `VARCHAR` | Classificação penal (ex: Roubo, Furto, Homicídio) |
| `nome_municipio` | `VARCHAR` | Cidade da ocorrência (ex: S.PAULO, CAMPINAS) |
| `bairro` | `VARCHAR` | Bairro georreferenciado |
| `latitude`, `longitude` | `FLOAT` | Coordenadas decimais WGS84 |
| `geom` | `GEOMETRY(Point, 4326)` | Geometria espacial nativa PostGIS |

---

## 2. Estratégias de Indexação de Alta Performance

### A. Índice Espacial GiST (Generalized Search Tree)
- **Definição:** `CREATE INDEX idx_fatos_geom_gist ON fatos_criminais USING GIST (geom);`
- **Índice Geodésico Funcional:** `CREATE INDEX idx_fatos_geog_gist ON fatos_criminais USING GIST (cast(geom as geography));`
- **Utilidade:** Permite buscas por raio geodésico em metros (`ST_DWithin`) sem converter graus para metros em tempo de execução.
- **Resultado:** Redução de tempo de consulta de **296 ms para 7.3 ms**.

### B. Índices Compostos B-Tree
- `idx_fatos_municipio_rubrica ON fatos_criminais (nome_municipio, rubrica)`
- `idx_fatos_municipio_ano_mes ON fatos_criminais (nome_municipio, ano, mes)`
- **Utilidade:** Acelera buscas filtradas do dashboard sem realizar scans sequenciais em 834k linhas.

### C. Índices Funcionais UPPER (Otimização para Chat IA)
- `idx_fatos_bairro_upper ON fatos_criminais (UPPER(bairro))`
- `idx_fatos_mun_bairro_upper ON fatos_criminais (UPPER(nome_municipio), UPPER(bairro))`
- **Utilidade:** Permite consultas case-insensitive instantâneas (< 20 ms) quando o usuário digita nomes de bairros e cidades no Chat IA.

---

## 3. Visões Materializadas (Materialized Views)

Agregações pesadas pré-computadas armazenadas fisicamente em disco:

1. **`mv_kpis_municipios`**: Total de crimes, diversidade de crimes e intervalo temporal por município.
   - Consulta instantânea dos cards de métricas em **0.06 ms** (ganho de 770x).
2. **`mv_tendencias_mensais`**: Agrupamento mensal pré-calculado por `(nome_municipio, ano, mes, rubrica)`.

---

## 4. Connection Pooling (SQLAlchemy)

Configuração otimizada para concorrência de usuários no Streamlit:
- `pool_size = 15`: 15 conexões persistentes abertas com o PostgreSQL.
- `max_overflow = 25`: Até 25 conexões extras sob pico de requisições.
- `pool_pre_ping = True`: Teste de liveness antes de reutilizar conexões.
- `pool_recycle = 1800`: Reciclagem periódica a cada 30 minutos.
