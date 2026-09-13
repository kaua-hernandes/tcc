# Relatório de Teste e Validação de Software: TDD em Pipeline de Dados e Modelagem Preditiva com IA

- **Aluno:** Kauã Hernandes
- **Disciplina:** Teste e Validação de Software II
- **Atividade:** Desenvolvimento Guiado por Testes (TDD) com Inteligência Artificial para o Projeto de TCC
- **Data de Execução e Entrega:** 10 de setembro de 2026
- **IA Utilizada:** Antigravity / Gemini 3.8 Flash (Google DeepMind)
- **Projeto de TCC:** Analytics Criminal SP (Sistema Inteligente de Análise Geoespacial e Modelagem Preditiva de Crimes em SP)

---

## 1. Contexto e Motivação do Projeto

O Trabalho de Conclusão de Curso (TCC) de **Kauã Hernandes** é centrado em Engenharia e Ciência de Dados para Segurança Pública, composto por:
1. **Banco de Dados Geoespacial (PostGIS/PostgreSQL):** Ingestão em massa e consultas analíticas de ocorrências policiais do Estado de São Paulo.
2. **Pipeline de Tratamento e Engenharia de Atributos:** Sanitização de coordenadas espaciais, remoção de registros corrompidos/duplicados e extração de variáveis temporais.
3. **Modelagem Preditiva de Criminalidade (Scikit-Learn):** Projeção de tendências de crimes por bairro e tipologia penal via Regressão Linear em séries temporais.
4. **Painel Interativo de Tomada de Decisão:** Interface tática com mapas de calor e copiloto analítico em Streamlit.

Como o foco do sistema é a ingestão, higienização e previsão analítica sobre grandes volumes de dados, o projeto não possui tela de login de usuário comum. Em consonância com a instrução do docente (*"caso seu projeto não tenha tela de login converse com um dos professores para mais informações"*), a metodologia de **TDD (Test-Driven Development)** foi direcionada ao **coração do sistema: a validação de integridade dos dados geoespaciais, a camada de consultas SQL/PostGIS e os modelos preditivos de Machine Learning**.

---

## 2. Identificação da Inteligência Artificial Utilizada

- **Nome da IA:** Antigravity
- **Modelo Fundacional:** **Gemini 3.8 Flash**
- **Desenvolvedor:** **Google DeepMind**

### Papel da IA no Processo:
A IA atuou como parceira de desenvolvimento (*Pair Programming*) com o aluno **Kauã Hernandes**, sendo responsável por:
1. **Mapeamento de Casos de Borda e Requisitos:** Identificação de anomalias críticas em dados policiais (coordenadas fora do Estado de SP, coordenadas em formato texto com vírgula, duplicidade de boletins e datas inválidas).
2. **Aplicação do Ciclo TDD (Red-Green-Refactor):** Criação dos casos de teste antes da validação do código, identificação de falhas e bugs lógicos nos modelos de predição e posterior correção.
3. **Isolamento de Testes Unitários:** Uso de mocks (`unittest.mock`) para validar as consultas SQL e PostGIS sem dependência externa do servidor de banco estar ativo no momento da execução dos testes.

---

## 3. Metodologia TDD (Test-Driven Development)

A construção das suítes de testes seguiu estritamente o ciclo clássico de TDD:

```
      🔴 RED (Falha Inicial)
 Escrever o teste para o requisito
            │
            ▼
      🟢 GREEN (Aprovação)
  Implementar / ajustar o código
            │
            ▼
      🔵 REFACTOR (Melhoria)
  Otimizar, tipar e tratar avisos
```

### 🔴 1ª Etapa: FASE RED (O Teste Aponta o Problema)
Durante a criação dos testes para o preditor de Machine Learning em `tests/test_models.py`, formulou-se o teste `test_prever_sem_treinamento_lanca_excecao` e a validação do índice temporal futuro.

Ao rodar a suíte inicial:
```bash
./venv/bin/python -m unittest tests/test_models.py
```
**Resultado Obtido na Fase RED (Evidência Real):**
```text
======================================================================
ERROR: test_prever_sem_treinamento_lanca_excecao (tests.test_models.TestCriminalPredictor)
----------------------------------------------------------------------
sklearn.exceptions.NotFittedError: This LinearRegression instance is not fitted yet.
```
Além disso, identificou-se que o método `prever_proximos_periodos` tentava ler `PERIODO_IDX` diretamente de `self.df`, gerando projeções com índice zerado e emitindo alertas de nomes de atributos do scikit-learn (`UserWarning: X does not have valid feature names`).

---

### 🟢 2ª Etapa: FASE GREEN (Fazer o Teste Passar)
Refatorou-se `src/models/predictor.py`:
- Criação do estado `self.treinado` e validação explícita levantando `RuntimeError` caso o método preditivo seja acionado antes do ajuste (`fit`).
- Salvamento do `self.ultimo_idx` referente ao último mês histórico real da série temporal agregada.
- Utilização de `pd.DataFrame({'PERIODO_IDX': ...})` para manter a consistência dos nomes de colunas esperadas pelo Scikit-learn.

Ao reexecutar:
```bash
./venv/bin/python -m unittest tests/test_models.py -v
```
**Resultado Obtido na Fase GREEN:**
```text
Ran 7 tests in 0.046s
OK
```

---

### 🔵 3ª Etapa: FASE REFACTOR (Isolamento e Qualidade)
- **Desacoplamento do Banco:** Em `tests/test_ingestion.py`, utilizou-se `unittest.mock.patch` sobre o `create_engine` e `pandas.read_sql`, garantindo que os testes de construção de query parametrizada e resiliência a falhas executem em milissegundos sem depender do PostgreSQL estar online.
- **Higienização Geográfica:** Em `tests/test_processing.py`, foram fixadas as regras do Bounding Box do Estado de São Paulo (Latitude: $-26^\circ$ a $-19^\circ$ / Longitude: $-53^\circ$ a $-44^\circ$).

---

## 4. Arquivos Utilizados no Projeto

| Arquivo | Camada | Descrição e Finalidade |
| :--- | :--- | :--- |
| [`tests/test_processing.py`](tests/test_processing.py) | Testes de Dados | Validação do pipeline de limpeza, Bounding Box de SP, conversão de vírgula e deduplicação. |
| [`tests/test_models.py`](tests/test_models.py) | Testes de ML | Validação do modelo preditivo de regressão linear, agregação temporal e projeção futura. |
| [`tests/test_ingestion.py`](tests/test_ingestion.py) | Testes de Query | Validação das consultas parametrizadas do PostGIS (lazy loading) e resiliência de conexão. |
| [`src/processing/clean_processor.py`](src/processing/clean_processor.py) | Produção | Classe `DataProcessor` responsável pela sanitização dos dados policiais. |
| [`src/models/predictor.py`](src/models/predictor.py) | Produção | Classe `CriminalPredictor` responsável pelas séries temporais e predições. |
| [`src/ingestion/database.py`](src/ingestion/database.py) | Produção | Classe `DatabaseManager` responsável pelas queries analíticas no PostgreSQL/PostGIS. |
| [`test.MD`](test.MD) | Documentação | Relatório comprobatório de conformidade acadêmica e evidências de execução. |

---

## 5. Detalhamento dos 16 Casos de Teste Executados

### A. Módulo de Processamento e Dados (`tests/test_processing.py`)
1. **`test_inicializacao_dataframe_vazio_lanca_excecao` (CT01):** Assegura que o processador rejeita dataframes nulos ou vazios (`ValueError`).
2. **`test_tratamento_coordenadas_virgula_e_string` (CT02):** Converte coordenadas em formato texto com vírgula (ex: `'-23,5505'`) para números `float`.
3. **`test_filtro_delimitacao_geografica_estado_sp` (CT03):** Descarta automaticamente coordenadas com erros de digitação ou localizadas fora dos limites territoriais do Estado de São Paulo.
4. **`test_engenharia_de_atributos_temporais` (CT04):** Valida a decomposição da data em novas variáveis analíticas: `ANO`, `MES`, `DIA`, `DIA_SEMANA` e `HORA`.
5. **`test_deduplicacao_por_numero_boletim` (CT05):** Garante a unicidade dos dados removendo boletins duplicados com mesmo `NUM_BO`.
6. **`test_pipeline_limpeza_completo` (CT06):** Executa o pipeline de ponta a ponta e verifica se o número final de registros íntegros é condizente.

### B. Módulo de Modelagem Preditiva (`tests/test_models.py`)
7. **`test_inicializacao_dataframe_vazio_lanca_excecao` (CT07):** Rejeita instanciar o modelo preditivo sem massa de dados (`ValueError`).
8. **`test_preparar_serie_temporal_case_insensitive_e_agregacao` (CT08):** Valida que a agregação agrupa crimes do mesmo mês e ignora diferenças de caixa alta/baixa nos nomes de bairros e rubricas.
9. **`test_preparar_serie_temporal_sem_correspondencia` (CT09):** Retorna `DataFrame` vazio sem erro ao consultar combinações inexistentes.
10. **`test_treinar_modelo_com_dados_insuficientes` (CT10):** Impede o treinamento do modelo caso a série histórica tenha menos de 3 períodos, evitando sobreajuste e modelos degenerados.
11. **`test_treinar_modelo_regressao_com_sucesso` (CT11):** Valida o ajuste da regressão linear, confirmando a presença de coeficientes angulares e lineares (`coef_`, `intercept_`).
12. **`test_prever_proximos_periodos_valores_nao_negativos` (CT12):** Garante que previsões de volume de crimes futuros nunca assumam valores negativos (`>= 0`).
13. **`test_prever_sem_treinamento_lanca_excecao` (CT13):** Previne erros em tempo de execução ao barrar chamadas de predição antes do método de treino (`RuntimeError`).

### C. Módulo de Consultas PostGIS (`tests/test_ingestion.py`)
14. **`test_query_lazy_loading_com_municipio` (CT14):** Valida se a query SQL do PostGIS aplica filtro parametrizado (`:municipio`) para proteção contra injeção de SQL.
15. **`test_query_lazy_loading_com_multiplas_rubricas` (CT15):** Valida a cláusula `rubrica = ANY(:rubricas)` com lista de crimes selecionados.
16. **`test_resiliencia_falha_de_banco_retorna_dataframe_vazio` (CT16):** Assegura que falhas de conexão com o banco não derrubam a aplicação, retornando `DataFrame` vazio de forma graciosa.

---

## 6. Evidência Completa de Execução dos Testes Automatizados

### Comando Executado no Terminal:
```bash
./venv/bin/python -m unittest discover tests -v
```

### Log de Saída da Execução (16 Testes Aprovados):
```text
test_query_lazy_loading_com_multiplas_rubricas (test_ingestion.TestDatabaseManager.test_query_lazy_loading_com_multiplas_rubricas)
CT02: Deve filtrar múltiplas rubricas via ANY(:rubricas) de forma parametrizada. ... ok
test_query_lazy_loading_com_municipio (test_ingestion.TestDatabaseManager.test_query_lazy_loading_com_municipio)
CT01: Deve aplicar filtro parametrizado (:municipio) para prevenir injeção de SQL. ... ok
test_resiliencia_falha_de_banco_retorna_dataframe_vazio (test_ingestion.TestDatabaseManager.test_resiliencia_falha_de_banco_retorna_dataframe_vazio)
CT03: Deve tratar falha de conexão com PostgreSQL retornando DataFrame vazio (sem quebrar a aplicação). ... ok
test_inicializacao_dataframe_vazio_lanca_excecao (test_models.TestCriminalPredictor.test_inicializacao_dataframe_vazio_lanca_excecao)
CT01: Deve lançar ValueError ao tentar inicializar o preditor sem dados. ... ok
test_preparar_serie_temporal_case_insensitive_e_agregacao (test_models.TestCriminalPredictor.test_preparar_serie_temporal_case_insensitive_e_agregacao)
CT02: A agregação da série deve somar ocorrências do mesmo mês e ser case-insensitive. ... ok
test_preparar_serie_temporal_sem_correspondencia (test_models.TestCriminalPredictor.test_preparar_serie_temporal_sem_correspondencia)
CT03: Deve retornar DataFrame vazio caso o bairro ou crime não existam. ... ok
test_prever_proximos_periodos_valores_nao_negativos (test_models.TestCriminalPredictor.test_prever_proximos_periodos_valores_nao_negativos)
CT06: As projeções de crimes para períodos futuros nunca devem ser negativas. ... ok
test_prever_sem_treinamento_lanca_excecao (test_models.TestCriminalPredictor.test_prever_sem_treinamento_lanca_excecao)
CT07: Tentar prever períodos futuros antes de treinar o modelo deve levantar RuntimeError. ... ok
test_treinar_modelo_com_dados_insuficientes (test_models.TestCriminalPredictor.test_treinar_modelo_com_dados_insuficientes)
CT04: Não deve treinar regressão com menos de 3 períodos temporais (retorna None). ... ok
test_treinar_modelo_regressao_com_sucesso (test_models.TestCriminalPredictor.test_treinar_modelo_regressao_com_sucesso)
CT05: Deve treinar com sucesso o modelo de regressão linear para série com >= 3 períodos. ... ok
test_deduplicacao_por_numero_boletim (test_processing.TestDataProcessor.test_deduplicacao_por_numero_boletim)
CT05: Deve eliminar ocorrências duplicadas baseadas no identificador único NUM_BO. ... ok
test_engenharia_de_atributos_temporais (test_processing.TestDataProcessor.test_engenharia_de_atributos_temporais)
CT04: Deve extrair corretamente novos atributos numéricos (ANO, MES, DIA, DIA_SEMANA, HORA). ... ok
test_filtro_delimitacao_geografica_estado_sp (test_processing.TestDataProcessor.test_filtro_delimitacao_geografica_estado_sp)
CT03: Deve descartar coordenadas que estejam fora dos limites geográficos do Estado de SP. ... ok
test_inicializacao_dataframe_vazio_lanca_excecao (test_processing.TestDataProcessor.test_inicializacao_dataframe_vazio_lanca_excecao)
CT01: Deve lançar ValueError se inicializado com DataFrame vazio ou nulo. ... ok
test_pipeline_limpeza_completo (test_processing.TestDataProcessor.test_pipeline_limpeza_completo)
CT06: Validação de ponta a ponta do pipeline completo de limpeza e transformação. ... ok
test_tratamento_coordenadas_virgula_e_string (test_processing.TestDataProcessor.test_tratamento_coordenadas_virgula_e_string)
CT02: Deve converter strings com vírgula ('-23,5505') para float numérico válido. ... ok

----------------------------------------------------------------------
Ran 16 tests in 0.069s

OK
```

---

## 7. Conclusão

A aplicação do **TDD com Inteligência Artificial** no projeto de **Kauã Hernandes** demonstrou que o Desenvolvimento Guiado por Testes é extremamente valioso não apenas em telas de interface clássicas (como login), mas sobretudo em **sistemas orientados a dados (Data-Driven Systems) e Machine Learning**:

1. **Robustez do Pipeline:** Garantia matemática de que coordenadas geográficas corrompidas ou fora de São Paulo jamais chegam ao banco PostGIS ou aos modelos.
2. **Confiabilidade Preditiva:** Eliminação de bugs de indexação temporal nas previsões e garantia de restrições de negócio (como previsões não-negativas).
3. **Segurança e Estabilidade:** Proteção contra SQL Injection e prevenção de indisponibilidade da aplicação Streamlit através de testes de resiliência.
