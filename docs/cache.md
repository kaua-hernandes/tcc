# Estratégias de Cache e Lazy Loading

Resumo das técnicas aplicadas para garantir que o dashboard Streamlit carregue de forma instantânea sobre uma base de 834k registros.

---

## 1. Lazy Loading Espacial do Mapa (< 1 ms)

- **O Problema:** Anteriormente, o sistema baixava até 15.000 linhas completas (13 colunas de texto, datas e horas) apenas para plotar os pontos no mapa de calor. Isso gerava um tráfego pesado de 5 MB a cada clique, travando o navegador.
- **A Solução:** Separação estrita em dois carregamentos independentes:
  1. `carregar_coordenadas_mapa_lazy()`: Busca **apenas** `latitude` e `longitude`. Executa em **0.65 ms** no PostgreSQL e gera um payload ultraleve de coordenadas.
  2. `carregar_dados_filtrados_db()`: Carrega dados tabulares de forma desacoplada para alimentar as tabelas e gráficos.
- **Impacto:** Redução de mais de **90%** na transferência de dados e renderização do mapa em tempo real.

---

## 2. Gerenciamento de Cache no Streamlit

### A. Cache de Recursos Globais (`@st.cache_resource`)
- Utilizado na função `get_db_manager()`.
- **Objetivo:** Garante que o `DatabaseManager` e o pool de 15 conexões com o PostgreSQL sejam criados uma única vez como Singleton, sobrevivendo a recarregamentos de página (evita estouro de conexões no banco).

### B. Cache de Dados com Expiração (`@st.cache_data`)
- `carregar_opcoes_filtros()` (`ttl=3600`): Cache de 1 hora para a lista de municípios e crimes, consultados prioritariamente da Materialized View.
- `carregar_coordenadas_mapa_lazy()` (`ttl=1800`): Cache de 30 minutos das coordenadas dos pontos com base nos filtros selecionados.
- `carregar_dados_filtrados_db()` (`ttl=1800`): Cache de 30 minutos para os dados tabulares analíticos.
- `chamar_gemini_com_cache()` (`ttl=3600`): Cache de relatórios de inteligência, economizando chamadas à API do Gemini para os mesmos filtros.

---

## 3. Resumo de Benefícios

- Elimina recálculos desnecessários entre interações do usuário.
- Otimiza o consumo de memória RAM tanto no servidor quanto no navegador.
- Respostas visuais imediatas ao alternar entre abas do painel.
