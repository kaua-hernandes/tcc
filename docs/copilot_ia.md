# Copiloto IA Criminal (Google Gemini + PostGIS)

Resumo do módulo de inteligência artificial conversacional e tática policial implementado em `src/models/ai_copilot.py`.

---

## 1. Visão Geral

O **Copiloto IA Criminal** é um assistente analítico desenhado para policiais, pesquisadores e gestores de segurança pública. Ele combina o poder generativo do **Google Gemini (3.6 Flash)** com **Grounding Factual** direto na base PostgreSQL de 834 mil registros.

---

## 2. Fluxo de Execução da Conversa

```
Usuário envia mensagem no Chat Lateral
                  │
                  ▼
   Extração de Entidades (Cidade, Bairro, Crime)
                  │
                  ▼
   Consulta Analítica no PostGIS (< 25 ms)
   - Total de ocorrências
   - Representatividade % local
   - Top 5 crimes do bairro
   - Horário de pico e dia mais crítico
                  │
                  ▼
   Construção de Prompt Enriquecido com Dados Reais
                  │
                  ▼
   Chamada à API Google Gemini (gemini-3.6-flash)
        │                             │
    (Sucesso)                   (Erro / Cota 429)
        │                             │
        ▼                             ▼
Resposta Analítica e       Fallback Analítico Estruturado
Recomendação Tática       com Dados Reais do PostGIS
```

---

## 3. Principais Componentes

### A. Extração Heurística de Entidades
- Mapeia variações de cidades: "São Paulo", "SP", "Sampa", "Capital" $\to$ `S.PAULO`; "Campinas" $\to$ `CAMPINAS`.
- Reconhece mais de 30 bairros populares paulistas (ex: Pinheiros, Sé, Mooca, República, Cambuí, Centro).
- Mapeia crimes para termos penais (Roubo, Furto, Homicídio, Tráfico de Drogas, Delitos Veiculares, etc.).

### B. Consulta Analítica no PostGIS (`consultar_dados_chat`)
- Executa em **menos de 25 milissegundos** graças aos índices funcionais `idx_fatos_bairro_upper` e `idx_fatos_mun_bairro_upper`.
- Calcula automaticamente o horário de pico mais crítico e o dia da semana com maior incidência.

### C. Resiliência a Cotas da API (Fallback Automático)
- Se a conta gratuita do Gemini atingir o limite de requisições por minuto (HTTP 429), o sistema não quebra.
- Ele gera automaticamente um relatório estruturado diretamente com os números do PostGIS, mantendo o usuário atendido com dados 100% verídicos.

### D. Interface Dupla no Streamlit
- **Chat Lateral (`st.sidebar`):** Permite conversar com a IA sem sair da tela de mapas e gráficos.
- **Aba Central 4 ("Central de Inteligência"):** Exibe o chat em modo amplo com download do histórico em `.txt` e geração de **Briefing Tático Oficial em PDF** direcionado ao Comando de Policiamento.
