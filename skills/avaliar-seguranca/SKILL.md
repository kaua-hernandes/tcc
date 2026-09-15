---
name: avaliar-seguranca-sistema
description: Avalia e audita aspectos de segurança, privacidade e boas práticas do sistema de software (TCC Analytics Criminal SP), gerando um relatório completo em formato Markdown (.md).
author: Kaua Hernandes Succi
version: 1.0.0
---

# Skill: Avaliação de Segurança de Software para Sistema de TCC

Esta Skill orienta o Agente Inteligente a realizar uma auditoria rigorosa de segurança no código-fonte, arquitetura de dados e infraestrutura do projeto **Analytics Criminal SP**, gerando ao final um relatório técnico estruturado em arquivo `.md`.

---

## 1. Objetivo da Skill

Capacitar o agente inteligente a inspecionar o repositório do projeto para:
1. Identificar potenciais vulnerabilidades de segurança (com base nas diretrizes OWASP Top 10 e CWE).
2. Avaliar a proteção e anonimização de dados pessoais em conformidade com a LGPD (Lei nº 13.709/2018).
3. Verificar a segurança no acesso a banco de dados relacional e geoespacial (PostgreSQL/PostGIS).
4. Analisar a gestão de credenciais, chaves de API e segredos de ambiente.
5. Inspecionar a segurança na integração de Inteligência Artificial Generativa (Google Gemini) e interfaces web (Streamlit).
6. Consolidar todas as constatações em um **Relatório de Avaliação de Segurança (.md)** detalhado com matriz de risco e recomendações práticas.

---

## 2. Escopo de Inspeção no Repositório

O agente deve analisar obrigatoriamente os seguintes arquivos e diretórios:
* **Acesso a Banco e Consultas:** `src/ingestion/database.py` (SQLAlchemy, PostGIS, parametrização de queries e índices).
* **Pipeline de Dados e Higienização:** `src/processing/clean_processor.py` (tratamento de dados brutos e validação de coordenadas).
* **Inteligência Artificial e LLM:** `src/models/ai_copilot.py` (injeção de prompt, gestão de chave de API e fallback).
* **Camada de Apresentação Web:** `src/visualization/app.py` (Streamlit, sanitização de inputs, renderização HTML e download de arquivos).
* **Infraestrutura e Contêineres:** `docker-compose.yml`, `.env.example`, `.env` e `.gitignore`.
* **Qualidade e Testes:** `tests/` (verificação de testes de segurança, injeção SQL e resiliência).

---

## 3. Critérios de Avaliação (Checklist de Segurança)

O agente deve avaliar rigorosamente os seguintes 7 pilares:

### Pilar 1: Injeção de SQL e Persistência
- [ ] As consultas SQL utilizam parâmetros nomeados (`:param`) com `sqlalchemy.text()` ou ORM para prevenir SQL Injection?
- [ ] Existem interpolações diretas de strings (`f"..."`, `.format()`, `%`) em cláusulas SQL críticas (`WHERE`, `ORDER BY`, `SELECT`)?
- [ ] O `Connection Pooling` está configurado para evitar esgotamento de conexões (ataques de Denial of Service / DoS)?

### Pilar 2: Gestão de Segredos e Credenciais
- [ ] As senhas do banco e a chave `GEMINI_API_KEY` são carregadas exclusivamente via variáveis de ambiente (`.env`)?
- [ ] O arquivo `.env` está devidamente listado e protegido no `.gitignore` para impedir vazamentos no GitHub?
- [ ] Há senhas padrão (*hardcoded*) fracas no `docker-compose.yml` sem aviso para ambiente produtivo?

### Pilar 3: Privacidade de Dados e LGPD (Lei nº 13.709/2018)
- [ ] A base de ocorrências armazena apenas dados públicos anonimizados?
- [ ] Nomes de vítimas, testemunhas, autores ou números de documentos pessoais (CPF, RG) estão ausentes da tabela `fatos_criminais`?
- [ ] O identificador de boletim (`num_bo`) e a localização geográfica expõem residências ou pessoas físicas?

### Pilar 4: Segurança de Contêineres e Infraestrutura (Docker)
- [ ] As portas do PostgreSQL (5432) estão expostas para todas as interfaces de rede (`0.0.0.0`) ou restritas a `localhost` (`127.0.0.1`)?
- [ ] O contêiner roda com usuário root ou usuário com privilégios mínimos?
- [ ] O volume de dados (`postgres_data`) está devidamente isolado?

### Pilar 5: Segurança da Camada de IA e Proteção contra Prompt Injection
- [ ] As mensagens enviadas pelo usuário no chat são sanitizadas antes de serem injetadas no prompt do Gemini?
- [ ] O sistema mitiga tentativas de Jailbreak ou Prompt Injection para desviar o papel do assistente policial?
- [ ] Há tratamento e fallback gracioso para falhas de cota (HTTP 429) ou indisponibilidade da API do Gemini sem expor *stack traces*?

### Pilar 6: Segurança no Frontend Streamlit e Sanitização de Entrada
- [ ] O uso de `st.markdown(..., unsafe_allow_html=True)` permite injeção de scripts maliciosos (Cross-Site Scripting - XSS)?
- [ ] As entradas dos seletores e sliders de volume possuem limites máximos definidos para evitar estouro de memória?
- [ ] A geração do relatório em PDF utiliza buffers em memória (`bytes`) em vez de arquivos temporários inseguros no disco?

### Pilar 7: Confiabilidade e Testes de Integridade
- [ ] A suíte de testes unitários cobre cenários adversos de banco desconectado e entradas anômalas?
- [ ] As dependências no `requirements.txt` possuem versões compatíveis e seguras?

---

## 4. Instruções de Execução para o Agente

1. **Leitura e Varredura:** Inspecione os arquivos listados no item 2 e execute testes conceituais para cada critério do checklist.
2. **Classificação de Severidade:** Para cada item identificado, classifique conforme a gravidade:
   - 🔴 **ALTO:** Risco iminente de invasão, vazamento de credenciais ou injeção de comandos.
   - 🟡 **MÉDIO:** Configuração insegura de rede, exposição local ou ausência de validação rígida de tipos.
   - 🟢 **BAIXO / MITIGADO:** Risco controlado com medidas mitigadoras existentes no código.
   - ℹ️ **INFORMATIVO:** Oportunidades de refatoração preventiva e boas práticas.
3. **Geração do Arquivo:** Crie obrigatoriamente o arquivo `relatorio_avaliacao_seguranca.md` na raiz do projeto com o relatório formal completo.

---

## 5. Estrutura Obrigatória do Relatório (.md)

O arquivo Markdown de saída gerado pelo agente deve conter as seguintes seções:
1. **Cabeçalho Institucional:** Nome do sistema, autor, instituição, orientador e data da auditoria.
2. **Resumo Executivo:** Visão geral da postura de segurança do sistema e pontuação geral.
3. **Metodologia Aplicada:** Normas e frameworks de referência (OWASP, LGPD, CIS Benchmarks).
4. **Resultados por Pilar de Segurança:** Detalhamento dos 7 pilares com trechos de código inspecionados.
5. **Matriz Consolidada de Riscos:** Tabela com ID, Categoria, Severidade, Status e Impacto.
6. **Pontos Fortes Identificados:** Medidas de segurança já implementadas e elogiáveis no projeto.
7. **Vulnerabilidades e Plano de Ação Recomendado:** Instruções práticas e código de correção para cada fragilidade.
8. **Parecer Conclusivo:** Conclusão formal sobre a maturidade de segurança do software.
