import os
import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from google import genai
from config.settings import Settings

env_path = os.path.join(Settings.BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("SecurityCopilot")

class SecurityCopilot:
    """
    Gerencia a inteligência artificial preditiva e conversacional via Google Gemini API
    integrada à base geoespacial do PostgreSQL / PostGIS.
    """

    MODELOS_DISPONIVEIS = [
        'gemini-3.6-flash',
        'gemini-3.5-flash',
        'gemini-2.5-flash-lite'
    ]

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                chave_limpa = self.api_key.strip('"').strip("'")
                self.client = genai.Client(api_key=chave_limpa)
            except Exception as e:
                logger.error(f"Erro ao inicializar genai.Client: {str(e)}")
                self.client = None

    def extrair_entidades_consulta(self, mensagem: str, municipio_padrao: str = "S.PAULO") -> Tuple[str, Optional[str], Optional[str]]:
        """
        Analisa a mensagem do usuário para identificar menções a cidades, bairros e tipologias criminais.
        """
        msg_upper = mensagem.upper()

        # 1. Mapeamento de Cidades
        mapa_cidades = {
            "SÃO PAULO": "S.PAULO",
            "SAO PAULO": "S.PAULO",
            "SP": "S.PAULO",
            "SAMPA": "S.PAULO",
            "CAPITAL": "S.PAULO",
            "CAMPINAS": "CAMPINAS",
            "GUARULHOS": "GUARULHOS",
            "SANTO ANDRÉ": "S.ANDRE",
            "SANTO ANDRE": "S.ANDRE",
            "SÃO BERNARDO": "S.BERNARDO DO CAMPO",
            "SAO BERNARDO": "S.BERNARDO DO CAMPO",
            "SÃO BERNARDO DO CAMPO": "S.BERNARDO DO CAMPO",
            "SAO BERNARDO DO CAMPO": "S.BERNARDO DO CAMPO",
            "OSASCO": "OSASCO",
            "SOROCABA": "SOROCABA",
            "RIBEIRÃO PRETO": "RIBEIRAO PRETO",
            "RIBEIRAO PRETO": "RIBEIRAO PRETO",
            "SANTOS": "SANTOS",
            "SÃO JOSÉ DOS CAMPOS": "S.JOSE DOS CAMPOS",
            "SAO JOSE DOS CAMPOS": "S.JOSE DOS CAMPOS",
            "SJC": "S.JOSE DOS CAMPOS",
            "MOGI": "MOGI DAS CRUZES",
            "MOGI DAS CRUZES": "MOGI DAS CRUZES"
        }
        
        municipio_detectado = None
        for alias, mun_oficial in mapa_cidades.items():
            padrao = r'\b' + re.escape(alias) + r'\b'
            if re.search(padrao, msg_upper):
                municipio_detectado = mun_oficial
                break

        if not municipio_detectado:
            municipio_detectado = municipio_padrao

        # 2. Mapeamento de Crimes
        crime_detectado = None
        mapa_crimes = {
            "ROUBO": "Roubo",
            "ASSALTO": "Roubo",
            "FURTO": "Furto",
            "HOMICIDIO": "Homicídio",
            "HOMICÍDIO": "Homicídio",
            "ASSASSINATO": "Homicídio",
            "MORTE": "Homicídio",
            "LESÃO": "Lesão corporal",
            "LESAO": "Lesão corporal",
            "AGRESSÃO": "Lesão corporal",
            "AGRESSAO": "Lesão corporal",
            "DROGA": "tráfico drogas",
            "DROGAS": "tráfico drogas",
            "TRÁFICO": "tráfico drogas",
            "TRAFICO": "tráfico drogas",
            "VEÍCULO": "Veículo",
            "VEICULO": "Veículo",
            "CARRO": "Veículo",
            "MOTO": "Veículo",
            "EXTORSÃO": "Extorsão",
            "EXTORSAO": "Extorsão",
            "FEMINICÍDIO": "Feminicídio",
            "FEMINICIDIO": "Feminicídio",
            "ARMA": "Porte de arma"
        }
        for termo, crime_nome in mapa_crimes.items():
            if re.search(r'\b' + re.escape(termo), msg_upper):
                crime_detectado = crime_nome
                break

        # 3. Mapeamento de Bairros Populares ou Padrões
        bairros_conhecidos = [
            "PINHEIROS", "SÉ", "SE", "MOOCA", "REPUBLICA", "REPÚBLICA", "VILA MARIANA",
            "SANTO AMARO", "ITAIM BIBI", "SANTANA", "BELA VISTA", "CONSOLAÇÃO", "CONSOLACAO",
            "CAMBUCI", "TATUAPÉ", "TATUAPE", "MORUMBI", "PERDIZES", "LAPA", "BRÁS", "BRAS",
            "LIBERDADE", "CAPÃO REDONDO", "CAPAO REDONDO", "IPIRANGA", "JARDINS", "JARDIM PAULISTA",
            "CENTRO", "CAMBUÍ", "CAMBUI", "VILA INDUSTRIAL", "TAQUARAL", "BARÃO GERALDO", "BARAO GERALDO",
            "GONZAGA", "BOQUEIRÃO", "BOQUEIRAO", "PONTA DA PRAIA", "VILA MATHIAS"
        ]
        bairro_detectado = None
        for b in bairros_conhecidos:
            padrao = r'\b' + re.escape(b) + r'\b'
            if re.search(padrao, msg_upper):
                # Normaliza grafia
                bairro_detectado = "SÉ" if b in ("SE", "SÉ") else ("REPÚBLICA" if b in ("REPUBLICA", "REPÚBLICA") else b)
                break

        # Tenta capturar "no bairro X" ou "em X" se não achou na lista
        if not bairro_detectado:
            match_bairro = re.search(r'(?:NO BAIRRO|NO|NA|EM)\s+([A-ZÀ-Úa-zà-ú0-9\s]+?)(?:\s+(?:EM|DE|SOBRE|QUAIS|QUANTOS|COM|\?|$))', mensagem, re.IGNORECASE)
            if match_bairro:
                candidato = match_bairro.group(1).strip().upper()
                ignorar = {"SÃO PAULO", "SAO PAULO", "SP", "CAMPINAS", "SANTOS", "GERAL", "TUDO", "CIDADE"}
                if candidato and candidato not in ignorar and len(candidato) > 2:
                    bairro_detectado = candidato

        return municipio_detectado, bairro_detectado, crime_detectado

    def conversar_chat(
        self, 
        mensagem: str, 
        historico: Optional[List[Dict[str, str]]] = None, 
        municipio_atual: str = "S.PAULO",
        db: Optional[Any] = None
    ) -> str:
        """
        Processa uma mensagem no chat interativo, consultando dados reais no PostgreSQL e
        gerando uma resposta conversacional e tática via Gemini API (com fallback resiliente).
        """
        historico = historico or []
        municipio_alvo, bairro_alvo, crime_alvo = self.extrair_entidades_consulta(mensagem, municipio_atual)

        # Consulta dados reais no PostgreSQL
        dados_banco: Optional[Dict[str, Any]] = None
        if db is not None:
            try:
                dados_banco = db.consultar_dados_chat(
                    municipio=municipio_alvo,
                    bairro=bairro_alvo,
                    crime=crime_alvo
                )
            except Exception as e:
                logger.error(f"Erro ao consultar banco para o chat: {e}")

        # Montagem do Grounding de Dados
        bloco_dados = ""
        if dados_banco and dados_banco.get("encontrado"):
            top_crimes_txt = ", ".join([f"{c[0]}: {c[1]:,}".replace(",", ".") for c in dados_banco['top_crimes']]) if dados_banco['top_crimes'] else "N/D"
            top_bairros_txt = ", ".join([f"{b[0]}: {b[1]:,}".replace(",", ".") for b in dados_banco['top_bairros']]) if dados_banco['top_bairros'] else ""
            hora_txt = f"{dados_banco['hora_pico']:02d}:00h ({dados_banco['total_hora_pico']} casos)" if dados_banco['hora_pico'] is not None else "Sem concentração pontual"
            dia_txt = f"{dados_banco['dia_semana_pico']} ({dados_banco['total_dia_pico']} casos)" if dados_banco['dia_semana_pico'] else "Distribuído"

            bloco_dados = f"""
            DADOS ESTATÍSTICOS REAIS EXTRAÍDOS DO POSTGRESQL / POSTGIS (Base 834k registros):
            - Município: {dados_banco['municipio']}
            - Bairro: {dados_banco['bairro'] or 'Geral do Município'}
            - Delito Filtrado: {dados_banco['crime'] or 'Todas as Ocorrências'}
            - Total de Ocorrências no Filtro: {dados_banco['total_filtro']:,}
            - Representatividade no Local: {dados_banco['percentual']}% do volume total
            - Principais Tipologias (Top Crimes): {top_crimes_txt}
            {f"- Bairros de Maior Concentração: {top_bairros_txt}" if top_bairros_txt else ""}
            - Horário Crítico de Pico: {hora_txt}
            - Dia Mais Incidente: {dia_txt}
            - Latência da Consulta no PostGIS: {dados_banco['latencia_ms']} ms
            """

        # Contexto de histórico recente (últimas 3 mensagens)
        historico_contexto = ""
        for h in historico[-4:]:
            papel = "Usuário" if h.get("role") == "user" else "Copiloto"
            historico_contexto += f"{papel}: {h.get('content')}\n"

        prompt = f"""
        Você é o Copiloto de Inteligência e Análise Criminal da plataforma Analytics Criminal SP.
        Você atua como um assessor analítico sênior para policiais, delegados, gestores de segurança pública e pesquisadores.

        DIRETRIZES DE RESPOSTA:
        1. Seja conversacional, prestativo e profissional. Responda em português brasileiro.
        2. Se houver DADOS ESTATÍSTICOS REAIS abaixo, baseie sua resposta neles com fidelidade absoluta aos números (NUNCA invente números diferentes).
        3. Destaque os números principais em negrito (volume total, percentual, horário de pico).
        4. Quando relevante, ofereça uma recomendação tática/operacional breve (ex: horários de patrulhamento preventivo ou alocação de viaturas).
        5. Se o usuário estiver apenas cumprimentando ou perguntando como você funciona, explique suas capacidades de consulta espacial e preditiva em São Paulo.

        {bloco_dados}

        HISTÓRICO DA CONVERSA:
        {historico_contexto}

        PERGUNTA DO USUÁRIO:
        {mensagem}
        """

        # Tentativa de chamada à API do Gemini
        if self.client:
            for mod in self.MODELOS_DISPONIVEIS:
                try:
                    response = self.client.models.generate_content(
                        model=mod,
                        contents=prompt
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as err:
                    logger.warning(f"Falha no modelo {mod}: {err}")
                    continue

        # FALLBACK INTELIGENTE CASO A API ATINJA COTA (429) OU FIQUE OFFLINE
        if dados_banco and dados_banco.get("encontrado"):
            return self._gerar_resposta_fallback_estruturada(dados_banco)

        return (
            "Olá! Sou o seu **Copiloto de Inteligência Criminal**. "
            "Posso consultar em tempo real mais de 834 mil ocorrências do PostgreSQL / PostGIS no Estado de SP. "
            "Você pode me perguntar, por exemplo: \n\n"
            "- *'Quais são os dados de roubo no bairro Pinheiros em São Paulo?'*\n"
            "- *'Quantos furtos ocorreram na Sé?'*\n"
            "- *'Qual o bairro com mais crimes em Campinas?'*\n"
            "- *'Como está a criminalidade em Santos?'*"
        )

    def _gerar_resposta_fallback_estruturada(self, dados: Dict[str, Any]) -> str:
        """Gera resposta formatada diretamente a partir dos dados do PostgreSQL se a API estiver indisponível."""
        loc = f"no bairro **{dados['bairro']}** ({dados['municipio']})" if dados.get('bairro') else f"em **{dados['municipio']}**"
        crime_txt = f"para o delito de **{dados['crime']}**" if dados.get('crime') else "em todas as ocorrências criminais"
        
        top_crimes_md = "\n".join([f"  {idx+1}. **{c[0]}:** {c[1]:,} casos".replace(",", ".") for idx, c in enumerate(dados.get('top_crimes', []))])
        top_bairros_md = ""
        if dados.get('top_bairros'):
            top_bairros_md = "\n**Bairros com maior incidência:**\n" + "\n".join([f"  - **{b[0]}:** {b[1]:,} casos".replace(",", ".") for b in dados['top_bairros']])

        hora_info = f"**{dados['hora_pico']:02d}:00h** ({dados['total_hora_pico']} casos)" if dados.get('hora_pico') is not None else "Sem pico isolado"
        dia_info = f"**{dados['dia_semana_pico']}**" if dados.get('dia_semana_pico') else "Variável"

        return f"""
Analisei diretamente os registros do **PostgreSQL / PostGIS** {loc} {crime_txt}:

📊 **Resumo Estatístico:**
- **Total de Ocorrências:** **{dados['total_filtro']:,}** casos registrados.
- **Representatividade:** Corresponde a **{dados['percentual']}%** do volume local apurado.
- **Horário de Pico Crítico:** {hora_info}.
- **Dia Mais Frequente:** {dia_info}.

🚨 **Distribuição de Ocorrências:**
{top_crimes_md}
{top_bairros_md}

⚡ **Recomendação Tática Operacional:**
Recomenda-se intensificação do policiamento ostensivo e posicionamento tático de viaturas na janela das **{max(0, (dados.get('hora_pico') or 20) - 1):02d}:00h às {min(23, (dados.get('hora_pico') or 20) + 1):02d}:00h**, visando a dissuasão no momento de maior vulnerabilidade.

*⏱️ Dados extraídos da base analítica em {dados.get('latencia_ms', 0)} ms.*
""".strip()

    def gerar_relatorio_tatico(
        self, 
        municipio: str, 
        total_crimes: int, 
        top_crimes: list, 
        top_bairros: list, 
        pico_horario: str
    ) -> str:
        """Gera Briefing Tático formal direcionado ao Comando de Policiamento."""
        bairros_str = ', '.join(top_bairros) if top_bairros else "Bairros Diversos"
        crimes_str = ', '.join(top_crimes) if top_crimes else "Ocorrências Gerais"

        prompt = f"""
        Você é um Especialista em Inteligência de Segurança Pública e Polícia Preditiva.
        Analise o resumo estatístico extraído do nosso modelo preditivo para a região selecionada e gere um Briefing Tático Operacional direcionado ao Comando de Policiamento.

        CONTEXTO TEMPORAL E TÉCNICO OBRIGATÓRIO:
        - Ano de Referência da Base Analítica: 2025 (Todos os registros e balizas devem refletir o ano de 2025).

        DADOS ESTATÍSTICOS EXTRAÍDOS DO MODELO:
        - Município: {municipio}
        - Total de Ocorrências no Filtro: {total_crimes}
        - Principais Delitos Identificados: {crimes_str}
        - Bairros de Maior Incidência (Hotspots): {bairros_str}
        - Janela Horária Crítica: {pico_horario}

        INSTRUÇÕES DE RESPOSTA:
        1. Formate como um 'DOCUMENTO INTERNO – RESERVADO' direcionado ao Comando da Região.
        2. Indique expressamente a data do documento dentro do ano de referência 2025.
        3. Apresente um Diagnóstico da Situação citando expressamente os bairros críticos ({bairros_str}) e a volumetria.
        4. Indique a Faixa Horária e Delitos de Maior Atenção.
        5. Recomende Ações Operacionais Práticas e Georreferenciadas.
        6. Mantenha um tom estritamente técnico e institucional de segurança pública.
        """

        if self.client:
            for mod in self.MODELOS_DISPONIVEIS:
                try:
                    response = self.client.models.generate_content(
                        model=mod,
                        contents=prompt,
                    )
                    if response and response.text:
                        return response.text
                except Exception as e:
                    logger.warning(f"[GEMINI API] Falha no modelo {mod}: {str(e)}")
                    continue

        return (
            f"DOCUMENTO INTERNO – RESERVADO\n"
            f"COMANDO REGIONAL DE POLICIAMENTO - {municipio.upper()}\n"
            f"REFERÊNCIA TEMPORAL: EXERCÍCIO 2025\n\n"
            f"1. DIAGNÓSTICO OPERACIONAL:\n"
            f"- Volume total monitorado: {total_crimes:,} ocorrências.\n"
            f"- Delitos predominantes: {crimes_str}.\n"
            f"- Bairros críticos: {bairros_str}.\n"
            f"- Janela horária de maior risco: {pico_horario}.\n\n"
            f"2. DIRETRIZES TÁTICAS:\n"
            f"- Priorizar pontos de visibilidade e saturação policial nas áreas críticas.\n"
            f"- Bloqueios intermitentes de trânsito e fiscalização ostensiva no horário de pico."
        )