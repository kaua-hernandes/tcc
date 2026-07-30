import os
import logging
from dotenv import load_dotenv
from google import genai
from config.settings import Settings

env_path = os.path.join(Settings.BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("SecurityCopilot")

class SecurityCopilot:
    """Gerencia a geração de relatórios táticos de inteligência policial via Gemini API."""

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

    def gerar_relatorio_tatico(
        self, 
        municipio: str, 
        total_crimes: int, 
        top_crimes: list, 
        top_bairros: list, 
        pico_horario: str
    ) -> str:
        """Envia métricas estatísticas detalhadas para o Gemini com data de referência rigida (2025)."""
        
        bairros_str = ', '.join(top_bairros) if top_bairros else "Bairros Diversos"
        crimes_str = ', '.join(top_crimes) if top_crimes else "Ocorrências Gerais"

        prompt = f"""
        Você é um Especialista em Inteligência de Segurança Pública e Polícia Preditiva.
        Analise o resumo estatístico extraído do nosso modelo preditivo para a região selecionada e gere um Briefing Tático Operacional direcionado ao Comando de Policiamento.

        CONTEXTO TEMPORAL E TÉCNICO OBRIGATÓRIO:
        - Ano de Referência da Base Analítica: 2025 (Todos os registros, datas e balizards temporais do relatório devem refletir o ano de 2025).

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
        5. Recomende Ações Operacionais Práticas e Georreferenciadas (ex: posicionamento de viaturas, ROCAM, pontos de estacionamento nos bairros citados).
        6. Mantenha um tom estritamente técnico e institucional de segurança pública.
        """

        if self.client:
            modelos_para_testar = [
                'gemini-3.5-flash',
                'gemini-2.0-flash',
                'gemini-2.0-flash-lite'
            ]
            
            for mod in modelos_para_testar:
                try:
                    response = self.client.models.generate_content(
                        model=mod,
                        contents=prompt,
                    )
                    if response and response.text:
                        return response.text
                except Exception as e:
                    print(f"\n[DIAGNÓSTICO GEMINI API] Falha no modelo {mod}: {str(e)}\n")
                    continue

        return "⚠️ Não foi possível obter resposta da API do Gemini. Verifique os logs no terminal."