"""
Testes Unitários - Copiloto IA Criminal (Gemini + PostgreSQL/PostGIS)
Projeto: TCC - Analytics Criminal SP
Aluno: Kauã Hernandes
Disciplina: Teste e Validação de Software II
Branch: feat/TCC
"""

import unittest
from unittest.mock import patch, MagicMock
from src.models.ai_copilot import SecurityCopilot
from src.ingestion.database import DatabaseManager


class TestSecurityCopilot(unittest.TestCase):
    def setUp(self):
        """Inicializa o SecurityCopilot para testes isolados."""
        self.copilot = SecurityCopilot()

    def test_extracao_entidades_completa(self):
        """CT01: Deve extrair corretamente cidade, bairro e crime a partir de linguagem natural."""
        msg = "Quero saber os roubos no bairro Pinheiros em São Paulo"
        mun, bairro, crime = self.copilot.extrair_entidades_consulta(msg)

        self.assertEqual(mun, "S.PAULO")
        self.assertEqual(bairro, "PINHEIROS")
        self.assertEqual(crime, "Roubo")

    def test_extracao_entidades_apenas_cidade(self):
        """CT02: Deve identificar apenas o município quando bairro e crime não forem citados."""
        msg = "Qual o panorama geral da criminalidade em Campinas?"
        mun, bairro, crime = self.copilot.extrair_entidades_consulta(msg)

        self.assertEqual(mun, "CAMPINAS")
        self.assertIsNone(bairro)
        self.assertIsNone(crime)

    def test_extracao_entidades_com_municipio_padrao(self):
        """CT03: Deve herdar o município de contexto ativo quando a cidade for omitida."""
        msg = "Quantos furtos ocorreram na Sé?"
        mun, bairro, crime = self.copilot.extrair_entidades_consulta(msg, municipio_padrao="S.PAULO")

        self.assertEqual(mun, "S.PAULO")
        self.assertEqual(bairro, "SÉ")
        self.assertEqual(crime, "Furto")

    @patch('src.models.ai_copilot.genai.Client')
    def test_conversar_chat_sucesso_com_gemini(self, mock_client_cls):
        """CT04: Deve chamar o Gemini com grounding dos dados e retornar resposta textual."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Em Pinheiros foram registrados 2.610 roubos."
        mock_client.models.generate_content.return_value = mock_response

        copilot = SecurityCopilot()
        copilot.client = mock_client

        mock_db = MagicMock()
        mock_db.consultar_dados_chat.return_value = {
            "encontrado": True,
            "municipio": "S.PAULO",
            "bairro": "PINHEIROS",
            "crime": "Roubo",
            "total_filtro": 2610,
            "percentual": 23.0,
            "top_crimes": [("Roubo", 2610), ("Furto", 6948)],
            "top_bairros": [],
            "hora_pico": 21,
            "total_hora_pico": 218,
            "dia_semana_pico": "Sexta-feira",
            "total_dia_pico": 450,
            "latencia_ms": 12.5
        }

        resp = copilot.conversar_chat(
            mensagem="Quantos roubos em Pinheiros?",
            municipio_atual="S.PAULO",
            db=mock_db
        )

        self.assertIn("Pinheiros", resp)
        self.assertTrue(mock_client.models.generate_content.called)

    def test_conversar_chat_fallback_resiliente_quando_gemini_indisponivel(self):
        """CT05: Deve ativar fallback estruturado se o Gemini falhar por cota (429) ou indisponibilidade."""
        copilot = SecurityCopilot()
        copilot.client = None  # Simula API offline ou sem chave

        mock_db = MagicMock()
        mock_db.consultar_dados_chat.return_value = {
            "encontrado": True,
            "municipio": "S.PAULO",
            "bairro": "PINHEIROS",
            "crime": "Roubo",
            "total_filtro": 2610,
            "percentual": 23.0,
            "top_crimes": [("Roubo", 2610)],
            "top_bairros": [],
            "hora_pico": 21,
            "total_hora_pico": 218,
            "dia_semana_pico": "Sexta-feira",
            "total_dia_pico": 450,
            "latencia_ms": 5.2
        }

        resp = copilot.conversar_chat(
            mensagem="Quais os roubos em Pinheiros?",
            municipio_atual="S.PAULO",
            db=mock_db
        )

        self.assertIn("PINHEIROS", resp)
        self.assertIn("2.610", resp)
        self.assertIn("21:00h", resp)
        self.assertIn("Sexta-feira", resp)
        self.assertIn("Recomendação Tática Operacional", resp)


class TestDatabaseChatQuery(unittest.TestCase):
    @patch('src.ingestion.database.create_engine')
    def test_consultar_dados_chat_estrutura_retorno(self, mock_create_engine):
        """CT06: Deve retornar dicionário padronizado mesmo se ocorrer erro na consulta SQL."""
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("Erro simulado no banco")
        mock_create_engine.return_value = mock_engine

        db = DatabaseManager()
        res = db.consultar_dados_chat(municipio="Campinas", bairro="Centro", crime="Furto")

        self.assertFalse(res["encontrado"])
        self.assertEqual(res["total_filtro"], 0)
        self.assertEqual(res["municipio"], "CAMPINAS")
        self.assertIsInstance(res["latencia_ms"], float)


if __name__ == "__main__":
    unittest.main()

