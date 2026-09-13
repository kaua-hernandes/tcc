"""
Testes Unitários - Camada de Acesso e Consultas PostGIS (Lazy Loading & Spatial Querying)
Projeto: TCC - Analytics Criminal SP
Aluno: Kauã Hernandes
Disciplina: Teste e Validação de Software II
Branch: feat/TCC
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.ingestion.database import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    @patch('src.ingestion.database.create_engine')
    def setUp(self, mock_create_engine):
        """Inicializa o DatabaseManager com mock do engine SQLAlchemy para testes unitários isolados."""
        self.mock_engine = MagicMock()
        mock_create_engine.return_value = self.mock_engine
        self.db = DatabaseManager()

    @patch('pandas.read_sql')
    def test_query_lazy_loading_com_municipio(self, mock_read_sql):
        """CT01: Deve aplicar filtro parametrizado (:municipio) para prevenir injeção de SQL."""
        mock_read_sql.return_value = pd.DataFrame({'num_bo': ['BO-100']})

        df = self.db.buscar_dados_lazy_loading(municipio="Campinas", limite=500)

        self.assertTrue(mock_read_sql.called)
        args, kwargs = mock_read_sql.call_args
        query_text = str(args[0])
        params = kwargs.get('params', {})

        self.assertIn("nome_municipio = :municipio", query_text)
        self.assertIn("LIMIT 500", query_text)
        self.assertEqual(params.get("municipio"), "Campinas")
        self.assertEqual(len(df), 1)

    @patch('pandas.read_sql')
    def test_query_lazy_loading_com_multiplas_rubricas(self, mock_read_sql):
        """CT02: Deve filtrar múltiplas rubricas via ANY(:rubricas) de forma parametrizada."""
        mock_read_sql.return_value = pd.DataFrame({'num_bo': ['BO-200']})

        rubricas = ["ROUBO DE VEÍCULO", "FURTO DE VEÍCULO"]
        df = self.db.buscar_dados_lazy_loading(rubricas=rubricas)

        args, kwargs = mock_read_sql.call_args
        query_text = str(args[0])
        params = kwargs.get('params', {})

        self.assertIn("rubrica = ANY(:rubricas)", query_text)
        self.assertEqual(params.get("rubricas"), rubricas)

    @patch('pandas.read_sql')
    def test_resiliencia_falha_de_banco_retorna_dataframe_vazio(self, mock_read_sql):
        """CT03: Deve tratar falha de conexão com PostgreSQL retornando DataFrame vazio (sem quebrar a aplicação)."""
        mock_read_sql.side_effect = Exception("Conexão recusada com PostgreSQL.")

        df = self.db.buscar_dados_lazy_loading(municipio="São Paulo")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    @patch('pandas.read_sql')
    def test_busca_por_raio_espacial_postgis(self, mock_read_sql):
        """CT04: Deve utilizar a função ST_DWithin geodésica do PostGIS com raio em metros parametrizado."""
        mock_read_sql.return_value = pd.DataFrame({'num_bo': ['BO-300'], 'distancia_metros': [450.2]})

        df = self.db.buscar_por_raio_espacial(latitude=-23.5505, longitude=-46.6333, raio_km=2.5)

        self.assertTrue(mock_read_sql.called)
        args, kwargs = mock_read_sql.call_args
        query_text = str(args[0])
        params = kwargs.get('params', {})

        self.assertIn("ST_DWithin", query_text)
        self.assertEqual(params.get("lat"), -23.5505)
        self.assertEqual(params.get("lon"), -46.6333)
        self.assertEqual(params.get("raio_metros"), 2500.0)
        self.assertEqual(len(df), 1)

    @patch('pandas.read_sql')
    def test_buscar_coordenadas_mapa_lazy(self, mock_read_sql):
        """CT06: Deve buscar exclusivamente latitude e longitude para renderização instantânea do mapa."""
        mock_read_sql.return_value = pd.DataFrame({'latitude': [-23.55], 'longitude': [-46.63]})

        df = self.db.buscar_coordenadas_mapa_lazy(municipio="S.PAULO", limite=1000)

        self.assertTrue(mock_read_sql.called)
        args, kwargs = mock_read_sql.call_args
        query_text = str(args[0])
        params = kwargs.get('params', {})

        self.assertIn("SELECT latitude, longitude", query_text)
        self.assertIn("LIMIT 1000", query_text)
        self.assertEqual(params.get("municipio"), "S.PAULO")
        self.assertEqual(len(df), 1)


if __name__ == '__main__':
    unittest.main()
