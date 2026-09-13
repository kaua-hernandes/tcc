"""
Testes Unitários - Pipeline de Limpeza e Processamento Geoespacial de Dados
Projeto: TCC - Analytics Criminal SP
Aluno: Kauã Hernandes
Disciplina: Teste e Validação de Software II
Metodologia: Test-Driven Development (TDD)
IA Assistente: Antigravity / Gemini 3.8 Flash (Google DeepMind)
"""

import unittest
import pandas as pd
import numpy as np
from src.processing.clean_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Prepara massa de dados de teste (Fixture) simulando registros do banco/SP."""
        self.dados_brutos = pd.DataFrame({
            'NUM_BO': ['BO-001', 'BO-002', 'BO-001', 'BO-003', 'BO-004'],
            'LATITUDE': ['-23,5505', -23.5600, '-23,5505', -10.0, None],
            'LONGITUDE': ['-46,6333', -46.6400, '-46,6333', -60.0, -46.6500],
            'DATA_OCORRENCIA_BO': ['2025-01-15', '2025-02-20', '2025-01-15', '2025-03-10', 'data_invalida'],
            'HORA_OCORRENCIA_BO': ['14:30:00', '22:15:00', '14:30:00', '08:00:00', 'hora_invalida'],
            'BAIRRO': ['Sé', 'Consolação', 'Sé', 'Fora do Estado', 'Sé'],
            'RUBRICA': ['Roubo', 'Furto', 'Roubo', 'Homicídio', 'Roubo']
        })

    def test_inicializacao_dataframe_vazio_lanca_excecao(self):
        """CT01: Deve lançar ValueError se inicializado com DataFrame vazio ou nulo."""
        with self.assertRaises(ValueError):
            DataProcessor(pd.DataFrame())

        with self.assertRaises(ValueError):
            DataProcessor(None)

    def test_tratamento_coordenadas_virgula_e_string(self):
        """CT02: Deve converter strings com vírgula ('-23,5505') para float numérico válido."""
        processor = DataProcessor(self.dados_brutos.copy())
        df_geo = processor.tratar_coordenadas_geograficas()

        self.assertTrue(pd.api.types.is_float_dtype(df_geo['LATITUDE']))
        self.assertTrue(pd.api.types.is_float_dtype(df_geo['LONGITUDE']))
        self.assertAlmostEqual(df_geo.loc[df_geo['NUM_BO'] == 'BO-001', 'LATITUDE'].iloc[0], -23.5505, places=4)

    def test_filtro_delimitacao_geografica_estado_sp(self):
        """CT03: Deve descartar coordenadas que estejam fora dos limites geográficos do Estado de SP."""
        processor = DataProcessor(self.dados_brutos.copy())
        df_geo = processor.tratar_coordenadas_geograficas()

        # BO-003 possui lat=-10 e lon=-60 (fora de SP), BO-004 possui lat nula
        self.assertNotIn('BO-003', df_geo['NUM_BO'].values)
        self.assertNotIn('BO-004', df_geo['NUM_BO'].values)
        
        # Garante que todas as latitudes/longitudes restantes estão no bounding box de SP
        self.assertTrue((df_geo['LATITUDE'] < -19).all() and (df_geo['LATITUDE'] > -26).all())
        self.assertTrue((df_geo['LONGITUDE'] < -44).all() and (df_geo['LONGITUDE'] > -53).all())

    def test_engenharia_de_atributos_temporais(self):
        """CT04: Deve extrair corretamente novos atributos numéricos (ANO, MES, DIA, DIA_SEMANA, HORA)."""
        processor = DataProcessor(self.dados_brutos.copy())
        df_temp = processor.engenharia_de_atributos_temporais()

        colunas_esperadas = ['ANO', 'MES', 'DIA', 'DIA_SEMANA', 'HORA']
        for col in colunas_esperadas:
            self.assertIn(col, df_temp.columns)

        # Valida BO-001: 2025-01-15 às 14:30:00
        registro_001 = df_temp[df_temp['NUM_BO'] == 'BO-001'].iloc[0]
        self.assertEqual(registro_001['ANO'], 2025)
        self.assertEqual(registro_001['MES'], 1)
        self.assertEqual(registro_001['DIA'], 15)
        self.assertEqual(registro_001['HORA'], 14)

    def test_deduplicacao_por_numero_boletim(self):
        """CT05: Deve eliminar ocorrências duplicadas baseadas no identificador único NUM_BO."""
        processor = DataProcessor(self.dados_brutos.copy())
        df_limpo = processor.executar_pipeline_limpeza()

        total_bo_001 = (df_limpo['NUM_BO'] == 'BO-001').sum()
        self.assertEqual(total_bo_001, 1)

    def test_pipeline_limpeza_completo(self):
        """CT06: Validação de ponta a ponta do pipeline completo de limpeza e transformação."""
        processor = DataProcessor(self.dados_brutos.copy())
        df_final = processor.executar_pipeline_limpeza()

        # Apenas BO-001 e BO-002 atendem todos os critérios de integridade
        self.assertEqual(len(df_final), 2)
        self.assertListEqual(sorted(df_final['NUM_BO'].tolist()), ['BO-001', 'BO-002'])


if __name__ == '__main__':
    unittest.main()
