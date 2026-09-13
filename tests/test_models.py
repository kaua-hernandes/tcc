"""
Testes Unitários - Modelagem Preditiva de Criminalidade (Scikit-Learn)
Projeto: TCC - Analytics Criminal SP
Aluno: Kauã Hernandes
Disciplina: Teste e Validação de Software II
Metodologia: Test-Driven Development (TDD)
IA Assistente: Antigravity / Gemini 3.8 Flash (Google DeepMind)
"""

import unittest
import pandas as pd
import numpy as np
from src.models.predictor import CriminalPredictor


class TestCriminalPredictor(unittest.TestCase):
    def setUp(self):
        """Massa de teste contendo série temporal de crimes por bairro e ano/mês."""
        self.dados_historicos = pd.DataFrame({
            'BAIRRO': ['Sé']*7 + ['Mooca']*2,
            'RUBRICA': ['Roubo']*7 + ['Furto']*2,
            'ANO': [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024],
            'MES': [1, 1, 2, 2, 3, 4, 5, 1, 2]  # Sé possui 5 meses distintos, com repetições em 1 e 2
        })

    def test_inicializacao_dataframe_vazio_lanca_excecao(self):
        """CT01: Deve lançar ValueError ao tentar inicializar o preditor sem dados."""
        with self.assertRaises(ValueError):
            CriminalPredictor(pd.DataFrame())

        with self.assertRaises(ValueError):
            CriminalPredictor(None)

    def test_preparar_serie_temporal_case_insensitive_e_agregacao(self):
        """CT02: A agregação da série deve somar ocorrências do mesmo mês e ser case-insensitive."""
        predictor = CriminalPredictor(self.dados_historicos)
        serie = predictor.preparar_dados_serie_temporal(bairro="sé", rubrica="roubo")

        self.assertFalse(serie.empty)
        self.assertIn('TOTAL_CRIMES', serie.columns)
        self.assertIn('PERIODO_IDX', serie.columns)
        # Deve agregar em 5 períodos distintos (meses 1, 2, 3, 4, 5)
        self.assertEqual(len(serie), 5)
        # O mês 1 teve 2 ocorrências agrupadas
        total_mes_1 = serie.loc[serie['MES'] == 1, 'TOTAL_CRIMES'].iloc[0]
        self.assertEqual(total_mes_1, 2)

    def test_preparar_serie_temporal_sem_correspondencia(self):
        """CT03: Deve retornar DataFrame vazio caso o bairro ou crime não existam."""
        predictor = CriminalPredictor(self.dados_historicos)
        serie = predictor.preparar_dados_serie_temporal(bairro="Bairro Inexistente", rubrica="Roubo")
        self.assertTrue(serie.empty)

    def test_treinar_modelo_com_dados_insuficientes(self):
        """CT04: Não deve treinar regressão com menos de 3 períodos temporais (retorna None)."""
        predictor = CriminalPredictor(self.dados_historicos)
        # Mooca só tem 2 períodos (meses 1 e 2)
        serie_insuficiente = predictor.preparar_dados_serie_temporal(bairro="Mooca", rubrica="Furto")
        modelo = predictor.treinar_modelo_regressao(serie_insuficiente)

        self.assertIsNone(modelo)

    def test_treinar_modelo_regressao_com_sucesso(self):
        """CT05: Deve treinar com sucesso o modelo de regressão linear para série com >= 3 períodos."""
        predictor = CriminalPredictor(self.dados_historicos)
        serie = predictor.preparar_dados_serie_temporal(bairro="Sé", rubrica="Roubo")
        modelo = predictor.treinar_modelo_regressao(serie)

        self.assertIsNotNone(modelo)
        self.assertTrue(hasattr(modelo, "coef_"))
        self.assertTrue(hasattr(modelo, "intercept_"))

    def test_prever_proximos_periodos_valores_nao_negativos(self):
        """CT06: As projeções de crimes para períodos futuros nunca devem ser negativas."""
        predictor = CriminalPredictor(self.dados_historicos)
        serie = predictor.preparar_dados_serie_temporal(bairro="Sé", rubrica="Roubo")
        predictor.treinar_modelo_regressao(serie)

        passos = 4
        previsoes = predictor.prever_proximos_periodos(passos_futuros=passos)

        self.assertEqual(len(previsoes), passos)
        self.assertTrue((previsoes >= 0).all())

    def test_prever_sem_treinamento_lanca_excecao(self):
        """CT07: Tentar prever períodos futuros antes de treinar o modelo deve levantar RuntimeError."""
        predictor = CriminalPredictor(self.dados_historicos)
        with self.assertRaises(RuntimeError):
            predictor.prever_proximos_periodos(passos_futuros=3)


if __name__ == '__main__':
    unittest.main()
