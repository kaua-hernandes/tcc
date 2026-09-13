import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

logger = logging.getLogger("CriminalPredictor")

class CriminalPredictor:
    """
    Componente responsável pela modelagem preditiva de séries temporais
    de criminalidade por município/bairro utilizando Scikit-learn.
    """
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("O DataFrame para predição não pode estar vazio.")
        self.df = df.copy()
        self.model = LinearRegression()
        self.ultimo_idx = 0
        self.treinado = False

    def preparar_dados_serie_temporal(self, bairro: str, rubrica: str) -> pd.DataFrame:
        """
        Filtra os dados por bairro e rubrica e agrega por mes/ano para criar a série temporal.
        """
        logger.info(f"Preparando dados para o bairro: {bairro} e rubrica: {rubrica}")
        
        # Normaliza os nomes das colunas para maiúsculas para evitar KeyError
        df_temp = self.df.copy()
        df_temp.columns = [c.upper() for c in df_temp.columns]

        subset = df_temp[
            (df_temp['BAIRRO'].astype(str).str.upper() == bairro.upper()) & 
            (df_temp['RUBRICA'].astype(str).str.upper() == rubrica.upper())
        ].copy()

        if subset.empty:
            logger.warning("Nenhum registro encontrado para os filtros informados.")
            return pd.DataFrame()

        serie_temporal = subset.groupby(['ANO', 'MES']).size().reset_index(name='TOTAL_CRIMES')
        serie_temporal['PERIODO_IDX'] = np.arange(len(serie_temporal)) 
        
        return serie_temporal

    def treinar_modelo_regressao(self, serie_temporal: pd.DataFrame):
        """
        Treina um modelo de regressão linear para identificar a tendência temporal de criminalidade.
        """
        if serie_temporal.empty or len(serie_temporal) < 3:
            logger.error("Dados insuficientes para treinar o modelo preditivo.")
            return None

        X = serie_temporal[['PERIODO_IDX']]
        y = serie_temporal['TOTAL_CRIMES']

        # Divide treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        logger.info(f"Modelo treinado com sucesso. Erro Quadrático Médio (MSE): {mse:.2f}")

        # Salva o último índice histórico para projeção contínua
        self.ultimo_idx = int(serie_temporal['PERIODO_IDX'].max())
        self.treinado = True

        return self.model

    def prever_proximos_periodos(self, passos_futuros: int = 3) -> np.ndarray:
        """
        Gera a previsão de ocorrências para os próximos meses com base na tendência aprendida.
        Garante projeção contínua no tempo e valores não-negativos.
        """
        if not self.treinado or not hasattr(self.model, "coef_"):
            raise RuntimeError("O modelo preditivo deve ser treinado antes de gerar projeções futuras.")

        futuro_df = pd.DataFrame({
            'PERIODO_IDX': np.arange(self.ultimo_idx + 1, self.ultimo_idx + 1 + passos_futuros)
        })
        
        previsoes = self.model.predict(futuro_df)
        return np.maximum(previsoes, 0)
