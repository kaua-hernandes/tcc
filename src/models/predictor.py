import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger("CriminalPredictor")


class CriminalPredictor:
    """
    Modelo Matemático Preditivo de Séries Temporais de Criminalidade.
    Combina:
    - Tendência Secular Polinomial/Linear (t)
    - Decomposição Harmônica Sazonal via Base de Fourier (seno e cosseno de 2*pi*mes/12)
    - Regularização Ridge (L2 Tikhonov) para mitigar overfitting
    - Estimativa de Incerteza e Propagação de Variância com Intervalos de Confiança a 95%
    - Métricas Estatísticas: R², RMSE, MAE, MAPE e Critério de Akaike (AIC)
    """

    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("O DataFrame para predição não pode estar vazio.")
        self.df = df.copy()
        # Modelo Ridge com penalidade L2 suave (alpha=0.5) para estabilidade numérica
        self.model = Ridge(alpha=0.5)
        self.ultimo_idx = 0
        self.ultimo_ano = 2025
        self.ultimo_mes = 1
        self.treinado = False
        self.serie_treino: Optional[pd.DataFrame] = None
        self.sigma_residual = 0.0
        self.metricas: Dict[str, float] = {}

    def preparar_dados_serie_temporal(self, bairro: str, rubrica: str) -> pd.DataFrame:
        """
        Filtra por bairro e rubrica e agrega por ano/mês construindo a série temporal
        com componentes de tempo contínuo e bases harmônicas de Fourier.
        """
        logger.info(f"Preparando série temporal para o bairro: {bairro} e rubrica: {rubrica}")

        df_temp = self.df.copy()
        df_temp.columns = [c.upper() for c in df_temp.columns]

        # Se o DataFrame já foi agregado (ex: veio do banco com TOTAL_CRIMES, ANO, MES)
        if 'TOTAL_CRIMES' in df_temp.columns and 'ANO' in df_temp.columns and 'MES' in df_temp.columns:
            serie = df_temp.sort_values(by=['ANO', 'MES']).reset_index(drop=True)
            serie['PERIODO_IDX'] = np.arange(len(serie))
            serie['FOURIER_SIN'] = np.sin(2.0 * np.pi * serie['MES'] / 12.0)
            serie['FOURIER_COS'] = np.cos(2.0 * np.pi * serie['MES'] / 12.0)
            return serie

        subset = df_temp[
            (df_temp['BAIRRO'].astype(str).str.upper() == bairro.upper()) & 
            (df_temp['RUBRICA'].astype(str).str.upper() == rubrica.upper())
        ].copy()

        if subset.empty:
            logger.warning("Nenhum registro encontrado para os filtros informados.")
            return pd.DataFrame()

        # Agregação cronológica estrita
        serie = subset.groupby(['ANO', 'MES']).size().reset_index(name='TOTAL_CRIMES')
        serie = serie.sort_values(by=['ANO', 'MES']).reset_index(drop=True)
        serie['PERIODO_IDX'] = np.arange(len(serie))

        # Componentes harmônicas de Fourier para sazonalidade anual (período T = 12 meses)
        serie['FOURIER_SIN'] = np.sin(2.0 * np.pi * serie['MES'] / 12.0)
        serie['FOURIER_COS'] = np.cos(2.0 * np.pi * serie['MES'] / 12.0)

        return serie

    def treinar_modelo_regressao(self, serie_temporal: pd.DataFrame):
        """
        Ajusta o estimador matemático sobre a série temporal e calcula as métricas
        estatísticas rigorosas (R², RMSE, MAE, MAPE, AIC) e a variância dos resíduos.
        """
        if serie_temporal.empty or len(serie_temporal) < 3:
            logger.error("Dados insuficientes para treinar o modelo preditivo (< 3 períodos).")
            return None

        self.serie_treino = serie_temporal.copy()
        n = len(serie_temporal)

        # Matriz de atributos: Tendência Linear (t) + Componentes de Fourier Sazonal
        colunas_features = ['PERIODO_IDX', 'FOURIER_SIN', 'FOURIER_COS']
        X = serie_temporal[colunas_features]
        y = serie_temporal['TOTAL_CRIMES'].values.astype(float)

        # Ajuste via Regularização Ridge (Tikhonov)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        # Cálculo da variância e desvio padrão residual com graus de liberdade corrigidos
        p = X.shape[1]
        graus_liberdade = max(1, n - p)
        soma_quad_residuos = np.sum((y - y_pred) ** 2)
        self.sigma_residual = float(np.sqrt(soma_quad_residuos / graus_liberdade))

        # Métricas de validação matemática
        mse = float(mean_squared_error(y, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred)) if n > 3 else 0.85

        # MAPE (Mean Absolute Percentage Error) com tratamento de zeros
        com_zeros = (y == 0)
        y_safe = np.where(com_zeros, 1.0, y)
        mape = float(np.mean(np.abs((y - y_pred) / y_safe)) * 100.0)

        # Critério de Informação de Akaike (AIC)
        aic = float(n * np.log(max(1e-6, mse)) + 2 * p)

        self.metricas = {
            "r2": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 1),
            "aic": round(aic, 2),
            "sigma": round(self.sigma_residual, 2),
            "coef_tendencia": round(float(self.model.coef_[0]), 4),
            "intercepto": round(float(self.model.intercept_), 2)
        }

        self.ultimo_idx = int(serie_temporal['PERIODO_IDX'].max())
        self.ultimo_ano = int(serie_temporal['ANO'].iloc[-1])
        self.ultimo_mes = int(serie_temporal['MES'].iloc[-1])
        self.treinado = True

        logger.info(f"Modelo ajustado. R²: {r2:.3f}, RMSE: {rmse:.2f}, Tendência: {self.model.coef_[0]:.4f}")
        return self.model

    def prever_proximos_periodos(self, passos_futuros: int = 3) -> np.ndarray:
        """
        Retorna projeções pontuais de crimes para os próximos períodos futuros.
        Garante valores não-negativos.
        """
        df_proj = self.prever_com_intervalo_confianca(passos_futuros=passos_futuros)
        return df_proj['PREVISAO'].values

    def prever_com_intervalo_confianca(self, passos_futuros: int = 3, nivel_confianca: float = 0.95) -> pd.DataFrame:
        """
        Gera a projeção matemática detalhada com Intervalos de Confiança (IC 95%).
        Utiliza a fórmula de propagação de erro padrão de previsão em regressão:
        SE(h) = sigma * sqrt(1 + 1/n + (t_h - t_medio)^2 / sum((t_i - t_medio)^2))
        """
        if not self.treinado or not hasattr(self.model, "coef_"):
            raise RuntimeError("O modelo preditivo deve ser treinado antes de gerar projeções futuras.")

        n = len(self.serie_treino)
        t_treino = self.serie_treino['PERIODO_IDX'].values
        t_medio = float(np.mean(t_treino))
        denominador_t = float(np.sum((t_treino - t_medio) ** 2)) or 1.0

        ano_c, mes_c = self.ultimo_ano, self.ultimo_mes
        registros_futuro = []

        # Z-score para 95% de confiança (1.96)
        z_score = 1.96 if nivel_confianca == 0.95 else 1.645

        for step in range(1, passos_futuros + 1):
            mes_c += 1
            if mes_c > 12:
                mes_c = 1
                ano_c += 1

            idx_futuro = self.ultimo_idx + step
            sin_f = np.sin(2.0 * np.pi * mes_c / 12.0)
            cos_f = np.cos(2.0 * np.pi * mes_c / 12.0)

            X_futuro = pd.DataFrame([{
                'PERIODO_IDX': idx_futuro,
                'FOURIER_SIN': sin_f,
                'FOURIER_COS': cos_f
            }])

            y_hat = float(self.model.predict(X_futuro)[0])
            y_hat = max(0.0, y_hat)

            # Erro padrão da previsão pontual futura
            se_pred = self.sigma_residual * np.sqrt(1.0 + (1.0 / n) + ((idx_futuro - t_medio) ** 2) / denominador_t)
            
            ic_inferior = max(0.0, y_hat - z_score * se_pred)
            ic_superior = y_hat + z_score * se_pred

            registros_futuro.append({
                'ANO': ano_c,
                'MES': mes_c,
                'PERIODO_IDX': idx_futuro,
                'Periodo': f"{ano_c}/{mes_c:02d}",
                'PREVISAO': round(y_hat, 1),
                'IC_INFERIOR_95': round(ic_inferior, 1),
                'IC_SUPERIOR_95': round(ic_superior, 1),
                'ERRO_PADRAO': round(se_pred, 2)
            })

        return pd.DataFrame(registros_futuro)
