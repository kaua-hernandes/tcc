import os
import logging
import pandas as pd
import numpy as np
from config.settings import Settings

log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(Settings.LOG_DIR, "pipeline_processing.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataProcessor")

class DataProcessor:
    def __init__(self, dataframe: pd.DataFrame):
        if dataframe is None or dataframe.empty:
            logger.critical("O DataFrame fornecido para processamento está vazio ou nulo.")
            raise ValueError("DataFrame inválido para inicialização do DataProcessor.")
        self.df = dataframe.copy()

    def tratar_coordenadas_geograficas(self) -> pd.DataFrame:
        logger.info("Iniciando tratamento de anomalias nas coordenadas geográficas...")
        try:
            if self.df['LATITUDE'].dtype == 'object':
                self.df['LATITUDE'] = self.df['LATITUDE'].astype(str).str.replace(',', '.', regex=False).str.strip()
            if self.df['LONGITUDE'].dtype == 'object':
                self.df['LONGITUDE'] = self.df['LONGITUDE'].astype(str).str.replace(',', '.', regex=False).str.strip()

            self.df['LATITUDE'] = pd.to_numeric(self.df['LATITUDE'], errors='coerce')
            self.df['LONGITUDE'] = pd.to_numeric(self.df['LONGITUDE'], errors='coerce')

            antes_drop = len(self.df)
            self.df = self.df.dropna(subset=['LATITUDE', 'LONGITUDE'])
            
            self.df = self.df[
                (self.df['LATITUDE'] < -19) & (self.df['LATITUDE'] > -26) &
                (self.df['LONGITUDE'] < -44) & (self.df['LONGITUDE'] > -53)
            ]
            
            logger.info(f"Filtro geográfico concluído. Descartados {antes_drop - len(self.df)} registros corrompidos.")
            return self.df
        except Exception as e:
            logger.error(f"Falha fatal no tratamento geoespacial. Erro: {str(e)}")
            raise e

    def engenharia_de_atributos_temporais(self) -> pd.DataFrame:
        logger.info("Executando engenharia de atributos temporais (Feature Engineering)...")
        try:
            self.df['DATA_OCORRENCIA_BO'] = pd.to_datetime(
                self.df['DATA_OCORRENCIA_BO'], format='%Y-%m-%d', errors='coerce'
            )
            self.df = self.df.dropna(subset=['DATA_OCORRENCIA_BO'])

            self.df['ANO'] = self.df['DATA_OCORRENCIA_BO'].dt.year
            self.df['MES'] = self.df['DATA_OCORRENCIA_BO'].dt.month
            self.df['DIA'] = self.df['DATA_OCORRENCIA_BO'].dt.day
            self.df['DIA_SEMANA'] = self.df['DATA_OCORRENCIA_BO'].dt.dayofweek
            
            self.df['HORA'] = self.df['HORA_OCORRENCIA_BO'].astype(str).str.split(':').str[0]
            self.df['HORA'] = pd.to_numeric(self.df['HORA'], errors='coerce').fillna(-1).astype(int)

            logger.info("Novos atributos numéricos (ANO, MES, DIA, DIA_SEMANA, HORA) injetados com sucesso.")
            return self.df
        except Exception as e:
            logger.error(f"Falha na engenharia de dados temporais. Erro: {str(e)}")
            raise e

    def executar_pipeline_limpeza(self) -> pd.DataFrame:
        logger.info("=== Iniciando Processo de Transformação e Modelagem Categórica ===")
        self.tratar_coordenadas_geograficas()
        self.engenharia_de_atributos_temporais()
        self.df = self.df.drop_duplicates(subset=['NUM_BO'])
        logger.info(f"=== Pipeline de Limpeza Concluído. Registros Finais Prontos: {len(self.df)} ===")
        return self.df