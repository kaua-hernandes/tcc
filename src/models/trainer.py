import os
import logging
import pandas as pd
from sklearn.cluster import KMeans
import joblib
from config.settings import Settings

log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(Settings.LOG_DIR, "pipeline_models.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelTrainer")

class CriminalModelTrainer:
    """Componente responsável pelo treinamento e persistência dos modelos de IA."""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.model_dir = os.path.join(Settings.BASE_DIR, "src", "models", "saved_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
    def treinar_clusterizacao_espacial(self) -> str:
        """Treina o algoritmo K-Means para mapear as zonas de maior risco criminal."""
        logger.info(f"Iniciando treinamento do modelo K-Means com {Settings.CLUSTER_K_MEANS} clusters...")
        try:
            if self.df is None or self.df.empty:
                raise ValueError("O DataFrame fornecido para treinamento está vazio.")

            col_lat = 'LATITUDE' if 'LATITUDE' in self.df.columns else 'latitude'
            col_lon = 'LONGITUDE' if 'LONGITUDE' in self.df.columns else 'longitude'

            if col_lat not in self.df.columns or col_lon not in self.df.columns:
                raise KeyError(f"As colunas de coordenadas ('{col_lat}', '{col_lon}') não foram encontradas no DataFrame.")

            coordenadas = self.df[[col_lat, col_lon]].dropna()

            if coordenadas.empty:
                raise ValueError("Não há coordenadas válidas suficientes para treinar o modelo de clusterização.")
            
            kmeans = KMeans(
                n_clusters=Settings.CLUSTER_K_MEANS, 
                random_state=Settings.ML_RANDOM_STATE, 
                n_init=10
            )
            kmeans.fit(coordenadas)
            
            model_path = os.path.join(self.model_dir, "kmeans_spatial_model.pkl")
            joblib.dump(kmeans, model_path)
            
            logger.info(f"Modelo preditivo espacial salvo com sucesso em: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Falha ao treinar o modelo preditivo. Erro: {str(e)}")
            raise e