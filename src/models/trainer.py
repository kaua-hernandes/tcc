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
        self.df = dataframe
        self.model_dir = os.path.join(Settings.BASE_DIR, "src", "models", "saved_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
    def treinar_clusterizacao_espacial(self) -> str:
        """Treina o algoritmo K-Means para mapear as zonas de maior risco criminal."""
        logger.info(f"Iniciando treinamento do modelo K-Means com {Settings.CLUSTER_K_MEANS} clusters...")
        try:
            # isola as variáveis de espaço para o algoritmo agrupar
            coordenadas = self.df[['LATITUDE', 'LONGITUDE']]
            
            kmeans = KMeans(
                n_clusters=Settings.CLUSTER_K_MEANS, 
                random_state=Settings.ML_RANDOM_STATE, 
                n_init=10
            )
            kmeans.fit(coordenadas)
            
            # caminha onde a IA treinada sera salva
            model_path = os.path.join(self.model_dir, "kmeans_spatial_model.pkl")
            joblib.dump(kmeans, model_path)
            
            logger.info(f"Modelo preditivo espacial salvo com sucesso em: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Falha ao treinar o modelo preditivo. Erro: {str(e)}")
            raise e