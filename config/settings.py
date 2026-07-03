import os
from pathlib import Path

class Settings:
    """Configurações globais do ecossistema de análise preditiva."""
    
    # caminhos base do projeto
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = os.path.join(BASE_DIR, "data")
    LOG_DIR: Path = os.path.join(BASE_DIR, "logs")
    
    # arquivos de dados
    RAW_DATA_NAME: str = "SPDadosCriminais_2025.xlsx"
    RAW_DATA_PATH: str = os.path.join(BASE_DIR, RAW_DATA_NAME)
    
    # colunas que serão utilizadas no pipeline de dados
    FEATURES_COLUMNS: list = [
        'NUM_BO', 'DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO', 
        'NOME_MUNICIPIO', 'BAIRRO', 'LATITUDE', 'LONGITUDE', 'RUBRICA'
    ]
    
    # config do banco de Dados PostGIS
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASS: str = os.getenv("DB_PASS", "tcc_password_2026")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "criminal_db")
    
    # parâmetros de machine learning (Hiperparâmetros)
    ML_RANDOM_STATE: int = 42
    ML_TEST_SIZE: float = 0.2
    CLUSTER_K_MEANS: int = 15
    
    @classmethod
    def inicializar_diretorios(cls) -> None:
        """Garante que as pastas de logs e dados existam no ambiente de dev."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

# executa a criação das pastas necessárias automaticamente ao importar
Settings.inicializar_diretorios()