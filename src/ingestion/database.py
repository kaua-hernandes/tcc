import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from config.settings import Settings

log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(Settings.LOG_DIR, "pipeline_ingestion.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DatabaseManager")


class DatabaseManager:
    """Gerencia a conexão e carga de dados no banco de dados PostgreSQL/PostGIS."""

    def __init__(self):
        # Constrói a URI de conexão com o Postgres
        self.db_url = (
            f"postgresql://{Settings.DB_USER}:{Settings.DB_PASS}"
            f"@{Settings.DB_HOST}:{Settings.DB_PORT}/{Settings.DB_NAME}"
        )
        self.engine = create_engine(self.db_url)

    def popular_banco_dados(self, df: pd.DataFrame) -> None:
        """Insere o DataFrame tratado na tabela do PostgreSQL."""
        logger.info("Iniciando a carga de dados no PostgreSQL/PostGIS...")
        try:
            # Garante que nomes de colunas fiquem em minúsculo para boa prática no Postgres
            df_postgres = df.copy()
            df_postgres.columns = [col.lower() for col in df_postgres.columns]

            # Envia o dataframe para a tabela 'ocorrencias_criminais'
            df_postgres.to_sql(
                name="ocorrencias_criminais",
                con=self.engine,
                if_exists="replace",  # Substitui a tabela se ela já existir
                index=False,
                chunksize=10000       # Insere em lotes de 10k para ser rápido na memória
            )
            logger.info(" Carga concluída com sucesso no PostgreSQL!")
        except Exception as e:
            logger.error(f"Falha ao popular o banco de dados. Erro: {str(e)}")
            raise e


if __name__ == "__main__":
    # Teste isolado do script de banco
    caminho_csv = os.path.join(Settings.BASE_DIR, "src", "processing", "dados_criminais_limpos.csv")
    if os.path.exists(caminho_csv):
        print("Carregando CSV limpo...")
        df_limpo = pd.read_csv(caminho_csv)
        db_manager = DatabaseManager()
        db_manager.popular_banco_dados(df_limpo)
    else:
        print("Execute o main.py primeiro para gerar o CSV limpo.")