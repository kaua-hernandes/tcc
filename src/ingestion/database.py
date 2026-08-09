import logging
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASS", "tcc_password_2026")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        db = os.getenv("DB_NAME", "criminal_db")
        
        self.engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")

    def popular_banco_dados(self, df: pd.DataFrame):
        """Cria a tabela via Pandas (inferindo tipos corretos) e aplica Bulk Insert."""
        print(f"Iniciando preparação de {len(df)} registros para o PostgreSQL...")

        df_clean = df.drop_duplicates(subset=['NUM_BO']).copy()
        
        if 'DATA_OCORRENCIA_BO' in df_clean.columns:
            df_clean['DATA_OCORRENCIA_BO'] = pd.to_datetime(df_clean['DATA_OCORRENCIA_BO'], errors='coerce')

        df_clean.columns = [col.lower() for col in df_clean.columns]

        try:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            
            print("Gravando dados no PostgreSQL via Bulk Insert (isso pode levar alguns segundos)...")
            with self.engine.begin() as conn:
                df_clean.to_sql(
                    'fatos_criminais', 
                    conn, 
                    if_exists='replace', 
                    index=False, 
                    method='multi', 
                    chunksize=10000
                )

            print("Criando índices espaciais e de performance...")
            with self.engine.begin() as conn:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fatos_municipio ON fatos_criminais (nome_municipio);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fatos_rubrica ON fatos_criminais (rubrica);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fatos_lat_long ON fatos_criminais (latitude, longitude);"))

            print("[SUCESSO TOTAL] Banco populado e indexado com sucesso!")
        except Exception as e:
            print(f"[ERRO CRÍTICO] Falha ao popular o banco: {e}")
            raise e

    def obter_municipios(self) -> list:
        """Busca a lista distinta de municípios diretamente do banco."""
        query = "SELECT DISTINCT nome_municipio FROM fatos_criminais WHERE nome_municipio IS NOT NULL ORDER BY nome_municipio;"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df['nome_municipio'].tolist()

    def obter_rubricas(self) -> list:
        """Busca a lista distinta de naturezas de crime diretamente do banco."""
        query = "SELECT DISTINCT rubrica FROM fatos_criminais WHERE rubrica IS NOT NULL ORDER BY rubrica;"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df['rubrica'].tolist()

    def buscar_dados_lazy_loading(self, municipio=None, rubricas=None, limite=15000) -> pd.DataFrame:
        """
        Lazy Loading otimizado: Consulta o PostgreSQL aplicando filtros 
        por município e múltiplos tipos de crime, registrando a query diretamente no arquivo de log.
        """
        query = """
            SELECT 
                num_bo, data_ocorrencia_bo AS data_ocorrencia, hora_ocorrencia_bo AS hora_ocorrencia, 
                ano, mes, dia, dia_semana, hora,
                rubrica, nome_municipio, bairro,
                latitude, longitude
            FROM fatos_criminais
            WHERE 1=1
        """
        params = {}
        
        if municipio:
            query += " AND nome_municipio = :municipio"
            params["municipio"] = municipio
            
        if rubricas:
            query += " AND rubrica = ANY(:rubricas)"
            params["rubricas"] = list(rubricas)
            
        query += f" LIMIT {limite};"
        
        logger = logging.getLogger("DatabaseManager")
        logger.info(f"[SQL EXECUTADA NO POSTGIS]\nQuery: {query.strip()}\nParâmetros: {params}")
        
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(query), conn, params=params)
        except Exception as e:
            logger.error(f"[DB ERROR] Falha ao executar query: {e}")
            return pd.DataFrame()