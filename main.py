import os
import logging
from config.settings import Settings
from src.ingestion.extractor import DataExtractor
from src.processing.clean_processor import DataProcessor
from src.models.trainer import CriminalModelTrainer
from src.ingestion.database import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MainPipeline")

def rodar_pipeline_principal():
    logger.info("======= INICIANDO SISTEMA INTEGRADO DE ANÁLISE PREDITIVA (TCC) =======")
          
    try:
        logger.info("[PASSO 1/5] Extraindo dados brutos...")
        extractor = DataExtractor(file_path=Settings.RAW_DATA_PATH, target_columns=Settings.FEATURES_COLUMNS)
        dados_brutos = extractor.executar_pipeline_extracao()
                  
        logger.info("[PASSO 2/5] Executando limpeza e tratamento geoespacial/temporal...")
        processor = DataProcessor(dataframe=dados_brutos)
        dados_tratados = processor.executar_pipeline_limpeza()
                  
        logger.info("[PASSO 3/5] Treinando modelos de clusterização espacial...")
        trainer = CriminalModelTrainer(dataframe=dados_tratados)
        modelo_salvo = trainer.treinar_clusterizacao_espacial()
                  
        logger.info("[PASSO 4/5] Exportando base limpa local...")
        caminho_csv_limpo = os.path.join(Settings.BASE_DIR, "src", "processing", "dados_criminais_limpos.csv")
        os.makedirs(os.path.dirname(caminho_csv_limpo), exist_ok=True)
        dados_tratados.to_csv(caminho_csv_limpo, index=False)
        logger.info(f"-> Base limpa exportada para: {caminho_csv_limpo}")
                  
        logger.info("[PASSO 5/5] Populando o Banco de Dados Relacional e Espacial...")
        db_manager = DatabaseManager()
        db_manager.popular_banco_dados(dados_tratados)

        logger.info("======= PIPELINE DE IA E BANCO EXECUTADOS COM SUCESSO TOTAL =======")
        return dados_tratados              

    except Exception as e:
        logger.critical(f"[ERRO CRÍTICO] O pipeline principal falhou. Detalhes: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    df_pronto = rodar_pipeline_principal()