import os
import logging
from config.settings import Settings
from src.ingestion.extractor import DataExtractor
from src.processing.clean_processor import DataProcessor
from src.models.trainer import CriminalModelTrainer
from src.ingestion.database import DatabaseManager  # <-- ADICIONE ESTA LINHA


def rodar_pipeline_principal():
    print("\n======= INICIANDO SISTEMA INTEGRADO DE ANÁLISE PREDITIVA =======")
    
    try:
        # PASSO 1: Ingestão e Carga dos Dados Brutos
        extractor = DataExtractor(file_path=Settings.RAW_DATA_PATH, target_columns=Settings.FEATURES_COLUMNS)
        dados_brutos = extractor.executar_pipeline_extracao()
        
        # PASSO 2: Processamento e Engenharia de Atributos (Features)
        processor = DataProcessor(dataframe=dados_brutos)
        dados_tratados = processor.executar_pipeline_limpeza()
        
        # PASSO 3: Treinamento da Inteligência Artificial
        trainer = CriminalModelTrainer(dataframe=dados_tratados)
        modelo_salvo = trainer.treinar_clusterizacao_espacial()
        
        # PASSO 4: Exportação CSV (para o Streamlit)
        caminho_csv_limpo = os.path.join(Settings.BASE_DIR, "src", "processing", "dados_criminais_limpos.csv")
        dados_tratados.to_csv(caminho_csv_limpo, index=False)
        print(f"Base limpa exportada para o dashboard: {caminho_csv_limpo}")
        
        # PASSO 5: Carga Automática no PostgreSQL/PostGIS  <-- ADICIONE ESTE BLOCO
        db_manager = DatabaseManager()
        db_manager.popular_banco_dados(dados_tratados)
        
        print("\n======= PIPELINE DE IA E BANCO EXECUTADOS COM SUCESSO TOTAL =======")
        return dados_tratados
        
    except Exception as e:
        print(f"\n[ERRO CRÍTICO] O pipeline principal quebrou. Detalhes: {e}")


if __name__ == "__main__":
    df_pronto = rodar_pipeline_principal()