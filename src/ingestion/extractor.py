import logging
import os
import pandas as pd
from typing import List
from config.settings import Settings

# conif de Logs para auditoria do pipeline
log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(Settings.LOG_DIR, "pipeline_ingestion.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataExtractor")

class DataExtractor:
    """Componente responsável pela extração e unificação de dados do repositório físico."""
    
    def __init__(self, file_path: str, target_columns: List[str]):
        self.file_path = file_path
        self.target_columns = target_columns
        
    def verificar_integridade_arquivo(self) -> bool:
        """Valida se o arquivo de dados existe e está acessível no diretório."""
        if not os.path.exists(self.file_path):
            logger.error(f"Arquivo não encontrado no caminho especificado: {self.file_path}")
            return False
        logger.info(f"Arquivo de origem identificado com sucesso: {self.file_path}")
        return True

    def extrair_aba_especifica(self, sheet_name: str) -> pd.DataFrame:
        """Extrai um DataFrame isolado de uma aba específica validando as colunas."""
        logger.info(f"Iniciando leitura da aba: {sheet_name}")
        try:
            # carrega a aba aplicando o filtro de memória do usecols
            df = pd.read_excel(self.file_path, sheet_name=sheet_name, usecols=self.target_columns)
            logger.info(f"Aba [{sheet_name}] carregada com sucesso. Registros: {len(df)}")
            return df
        except ValueError as val_err:
            logger.critical(f"Erro de colunas na aba [{sheet_name}]. Verifique o dicionário de dados. Erro: {val_err}")
            raise val_err
        except Exception as e:
            logger.critical(f"Falha inesperada ao ler a aba [{sheet_name}]. Detalhes: {str(e)}")
            raise e

    def executar_pipeline_extracao(self) -> pd.DataFrame:
        """Orquestra a leitura de todo o ano de 2025 unificando os semestres."""
        logger.info("=== Iniciando Processo de Ingestão de Dados - SSP-SP ===")
        
        if not self.verificar_integridade_arquivo():
            raise FileNotFoundError(f"Impossível iniciar extração sem o arquivo base: {self.file_path}")
            
        try:
            # extração modular das abas divididas na planilha
            df_semestre_1 = self.extrair_aba_especifica(sheet_name="JAN_A_JUN_2025")
            df_semestre_2 = self.extrair_aba_especifica(sheet_name="JUL_A_DEZ_2025")
            
            logger.info("Concatenando matrizes de dados do primeiro e segundo semestre...")
            df_unificado = pd.concat([df_semestre_1, df_semestre_2], ignore_index=True)
            
            logger.info(f"Unificação concluída. Volumetria final em memória: {len(df_unificado)} linhas.")
            return df_unificado
            
        except Exception as err:
            logger.error(f"O pipeline de ingestão falhou e foi abortado. Motivo: {str(err)}")
            raise err

if __name__ == "__main__":
    extractor = DataExtractor(file_path=Settings.RAW_DATA_PATH, target_columns=Settings.FEATURES_COLUMNS)
    try:
        dados_brutos = extractor.executar_pipeline_extracao()
        print(dados_brutos.head())
    except Exception as e:
        print(f"Execução falhou conforme esperado nos logs: {e}")