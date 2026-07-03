import pandas as pd
import numpy as np

def carregar_e_limpar_2025(nome_arquivo):
    # Selecionamos apenas as colunas essenciais para o mapa e para a IA
    colunas_uteis = [
        'NUM_BO', 'DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO', 
        'NOME_MUNICIPIO', 'BAIRRO', 'LATITUDE', 'LONGITUDE', 'RUBRICA'
    ]
    
    # 1. Lendo o Primeiro Semestre
    print("1/3. Carregando dados de Janeiro a Junho/2025 (Aba 1)...")
    df1 = pd.read_excel(nome_arquivo, sheet_name='JAN_A_JUN_2025', usecols=colunas_uteis)
    print(f"   -> {len(df1)} registros carregados do 1º semestre.")
    
    # 2. Lendo o Segundo Semestre
    print("2/3. Carregando dados de Julho a Dezembro/2025 (Aba 2)...")
    df2 = pd.read_excel(nome_arquivo, sheet_name='JUL_A_DEZ_2025', usecols=colunas_uteis)
    print(f"   -> {len(df2)} registros carregados do 2º semestre.")
    
    # 3. Juntando as duas partes (Concatenação)
    print("3/3. Unificando as duas bases e iniciando a faxina...")
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"-> Total unificado: {len(df)} registros na memória.")
    
    # --- PROCESSO DE LIMPEZA CORRIGIDO (ETL) ---
    
    # Se os dados vieram como texto, removemos espaços em branco antes de tratar
    if df['LATITUDE'].dtype == 'object':
        df['LATITUDE'] = df['LATITUDE'].astype(str).str.replace(',', '.', regex=False).str.strip()
        df['LONGITUDE'] = df['LONGITUDE'].astype(str).str.replace(',', '.', regex=False).str.strip()
    
    # A BALA DE PRATA: Força virar número. Tudo que for lixo de texto (tipo 'NUL.L') vira NaN
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    
    # Agora sim, dropamos tudo que sobrou de nulo ou NaN
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    
    # Remove coordenadas zeradas ou fora do padrão
    df = df[(df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0)]
    
    # Remove boletins duplicados para não enviesar o modelo
    df = df.drop_duplicates(subset=['NUM_BO'])
    
    print(f"\n[SUCESSO] Limpeza concluída!")
    print(f"-> Sobraram {len(df)} ocorrências válidas com geo-localização para o nosso mapa.")
    
    return df

if __name__ == "__main__":
    arquivo_real = "SPDadosCriminais_2025.xlsx" 
    
    # Executa o pipeline completo
    df_final = carregar_e_limpar_2025(arquivo_real)
    
    # Mostra uma prévia das primeiras linhas processadas
    print("\nVisualização dos dados limpos (Primeiras 5 linhas):")
    print(df_final.head())