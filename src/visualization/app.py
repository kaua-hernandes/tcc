import os
import sys

path_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if path_raiz not in sys.path:
    sys.path.append(path_raiz)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from config.settings import Settings

st.set_page_config(page_title="Analytics Criminal SP", layout="wide")

st.title("Sistema de Análise Preditiva e Mapeamento Criminal - 2025")
st.markdown("---")

@st.cache_data
def carregar_dados_reais_ssp():
    caminho_csv = os.path.join(Settings.BASE_DIR, "src", "processing", "dados_criminais_limpos.csv")
    if os.path.exists(caminho_csv):
        # carrega a base real gerada pelo pipeline
        df = pd.read_csv(caminho_csv)
        return df
    return pd.DataFrame()

df_real_completo = carregar_dados_reais_ssp()

# Tenta carregar o modelo de IA
model_path = os.path.join(Settings.BASE_DIR, "src", "models", "saved_models", "kmeans_spatial_model.pkl")
kmeans_model = None
if os.path.exists(model_path):
    kmeans_model = joblib.load(model_path)

# FILTROS OPERACIONAIS NA BARRA LATERAL
st.sidebar.header("Filtros Operacionais")

if not df_real_completo.empty:
    lista_municipios = sorted(df_real_completo['NOME_MUNICIPIO'].dropna().unique())
    default_idx = lista_municipios.index('S.PAULO') if 'S.PAULO' in lista_municipios else 0
    municipio_selecionado = st.sidebar.selectbox("Selecione o Município", lista_municipios, index=default_idx)
    
    # captura as rubricas/crimes reais da base para o filtro
    lista_crimes = sorted(df_real_completo['RUBRICA'].dropna().unique())
    crimes_selecionados = st.sidebar.multiselect(
        "Natureza da Ocorrência", 
        lista_crimes,
        default=lista_crimes[:2]
    )
else:
    municipio_selecionado = "S.PAULO"
    crimes_selecionados = []

if not df_real_completo.empty:
    df_filtrado = df_real_completo[df_real_completo['NOME_MUNICIPIO'] == municipio_selecionado]
    if crimes_selecionados:
        df_filtrado = df_filtrado[df_filtrado['RUBRICA'].isin(crimes_selecionados)]
        
    volumetria_atual = len(df_filtrado)
    
    # para o mapa não travar com 500k pontos, extraímos uma amostra real do dado filtrado
    if volumetria_atual > 3000:
        df_amostra_mapa = df_filtrado.sample(n=3000, random_state=Settings.ML_RANDOM_STATE)
    else:
        df_amostra_mapa = df_filtrado
else:
    df_filtrado = pd.DataFrame()
    df_amostra_mapa = pd.DataFrame()
    volumetria_atual = 0

# CONFIGURAÇÃO DO LAYOUT
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Mapa de Calor Espacial de Risco")
    
    # define a centralização do mapa com base nos dados filtrados
    if not df_amostra_mapa.empty:
        centro_lat = df_amostra_mapa['LATITUDE'].mean()
        centro_lon = df_amostra_mapa['LONGITUDE'].mean()
        zoom = 12 if municipio_selecionado == 'S.PAULO' else 9
    else:
        centro_lat, centro_lon, zoom = -23.550520, -46.633308, 11
        
    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=zoom, tiles="CartoDB dark_matter")
    
    # plota a mancha de calor baseada no filtro do usuário
    if not df_amostra_mapa.empty:
        dados_calor = df_amostra_mapa[['LATITUDE', 'LONGITUDE']].values.tolist()
        HeatMap(dados_calor, radius=15, blur=10, min_opacity=0.4).add_to(m)
        
    # plota os marcadores fixos da IA
    if kmeans_model is not None and municipio_selecionado == 'S.PAULO':
        centros = kmeans_model.cluster_centers_
        for idx, centro in enumerate(centros):
            folium.Marker(
                location=[centro[0], centro[1]],
                popup=f"Zona Crítica Geral {idx+1}",
                icon=folium.Icon(color="red", icon="warning", prefix="fa")
            ).add_to(m)
            
    st_folium(m, width=850, height=500, key=f"mapa_{municipio_selecionado}_{len(crimes_selecionados)}")

with col2:
    st.subheader("Estatísticas IA")
    st.metric(label="Ocorrências no Filtro Atual", value=f"{volumetria_atual:,}".replace(",", "."))
    st.metric(label="Total da Base Consolidada", value="834.385")
    
    #st.success("🤖 Filtros conectados com sucesso à base processada.")
    #st.info("O dinamismo em tela agora consome amostragens reais calculadas em memória.")