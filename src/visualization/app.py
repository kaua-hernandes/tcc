import os
import sys

path_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if path_raiz not in sys.path:
    sys.path.append(path_raiz)

import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import plotly.express as px
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from fpdf import FPDF
from config.settings import Settings
from src.models.ai_copilot import SecurityCopilot

# 1. Configuração da Página e Estilização Customizada CSS
st.set_page_config(page_title="Analytics Criminal SP", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            padding: 12px 18px;
            border-radius: 10px;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.04);
            text-align: center;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #6c757d !important;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            color: #1e293b !important;
            font-weight: 700;
        }
        .header-title {
            color: #0f172a;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0px;
        }
        .ai-box {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-left: 5px solid #2563eb;
            padding: 20px;
            border-radius: 8px;
            color: #1e293b;
            font-size: 0.95rem;
            line-height: 1.6;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Título Principal
st.markdown('<p class="header-title">🛡️ Sistema de Análise Preditiva e Mapeamento Criminal - 2025</p>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def carregar_dados_reais_ssp():
    caminho_csv = os.path.join(Settings.BASE_DIR, "src", "processing", "dados_criminais_limpos.csv")
    if os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv)
        return df
    return pd.DataFrame()

# FUNÇÃO AUXILIAR PARA GERAR O PDF EM MEMÓRIA
def gerar_pdf_bytes(texto_relatorio: str, municipio: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, f"RELATORIO TATICO DE INTELIGENCIA - {municipio.upper()}", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=10)
    # Trata caracteres utf-8 incompatíveis com fontes padrões do fpdf
    texto_limpo = texto_relatorio.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, txt=texto_limpo)
    
    return bytes(pdf.output())

# FUNÇÃO COM CACHE DO STREAMLIT PARA A API DO GEMINI
@st.cache_data(show_spinner=False, ttl=3600)
def chamar_gemini_com_cache(municipio: str, total_crimes: int, top_crimes_tuple: tuple, top_bairros_tuple: tuple, pico_horario: str) -> str:
    copilot = SecurityCopilot()
    return copilot.gerar_relatorio_tatico(
        municipio=municipio,
        total_crimes=total_crimes,
        top_crimes=list(top_crimes_tuple),
        top_bairros=list(top_bairros_tuple),
        pico_horario=pico_horario
    )

df_real_completo = carregar_dados_reais_ssp()

# Tenta carregar o modelo de IA
model_path = os.path.join(Settings.BASE_DIR, "src", "models", "saved_models", "kmeans_spatial_model.pkl")
kmeans_model = None
if os.path.exists(model_path):
    kmeans_model = joblib.load(model_path)

# FILTROS OPERACIONAIS NA BARRA LATERAL
st.sidebar.markdown("### 🔍 Filtros Operacionais")

if not df_real_completo.empty:
    lista_municipios = sorted(df_real_completo['NOME_MUNICIPIO'].dropna().unique())
    default_idx = lista_municipios.index('S.PAULO') if 'S.PAULO' in lista_municipios else 0
    municipio_selecionado = st.sidebar.selectbox("Selecione o Município", lista_municipios, index=default_idx)
    
    lista_crimes = sorted(df_real_completo['RUBRICA'].dropna().unique())
    crimes_selecionados = st.sidebar.multiselect(
        "Natureza da Ocorrência", 
        lista_crimes,
        default=lista_crimes[:2]
    )
else:
    municipio_selecionado = "S.PAULO"
    crimes_selecionados = []

# Filtragem dos dados
if not df_real_completo.empty:
    df_filtrado = df_real_completo[df_real_completo['NOME_MUNICIPIO'] == municipio_selecionado]
    if crimes_selecionados:
        df_filtrado = df_filtrado[df_filtrado['RUBRICA'].isin(crimes_selecionados)]
        
    volumetria_atual = len(df_filtrado)
    
    if volumetria_atual > 2500:
        df_amostra_mapa = df_filtrado.sample(n=2500, random_state=Settings.ML_RANDOM_STATE)
    else:
        df_amostra_mapa = df_filtrado
else:
    df_filtrado = pd.DataFrame()
    df_amostra_mapa = pd.DataFrame()
    volumetria_atual = 0

# --- PAINEL DE MÉTRICAS (KPI CARDS) ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Município Selecionado", municipio_selecionado)
kpi2.metric("Ocorrências no Filtro", f"{volumetria_atual:,}".replace(",", "."))
kpi3.metric("Total Base Limpa (SP)", "834.385")
kpi4.metric("Status do Modelo IA", "Ativo (K-Means + Gemini)" if kmeans_model else "Offline")

st.markdown("<br>", unsafe_allow_html=True)

# --- 1. SEÇÃO DO MAPA EXPANDIDO ---
st.markdown("##### 📍 Mapa Espacial de Risco (Hotspots)")

if not df_amostra_mapa.empty:
    centro_lat = df_amostra_mapa['LATITUDE'].mean()
    centro_lon = df_amostra_mapa['LONGITUDE'].mean()
    zoom = 13 if municipio_selecionado in ['S.PAULO', 'CAMPINAS', 'SANTOS'] else 12
else:
    centro_lat, centro_lon, zoom = -23.550520, -46.633308, 11
    
m = folium.Map(location=[centro_lat, centro_lon], zoom_start=zoom, tiles="OpenStreetMap")

if not df_amostra_mapa.empty:
    dados_calor = df_amostra_mapa[['LATITUDE', 'LONGITUDE']].values.tolist()
    
    gradiente_personalizado = {
        0.25: '#4B0082',
        0.45: '#0000FF',
        0.65: '#00FF00',
        0.85: '#FFFF00',
        1.00: '#FF0000'
    }
    
    HeatMap(
        dados_calor, 
        radius=18, 
        blur=12, 
        min_opacity=0.4,
        gradient=gradiente_personalizado
    ).add_to(m)

# Renderiza o Mapa
st_folium(
    m, 
    use_container_width=True, 
    height=580, 
    key=f"mapa_{municipio_selecionado}_{len(crimes_selecionados)}",
    returned_objects=[]
)

st.markdown("<br>", unsafe_allow_html=True)

# --- 2. SEÇÃO EXPANSÍVEL DE ESTATÍSTICAS E IA COPILOT ---
with st.expander("📊 Clique aqui para abrir o Diagnóstico Analítico e Relatório da IA", expanded=False):
    if not df_filtrado.empty:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            # GRÁFICO 1: Top Crimes
            top_crimes_df = df_filtrado['RUBRICA'].value_counts().head(5).reset_index()
            top_crimes_df.columns = ['Crime', 'Quantidade']
            
            top_crimes_df['Crime_Curto'] = top_crimes_df['Crime'].apply(
                lambda x: x[:32] + '...' if len(x) > 32 else x
            )
            
            fig_barras = px.bar(
                top_crimes_df, 
                x='Quantidade', 
                y='Crime_Curto', 
                orientation='h',
                title="Top 5 Ocorrências mais Frequentes",
                text='Quantidade',
                color='Quantidade',
                color_continuous_scale='Reds'
            )
            fig_barras.update_traces(textposition='outside')
            fig_barras.update_layout(
                yaxis={'categoryorder':'total ascending', 'title': ''}, 
                xaxis={'title': 'Volume de Ocorrências'},
                showlegend=False, 
                height=300,
                margin=dict(l=10, r=30, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_barras, use_container_width=True)

        with col_g2:
            # GRÁFICO 2: Distribuição Mensal
            if 'MES' in df_filtrado.columns:
                crimes_mes = df_filtrado['MES'].value_counts().sort_index().reset_index()
                crimes_mes.columns = ['Mês', 'Ocorrências']
                
                mapa_meses = {
                    1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                    7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
                }
                crimes_mes['Mês_Nome'] = crimes_mes['Mês'].map(mapa_meses)
                
                fig_linha = px.line(
                    crimes_mes, 
                    x='Mês_Nome', 
                    y='Ocorrências', 
                    markers=True,
                    title="Série Temporal Mensal (2025)"
                )
                fig_linha.update_traces(line_color='#ef4444', line_width=3)
                fig_linha.update_layout(
                    height=300,
                    xaxis={'title': 'Mês'},
                    yaxis={'title': 'Total de Ocorrências'},
                    margin=dict(l=10, r=10, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_linha, use_container_width=True)

        st.markdown("---")
        
        # --- 🤖 MÓDULO INTELIGENTE: GERADOR DE BRIEFING TÁTICO COM CONTEXTO DE BAIRROS ---
        st.subheader("🤖 Assistente de Inteligência Policial (LLM Copilot)")
        
        top_crimes_lista = tuple(top_crimes_df['Crime'].tolist()[:3]) if not top_crimes_df.empty else ("Geral",)
        
        # Extrai os 3 bairros com mais ocorrências no filtro
        top_bairros_lista = ("Bairros Diversos",)
        if 'BAIRRO' in df_filtrado.columns:
            bairros_top = df_filtrado['BAIRRO'].dropna().value_counts().head(3).index.tolist()
            if bairros_top:
                top_bairros_lista = tuple(bairros_top)

        pico_horario = "Período Noturno"
        if 'HORA' in df_filtrado.columns:
            hora_frequente = df_filtrado[df_filtrado['HORA'] != -1]['HORA'].mode()
            if not hora_frequente.empty:
                h = hora_frequente.iloc[0]
                pico_horario = f"Concentração elevada por volta das {h}:00h"

        chave_estado_relatorio = f"relatorio_{municipio_selecionado}_{len(crimes_selecionados)}"

        col_btn1, col_btn2 = st.columns([2, 5])
        
        with col_btn1:
            gerar_clicado = st.button("✨ Gerar Briefing Tático do Município com IA")

        if gerar_clicado:
            with st.spinner("Analisando manchas criminais e bairros críticos com o Gemini 3.5..."):
                relatorio_res = chamar_gemini_com_cache(
                    municipio=municipio_selecionado,
                    total_crimes=volumetria_atual,
                    top_crimes_tuple=top_crimes_lista,
                    top_bairros_tuple=top_bairros_lista,
                    pico_horario=pico_horario
                )
                st.session_state[chave_estado_relatorio] = relatorio_res

        # Exibe o relatório salvo em memória e o Botão de Download em PDF
        if chave_estado_relatorio in st.session_state:
            texto_relatorio = st.session_state[chave_estado_relatorio]
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="ai-box">{texto_relatorio}</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Botão para baixar o relatório em PDF
            pdf_bytes = gerar_pdf_bytes(texto_relatorio, municipio_selecionado)
            st.download_button(
                label="📄 Baixar Relatório Tático em PDF",
                data=pdf_bytes,
                file_name=f"Relatorio_Tatico_{municipio_selecionado}.pdf",
                mime="application/pdf"
            )

    else:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")