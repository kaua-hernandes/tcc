import os
import sys

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

import logging
from config.settings import Settings

os.makedirs(Settings.LOG_DIR, exist_ok=True)
log_arquivo = os.path.join(Settings.LOG_DIR, "pipeline_streamlit.log")

log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_arquivo, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StreamlitApp")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import plotly.express as px
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from fpdf import FPDF
from src.models.ai_copilot import SecurityCopilot
from src.models.predictor import CriminalPredictor
from src.ingestion.database import DatabaseManager

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

st.markdown('<p class="header-title">🛡️ Sistema de Análise Preditiva e Mapeamento Criminal - PostGIS</p>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def get_db_manager():
    logger.info("[DB Resource] Conexão com o PostgreSQL/PostGIS inicializada.")
    return DatabaseManager()

db = get_db_manager()

@st.cache_data(ttl=3600)
def carregar_opcoes_filtros():
    logger.info("[CACHE MISS] Listas da barra lateral não estavam em cache. Consultando o banco PostgreSQL...")
    try:
        municipios = db.obter_municipios()
        rubricas = db.obter_rubricas()
        logger.info(f"[DB QUERY SUCCESS] Carregados {len(municipios)} municípios e {len(rubricas)} rubricas do banco.")
        return municipios, rubricas
    except Exception as e:
        logger.error(f"[DB ERROR] Falha ao carregar opções de filtros: {e}")
        return [], []

lista_municipios, lista_crimes = carregar_opcoes_filtros()

@st.cache_data(ttl=1800)
def carregar_dados_filtrados_db(municipio: str, rubricas_tuple: tuple, limite: int):
    logger.info(f"[CACHE MISS] Dados ausentes em memória. Executando consulta SQL no PostGIS para -> Município: {municipio} | Crimes: {rubricas_tuple}")
    
    df_resultado = db.buscar_dados_lazy_loading(
        municipio=municipio, 
        rubricas=list(rubricas_tuple) if rubricas_tuple else None, 
        limite=limite
    )
    logger.info(f"[DB QUERY SUCCESS] Query executada com sucesso. Registros retornados: {len(df_resultado)}")
    return df_resultado

def gerar_pdf_bytes(texto_relatorio: str, municipio: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, f"RELATORIO TATICO DE INTELIGENCIA - {municipio.upper()}", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=10)
    texto_limpo = texto_relatorio.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, txt=texto_limpo)
    
    return bytes(pdf.output())

@st.cache_data(show_spinner=False, ttl=3600)
def chamar_gemini_com_cache(municipio: str, total_crimes: int, top_crimes_tuple: tuple, top_bairros_tuple: tuple, pico_horario: str) -> str:
    logger.info(f"[AI CACHE MISS] Chamando API externa do Gemini (Cache Miss) para o município: {municipio}")
    copilot = SecurityCopilot()
    return copilot.gerar_relatorio_tatico(
        municipio=municipio,
        total_crimes=total_crimes,
        top_crimes=list(top_crimes_tuple),
        top_bairros=list(top_bairros_tuple),
        pico_horario=pico_horario
    )

model_path = os.path.join(Settings.BASE_DIR, "src", "models", "saved_models", "kmeans_spatial_model.pkl")
kmeans_model = None
if os.path.exists(model_path):
    kmeans_model = joblib.load(model_path)

st.sidebar.markdown("### 🔍 Filtros Operacionais")

default_idx = lista_municipios.index('S.PAULO') if 'S.PAULO' in lista_municipios else 0
municipio_selecionado = st.sidebar.selectbox("Selecione o Município", lista_municipios if lista_municipios else ["S.PAULO"], index=default_idx)

crimes_padrao = lista_crimes[:2] if len(lista_crimes) >= 2 else lista_crimes
crimes_selecionados = st.sidebar.multiselect(
    "Natureza da Ocorrência", 
    lista_crimes,
    default=crimes_padrao
)

logger.info(f"[UI ACTION] Renderizando painel para o filtro -> Município: {municipio_selecionado} | Seleções: {len(crimes_selecionados)}")

with st.spinner("Consultando dados espaciais no PostgreSQL..."):
    try:
        if carregar_dados_filtrados_db.has_cached_value(municipio_selecionado, tuple(crimes_selecionados), 15000):
            logger.info(f"[CACHE HIT] Dados recuperados instantaneamente da RAM para -> Município: {municipio_selecionado} | Seleções: {len(crimes_selecionados)}")
        else:
            logger.info(f"[CACHE MISS] Dados não encontrados em cache para -> Município: {municipio_selecionado}")
    except Exception:
        pass

    df_filtrado = carregar_dados_filtrados_db(
        municipio=municipio_selecionado,
        rubricas_tuple=tuple(crimes_selecionados),
        limite=15000
    )

volumetria_atual = len(df_filtrado)

if volumetria_atual > 2500:
    df_amostra_mapa = df_filtrado.sample(n=2500, random_state=Settings.ML_RANDOM_STATE)
else:
    df_amostra_mapa = df_filtrado

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Município Selecionado", municipio_selecionado)
kpi2.metric("Ocorrências no Filtro", f"{volumetria_atual:,}".replace(",", "."))
kpi3.metric("Repositório PostGIS", "Ativo & Normalizado")
kpi4.metric("Status do Modelo IA", "Ativo (K-Means + Gemini)" if kmeans_model else "Offline")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("##### 📍 Mapa Espacial de Risco (Hotspots PostGIS)")

if not df_amostra_mapa.empty:
    centro_lat = df_amostra_mapa['latitude'].mean()
    centro_lon = df_amostra_mapa['longitude'].mean()
    zoom = 13 if municipio_selecionado in ['S.PAULO', 'CAMPINAS', 'SANTOS'] else 12
else:
    centro_lat, centro_lon, zoom = -23.550520, -46.633308, 11
    
m = folium.Map(location=[centro_lat, centro_lon], zoom_start=zoom, tiles="OpenStreetMap")

if not df_amostra_mapa.empty:
    dados_calor = df_amostra_mapa[['latitude', 'longitude']].values.tolist()
    
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

st_folium(
    m, 
    use_container_width=True, 
    height=580, 
    key=f"mapa_{municipio_selecionado}_{len(crimes_selecionados)}",
    returned_objects=[]
)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📊 Clique aqui para abrir o Diagnóstico Analítico, Relatório da IA e Previsão Preditiva", expanded=False):
    if not df_filtrado.empty:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            top_crimes_df = df_filtrado['rubrica'].value_counts().head(5).reset_index()
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
            if 'mes' in df_filtrado.columns:
                crimes_mes = df_filtrado['mes'].value_counts().sort_index().reset_index()
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
        
        # --- SEÇÃO DE ANÁLISE PREDITIVA (SCI KIT-LEARN) ---
        st.subheader("📈 Projeção Preditiva de Criminalidade (Scikit-learn)")
        
        col_p1, col_p2 = st.columns(2)
        bairros_disponiveis = df_filtrado['bairro'].dropna().unique().tolist() if 'bairro' in df_filtrado.columns else []
        
        with col_p1:
            bairro_escolhido = st.selectbox("Selecione o Bairro para Previsão", bairros_disponiveis if bairros_disponiveis else ["GERAL"])
            
        with col_p2:
            rubrica_escolhida = st.selectbox("Selecione a Rubrica para Previsão", crimes_selecionados if crimes_selecionados else ["Geral"])

        if st.button("🚀 Executar Modelo Preditivo"):
            with st.spinner("Treinando modelo estatístico com Scikit-learn..."):
                preditor = CriminalPredictor(df_filtrado)
                serie_hist = preditor.preparar_dados_serie_temporal(bairro=bairro_escolhido, rubrica=rubrica_escolhida)
                
                if not serie_hist.empty and len(serie_hist) >= 3:
                    modelo = preditor.treinar_modelo_regressao(serie_hist)
                    
                    if modelo:
                        passos = 3
                        ultimo_idx = serie_hist['PERIODO_IDX'].max()
                        futuro_indices = np.arange(ultimo_idx + 1, ultimo_idx + 1 + passos).reshape(-1, 1)
                        previsoes_futuras = modelo.predict(futuro_indices)
                        previsoes_futuras = np.maximum(previsoes_futuras, 0)
                        
                        df_historico_plot = serie_hist[['ANO', 'MES', 'TOTAL_CRIMES']].copy()
                        df_historico_plot['Tipo'] = 'Histórico Real'
                        df_historico_plot['Periodo'] = df_historico_plot['ANO'].astype(str) + "/" + df_historico_plot['MES'].astype(str)
                        
                        ultimo_ano = serie_hist['ANO'].iloc[-1]
                        ultimo_mes = serie_hist['MES'].iloc[-1]
                        
                        lista_futuro = []
                        ano_c, mes_c = ultimo_ano, ultimo_mes
                        for val_prev in previsoes_futuras:
                            mes_c += 1
                            if mes_c > 12:
                                mes_c = 1
                                ano_c += 1
                            lista_futuro.append({
                                'ANO': ano_c,
                                'MES': mes_c,
                                'TOTAL_CRIMES': val_prev,
                                'Tipo': 'Projeção Preditiva',
                                'Periodo': f"{ano_c}/{mes_c}"
                            })
                            
                        df_futuro_plot = pd.DataFrame(lista_futuro)
                        df_combinado = pd.concat([df_historico_plot, df_futuro_plot], ignore_index=True)
                        
                        fig_pred = px.line(
                            df_combinado,
                            x='Periodo',
                            y='TOTAL_CRIMES',
                            color='Tipo',
                            markers=True,
                            title=f"Tendência e Projeção de Ocorrências para: {bairro_escolhido}"
                        )
                        fig_pred.update_traces(line_width=3)
                        fig_pred.update_layout(
                            height=350,
                            xaxis={'title': 'Período (Ano/Mês)'},
                            yaxis={'title': 'Volume Estimado de Crimes'},
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                        st.success("✨ Modelo preditivo executado com sucesso e integrado ao pipeline de dados!")
                else:
                    st.warning("⚠️ Dados insuficientes para este bairro/crime. Selecione outro filtro com mais histórico de ocorrências.")

        st.markdown("---")
        
        st.subheader("🤖 Assistente de Inteligência Policial (LLM Copilot)")
        
        top_crimes_lista = tuple(top_crimes_df['Crime'].tolist()[:3]) if not top_crimes_df.empty else ("Geral",)
        
        top_bairros_lista = ("Bairros Diversos",)
        if 'bairro' in df_filtrado.columns:
            bairros_top = df_filtrado['bairro'].dropna().value_counts().head(3).index.tolist()
            if bairros_top:
                top_bairros_lista = tuple(bairros_top)

        pico_horario = "Período Noturno"
        if 'hora' in df_filtrado.columns:
            hora_frequente = df_filtrado[df_filtrado['hora'] != -1]['hora'].mode()
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

        if chave_estado_relatorio in st.session_state:
            texto_relatorio = st.session_state[chave_estado_relatorio]
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="ai-box">{texto_relatorio}</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            pdf_bytes = gerar_pdf_bytes(texto_relatorio, municipio_selecionado)
            logger.info(f"[PDF EXPORT] Relatório PDF gerado em memória para o município: {municipio_selecionado}")
            
            st.download_button(
                label="📄 Baixar Relatório Tático em PDF",
                data=pdf_bytes,
                file_name=f"Relatorio_Tatico_{municipio_selecionado}.pdf",
                mime="application/pdf"
            )

    else:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")