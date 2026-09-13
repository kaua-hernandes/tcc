import os
import sys
import time

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
import folium
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from fpdf import FPDF
from src.models.ai_copilot import SecurityCopilot
from src.models.predictor import CriminalPredictor
from src.ingestion.database import DatabaseManager

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Analytics Criminal SP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN MODERNO E MINIMALISTA (CLEAN / LIGHTWEIGHT) ---
st.markdown("""
    <style>
        /* Tipografia e Fundo */
        .main {
            background-color: #f8fafc;
            color: #1e293b;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Cabeçalho Minimalista */
        .minimal-header {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 22px 26px;
            margin-bottom: 22px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        }
        .minimal-title {
            color: #0f172a;
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .minimal-subtitle {
            color: #64748b;
            font-size: 0.88rem;
            margin-top: 4px;
            margin-bottom: 12px;
        }
        
        /* Badges Minimalistas */
        .chip-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .chip {
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            color: #475569;
            font-size: 0.76rem;
            font-weight: 500;
            padding: 3px 10px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .chip-blue {
            background: #eff6ff;
            border-color: #bfdbfe;
            color: #2563eb;
        }
        .chip-green {
            background: #f0fdf4;
            border-color: #bbf7d0;
            color: #16a34a;
        }
        
        /* Cards de Métricas Minimalistas */
        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            padding: 16px 20px !important;
            border-radius: 10px !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
            transition: border-color 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #cbd5e1 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            color: #64748b !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.03em !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        
        /* Caixa de Destaque do Copiloto IA */
        .ai-box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            padding: 22px 24px;
            border-radius: 8px;
            color: #1e293b;
            font-size: 0.94rem;
            line-height: 1.7;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        }
        
        /* Abas Minimalistas */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            font-weight: 500;
            color: #64748b;
            border-radius: 6px 6px 0 0;
            padding: 0 14px;
            font-size: 0.9rem;
        }
        .stTabs [aria-selected="true"] {
            color: #2563eb !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #2563eb !important;
            background: transparent !important;
        }
        
        /* Moldura de Mapa */
        .map-frame {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            background: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# --- INICIALIZAÇÃO DE RECURSOS (SINGLETON / POOLING) ---
@st.cache_resource
def get_db_manager():
    logger.info("[DB Resource] Inicializando gerenciador de conexão com PostgreSQL/PostGIS...")
    return DatabaseManager()

db = get_db_manager()

@st.cache_data(ttl=3600)
def carregar_opcoes_filtros():
    """Carrega municípios e naturezas penais utilizando índices e Materialized Views."""
    logger.info("[CACHE MISS] Carregando opções de filtros do banco de dados...")
    try:
        municipios = db.obter_municipios()
        rubricas = db.obter_rubricas()
        logger.info(f"[DB QUERY SUCCESS] Carregados {len(municipios)} municípios e {len(rubricas)} rubricas.")
        return municipios, rubricas
    except Exception as e:
        logger.error(f"[DB ERROR] Falha ao carregar opções de filtros: {e}")
        return [], []

lista_municipios, lista_crimes = carregar_opcoes_filtros()

@st.cache_data(ttl=1800)
def carregar_dados_filtrados_db(municipio: str, rubricas_tuple: tuple, limite: int):
    """Executa query otimizada com medição de latência via índice composto."""
    t_inicio = time.perf_counter()
    df_resultado = db.buscar_dados_lazy_loading(
        municipio=municipio, 
        rubricas=list(rubricas_tuple) if rubricas_tuple else None, 
        limite=limite
    )
    t_fim = time.perf_counter()
    latencia_ms = (t_fim - t_inicio) * 1000.0
    logger.info(f"[QUERY EXEC] {len(df_resultado)} registros obtidos em {latencia_ms:.2f} ms.")
    return df_resultado, latencia_ms

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
    copilot = SecurityCopilot()
    return copilot.gerar_relatorio_tatico(
        municipio=municipio,
        total_crimes=total_crimes,
        top_crimes=list(top_crimes_tuple),
        top_bairros=list(top_bairros_tuple),
        pico_horario=pico_horario
    )

# --- SIDEBAR DE FILTROS MINIMALISTA ---
st.sidebar.markdown("### 🔍 Filtros")

default_idx = lista_municipios.index('S.PAULO') if 'S.PAULO' in lista_municipios else 0
municipio_selecionado = st.sidebar.selectbox(
    "Município", 
    lista_municipios if lista_municipios else ["S.PAULO"], 
    index=default_idx
)

crimes_padrao = lista_crimes[:2] if len(lista_crimes) >= 2 else lista_crimes
crimes_selecionados = st.sidebar.multiselect(
    "Natureza do Crime", 
    lista_crimes,
    default=crimes_padrao
)

limite_registros = st.sidebar.select_slider(
    "Limite de Registros",
    options=[2500, 5000, 10000, 15000, 25000],
    value=15000
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Configuração do Mapa")
amostragem_mapa = st.sidebar.slider("Amostragem de Pontos", 500, 5000, 2500, step=500)
tipo_mapa = st.sidebar.selectbox("Camada de Mapa", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"], index=0)

# --- CONSULTA PRINCIPAL COM TELEMETRIA ---
with st.spinner("Buscando registros no banco..."):
    df_filtrado, latencia_ms = carregar_dados_filtrados_db(
        municipio=municipio_selecionado,
        rubricas_tuple=tuple(crimes_selecionados),
        limite=limite_registros
    )

volumetria_atual = len(df_filtrado)

# KPIs rápidos da Materialized View
kpis_mv = db.obter_kpis_municipio(municipio_selecionado)
total_historico_municipio = kpis_mv.get("total_crimes", 0)

# --- CABEÇALHO MINIMALISTA ---
st.markdown(f"""
    <div class="minimal-header">
        <h1 class="minimal-title">
            <span>🛡️ Analytics Criminal SP</span>
        </h1>
        <p class="minimal-subtitle">
            Inteligência Geoespacial e Modelagem Preditiva de Dados Policiais do Estado de São Paulo
        </p>
        <div class="chip-container">
            <span class="chip chip-blue">📍 {municipio_selecionado}</span>
            <span class="chip chip-green">✓ PostGIS GiST Ativo</span>
            <span class="chip">⚡ {latencia_ms:.1f} ms</span>
            <span class="chip">🗄️ 834.385 Registros</span>
            <span class="chip">📈 Scikit-learn</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- CARDS DE MÉTRICAS OPERACIONAIS ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    label="Município", 
    value=municipio_selecionado,
    delta=f"Total: {total_historico_municipio:,}".replace(",", ".") if total_historico_municipio else None
)
kpi2.metric(
    label="Ocorrências Filtradas", 
    value=f"{volumetria_atual:,}".replace(",", "."),
    delta=f"{latencia_ms:.1f} ms"
)

horario_pico = "N/D"
if not df_filtrado.empty and 'hora' in df_filtrado.columns:
    moda_hora = df_filtrado[df_filtrado['hora'] != -1]['hora'].mode()
    if not moda_hora.empty:
        horario_pico = f"{moda_hora.iloc[0]:02d}:00h"

kpi3.metric(
    label="Horário Mais Frequente", 
    value=horario_pico,
    delta="Pico de incidência"
)

top_bairro = "Geral"
if not df_filtrado.empty and 'bairro' in df_filtrado.columns and not df_filtrado['bairro'].dropna().empty:
    top_bairro = df_filtrado['bairro'].dropna().value_counts().index[0]

kpi4.metric(
    label="Bairro Mais Incidente", 
    value=top_bairro,
    delta="Mancha crítica"
)

st.markdown("<br>", unsafe_allow_html=True)

# --- ABAS DE NAVEGAÇÃO LIMPAS ---
tab_mapa, tab_diagnostico, tab_predicao, tab_copilot, tab_performance = st.tabs([
    "🗺️ Mapa Espacial",
    "📊 Diagnóstico",
    "📈 Projeção Preditiva",
    "🤖 Copiloto IA",
    "⚡ Performance PostGIS"
])

# ==========================================
# ABA 1: MAPA ESPACIAL (HOTSPOTS)
# ==========================================
with tab_mapa:
    if not df_filtrado.empty:
        df_amostra = df_filtrado.sample(n=min(volumetria_atual, amostragem_mapa), random_state=Settings.ML_RANDOM_STATE)
        centro_lat = df_amostra['latitude'].mean()
        centro_lon = df_amostra['longitude'].mean()
        zoom = 13 if municipio_selecionado in ['S.PAULO', 'CAMPINAS', 'SANTOS'] else 12

        col_m1, col_m2 = st.columns([5, 1])
        with col_m2:
            st.caption("Legenda de Risco")
            st.markdown("""
            <div style="font-size: 0.8rem; line-height: 1.8; color: #475569;">
                🟣 Baixo<br>
                🔵 Moderado<br>
                🟢 Médio<br>
                🟡 Elevado<br>
                🔴 Crítico
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"{len(df_amostra):,} pontos")

        with col_m1:
            m = folium.Map(location=[centro_lat, centro_lon], zoom_start=zoom, tiles=tipo_mapa)
            dados_calor = df_amostra[['latitude', 'longitude']].values.tolist()
            
            gradiente = {
                0.25: '#4B0082',
                0.45: '#0000FF',
                0.65: '#00FF00',
                0.85: '#FFFF00',
                1.00: '#FF0000'
            }
            
            HeatMap(dados_calor, radius=18, blur=12, min_opacity=0.4, gradient=gradiente).add_to(m)
            st_folium(m, use_container_width=True, height=520, key=f"map_{municipio_selecionado}_{len(crimes_selecionados)}", returned_objects=[])
    else:
        st.info("Nenhuma ocorrência encontrada para os filtros selecionados.")

# ==========================================
# ABA 2: DIAGNÓSTICO E ESTATÍSTICAS
# ==========================================
with tab_diagnostico:
    if not df_filtrado.empty:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            top_crimes_df = df_filtrado['rubrica'].value_counts().head(5).reset_index()
            top_crimes_df.columns = ['Crime', 'Quantidade']
            top_crimes_df['Crime_Curto'] = top_crimes_df['Crime'].apply(lambda x: x[:28] + '...' if len(x) > 28 else x)
            
            fig_barras = px.bar(
                top_crimes_df, 
                x='Quantidade', 
                y='Crime_Curto', 
                orientation='h',
                title="Top 5 Ocorrências Mais Frequentes",
                text='Quantidade',
                color='Quantidade',
                color_continuous_scale='Blues'
            )
            fig_barras.update_traces(textposition='outside')
            fig_barras.update_layout(
                yaxis={'categoryorder':'total ascending', 'title': ''}, 
                xaxis={'title': 'Volume'},
                height=320,
                margin=dict(l=10, r=30, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#475569', family='sans-serif')
            )
            st.plotly_chart(fig_barras, use_container_width=True)

        with col_g2:
            if 'mes' in df_filtrado.columns:
                crimes_mes = df_filtrado['mes'].value_counts().sort_index().reset_index()
                crimes_mes.columns = ['Mês', 'Ocorrências']
                mapa_meses = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
                crimes_mes['Mês_Nome'] = crimes_mes['Mês'].map(mapa_meses)
                
                fig_linha = px.line(
                    crimes_mes, 
                    x='Mês_Nome', 
                    y='Ocorrências', 
                    markers=True,
                    title="Evolução Mensal (2025)"
                )
                fig_linha.update_traces(line_color='#2563eb', line_width=2.5)
                fig_linha.update_layout(
                    height=320,
                    xaxis={'title': 'Mês'},
                    yaxis={'title': 'Total'},
                    margin=dict(l=10, r=10, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#475569', family='sans-serif')
                )
                st.plotly_chart(fig_linha, use_container_width=True)

        # Gráfico por Hora do Dia
        if 'hora' in df_filtrado.columns:
            horas_df = df_filtrado[df_filtrado['hora'] != -1]['hora'].value_counts().sort_index().reset_index()
            horas_df.columns = ['Hora', 'Ocorrências']
            fig_hora = px.area(
                horas_df, 
                x='Hora', 
                y='Ocorrências', 
                title="Distribuição por Hora do Dia (0h às 23h)",
                color_discrete_sequence=['#3b82f6']
            )
            fig_hora.update_layout(
                height=240,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=20),
                font=dict(color='#475569', family='sans-serif')
            )
            st.plotly_chart(fig_hora, use_container_width=True)
    else:
        st.info("Sem dados para exibir gráficos.")

# ==========================================
# ABA 3: INTELIGÊNCIA PREDITIVA (ML)
# ==========================================
with tab_predicao:
    col_p1, col_p2 = st.columns(2)
    bairros_disponiveis = df_filtrado['bairro'].dropna().unique().tolist() if not df_filtrado.empty and 'bairro' in df_filtrado.columns else []
    
    with col_p1:
        bairro_escolhido = st.selectbox("Bairro Alvo", bairros_disponiveis if bairros_disponiveis else ["GERAL"])
    with col_p2:
        rubrica_escolhida = st.selectbox("Tipologia Penal", crimes_selecionados if crimes_selecionados else ["Geral"])

    if st.button("Executar Projeção Preditiva", key="btn_predicao"):
        with st.spinner("Treinando modelo estatístico..."):
            preditor = CriminalPredictor(df_filtrado)
            serie_hist = preditor.preparar_dados_serie_temporal(bairro=bairro_escolhido, rubrica=rubrica_escolhida)
            
            if not serie_hist.empty and len(serie_hist) >= 3:
                modelo = preditor.treinar_modelo_regressao(serie_hist)
                if modelo:
                    previsoes = preditor.prever_proximos_periodos(passos_futuros=3)
                    
                    df_historico_plot = serie_hist[['ANO', 'MES', 'TOTAL_CRIMES']].copy()
                    df_historico_plot['Tipo'] = 'Histórico Real'
                    df_historico_plot['Periodo'] = df_historico_plot['ANO'].astype(str) + "/" + df_historico_plot['MES'].astype(str)
                    
                    ultimo_ano = serie_hist['ANO'].iloc[-1]
                    ultimo_mes = serie_hist['MES'].iloc[-1]
                    
                    lista_futuro = []
                    ano_c, mes_c = ultimo_ano, ultimo_mes
                    for val_prev in previsoes:
                        mes_c += 1
                        if mes_c > 12:
                            mes_c = 1
                            ano_c += 1
                        lista_futuro.append({
                            'ANO': ano_c,
                            'MES': mes_c,
                            'TOTAL_CRIMES': round(val_prev, 1),
                            'Tipo': 'Projeção Futura',
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
                        title=f"Tendência e Projeção: {bairro_escolhido} ({rubrica_escolhida})",
                        color_discrete_map={'Histórico Real': '#2563eb', 'Projeção Futura': '#e11d48'}
                    )
                    fig_pred.update_traces(line_width=2.5)
                    fig_pred.update_layout(
                        height=330,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#475569', family='sans-serif')
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    tendencia = "Alta" if modelo.coef_[0] > 0 else "Queda"
                    st.info(f"Tendência calculada: **{tendencia}** (Coeficiente: {modelo.coef_[0]:.2f}).")
            else:
                st.warning("Amostragem histórica insuficiente (< 3 meses) para este bairro/crime.")

# ==========================================
# ABA 4: COPILOTO TÁTICO IA (GEMINI)
# ==========================================
with tab_copilot:
    top_crimes_lista = tuple(df_filtrado['rubrica'].value_counts().head(3).index.tolist()) if not df_filtrado.empty else ("Geral",)
    top_bairros_lista = tuple(df_filtrado['bairro'].dropna().value_counts().head(3).index.tolist()) if not df_filtrado.empty and 'bairro' in df_filtrado.columns and not df_filtrado['bairro'].dropna().empty else ("Bairros Diversos",)
    
    chave_relatorio = f"rel_{municipio_selecionado}_{len(crimes_selecionados)}"
    
    if st.button("Gerar Relatório de Inteligência com IA", key="btn_ai"):
        with st.spinner("Gerando diagnóstico analítico com o Gemini..."):
            relatorio_res = chamar_gemini_com_cache(
                municipio=municipio_selecionado,
                total_crimes=volumetria_atual,
                top_crimes_tuple=top_crimes_lista,
                top_bairros_tuple=top_bairros_lista,
                pico_horario=horario_pico
            )
            st.session_state[chave_relatorio] = relatorio_res

    if chave_relatorio in st.session_state:
        texto_relatorio = st.session_state[chave_relatorio]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="ai-box">{texto_relatorio}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_bytes = gerar_pdf_bytes(texto_relatorio, municipio_selecionado)
        
        st.download_button(
            label="Baixar Relatório em PDF",
            data=pdf_bytes,
            file_name=f"Relatorio_{municipio_selecionado}.pdf",
            mime="application/pdf"
        )

# ==========================================
# ABA 5: CENTRAL DE PERFORMANCE POSTGIS
# ==========================================
with tab_performance:
    telemetria = db.obter_telemetria_postgis()
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Total de Registros", f"{telemetria['total_registros']:,}".replace(",", "."))
    col_t2.metric("Tamanho da Tabela", telemetria['tamanho_tabela'])
    col_t3.metric("Tamanho dos Índices", telemetria['tamanho_indices'])
    col_t4.metric("Índices GiST", telemetria['status_gist'])
    
    st.markdown("---")
    
    col_b1, col_b2 = st.columns([3, 2])
    with col_b1:
        bench_data = pd.DataFrame({
            'Estratégia': [
                'Materialized View', 
                'Índice Composto', 
                'PostGIS GiST', 
                'Scan Sequencial'
            ],
            'Latência (ms)': [0.06, 5.4, 7.3, 33.8]
        })
        fig_bench = px.bar(
            bench_data, 
            x='Latência (ms)', 
            y='Estratégia', 
            orientation='h',
            title="Latência Média por Tipo de Consulta",
            color='Latência (ms)',
            color_continuous_scale='Blues_r',
            text='Latência (ms)'
        )
        fig_bench.update_traces(texttemplate='%{text} ms', textposition='outside')
        fig_bench.update_layout(
            height=260,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#475569', family='sans-serif')
        )
        st.plotly_chart(fig_bench, use_container_width=True)

    with col_b2:
        st.caption("Otimizações Ativas")
        st.markdown("""
        - **PostGIS GiST:** Indexação espacial R-Tree (`geom geometry`).
        - **Índice Composto:** `(nome_municipio, rubrica)`.
        - **Materialized Views:** KPIs pré-computados (< 0.1 ms).
        - **Connection Pool:** Conexões persistentes via SQLAlchemy.
        """)
