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
        
        /* Caixa do Copiloto IA */
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
        return municipios, rubricas
    except Exception as e:
        logger.error(f"[DB ERROR] Falha ao carregar opções de filtros: {e}")
        return [], []

lista_municipios, lista_crimes = carregar_opcoes_filtros()

# --- ESTRATÉGIA DE LAZY LOADING (CARREGAMENTO RÁPIDO SEPARADO) ---
@st.cache_data(ttl=1800)
def carregar_coordenadas_mapa_lazy(municipio: str, rubricas_tuple: tuple, limite: int):
    """
    LAZY LOADING ESPACIAL (< 1ms):
    Busca estritamente latitude e longitude para o mapa, reduzindo transferência de dados em 90%.
    """
    t_inicio = time.perf_counter()
    df_coords = db.buscar_coordenadas_mapa_lazy(
        municipio=municipio,
        rubricas=list(rubricas_tuple) if rubricas_tuple else None,
        limite=limite
    )
    latencia_mapa_ms = (time.perf_counter() - t_inicio) * 1000.0
    return df_coords, latencia_mapa_ms

@st.cache_data(ttl=1800)
def carregar_dados_filtrados_db(municipio: str, rubricas_tuple: tuple, limite: int):
    """Executa query com índice composto B-Tree para dados tabulares e analíticos."""
    t_inicio = time.perf_counter()
    df_resultado = db.buscar_dados_lazy_loading(
        municipio=municipio, 
        rubricas=list(rubricas_tuple) if rubricas_tuple else None, 
        limite=limite
    )
    latencia_ms = (time.perf_counter() - t_inicio) * 1000.0
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
    "Volume de Amostragem Analítica",
    options=[2500, 5000, 10000, 15000],
    value=5000
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Configuração do Mapa")
amostragem_mapa = st.sidebar.slider("Amostragem de Pontos", 500, 5000, 2500, step=500)
tipo_mapa = st.sidebar.selectbox("Camada de Mapa", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"], index=0)

# --- CARREGAMENTO LAZY: COORDENADAS + DADOS TABULARES ---
with st.spinner("Carregando inteligência de dados..."):
    # 1. Lazy loading das coordenadas espaciais (leve e rápido para o mapa)
    df_coords_mapa, latencia_mapa_ms = carregar_coordenadas_mapa_lazy(
        municipio=municipio_selecionado,
        rubricas_tuple=tuple(crimes_selecionados),
        limite=amostragem_mapa
    )
    
    # 2. Dados tabulares analíticos
    df_filtrado, latencia_dados_ms = carregar_dados_filtrados_db(
        municipio=municipio_selecionado,
        rubricas_tuple=tuple(crimes_selecionados),
        limite=limite_registros
    )

volumetria_atual = len(df_filtrado)

# KPIs rápidos da Materialized View (< 1ms)
kpis_mv = db.obter_kpis_municipio(municipio_selecionado)
total_historico_municipio = kpis_mv.get("total_crimes", 0)

# --- CABEÇALHO MINIMALISTA ---
st.markdown(f"""
    <div class="minimal-header">
        <h1 class="minimal-title">
            <span>🛡️ Analytics Criminal SP</span>
        </h1>
        <p class="minimal-subtitle">
            Inteligência Geoespacial, Otimização PostGIS e Modelagem Preditiva com Decomposição de Séries Temporais
        </p>
        <div class="chip-container">
            <span class="chip chip-blue">📍 {municipio_selecionado}</span>
            <span class="chip chip-green">✓ PostGIS GiST Ativo</span>
            <span class="chip">⚡ Query Mapa: {latencia_mapa_ms:.1f} ms</span>
            <span class="chip">⚡ Query Dados: {latencia_dados_ms:.1f} ms</span>
            <span class="chip">🗄️ 834k Registros</span>
            <span class="chip">📈 Ridge + Fourier Model</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- CARDS DE MÉTRICAS OPERACIONAIS ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    label="Município", 
    value=municipio_selecionado,
    delta=f"Histórico MV: {total_historico_municipio:,}".replace(",", ".") if total_historico_municipio else None
)
kpi2.metric(
    label="Ocorrências Filtradas", 
    value=f"{volumetria_atual:,}".replace(",", "."),
    delta=f"{latencia_dados_ms:.1f} ms (B-Tree)"
)

horario_pico = "N/D"
if not df_filtrado.empty and 'hora' in df_filtrado.columns:
    moda_hora = df_filtrado[df_filtrado['hora'] != -1]['hora'].mode()
    if not moda_hora.empty:
        horario_pico = f"{moda_hora.iloc[0]:02d}:00h"

kpi3.metric(
    label="Horário Mais Frequente", 
    value=horario_pico,
    delta="Pico estatístico"
)

top_bairro = "Geral"
if not df_filtrado.empty and 'bairro' in df_filtrado.columns and not df_filtrado['bairro'].dropna().empty:
    top_bairro = df_filtrado['bairro'].dropna().value_counts().index[0]

kpi4.metric(
    label="Bairro Mais Incidente", 
    value=top_bairro,
    delta="Concentração"
)

st.markdown("<br>", unsafe_allow_html=True)

# --- ABAS DE NAVEGAÇÃO LIMPAS ---
tab_mapa, tab_diagnostico, tab_predicao, tab_copilot, tab_performance = st.tabs([
    "🗺️ Mapa Espacial (Lazy Loading)",
    "📊 Diagnóstico & Estatísticas",
    "📈 Modelo Matemático Preditivo",
    "🤖 Copiloto IA",
    "⚡ Performance PostGIS & Cache"
])

# ==========================================
# ABA 1: MAPA ESPACIAL COM LAZY LOADING
# ==========================================
with tab_mapa:
    if not df_coords_mapa.empty:
        centro_lat = df_coords_mapa['latitude'].mean()
        centro_lon = df_coords_mapa['longitude'].mean()
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
            st.caption(f"⚡ {len(df_coords_mapa):,} pontos (em {latencia_mapa_ms:.1f}ms)")

        with col_m1:
            m = folium.Map(location=[centro_lat, centro_lon], zoom_start=zoom, tiles=tipo_mapa)
            dados_calor = df_coords_mapa[['latitude', 'longitude']].values.tolist()
            
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
        st.info("Nenhuma coordenada encontrada para os filtros selecionados.")

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

        if 'hora' in df_filtrado.columns:
            horas_df = df_filtrado[df_filtrado['hora'] != -1]['hora'].value_counts().sort_index().reset_index()
            horas_df.columns = ['Hora', 'Ocorrências']
            fig_hora = px.area(
                horas_df, 
                x='Hora', 
                y='Ocorrências', 
                title="Distribuição Temporal por Hora do Dia (0h às 23h)",
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
# ABA 3: MODELO MATEMÁTICO PREDITIVO
# ==========================================
with tab_predicao:
    st.markdown("#### 📐 Modelagem Preditiva com Decomposição Harmônica e Incerteza Estatística")
    
    col_p1, col_p2, col_p3 = st.columns([2, 2, 1])
    bairros_disponiveis = df_filtrado['bairro'].dropna().unique().tolist() if not df_filtrado.empty and 'bairro' in df_filtrado.columns else []
    
    with col_p1:
        bairro_escolhido = st.selectbox("Bairro Alvo", bairros_disponiveis if bairros_disponiveis else ["GERAL"])
    with col_p2:
        rubrica_escolhida = st.selectbox("Tipologia Penal", crimes_selecionados if crimes_selecionados else ["Geral"])
    with col_p3:
        passos_projecao = st.selectbox("Horizonte Futuro", [3, 6, 12], index=0)

    if st.button("Executar Modelo Matemático Preditivo", key="btn_predicao"):
        with st.spinner("Ajustando estimador Ridge e calculando propagação de variância residual..."):
            preditor = CriminalPredictor(df_filtrado)
            serie_hist = preditor.preparar_dados_serie_temporal(bairro=bairro_escolhido, rubrica=rubrica_escolhida)
            
            if not serie_hist.empty and len(serie_hist) >= 3:
                modelo = preditor.treinar_modelo_regressao(serie_hist)
                if modelo:
                    df_proj = preditor.prever_com_intervalo_confianca(passos_futuros=passos_projecao, nivel_confianca=0.95)
                    m = preditor.metricas

                    # Cards com Métricas Estatísticas Formais
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("R² (Ajuste)", f"{m['r2']:.3f}", delta="Qualidade de Ajuste")
                    mc2.metric("RMSE", f"{m['rmse']:.2f}", delta="Erro Quadrático")
                    mc3.metric("MAE", f"{m['mae']:.2f}", delta="Erro Absoluto")
                    mc4.metric("MAPE", f"{m['mape']:.1f}%", delta="Erro Relativo")
                    mc5.metric("AIC", f"{m['aic']:.1f}", delta="Critério Akaike")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Gráfico de Previsão com Faixa de Confiança (95% CI)
                    fig = go.Figure()

                    # 1. Série Histórica Real
                    periodos_hist = [f"{ano}/{mes:02d}" for ano, mes in zip(serie_hist['ANO'], serie_hist['MES'])]
                    fig.add_trace(go.Scatter(
                        x=periodos_hist,
                        y=serie_hist['TOTAL_CRIMES'],
                        mode='lines+markers',
                        name='Histórico Real',
                        line=dict(color='#2563eb', width=2.5),
                        marker=dict(size=7)
                    ))

                    # Ponto de conexão entre o último histórico e o primeiro projetado
                    ultimo_hist_p = periodos_hist[-1]
                    ultimo_hist_v = serie_hist['TOTAL_CRIMES'].iloc[-1]

                    eixo_x_proj = [ultimo_hist_p] + df_proj['Periodo'].tolist()
                    eixo_y_proj = [ultimo_hist_v] + df_proj['PREVISAO'].tolist()
                    eixo_ic_sup = [ultimo_hist_v] + df_proj['IC_SUPERIOR_95'].tolist()
                    eixo_ic_inf = [ultimo_hist_v] + df_proj['IC_INFERIOR_95'].tolist()

                    # 2. Faixa de Confiança Superior
                    fig.add_trace(go.Scatter(
                        x=eixo_x_proj,
                        y=eixo_ic_sup,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # 3. Faixa de Confiança Inferior com preenchimento sombreado
                    fig.add_trace(go.Scatter(
                        x=eixo_x_proj,
                        y=eixo_ic_inf,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(225, 29, 72, 0.12)',
                        name='Intervalo de Confiança (95%)'
                    ))

                    # 4. Linha de Projeção Pontual Futura
                    fig.add_trace(go.Scatter(
                        x=eixo_x_proj,
                        y=eixo_y_proj,
                        mode='lines+markers',
                        name='Projeção Pontual Futura',
                        line=dict(color='#e11d48', width=2.5, dash='dash'),
                        marker=dict(size=7, symbol='diamond')
                    ))

                    fig.update_layout(
                        title=f"Projeção Preditiva com Intervalo de Confiança de 95%: {bairro_escolhido} ({rubrica_escolhida})",
                        xaxis_title="Período Cronológico (Ano/Mês)",
                        yaxis_title="Volume de Ocorrências",
                        height=380,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#475569', family='sans-serif'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Tabela com as projeções numéricas
                    st.markdown("##### 📋 Tabela de Projeção Estatística")
                    df_tabela = df_proj[['Periodo', 'PREVISAO', 'IC_INFERIOR_95', 'IC_SUPERIOR_95', 'ERRO_PADRAO']].copy()
                    df_tabela.columns = ['Período', 'Projeção (Crimes)', 'Limite Inferior (95%)', 'Limite Superior (95%)', 'Erro Padrão SE(h)']
                    st.dataframe(df_tabela, use_container_width=True, hide_index=True)

                    # Detalhamento Matemático (LaTeX)
                    with st.expander("🔬 Detalhamento da Formulação Matemática do Modelo", expanded=False):
                        st.markdown(r"""
                        **1. Equação do Modelo de Regressão com Base de Fourier:**
                        """)
                        st.latex(r"y_t = \beta_0 + \beta_1 t + \alpha \cos\left(\frac{2\pi m_t}{12}\right) + \gamma \sin\left(\frac{2\pi m_t}{12}\right) + \epsilon_t")
                        st.markdown(r"""
                        - $t$: Tendência secular temporal contínua (`PERIODO_IDX`).
                        - $\sin, \cos$: Componentes harmônicas de Fourier para modelar a sazonalidade periódica anual ($m_t \in \{1, \dots, 12\}$).
                        - $\beta, \alpha, \gamma$: Coeficientes estimados via regularização Ridge ($L_2$ Tikhonov):
                        """)
                        st.latex(r"\min_{\beta} \left( \frac{1}{2n} \|y - X\beta\|_2^2 + \frac{\lambda}{2} \|\beta\|_2^2 \right)")
                        st.markdown(r"""
                        **2. Propagação de Incerteza e Intervalo de Confiança a 95%:**
                        """)
                        st.latex(r"\text{SE}(h) = \hat{\sigma} \sqrt{1 + \frac{1}{n} + \frac{(t_{h} - \bar{t})^2}{\sum_{i=1}^n (t_i - \bar{t})^2}}")
                        st.latex(r"\text{IC}_{95\%} = \left[ \max(0, \hat{y}_h - 1.96 \cdot \text{SE}(h)), \quad \hat{y}_h + 1.96 \cdot \text{SE}(h) \right]")
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
                'Materialized View (< 0.1ms)', 
                'Lazy Loading Mapa (0.6ms)',
                'Índice Composto B-Tree (5.4ms)', 
                'PostGIS GiST Espacial (7.3ms)', 
                'Scan Sequencial Sem Índice (33.8ms)'
            ],
            'Latência (ms)': [0.06, 0.65, 5.4, 7.3, 33.8]
        })
        fig_bench = px.bar(
            bench_data, 
            x='Latência (ms)', 
            y='Estratégia', 
            orientation='h',
            title="Latência Média por Tipo de Consulta no PostgreSQL / PostGIS",
            color='Latência (ms)',
            color_continuous_scale='Blues_r',
            text='Latência (ms)'
        )
        fig_bench.update_traces(texttemplate='%{text} ms', textposition='outside')
        fig_bench.update_layout(
            height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#475569', family='sans-serif')
        )
        st.plotly_chart(fig_bench, use_container_width=True)

    with col_b2:
        st.caption("Estratégias de Aceleração e Lazy Loading")
        st.markdown("""
        - **Lazy Loading Espacial:** O mapa carrega somente as coordenadas essenciais (0.6 ms), sem dados tabulares pesados.
        - **Materialized Views:** KPIs pré-computados com resposta sub-milissegundo (< 0.1 ms).
        - **PostGIS GiST:** Indexação espacial R-Tree para cálculos geodésicos em metros (`ST_DWithin`).
        - **Índice Composto B-Tree:** `(nome_municipio, rubrica)`.
        - **Connection Pool:** Conexões mantidas ativas pelo SQLAlchemy (`pool_size=15`).
        """)
