import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# Configuração da página
# -------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard de Análise de Tarefas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Mapeamento de colunas
# -------------------------------------------------
COLUMN_MAPPING = {
    'Quadro': 'Quadro',
    'Cliente': 'Cliente',
    'Grupo': 'Grupo',
    'Projeto': 'Projeto',
    'ID da tarefa principal': 'ID_Tarefa',
    'Título da tarefa principal': 'Titulo_Tarefa',
    'Tipo de tarefa': 'Tipo_Tarefa',
    'Equipe': 'Equipe',
    'Centro de custo': 'CC',
    'Para': 'Para',
    'ID da Tarefa': 'ID_Tarefa_Secundaria',
    'Tarefa': 'Tarefa',
    'Urgente': 'Urgente',
    'Prioridade': 'Prioridade',
    'Aberta por': 'Tarefa_Aberta',
    'Criada em': 'Tarefa_Criada',
    'Entrega desejada': 'Tarefa_Entrega_Desejada',
    'Entrega estimada': 'Tarefa_Entrega_Estimada',
    'Fechada em': 'Tarefa_Fechada',
    'Esforço estimado h': 'Tarefa_Esforco_Estimado',
    'Primeiro esforço estimado h': 'Tarefa_Esforco_Primeiro',
    'Já registradas h': 'Tarefa_Esforco_Registradas',
    'Já registradas em subtarefas': 'Tarefa_Esforco_Registradas_Sub',
    '%': 'Tarefa_Esforco_Percentual',
    'Etapa': 'Etapa',
    'Fase': 'Fase',
    'Reaberta?': 'Tarefa_Reaberta',
    'Tags': 'Tags',
    'Código customizado de cliente': 'Codigo_Cliente',
    'Horas restantes h': 'Horas_Restantes'
}

# -------------------------------------------------
# Funções auxiliares
# -------------------------------------------------
@st.cache_data
def load_uploaded_file(file) -> pd.DataFrame:
    """Lê CSV ou Excel enviado pelo usuário."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file, encoding='utf-8', encoding_errors='ignore')
    return pd.read_excel(file)

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e padroniza colunas básicas."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAPPING)

    # remove colunas totalmente vazias
    df = df.dropna(axis=1, how='all')

    # trata valores estranhos
    df = df.replace(['-', 'NaN', 'nan', ''], np.nan)

    # datas
    date_cols = [
        'Tarefa_Criada',
        'Tarefa_Entrega_Desejada',
        'Tarefa_Entrega_Estimada',
        'Tarefa_Fechada'
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # numéricos
    num_cols = [
        'Tarefa_Esforco_Estimado',
        'Tarefa_Esforco_Primeiro',
        'Tarefa_Esforco_Registradas',
        'Tarefa_Esforco_Registradas_Sub',
        'Tarefa_Esforco_Percentual',
        'Horas_Restantes'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # booleanos
    if 'Urgente' in df.columns:
        df['Urgente'] = df['Urgente'].map({'Sim': True, 'Não': False}).fillna(False)

    if 'Tarefa_Reaberta' in df.columns:
        df['Tarefa_Reaberta'] = df['Tarefa_Reaberta'].map({'Sim': True, 'Não': False}).fillna(False)
    
    # Prioridade - converter para categórica ordenada
    if 'Prioridade' in df.columns:
        priority_order = {'Baixa': 1, 'Média': 2, 'Alta': 3, 'Urgente': 4}
        df['Prioridade_Num'] = df['Prioridade'].map(priority_order)
    
    return df

def adicionar_colunas_analise(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas específicas das análises obrigatórias."""
    df = df.copy()

    # SLA / Distância em dias (Entrega desejada × Fechada)
    if {'Tarefa_Entrega_Desejada', 'Tarefa_Fechada'}.issubset(df.columns):
        df['SLA_Dias'] = (df['Tarefa_Fechada'] - df['Tarefa_Entrega_Desejada']).dt.days
        df['Distancia_Dias'] = df['SLA_Dias']
        
        # Classificar SLA
        df['SLA_Status'] = np.where(
            df['SLA_Dias'] > 0,
            "Atrasada",
            np.where(df['SLA_Dias'].isna(), "Sem data", "No prazo")
        )

    # Eficiência de esforço (%)
    if {'Tarefa_Esforco_Registradas', 'Tarefa_Esforco_Estimado'}.issubset(df.columns):
        df['Eficiencia'] = (
            df['Tarefa_Esforco_Registradas'] / df['Tarefa_Esforco_Estimado']
        ) * 100
        
        # Classificar eficiência
        conditions = [
            df['Eficiencia'] < 50,
            df['Eficiencia'] <= 100,
            df['Eficiencia'] > 100
        ]
        choices = ['Baixa', 'Normal', 'Alta']
        df['Eficiencia_Categoria'] = np.select(conditions, choices, default='Normal')

    return df

def calcular_metricas(df: pd.DataFrame) -> dict:
    """Calcula métricas gerais e tempos por tipo/cliente."""
    m = {}

    m['total_tarefas'] = len(df)
    
    if 'Tarefa_Reaberta' in df.columns:
        m['tarefas_reabertas'] = df['Tarefa_Reaberta'].sum()
        m['perc_reabertas'] = df['Tarefa_Reaberta'].mean() * 100
    else:
        m['tarefas_reabertas'] = 0
        m['perc_reabertas'] = 0

    if 'Tarefa_Esforco_Registradas' in df.columns:
        m['total_horas'] = df['Tarefa_Esforco_Registradas'].sum()
        m['media_horas_por_tarefa'] = df['Tarefa_Esforco_Registradas'].mean()
    else:
        m['total_horas'] = 0
        m['media_horas_por_tarefa'] = 0
    
    if 'Eficiencia' in df.columns:
        m['eficiencia_media'] = df['Eficiencia'].mean()
        m['outliers_eficiencia'] = len(df[(df['Eficiencia'] > 100) | (df['Eficiencia'] < 50)])
    
    if 'SLA_Status' in df.columns:
        sla_stats = df['SLA_Status'].value_counts(normalize=True) * 100
        m['sla_no_prazo'] = sla_stats.get("No prazo", 0)
        m['sla_atrasadas'] = sla_stats.get("Atrasada", 0)

    # tempo médio geral (Criada x Fechada)
    m['tempo_medio_dias'] = None
    if {'Tarefa_Criada', 'Tarefa_Fechada'}.issubset(df.columns):
        fechadas = df.dropna(subset=['Tarefa_Criada', 'Tarefa_Fechada']).copy()
        if not fechadas.empty:
            dias = (fechadas['Tarefa_Fechada'] - fechadas['Tarefa_Criada']).dt.days
            m['tempo_medio_dias'] = dias.mean()
            m['tempo_mediano_dias'] = dias.median()

    return m

# -------------------------------------------------
# Funções para criar gráficos com Altair (ATUALIZADAS)
# -------------------------------------------------
def criar_grafico_barras_horizontais(df, x_col, y_col, title, limit=10, color="#4CAF50"):
    """Cria gráfico de barras horizontais."""
    if y_col not in df.columns:
        return None
    
    top_data = df[y_col].value_counts().head(limit).reset_index()
    top_data.columns = ['Categoria', 'Quantidade']
    
    chart = alt.Chart(top_data).mark_bar(color=color).encode(
        x=alt.X('Quantidade:Q', title='Quantidade'),
        y=alt.Y('Categoria:N', sort='-x', title='Categoria')
    ).properties(
        title=title,
        height=300
    )
    
    return chart

def criar_grafico_barras(df, x_col, y_col, title, color="#2196F3"):
    """Cria gráfico de barras verticais."""
    if x_col not in df.columns or y_col not in df.columns:
        return None
    
    chart_data = df.groupby(x_col)[y_col].sum().reset_index()
    
    chart = alt.Chart(chart_data).mark_bar(color=color).encode(
        x=alt.X(f'{x_col}:N', title=x_col, axis=alt.Axis(labelAngle=45)),
        y=alt.Y(f'{y_col}:Q', title=y_col)
    ).properties(
        title=title,
        height=300
    )
    
    return chart

def criar_grafico_pizza(df, col, title, colors=None):
    """Cria gráfico de pizza/donut."""
    if col not in df.columns:
        return None
    
    chart_data = df[col].value_counts().reset_index()
    chart_data.columns = ['Categoria', 'Quantidade']
    
    # Usar cores personalizadas se fornecidas
    if colors and len(colors) >= len(chart_data):
        color_scale = alt.Scale(
            domain=chart_data['Categoria'].tolist(),
            range=colors[:len(chart_data)]
        )
    else:
        # Cores padrão
        color_scale = alt.Scale(scheme='category10')
    
    chart = alt.Chart(chart_data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Quantidade", type="quantitative"),
        color=alt.Color(field="Categoria", type="nominal", scale=color_scale),
        tooltip=['Categoria', 'Quantidade']
    ).properties(
        title=title,
        height=300,
        width=300
    )
    
    return chart

def criar_grafico_dispersao(df, x_col, y_col, color_col, title, color="#FF5722"):
    """Cria gráfico de dispersão."""
    if x_col not in df.columns or y_col not in df.columns:
        return None
    
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(f'{x_col}:Q', title=x_col),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        color=alt.Color(f'{color_col}:N', title=color_col) if color_col in df.columns else alt.value(color),
        tooltip=[x_col, y_col, color_col] if color_col in df.columns else [x_col, y_col]
    ).properties(
        title=title,
        height=400
    )
    
    return chart

def criar_histograma(df, col, title, bins=30, color="#9C27B0"):
    """Cria histograma."""
    if col not in df.columns:
        return None
    
    chart = alt.Chart(df).mark_bar(color=color).encode(
        alt.X(f'{col}:Q', bin=alt.Bin(maxbins=bins), title=col),
        alt.Y('count()', title='Frequência')
    ).properties(
        title=title,
        height=300
    )
    
    return chart

# -------------------------------------------------
# Header Principal
# -------------------------------------------------
st.title("📊 Dashboard de Análise de Tarefas")
st.markdown("---")

# -------------------------------------------------
# Upload na barra lateral
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📁 Envie um arquivo",
        type=["csv", "xlsx"],
        help="Arquivos CSV ou Excel com dados de tarefas"
    )
    
# Inicializar variáveis de filtro
filtro_cliente = None
filtro_tipo = None
filtro_prioridade = None

# Inicializar cores com valores padrão
cor_primaria = "#2196F3"
cor_secundaria = "#4CAF50"

if uploaded_file:
    # Carregar e preparar dados UMA VEZ
    df_raw = load_uploaded_file(uploaded_file)
    df_base = preparar_dados(df_raw)
    df_base = adicionar_colunas_analise(df_base)
    
    with st.sidebar:
        st.success("✅ Arquivo carregado com sucesso!")
        
        st.markdown("---")
        st.header("🔍 Filtros")
        
        # Filtros usando df_base para as opções
        if 'Cliente' in df_base.columns:
            clientes = ['Todos'] + sorted(df_base['Cliente'].dropna().unique().tolist())
            filtro_cliente = st.selectbox("Cliente", clientes)
        
        if 'Tipo_Tarefa' in df_base.columns:
            tipos = ['Todos'] + sorted(df_base['Tipo_Tarefa'].dropna().unique().tolist())
            filtro_tipo = st.selectbox("Tipo de Tarefa", tipos)
        
        if 'Prioridade' in df_base.columns:
            prioridades = ['Todos'] + sorted(df_base['Prioridade'].dropna().unique().tolist())
            filtro_prioridade = st.selectbox("Prioridade", prioridades)
        
        st.markdown("---")
        st.header("🎨 Configurações de Cores")
        
        # Cores personalizadas que serão usadas em TODOS os gráficos
        cor_primaria = st.color_picker("Cor Primária", "#2196F3", 
                                      help="Cor para gráficos principais")
        cor_secundaria = st.color_picker("Cor Secundária", "#4CAF50", 
                                        help="Cor para gráficos secundários")
        cor_terciaria = st.color_picker("Cor Terciária", "#FF5722", 
                                       help="Cor para gráficos complementares")
        
        # Paleta de cores para gráficos de pizza
        st.markdown("**Paleta para gráficos de categoria:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            cor_cat1 = st.color_picker("Cat 1", "#4CAF50", key="cat1")
        with col2:
            cor_cat2 = st.color_picker("Cat 2", "#2196F3", key="cat2")
        with col3:
            cor_cat3 = st.color_picker("Cat 3", "#FF9800", key="cat3")
        
        paleta_cores = [cor_cat1, cor_cat2, cor_cat3, "#9C27B0", "#F44336", "#00BCD4"]
        
        st.markdown("---")
        st.header("ℹ️ Sobre")
        st.info("""
        Análise baseada na Arquitetura Medalhão:
        - Camada Bronze: Dados brutos
        - Camada Prata: Dados tratados
        - Camada Ouro: Indicadores e métricas
        """)
    
    # APLICAR FILTROS ao DataFrame
    df_filtrado = df_base.copy()
    
    if filtro_cliente and filtro_cliente != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Cliente'] == filtro_cliente]
    
    if filtro_tipo and filtro_tipo != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Tipo_Tarefa'] == filtro_tipo]
    
    if filtro_prioridade and filtro_prioridade != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Prioridade'] == filtro_prioridade]
    
    # Calcular métricas com dados FILTRADOS
    metricas = calcular_metricas(df_filtrado)
    
    # Mostrar filtros aplicados
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Filtros Aplicados")
    filtros_ativos = []
    if filtro_cliente and filtro_cliente != 'Todos':
        filtros_ativos.append(f"**Cliente:** {filtro_cliente}")
    if filtro_tipo and filtro_tipo != 'Todos':
        filtros_ativos.append(f"**Tipo:** {filtro_tipo}")
    if filtro_prioridade and filtro_prioridade != 'Todos':
        filtros_ativos.append(f"**Prioridade:** {filtro_prioridade}")
    
    if filtros_ativos:
        for filtro in filtros_ativos:
            st.sidebar.info(filtro)
        st.sidebar.metric("Tarefas Filtradas", len(df_filtrado))
    else:
        st.sidebar.info("Nenhum filtro aplicado")
    
else:
    st.info("👈 Use a barra lateral para enviar um arquivo CSV ou Excel com dados de tarefas para iniciar a análise.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Tarefas", "0")
    with col2:
        st.metric("Horas Totais", "0 h")
    with col3:
        st.metric("SLA no Prazo", "0%")
    with col4:
        st.metric("Eficiência Média", "0%")
    
    st.stop()

# -------------------------------------------------
# Visão geral (métricas) - AGORA COM DADOS FILTRADOS
# -------------------------------------------------
st.header("📈 Visão Geral")

# Mostrar informações dos filtros
if filtros_ativos:
    st.info(f"**Filtros ativos:** {', '.join([f.split(':')[1].strip() for f in filtros_ativos])}")

# Mostrar paleta de cores ativa
with st.expander("🎨 Paleta de Cores Ativa", expanded=False):
    col_cor1, col_cor2, col_cor3 = st.columns(3)
    with col_cor1:
        st.color_picker("Primária", cor_primaria, disabled=True)
    with col_cor2:
        st.color_picker("Secundária", cor_secundaria, disabled=True)
    with col_cor3:
        st.color_picker("Terciária", "#FF5722", disabled=True)

# Primeira linha de métricas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total de Tarefas",
        value=f"{metricas['total_tarefas']:,}",
        help="Número total de tarefas no dataset"
    )

with col2:
    st.metric(
        label="Horas Registradas",
        value=f"{metricas['total_horas']:,.1f} h",
        help="Total de horas registradas em todas as tarefas"
    )

with col3:
    if metricas['tempo_medio_dias'] is not None:
        valor = f"{metricas['tempo_medio_dias']:.1f} dias"
    else:
        valor = "—"
    st.metric(
        label="Tempo Médio",
        value=valor,
        help="Tempo médio entre criação e fechamento"
    )

with col4:
    st.metric(
        label="Tarefas Reabertas",
        value=f"{metricas['perc_reabertas']:.1f}%",
        help="Percentual de tarefas que foram reabertas"
    )

# Segunda linha de métricas
col5, col6, col7, col8 = st.columns(4)

with col5:
    if 'eficiencia_media' in metricas:
        st.metric(
            label="Eficiência Média",
            value=f"{metricas['eficiencia_media']:.1f}%",
            help="Média da eficiência (horas registradas/estimadas)"
        )

with col6:
    if 'sla_no_prazo' in metricas:
        st.metric(
            label="SLA no Prazo",
            value=f"{metricas['sla_no_prazo']:.1f}%",
            help="Percentual de tarefas entregues no prazo"
        )

with col7:
    if 'outliers_eficiencia' in metricas:
        st.metric(
            label="Outliers de Eficiência",
            value=f"{metricas['outliers_eficiencia']}",
            help="Tarefas com eficiência <50% ou >100%"
        )

with col8:
    if 'media_horas_por_tarefa' in metricas:
        st.metric(
            label="Média Horas/Tarefa",
            value=f"{metricas['media_horas_por_tarefa']:.1f} h",
            help="Média de horas por tarefa"
        )

st.markdown("---")

# -------------------------------------------------
# Gráficos principais - Primeira linha
# -------------------------------------------------
st.header("📊 Gráficos Principais")

col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    # 1. Tarefas reabertas (Pizza)
    if 'Tarefa_Reaberta' in df_filtrado.columns:
        st.subheader("🔄 Tarefas Reabertas")
        
        # Converter booleanos para texto para melhor visualização
        df_pizza = df_filtrado.copy()
        df_pizza['Tarefa_Reaberta_Texto'] = df_pizza['Tarefa_Reaberta'].map({True: 'Reabertas', False: 'Não Reabertas'})
        
        pizza_chart = criar_grafico_pizza(df_pizza, 'Tarefa_Reaberta_Texto', 
                                         'Distribuição de Tarefas Reabertas',
                                         colors=[cor_primaria, cor_secundaria])
        if pizza_chart:
            st.altair_chart(pizza_chart, use_container_width=True)
        
        # Estatísticas
        reabertas_stats = df_filtrado['Tarefa_Reaberta'].value_counts()
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.info(f"**Não Reabertas:** {reabertas_stats.get(False, 0):,}")
        with col_stat2:
            st.warning(f"**Reabertas:** {reabertas_stats.get(True, 0):,}")

with col_graf2:
    # 2. Horas por Cliente (Top 10) - AGORA COM FILTROS
    if {'Cliente', 'Tarefa_Esforco_Registradas'}.issubset(df_filtrado.columns):
        st.subheader("🏢 Top 10 Clientes por Horas")
        
        # Se já está filtrado por um cliente específico, mostrar apenas ele
        if filtro_cliente and filtro_cliente != 'Todos':
            cliente_especifico = df_filtrado[df_filtrado['Cliente'] == filtro_cliente]
            if not cliente_especifico.empty:
                horas = cliente_especifico['Tarefa_Esforco_Registradas'].sum()
                st.info(f"**Cliente selecionado:** {filtro_cliente}")
                st.metric("Total de Horas", f"{horas:,.1f} h")
        else:
            # Mostrar top 10 clientes
            top_clientes = df_filtrado.groupby('Cliente')['Tarefa_Esforco_Registradas'].sum().nlargest(10).reset_index()
            
            if not top_clientes.empty:
                chart = alt.Chart(top_clientes).mark_bar(color=cor_primaria).encode(
                    x=alt.X('Tarefa_Esforco_Registradas:Q', title='Horas Registradas'),
                    y=alt.Y('Cliente:N', sort='-x', title='Cliente'),
                    tooltip=['Cliente', 'Tarefa_Esforco_Registradas']
                ).properties(
                    height=400
                )
                
                st.altair_chart(chart, use_container_width=True)

# -------------------------------------------------
# Gráficos principais - Segunda linha
# -------------------------------------------------
col_graf3, col_graf4 = st.columns(2)

with col_graf3:
    # 3. Horas por Equipe
    if {'Equipe', 'Tarefa_Esforco_Registradas'}.issubset(df_filtrado.columns):
        st.subheader("👥 Horas por Equipe")
        
        horas_equipe = df_filtrado.groupby('Equipe')['Tarefa_Esforco_Registradas'].sum().reset_index()
        
        if not horas_equipe.empty:
            chart = alt.Chart(horas_equipe).mark_bar(color=cor_secundaria).encode(
                x=alt.X('Equipe:N', title='Equipe', axis=alt.Axis(labelAngle=45)),
                y=alt.Y('Tarefa_Esforco_Registradas:Q', title='Horas Registradas'),
                tooltip=['Equipe', 'Tarefa_Esforco_Registradas']
            ).properties(
                height=300
            )
            
            st.altair_chart(chart, use_container_width=True)

with col_graf4:
    # 4. Distribuição por Prioridade
    if 'Prioridade' in df_filtrado.columns:
        st.subheader("🎯 Distribuição por Prioridade")
        
        # Se já está filtrado por uma prioridade específica
        if filtro_prioridade and filtro_prioridade != 'Todos':
            st.info(f"**Prioridade selecionada:** {filtro_prioridade}")
            contagem = len(df_filtrado[df_filtrado['Prioridade'] == filtro_prioridade])
            st.metric("Tarefas com esta prioridade", contagem)
        else:
            # Gráfico de barras com cores da paleta
            prioridade_data = df_filtrado['Prioridade'].value_counts().reset_index()
            prioridade_data.columns = ['Prioridade', 'Quantidade']
            
            if not prioridade_data.empty:
                chart = alt.Chart(prioridade_data).mark_bar().encode(
                    x=alt.X('Prioridade:N', title='Prioridade', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Quantidade:Q', title='Quantidade de Tarefas'),
                    color=alt.Color('Prioridade:N', scale=alt.Scale(
                        domain=prioridade_data['Prioridade'].tolist(),
                        range=paleta_cores[:len(prioridade_data)]
                    )),
                    tooltip=['Prioridade', 'Quantidade']
                ).properties(
                    height=300,
                    title='Quantidade por Prioridade'
                )
                
                st.altair_chart(chart, use_container_width=True)
        
        # Estatísticas de prioridade
        prioridades = df_filtrado['Prioridade'].value_counts()
        col_pri1, col_pri2, col_pri3 = st.columns(3)
        with col_pri1:
            st.error(f"**Alta:** {prioridades.get('Alta', 0):,}")
        with col_pri2:
            st.warning(f"**Média:** {prioridades.get('Média', 0):,}")
        with col_pri3:
            st.success(f"**Baixa:** {prioridades.get('Baixa', 0):,}")

st.markdown("---")

# -------------------------------------------------
# Análise de SLA
# -------------------------------------------------
st.header("⏱️ Análise de SLA")

if 'SLA_Status' in df_filtrado.columns:
    col_sla1, col_sla2 = st.columns(2)
    
    with col_sla1:
        st.subheader("📊 Status do SLA")
        
        sla_data = df_filtrado['SLA_Status'].value_counts().reset_index()
        sla_data.columns = ['Status', 'Quantidade']
        
        if not sla_data.empty:
            # Mapa de cores usando a paleta
            cores_sla = [cor_secundaria, "#F44336", "#FF9800"]
            color_scale = alt.Scale(
                domain=['No prazo', 'Atrasada', 'Sem data'],
                range=cores_sla
            )
            
            chart = alt.Chart(sla_data).mark_arc(innerRadius=50).encode(
                theta='Quantidade:Q',
                color=alt.Color('Status:N', scale=color_scale),
                tooltip=['Status', 'Quantidade']
            ).properties(
                height=300,
                width=300
            )
            
            st.altair_chart(chart, use_container_width=True)
    
    with col_sla2:
        st.subheader("📈 Distribuição do SLA (Dias)")
        
        if 'SLA_Dias' in df_filtrado.columns:
            df_sla = df_filtrado.dropna(subset=['SLA_Dias']).copy()
            
            if not df_sla.empty:
                # Filtrar outliers extremos para melhor visualização
                df_sla_filtered = df_sla[(df_sla['SLA_Dias'] >= -30) & (df_sla['SLA_Dias'] <= 60)]
                
                chart = criar_histograma(df_sla_filtered, 'SLA_Dias', 
                                        'Distribuição do SLA (entre -30 e 60 dias)', 
                                        bins=30, color=cor_primaria)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
                
                # Estatísticas do SLA
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Média", f"{df_sla['SLA_Dias'].mean():.1f} dias")
                with col_stat2:
                    st.metric("Mediana", f"{df_sla['SLA_Dias'].median():.1f} dias")
                with col_stat3:
                    atrasadas = len(df_sla[df_sla['SLA_Dias'] > 0])
                    st.metric("Atrasadas", f"{atrasadas:,}")

# -------------------------------------------------
# Análise de Eficiência
# -------------------------------------------------
st.markdown("---")
st.header("🎯 Análise de Eficiência")

if 'Eficiencia' in df_filtrado.columns:
    col_eff1, col_eff2 = st.columns(2)
    
    with col_eff1:
        st.subheader("📊 Categorias de Eficiência")
        
        if 'Eficiencia_Categoria' in df_filtrado.columns:
            eff_data = df_filtrado['Eficiencia_Categoria'].value_counts().reset_index()
            eff_data.columns = ['Categoria', 'Quantidade']
            
            if not eff_data.empty:
                color_scale = alt.Scale(
                    domain=['Baixa', 'Normal', 'Alta'],
                    range=[cor_secundaria, cor_primaria, "#FF5722"]
                )
                
                chart = alt.Chart(eff_data).mark_bar().encode(
                    x=alt.X('Categoria:N', title='Categoria'),
                    y=alt.Y('Quantidade:Q', title='Quantidade de Tarefas'),
                    color=alt.Color('Categoria:N', scale=color_scale, legend=None),
                    tooltip=['Categoria', 'Quantidade']
                ).properties(
                    height=300
                )
                
                st.altair_chart(chart, use_container_width=True)
    
    with col_eff2:
        st.subheader("📈 Distribuição da Eficiência")
        
        chart = criar_histograma(df_filtrado, 'Eficiencia', 
                                'Distribuição da Eficiência (%)', 
                                bins=30, color=cor_primaria)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        
        # Outliers
        outliers = df_filtrado[(df_filtrado['Eficiencia'] > 100) | (df_filtrado['Eficiencia'] < 50)]
        if len(outliers) > 0:
            st.warning(f"**Outliers detectados:** {len(outliers):,} tarefas")

# -------------------------------------------------
# Análise de Tempo
# -------------------------------------------------
st.markdown("---")
st.header("⏰ Análise de Tempo")

col_time1, col_time2 = st.columns(2)

with col_time1:
    # Tempo por Tipo de Tarefa
    if {'Tipo_Tarefa', 'Tarefa_Criada', 'Tarefa_Fechada'}.issubset(df_filtrado.columns):
        st.subheader("⏱️ Tempo por Tipo de Tarefa")
        
        tmp = df_filtrado.dropna(subset=['Tarefa_Criada', 'Tarefa_Fechada']).copy()
        
        if not tmp.empty:
            tmp['Dias'] = (tmp['Tarefa_Fechada'] - tmp['Tarefa_Criada']).dt.days
            
            tempo_tipo = tmp.groupby('Tipo_Tarefa')['Dias'].mean().sort_values(ascending=False).head(10).reset_index()
            
            if not tempo_tipo.empty:
                chart = alt.Chart(tempo_tipo).mark_bar(color="#9C27B0").encode(
                    x=alt.X('Dias:Q', title='Dias Médios'),
                    y=alt.Y('Tipo_Tarefa:N', sort='-x', title='Tipo de Tarefa'),
                    tooltip=['Tipo_Tarefa', 'Dias']
                ).properties(
                    height=400
                )
                
                st.altair_chart(chart, use_container_width=True)

with col_time2:
    # Tempo por Cliente
    if {'Cliente', 'Tarefa_Criada', 'Tarefa_Fechada'}.issubset(df_filtrado.columns):
        st.subheader("🏢 Tempo por Cliente")
        
        tmp = df_filtrado.dropna(subset=['Tarefa_Criada', 'Tarefa_Fechada']).copy()
        
        if not tmp.empty:
            tmp['Dias'] = (tmp['Tarefa_Fechada'] - tmp['Tarefa_Criada']).dt.days
            
            tempo_cliente = tmp.groupby('Cliente')['Dias'].mean().sort_values(ascending=False).head(10).reset_index()
            
            if not tempo_cliente.empty:
                chart = alt.Chart(tempo_cliente).mark_bar(color=cor_secundaria).encode(
                    x=alt.X('Dias:Q', title='Dias Médios'),
                    y=alt.Y('Cliente:N', sort='-x', title='Cliente'),
                    tooltip=['Cliente', 'Dias']
                ).properties(
                    height=400
                )
                
                st.altair_chart(chart, use_container_width=True)

# -------------------------------------------------
# Tabela de dados
# -------------------------------------------------
st.markdown("---")
st.header("📋 Dados Detalhados")

with st.expander("🔍 Ver dados completos", expanded=False):
    # Filtros para a tabela
    col_filt1, col_filt2 = st.columns(2)
    with col_filt1:
        mostrar_colunas = st.multiselect(
            "Selecione colunas para exibir",
            options=df_filtrado.columns.tolist(),
            default=df_filtrado.columns.tolist()[:8]
        )
    
    with col_filt2:
        linhas_mostrar = st.slider("Número de linhas para exibir", 10, 100, 50)
    
    # Exibir tabela
    st.dataframe(
        df_filtrado[mostrar_colunas] if mostrar_colunas else df_filtrado.head(linhas_mostrar),
        use_container_width=True,
        height=400
    )

# -------------------------------------------------
# Resumo estatístico
# -------------------------------------------------
st.markdown("---")
st.header("📊 Resumo Estatístico")

col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    st.subheader("📈 Estatísticas Numéricas")
    numeric_cols = df_filtrado.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(df_filtrado[numeric_cols].describe(), use_container_width=True)

with col_res2:
    st.subheader("📋 Contagem por Categoria")
    categorical_cols = df_filtrado.select_dtypes(include=['object', 'bool']).columns.tolist()[:3]
    for col in categorical_cols:
        if col in df_filtrado.columns:
            st.write(f"**{col}:**")
            st.write(df_filtrado[col].value_counts().head(5))

with col_res3:
    st.subheader("📅 Estatísticas de Datas")
    date_cols = df_filtrado.select_dtypes(include=['datetime64']).columns.tolist()
    for col in date_cols[:2]:
        if col in df_filtrado.columns and not df_filtrado[col].isna().all():
            st.write(f"**{col}:**")
            st.write(f"Início: {df_filtrado[col].min().date()}")
            st.write(f"Fim: {df_filtrado[col].max().date()}")

# -------------------------------------------------
# Botão para limpar filtros
# -------------------------------------------------
st.markdown("---")
if st.button("🔄 Limpar Todos os Filtros"):
    st.rerun()

# -------------------------------------------------
# Rodapé
# -------------------------------------------------
st.markdown("---")
st.caption(f"📅 Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.caption(f"📊 Tarefas analisadas: {len(df_filtrado):,} de {len(df_base):,} total")
st.caption("🎨 Cores ativas: " + f"Primária: {cor_primaria}, Secundária: {cor_secundaria}")
st.caption("Dashboard de Análise de Tarefas - Baseado na Arquitetura Medalhão")
