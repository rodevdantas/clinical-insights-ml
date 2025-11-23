import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import datetime

# --- CONFIGURAÇÃO INICIAL E CARREGAMENTO DE DADOS ---

# Diretório base onde está o app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho correto do arquivo final (versão atualizada do projeto)
PATH_DATASET_FINAL = os.path.join(BASE_DIR, "data", "pacientes_engajamento_score.csv")

# Mapeamentos
CLUSTER_RFM_MAP = {
    -1: "Inativo / Sem Histórico (R$ 0)",

    1: "RFM 1 - Valor Alto e Ativo",
    2: "RFM 2 - Valor Relevante e Estável",
    3: "RFM 3 - Valor Moderado e Vulnerável",
    0: "RFM 0 - Baixo Valor e Baixa Atividade",
}
CLUSTER_MAP_ORDER = sorted(CLUSTER_RFM_MAP.keys())

CLUSTER_COLOR_MAP = {
    "RFM 1 - Valor Alto e Ativo": '#1f77b4', # AZUL - VIPS 
    "RFM 0 - Baixo Valor e Baixa Atividade": '#2ca02c', # VERDE - Alerta Leve 
    "RFM 2 - Valor Relevante e Estável": '#ff7f0e', # LARANJA - Alerta Moderado
    "RFM 3 - Valor Moderado e Vulnerável": '#d62728', # VERMELHO - Risco 
    "Inativo / Sem Histórico (R$ 0)": '#cccccc', # Cinza
}

PLANO_MAP = {
    0: 'Plano Popular (R$ 0)',
    1: 'Plano Executivo (R$ 100)',
    2: 'Plano Premium (R$ 500)'
}
PLANO_REVERSE_MAP = {'Popular': 0, 'Executivo': 1, 'Premium': 2}

@st.cache_data
def load_data():
    if not os.path.exists(PATH_DATASET_FINAL):
        st.error(f" O arquivo final não foi encontrado:\n{PATH_DATASET_FINAL}")
        return pd.DataFrame(), 0, 0
    try:
        df_final = pd.read_csv(PATH_DATASET_FINAL)
    except Exception as e:
        st.error(f" Erro ao carregar o dataset_final.csv:\n{e}")
        return pd.DataFrame(), 0, 0

    # Recálculo da idade
    DATA_HOJE_FIXA = pd.to_datetime("2025-05-20")
    df_final['data_nascimento'] = pd.to_datetime(df_final['data_nascimento'], errors='coerce')
    df_final['idade'] = (DATA_HOJE_FIXA - df_final['data_nascimento']).dt.days // 365

    # Mapeia plano
    df_final['plano_saude_cod'] = df_final['plano_saude'].map(PLANO_REVERSE_MAP)
    df_final['plano_desc'] = df_final['plano_saude_cod'].map(PLANO_MAP)

    # Mapeia cluster
    df_final['cluster_rfm_desc'] = df_final['cluster_rfm'].map(CLUSTER_RFM_MAP).fillna("Inativo / Não Classificado")

    # KPIs
    df_ativos = df_final[df_final['valor_monetario'] > 0]
    avg_recency = df_ativos['recencia_dias'].mean().round(0) if not df_ativos.empty else 0
    num_pacientes_ativos = len(df_ativos)

    return df_final, avg_recency, num_pacientes_ativos

# --- ESTRUTURA DO STREAMLIT APP ---

df_dados, avg_recency_ativos, num_pacientes_ativos = load_data()

st.set_page_config(
    page_title="Dashboard de Otimização Clínica (ML)",
    layout="wide"
)

if not df_dados.empty:
    st.title("Dashboard de Otimização de Engajamento Clínico")
    st.subheader("Análise de Clusters e Previsão de Frequência")
    st.markdown("""
    Este dashboard é uma **solução de intervenção de risco** que utiliza Machine Learning (não-supervisionado e regressão) para ajudar a clínica a priorizar quais pacientes precisam de contato imediato.
    O principal objetivo é **transformar a inatividade (Recência) em receita acionável**.

    **Como funciona a análise:**
    1. **Segmentação (K-Means RFM):** Os pacientes são agrupados em perfis com base na Recência (dias sem consulta), Frequência (média de consultas por ano) e Valor Monetário.
    2. **Previsão:** Um modelo de Regressão cria um **Score de Engajamento** (Consultas/Ano Previstas). A variável **Recência** foi identificada com **98.5% de importância** para esta previsão.
    3. **Ação:** Os gráficos e a tabela final combinam esses insights para mostrar quem está sumido e quem tem o maior valor potencial de ser perdido.
    """)
    # --- FIM DA DESCRIÇÃO LONGA ---

    # 1. KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Pacientes Ativos (com consultas)",
        value=f"{num_pacientes_ativos:,}".replace(",", "."),
        delta=f"Total na Base: {len(df_dados):,}".replace(",", "."),
        delta_color="off"
    )
    col2.metric(
        label="Recência Média de ATIVOS (Dias)",
        value=f"{avg_recency_ativos:.0f} dias",
        delta_color="off",
        help="Média de dias desde a última consulta, considerando apenas pacientes que já realizaram consultas (Valor Total > R$0)."
    )
    col3.metric(
        label="Score Médio de Engajamento Previsto",
        value=f"{df_dados['frequencia_prevista_reg'].mean():.2f} consultas/ano",
        delta_color="off",
        help="Frequência anual média de consultas prevista pelo modelo de Regressão."
    )
    st.divider()

    # --- FILTRAGEM DE DADOS PARA O GRÁFICO (Ativos + Sem Histórico) ---
    df_ativos_grafico = df_dados[df_dados['cluster_rfm'] != -1].copy()

    # 2. Gráfico Principal (Dispersão: Recência vs. Monetário com Cluster RFM)
    st.header("1. Risco de Evasão por Cluster")
    st.markdown(
    """
    Este gráfico ilustra como os pacientes ativos se distribuem em termos de **recência** (dias desde a última consulta) e **valor monetário**, separados pelos 4 perfis comportamentais definidos na clusterização RFM.

    Embora as métricas estatísticas (Elbow e Silhouette Score) tenham indicado K=2 como a solução mais compacta, a escolha de K=4 foi mantida por razões de interpretabilidade e estratégia: cada grupo formado apresenta padrões distintos de risco e valor, essenciais para ações diferentes de retenção.

    O eixo Y está em escala logarítmica devido ao tratamento de outliers, evidenciando melhor a separação dos clusters)

    """
)


    # Criando o gráfico interativo com Plotly: Cor pelo NOVO Cluster RFM
    fig = px.scatter(
        df_ativos_grafico,
        x='recencia_dias',
        y='valor_monetario',
        color='cluster_rfm_desc',
        color_discrete_map=CLUSTER_COLOR_MAP,
        log_y=True,
        size='frequencia_prevista_reg',
        hover_data=['nome', 'idade', 'plano_saude', 'frequencia_prevista_reg', 'cluster_rfm_desc'],
        # Adicionando cluster_rfm_desc no hover
        title='Recência vs. Valor Monetário por Cluster'
    )

    # --- AJUSTES FINAIS DE LAYOUT ---
    max_recency_ativa = df_ativos_grafico['recencia_dias'].max() if not df_ativos_grafico.empty else 400
    fig.update_layout(
        xaxis_title="Recência (dias desde a última consulta)",
        yaxis_title="Valor Total Gasto (R$)",
        legend_title="Novo Perfil RFM",
        height=450,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    fig.update_xaxes(range=[0, max_recency_ativa + 10])
    fig.update_yaxes(range=[np.log10(50), np.log10(6000)])
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))

    st.plotly_chart(fig, use_container_width=True)

    # --- NOVO GRÁFICO DE DESTAQUE: SCORE DE ENGAJAMENTO POR CLUSTER ---
    st.header("2. Score de Engajamento por Cluster")
    st.markdown(
    """
    Este gráfico apresenta a **frequência média prevista de consultas/ano** para cada cluster, resultado do modelo de regressão aplicado após a segmentação. Essa métrica indica a tendência anual de retorno dos pacientes.

    Note que alguns clusters possuem valores próximos. Isso significa que o **engajamento previsto** - isoladamente - é quase igual, o que não significa que devem ter tratamentos iguais.

    A diferença real entre os clusters aparece quando combinamos os fatores do modelo RFM:

    - **Recência**: quanto maior, maior o risco de churn.
    - **Valor Monetário**: indica o impacto financeiro de cada perfil.
    - **Frequência histórica**: consistência e padrão de uso ao longo do tempo.

    Assim, mesmo com scores de engajamento quase idênticos, dois clusters podem ter **riscos e prioridades muito diferentes**, pois o risco de churn deriva do conjunto das variáveis RFM e não de um único indicador.

    O modelo de regressão apresentou um **R² em torno de 47%**, um resultado sólido para dados de comportamento humano em saúde, onde há alta variabilidade individual. O objetivo aqui é revelar **tendências médias**, e não prever com precisão cada paciente isoladamente.
    """
)


    # Calculando o Score Médio por Cluster (excluindo Inativos -1)
    df_cluster_score = df_dados[df_dados['cluster_rfm'] != -1].groupby('cluster_rfm_desc')['frequencia_prevista_reg'].mean().reset_index()
    df_cluster_score.columns = ['Perfil RFM', 'Score Médio Previsto (Consultas/Ano)']
    df_cluster_score = df_cluster_score.sort_values(by='Score Médio Previsto (Consultas/Ano)', ascending=False)

    # Criando o gráfico de barras
    fig_bar = px.bar(
        df_cluster_score,
        x='Perfil RFM',
        y='Score Médio Previsto (Consultas/Ano)',
        color='Perfil RFM',
        color_discrete_map=CLUSTER_COLOR_MAP,
        text='Score Médio Previsto (Consultas/Ano)',
        title='Score Médio de Engajamento por Cluster'
    )

    # Formatação do texto no topo das barras
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.update_yaxes(title='N° de Consultas Previstas', range=[0, df_cluster_score['Score Médio Previsto (Consultas/Ano)'].max() * 1.1])
    fig_bar.update_layout(height=450, margin=dict(t=50, b=0, l=0, r=0))

    st.plotly_chart(fig_bar, use_container_width=True)
    st.divider()  # Adiciona um divisor após a seção 1 completa

    # 3. Análise Acionável: Tabela de Pacientes Críticos
    st.header("3. Top 10 Pacientes com Maior Inatividade e Maior Valor Histórico")
    st.markdown("""
Esta lista apresenta os pacientes ordenados pela combinação de:

- **Maior tempo de inatividade (Recência)**  
- **Maior Valor Monetário Histórico**

A ordenação reflete apenas esses dois critérios, sem representar risco estatístico ou prioridade clínica.
O objetivo é facilitar análises operacionais dentro de cada Perfil RFM.
""")

    # Filtro de Seleção do NOVO Cluster
    selected_cluster_rfm = st.selectbox(
        "Selecione um Perfil RFM para Análise Detalhada:",
        options=["Todos os Perfis"] + [CLUSTER_RFM_MAP[k] for k in CLUSTER_MAP_ORDER],
        format_func=lambda x: x if x != "Todos os Perfis" else x,
        help="Use os Novos Clusters RFM para focar em grupos com comportamento transacional específico."
    )

    if selected_cluster_rfm != "Todos os Perfis":
        df_filtered = df_dados[df_dados['cluster_rfm_desc'] == selected_cluster_rfm]
    else:
        df_filtered = df_dados.copy()

    # Filtra e Ranqueia: Pacientes que sumiram (Alta Recência) e eram valiosos (Monetário)
    # Excluindo pacientes com recência muito alta que ainda não consultaram (cluster -1)
    df_risco = df_filtered[df_filtered['cluster_rfm'] != -1].sort_values(
        by=['recencia_dias', 'valor_monetario'],
        ascending=[False, False]
    ).head(10)  # <--- CORRIGIDO: Agora exibe os 10 mais críticos

    df_display = df_risco[['id_paciente', 'nome', 'cluster_rfm_desc', 'idade', 'plano_saude', 'recencia_dias', 'valor_monetario', 'frequencia_prevista_reg']].copy()
    df_display.columns = [
        'ID', 'Nome', 'Perfil RFM', 'Idade', 'Plano (Original)',
        'Dias de Inatividade', 'Valor Total Gasto (R$)', 'Score Engajamento (Previsão)'
    ]

    # Formata colunas para melhor visualização
    df_display['Valor Total Gasto (R$)'] = df_display['Valor Total Gasto (R$)'].map('R$ {:,.2f}'.format)
    df_display['Score Engajamento (Previsão)'] = df_display['Score Engajamento (Previsão)'].map('{:.2f} consultas/ano'.format)

    st.dataframe(df_display, use_container_width=True)

    # 4. Conclusão para Aula
    st.markdown("---")
    st.info(
        """
        O projeto busca resolver o problema de evasão (Churn) de pacientes, transformando grandes volumes de dados brutos em uma estratégia de intervenção clara e priorizada.

        **Principais entregas para a tomada de decisão:**
        1. **Fator de Risco Dominante:** O modelo de **Regressão (Random Forest)** confirmou que a **Recência** (tempo sem consulta) é o fator mais importante, contribuindo com **98.5%** para a previsão de engajamento futuro.
        2. **Clusterização:** O modelo **K-Means RFM** criou perfis baseados na matriz **RFM: Recência, Frequência e Valor Monetário**.
           * O gráfico de barras no dashboard permite à gerência alocar recursos de marketing e comunicação exatamente para o perfil de maior potencial de retorno (os Clusters de maior Score).
        3. **Top 10**: A lista de pacientes da seção 3 prioriza os pacientes com a maior Recência.
           * **Ação Prática:** A equipe de Churn deve direcionar o esforço de contato para estes pacientes, **otimizando o recurso** e focando na Recência para converter inatividade em receita.
        """
    )


