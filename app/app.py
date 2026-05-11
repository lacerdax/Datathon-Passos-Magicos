import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pickle
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Configuração da página ────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos — Análise Educacional",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilo CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #6E8FA3;
        margin-bottom: 8px;
    }
    .metric-title { font-size: 13px; color: #666; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
    .metric-delta { font-size: 12px; color: #5DA58C; }
    .risco-alto {
        background: #FEECEC; border-left: 4px solid #E05C5C;
        border-radius: 8px; padding: 12px 16px;
    }
    .risco-medio {
        background: #FEF3E2; border-left: 4px solid #E8934A;
        border-radius: 8px; padding: 12px 16px;
    }
    .risco-baixo {
        background: #E8F5EE; border-left: 4px solid #5DA58C;
        border-radius: 8px; padding: 12px 16px;
    }
    .section-title {
        font-size: 20px; font-weight: 700;
        color: #1a1a2e; margin: 24px 0 12px 0;
        border-bottom: 2px solid #eee; padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────
PEDRA_ORDER  = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
CORES_PEDRA  = {'Quartzo':'#6E8FA3','Ágata':'#5DA58C','Ametista':'#8A6FAC','Topázio':'#D4A847'}
CORES_ANOS   = {2022:'#4A7B9D', 2023:'#6BAE8E', 2024:'#C4873A'}
ANOS         = [2022, 2023, 2024]

FEATURES = ['IAA','IEG','IPS','IPP','IDA','IAN','Mat','Por','Ing',
            'Pedra_num','Genero_num','Inst_num','Defasagem','Ano']

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
SRC_DIR   = os.path.join(BASE_DIR, 'src')

plt.rcParams.update({
    'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'--',
    'figure.facecolor':'white','axes.facecolor':'#FAFAF8','font.size':10
})

# ── Carregamento de dados ─────────────────────────────────────
@st.cache_data
def carregar_dados():
    path = os.path.join(DATA_DIR, 'df_consolidado.pkl')
    if not os.path.exists(path):
        return None
    df = pd.read_pickle(path)
    pedra_map = {'Agata':'Ágata','INCLUIR':'Quartzo'}
    df['Pedra'] = df['Pedra'].replace(pedra_map)
    df['Pedra'] = pd.Categorical(df['Pedra'], categories=PEDRA_ORDER, ordered=True)
    df['Genero'] = df['Genero'].replace({'Menina':'Feminino','Menino':'Masculino'})
    df['em_risco'] = ((df['INDE'] < 6.5) | (df['Defasagem'] <= -2)).astype(int)
    df['Pedra_num'] = df['Pedra'].map({'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4})
    df['Genero_num'] = df['Genero'].map({'Feminino':0,'Masculino':1})
    df['Inst_num'] = 0
    return df

@st.cache_resource
def carregar_modelo():
    path = os.path.join(DATA_DIR, 'modelo_risco_passos.pkl')
    if not os.path.exists(path):
        return None, None
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    return saved['model'], saved['features']

df       = carregar_dados()
modelo, features_modelo = carregar_modelo()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    pagina = st.radio(
        "Navegação",
        ["Visão Geral", "Análise por Indicador", "Previsão de Risco", "Sobre o Projeto"],
        index=0
    )
    st.markdown("---")
    if df is not None:
        st.markdown(f"**Dataset:** {len(df):,} registros")
        st.markdown(f"**Período:** 2022 – 2024")
        st.markdown(f"**Alunos únicos:** {df['RA'].nunique():,}")
    st.markdown("---")
    st.caption("Datathon PosTech · Fase 5")

# ════════════════════════════════════════════════════════════
# PÁGINA 1 — VISÃO GERAL
# ════════════════════════════════════════════════════════════
if pagina == "Visão Geral":
    st.title("Passos Mágicos — Análise Educacional 2022–2024")
    st.markdown("Acompanhamento do desenvolvimento educacional de crianças e jovens em vulnerabilidade social.")

    if df is None:
        st.error("Dataset não encontrado em `data/df_consolidado.pkl`. Execute os notebooks primeiro.")
        st.stop()

    # KPIs
    st.markdown('<div class="section-title">Indicadores Gerais</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    df24 = df[df['Ano']==2024]
    df22 = df[df['Ano']==2022]

    with col1:
        total = len(df24)
        st.metric("Alunos em 2024", f"{total:,}", f"+{total - len(df22):,} vs 2022")
    with col2:
        inde_24 = df24['INDE'].mean()
        inde_22 = df22['INDE'].mean()
        st.metric("INDE médio 2024", f"{inde_24:.2f}", f"+{inde_24-inde_22:.2f} vs 2022")
    with col3:
        topazio_pct = (df24['Pedra']=='Topázio').mean()*100
        topazio_22  = (df22['Pedra']=='Topázio').mean()*100
        st.metric("Em Topázio", f"{topazio_pct:.0f}%", f"+{topazio_pct-topazio_22:.0f}pp vs 2022")
    with col4:
        risco_pct = df24['em_risco'].mean()*100
        risco_22  = df22['em_risco'].mean()*100
        st.metric("Em risco 2024", f"{risco_pct:.1f}%", f"{risco_pct-risco_22:.1f}pp vs 2022",
                  delta_color="inverse")
    with col5:
        ian_ade = (df24['IAN'] >= 7).mean()*100
        ian_22  = (df22['IAN'] >= 7).mean()*100
        st.metric("Nível adequado", f"{ian_ade:.0f}%", f"+{ian_ade-ian_22:.0f}pp vs 2022")

    st.markdown("---")

    # Gráficos visão geral
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("INDE médio por ano e Pedra")
        fig, ax = plt.subplots(figsize=(7, 4))
        inde_pedra = df.groupby(['Ano','Pedra'], observed=True)['INDE'].mean().unstack()
        for p in PEDRA_ORDER:
            if p in inde_pedra.columns:
                ax.plot(ANOS, inde_pedra[p].reindex(ANOS), marker='o',
                        label=p, color=CORES_PEDRA[p], linewidth=2.5, markersize=8)
        ax.set_xticks(ANOS); ax.set_ylim(5, 10)
        ax.axhline(7, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.legend(fontsize=9); ax.set_ylabel('INDE médio')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("Composição por Pedra (%)")
        fig, ax = plt.subplots(figsize=(7, 4))
        pedra_comp = df.groupby(['Ano','Pedra'], observed=True).size().unstack(fill_value=0)
        pedra_pct  = pedra_comp.div(pedra_comp.sum(axis=1), axis=0)*100
        bottom = np.zeros(3)
        for p in PEDRA_ORDER:
            if p in pedra_pct.columns:
                vals = pedra_pct[p].reindex(ANOS, fill_value=0).values
                bars = ax.bar(ANOS, vals, bottom=bottom, label=p,
                              color=CORES_PEDRA[p], edgecolor='white', width=0.5)
                for bar, v in zip(bars, vals):
                    if v > 5:
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+v/2,
                                f'{v:.0f}%', ha='center', va='center',
                                fontsize=9, color='white', fontweight='bold')
                bottom += vals
        ax.set_xticks(ANOS); ax.set_ylabel('%')
        ax.legend(fontsize=9, bbox_to_anchor=(1,1))
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Defasagem (IAN) por ano")
        fig, ax = plt.subplots(figsize=(7, 4))
        # IAN tem apenas 3 valores reais: 2.5 (Severa), 5.0 (Moderada), 10.0 (Adequado)
        df['cat_IAN'] = df['IAN'].map({2.5:'Severa (2.5)', 5.0:'Moderada (5.0)', 10.0:'Adequado (10.0)'})
        df['cat_IAN'] = df['cat_IAN'].fillna(
            df['IAN'].apply(lambda x: 'Severa (2.5)' if x < 5 else ('Adequado (10.0)' if x >= 7 else 'Moderada (5.0)'))
        )
        ian_cat = df.groupby(['Ano','cat_IAN']).size().unstack(fill_value=0)
        ian_pct = ian_cat.div(ian_cat.sum(axis=1), axis=0)*100
        bottom = np.zeros(3)
        for cat, cor in [('Severa (2.5)','#D94F4F'),
                         ('Moderada (5.0)','#E8934A'),
                         ('Adequado (10.0)','#4EA87A')]:
            vals = ian_pct[cat].reindex(ANOS, fill_value=0).values if cat in ian_pct.columns else np.zeros(3)
            ax.bar(ANOS, vals, bottom=bottom, color=cor, label=cat,
                   edgecolor='white', width=0.5)
            for i, v in enumerate(vals):
                if v > 3:
                    ax.text(ANOS[i], bottom[i]+v/2, f'{v:.0f}%',
                            ha='center', va='center', fontsize=9,
                            color='white', fontweight='bold')
            bottom += vals
        ax.set_xticks(ANOS); ax.set_ylabel('%'); ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_d:
        st.subheader("Notas por disciplina")
        fig, ax = plt.subplots(figsize=(7, 4))
        disc = df.groupby('Ano')[['Mat','Por','Ing']].mean()
        x = np.arange(3); w = 0.25
        for i, (col, c, label) in enumerate([('Mat','#4A7B9D','Matemática'),
                                               ('Por','#6BAE8E','Português'),
                                               ('Ing','#C4873A','Inglês')]):
            vals = disc[col].reindex(ANOS).values
            bars = ax.bar(x+i*w, vals, w, label=label, color=c, alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x()+bar.get_width()/2, v+0.05,
                            f'{v:.1f}', ha='center', fontsize=8)
        ax.set_xticks(x+w); ax.set_xticklabels(ANOS)
        ax.set_ylim(0, 10); ax.set_ylabel('Nota média'); ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════
# PÁGINA 2 — ANÁLISE POR INDICADOR
# ════════════════════════════════════════════════════════════
elif pagina == "Análise por Indicador":
    st.title("Análise por Indicador")

    if df is None:
        st.error("Dataset não encontrado. Execute os notebooks primeiro.")
        st.stop()

    indicador = st.selectbox(
        "Selecione o indicador",
        ['INDE','IDA','IEG','IAA','IPS','IPP','IPV','IAN'],
        index=0
    )

    nomes = {
        'INDE':'Índice de Desenvolvimento Educacional',
        'IDA':'Desempenho Acadêmico',
        'IEG':'Engajamento',
        'IAA':'Autoavaliação',
        'IPS':'Psicossocial',
        'IPP':'Psicopedagógico',
        'IPV':'Ponto de Virada',
        'IAN':'Adequação de Nível'
    }
    st.markdown(f"**{nomes[indicador]}** — evolução por ano e distribuição por Pedra")

    col1, col2 = st.columns(2)

    with col1:
        # Boxplot por ano
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [df[df['Ano']==a][indicador].dropna().values for a in ANOS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', linewidth=2.5))
        for patch, c in zip(bp['boxes'], CORES_ANOS.values()):
            patch.set_facecolor(c); patch.set_alpha(0.85)
        for i, d in enumerate(data):
            if len(d):
                ax.text(i+1, np.median(d)+0.1, f'{np.median(d):.2f}',
                        ha='center', fontsize=9, fontweight='bold')
        ax.set_xticklabels(ANOS)
        ax.set_title(f'{indicador} — distribuição por ano', fontweight='bold')
        ax.set_ylabel(indicador)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        # Média por Pedra e ano
        fig, ax = plt.subplots(figsize=(6, 4))
        ind_pedra = df.groupby(['Ano','Pedra'], observed=True)[indicador].mean().unstack()
        for p in PEDRA_ORDER:
            if p in ind_pedra.columns:
                vals = ind_pedra[p].reindex(ANOS)
                ax.plot(ANOS, vals, marker='o', label=p,
                        color=CORES_PEDRA[p], linewidth=2.5, markersize=8)
        ax.set_xticks(ANOS); ax.legend(fontsize=9)
        ax.set_title(f'{indicador} médio por Pedra', fontweight='bold')
        ax.set_ylabel(f'{indicador} médio')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Correlações
    st.subheader(f"Correlação de {indicador} com outros indicadores")
    outros = [i for i in ['INDE','IDA','IEG','IAA','IPS','IPP','IPV','IAN'] if i != indicador]
    corrs = []
    for col in outros:
        d_c = df[[indicador, col]].dropna()
        if len(d_c) > 30:
            r, p = stats.pearsonr(d_c[indicador], d_c[col])
            corrs.append({'Indicador': col, 'r': r, 'p': p})
    corrs_df = pd.DataFrame(corrs).sort_values('r', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ['#4EA87A' if r > 0 else '#D94F4F' for r in corrs_df['r']]
    ax.barh(corrs_df['Indicador'], corrs_df['r'], color=colors, edgecolor='white', alpha=0.85)
    ax.axvline(0, color='gray', linewidth=0.8)
    for i, row in corrs_df.iterrows():
        ax.text(row['r'] + (0.01 if row['r'] >= 0 else -0.01),
                corrs_df.index.get_loc(i),
                f"r={row['r']:.2f}", va='center', fontsize=9)
    ax.set_xlabel('Pearson r')
    ax.set_title(f'Correlação com {indicador}', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Gráficos salvos
    st.subheader("Gráficos detalhados da análise")
    mapa_graficos = {
        'IAN': 'p1_ian_defasagem.png', 'IDA': 'p2_ida_desempenho.png',
        'IEG': 'p3_ieg_engajamento.png', 'IAA': 'p4_iaa_autoavaliacao.png',
        'IPS': 'p5_ips_psicossocial.png', 'IPP': 'p6_ipp_psicopedagogico.png',
        'IPV': 'p7_ipv_ponto_virada.png', 'INDE': 'p8_multidimensionalidade.png'
    }
    if indicador in mapa_graficos:
        img_path = os.path.join(SRC_DIR, mapa_graficos[indicador])
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info(f"Imagem `{mapa_graficos[indicador]}` não encontrada em `src/`. Execute os notebooks para gerá-la.")

# ════════════════════════════════════════════════════════════
# PÁGINA 3 — PREVISÃO DE RISCO
# ════════════════════════════════════════════════════════════
elif pagina == "Previsão de Risco":
    st.title("Previsão de Risco de Defasagem")
    st.markdown("Insira os indicadores do aluno para calcular a probabilidade de risco.")

    if modelo is None:
        st.error("Modelo não encontrado em `data/modelo_risco_passos.pkl`. Execute o notebook 03 primeiro.")
        st.stop()

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.subheader("Dados do aluno")

        with st.expander("Indicadores educacionais", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                ida = st.slider("IDA — Desempenho acadêmico", 0.0, 10.0, 6.0, 0.1)
                ieg = st.slider("IEG — Engajamento",          0.0, 10.0, 6.0, 0.1)
                ian = st.slider("IAN — Adequação de nível",   0.0, 10.0, 6.0, 0.1)
                iaa = st.slider("IAA — Autoavaliação",        0.0, 10.0, 6.0, 0.1)
            with c2:
                ips = st.slider("IPS — Psicossocial",         0.0, 10.0, 6.0, 0.1)
                ipp = st.slider("IPP — Psicopedagógico",      0.0, 10.0, 6.0, 0.1)
                ipv = st.slider("IPV — Ponto de virada",      0.0, 10.0, 6.0, 0.1)

        with st.expander("Notas por disciplina", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1: mat = st.slider("Matemática", 0.0, 10.0, 6.0, 0.1)
            with c2: por = st.slider("Português",  0.0, 10.0, 6.0, 0.1)
            with c3: ing = st.slider("Inglês",     0.0, 10.0, 6.0, 0.1)

        with st.expander("Informações gerais", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                pedra = st.selectbox("Pedra atual", PEDRA_ORDER, index=0)
                pedra_num_val = {'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4}[pedra]
            with c2:
                genero = st.selectbox("Gênero", ['Feminino','Masculino'])
                genero_num_val = 0 if genero == 'Feminino' else 1
            with c3:
                defasagem = st.number_input("Defasagem (anos)", -5, 2, 0, 1)

        ano_aluno = st.selectbox("Ano de referência", [2022, 2023, 2024], index=2)

    with col_result:
        st.subheader("Resultado")

        entrada = np.array([[iaa, ieg, ips, ipp, ida, ian,
                             mat, por, ing,
                             pedra_num_val, genero_num_val, 0,
                             defasagem, ano_aluno]])

        prob = modelo.predict_proba(entrada)[0][1]
        threshold = 0.40

        if prob >= 0.65:
            nivel = "alto"
            cor_classe = "risco-alto"
            emoji = "🔴"
            msg = "Aluno com alto risco de defasagem. Intervenção imediata recomendada."
        elif prob >= threshold:
            nivel = "moderado"
            cor_classe = "risco-medio"
            emoji = "🟡"
            msg = "Risco moderado. Acompanhamento próximo recomendado."
        else:
            nivel = "baixo"
            cor_classe = "risco-baixo"
            emoji = "🟢"
            msg = "Risco baixo. Manter acompanhamento regular."

        st.markdown(f"""
        <div class="{cor_classe}">
            <div style="font-size:36px; font-weight:700; margin-bottom:4px;">
                {emoji} {prob:.1%}
            </div>
            <div style="font-size:14px; font-weight:600; margin-bottom:6px;">
                Probabilidade de risco — nível {nivel}
            </div>
            <div style="font-size:13px; color:#555;">
                {msg}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Gauge visual
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.5)
        ax.axis('off')

        # Fundo colorido
        from matplotlib.patches import FancyArrowPatch
        ax.barh(0.2, 0.40, left=0.00, height=0.15, color='#4EA87A', alpha=0.8)
        ax.barh(0.2, 0.25, left=0.40, height=0.15, color='#E8934A', alpha=0.8)
        ax.barh(0.2, 0.35, left=0.65, height=0.15, color='#D94F4F', alpha=0.8)

        # Indicador
        ax.axvline(prob, ymin=0.2, ymax=0.95, color='#1a1a2e', linewidth=3)
        ax.scatter([prob], [0.42], color='#1a1a2e', s=120, zorder=5)

        ax.text(0.20, 0.05, 'Baixo', ha='center', fontsize=9, color='#2E7A5B', fontweight='bold')
        ax.text(0.52, 0.05, 'Moderado', ha='center', fontsize=9, color='#854F0B', fontweight='bold')
        ax.text(0.82, 0.05, 'Alto', ha='center', fontsize=9, color='#8B1A1A', fontweight='bold')
        ax.text(prob, 0.46, f'{prob:.1%}', ha='center', fontsize=11, fontweight='bold', color='#1a1a2e')

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("**Fatores de maior impacto no modelo:**")
        fatores = {
            'Desempenho (IDA)': ida,
            'Engajamento (IEG)': ieg,
            'Defasagem': defasagem,
            'Ponto de Virada (IPV)': ipv,
            'Psicopedag. (IPP)': ipp
        }
        for fator, valor in fatores.items():
            ref = 6.0
            if fator == 'Defasagem':
                alerta = "⚠️" if valor <= -2 else "✅"
            else:
                alerta = "⚠️" if valor < ref else "✅"
            st.markdown(f"{alerta} **{fator}:** {valor:.1f}")

    # Tabela de histórico de consultas na sessão
    st.markdown("---")
    st.subheader("Histórico de consultas nesta sessão")
    if 'historico' not in st.session_state:
        st.session_state.historico = []

    if st.button("Salvar esta consulta"):
        st.session_state.historico.append({
            'Pedra': pedra, 'Gênero': genero, 'IDA': ida, 'IEG': ieg,
            'IAN': ian, 'Defasagem': defasagem,
            'P(risco)': f"{prob:.1%}", 'Nível': nivel.capitalize()
        })
        st.success("Consulta salva!")

    if st.session_state.historico:
        st.dataframe(pd.DataFrame(st.session_state.historico), use_container_width=True)
    else:
        st.info("Nenhuma consulta salva ainda. Ajuste os sliders e clique em 'Salvar esta consulta'.")

# ════════════════════════════════════════════════════════════
# PÁGINA 4 — SOBRE O PROJETO
# ════════════════════════════════════════════════════════════
elif pagina == "Sobre o Projeto":
    st.title("Sobre o Projeto")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Datathon PosTech — Fase 5

        Este projeto foi desenvolvido como entrega final do curso de **Data Analytics da PosTech (FIAP)**,
        utilizando dados reais da **Associação Passos Mágicos**.

        ### A Associação
        A Passos Mágicos tem 35 anos de atuação, transformando a vida de crianças e jovens
        de baixa renda em Embu-Guaçu (SP) por meio de educação de qualidade, apoio psicológico
        e ampliação de visão de mundo.

        ### Metodologia
        Foram analisados dados do PEDE (Pesquisa Extensiva do Desenvolvimento Educacional)
        dos anos de 2022, 2023 e 2024, totalizando ~3.000 registros de alunos.

        ### Principais achados
        | Métrica | 2022 | 2024 | Variação |
        |---|---|---|---|
        | INDE médio | 7,04 | 7,40 | +0,36 |
        | Em Topázio | 15% | 28% | +13pp |
        | Defasagem severa | 3,3% | 0,3% | -3pp |
        | Em risco | 38% | 20% | -18pp |

        ### Modelo preditivo
        O modelo XGBoost atingiu **AUC-ROC de 0,997**, permitindo identificar
        alunos em risco de defasagem antes que a queda ocorra.

        Os principais preditores de risco são:
        - Desempenho acadêmico (IDA)
        - Engajamento (IEG)
        - Defasagem atual
        - Ponto de virada (IPV)
        """)

    with col2:
        st.markdown("### Indicadores")
        indicadores_info = {
            'INDE': 'Índice global (0–10)',
            'IDA':  'Desempenho acadêmico',
            'IEG':  'Engajamento',
            'IAA':  'Autoavaliação',
            'IPS':  'Psicossocial',
            'IPP':  'Psicopedagógico',
            'IPV':  'Ponto de virada',
            'IAN':  'Adequação de nível',
        }
        for ind, desc in indicadores_info.items():
            st.markdown(f"**{ind}** — {desc}")

        st.markdown("---")
        st.markdown("### Sistema de Pedras")
        pedras_info = {
            'Quartzo':  'INDE 0–6,0',
            'Ágata':    'INDE 6,0–7,0',
            'Ametista': 'INDE 7,0–8,0',
            'Topázio':  'INDE 8,0–10,0',
        }
        for pedra, faixa in pedras_info.items():
            cor = CORES_PEDRA[pedra]
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{cor};border-radius:2px;margin-right:6px;"></span>'
                f'**{pedra}** — {faixa}',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Tecnologias")
        techs = ['Python 3.11', 'Pandas', 'XGBoost', 'SHAP',
                 'Scikit-learn', 'Matplotlib', 'Streamlit']
        for t in techs:
            st.markdown(f"- {t}")
