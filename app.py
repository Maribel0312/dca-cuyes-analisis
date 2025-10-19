import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

# Configuración de la página
st.set_page_config(page_title="Análisis DCA - Engorde de Cuyes", layout="wide", page_icon="🐹")

# ENCABEZADO PERSONALIZADO
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='text-align: center; color: #1f77b4;'>Análisis Estadístico de Diseños Experimentales</h1>
    <h3 style='text-align: center; color: #2c3e50;'>Diseño Completamente al Azar (DCA) - Engorde de Cuyes</h3>
    <p style='text-align: center; font-size: 18px;'><b>Nombre:</b> Dina Maribel Yana Yucra | <b>Código:</b> 221086</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.title("📋 Navegación")

# Configuración del experimento
st.sidebar.markdown("---")
st.sidebar.title("⚙️ Configuración")

with st.sidebar.expander("🔧 Personalizar Experimento", expanded=False):
    st.markdown("**Parámetros del Diseño:**")
    
    n_tratamientos_custom = st.slider(
        "Número de tratamientos", 
        min_value=2, 
        max_value=6, 
        value=4,
        help="Cantidad de dietas diferentes a comparar"
    )
    
    n_repeticiones_custom = st.number_input(
        "Repeticiones por tratamiento",
        min_value=5,
        max_value=30,
        value=15,
        help="Número de unidades experimentales por tratamiento"
    )
    
    alpha_custom = st.select_slider(
        "Nivel de significancia (α)",
        options=[0.01, 0.05, 0.10],
        value=0.05,
        help="Probabilidad de error tipo I"
    )
    
    st.info(f"📊 Configuración: {n_tratamientos_custom} tratamientos × {n_repeticiones_custom} repeticiones = {n_tratamientos_custom * n_repeticiones_custom} observaciones")

st.sidebar.markdown("---")

seccion = st.sidebar.radio(
    "Seleccione una sección:",
    ["🏠 Inicio", "📚 Teoría", "📊 Modelos Experimentales", "📤 Mis Datos", "📈 Comparación de Modelos"]
)

modelo_seleccionado = None
if seccion == "📊 Modelos Experimentales":
    modelo_seleccionado = st.sidebar.selectbox(
        "Seleccione el Modelo:",
        ["Modelo 1: Balanceado", "Modelo 2: No Balanceado", 
         "Modelo 3: Bal-Bal (Sub)", "Modelo 4: Bal-NoBal (Sub)",
         "Modelo 5: NoBal-Bal (Sub)", "Modelo 6: NoBal-NoBal (Sub)"]
    )

# Funciones para generar datos
def generar_datos_modelo1():
    np.random.seed(42)
    datos = []
    medias = {"T1": 520, "T2": 580, "T3": 545, "T4": 595}
    desv = {"T1": 28, "T2": 32, "T3": 25, "T4": 35}
    
    for trat in ["T1", "T2", "T3", "T4"]:
        for i in range(15):
            peso_inicial = np.random.normal(250, 15)
            ganancia = np.random.normal(medias[trat], desv[trat])
            datos.append({
                "Cuy": i+1,
                "Tratamiento": trat,
                "Peso_Inicial_g": round(peso_inicial, 1),
                "Peso_Final_g": round(peso_inicial + ganancia, 1),
                "Ganancia_Peso_g": round(ganancia, 1)
            })
    return pd.DataFrame(datos)

def generar_datos_modelo2():
    np.random.seed(123)
    datos = []
    medias = {"T1": 515, "T2": 590, "T3": 540, "T4": 605}
    desv = {"T1": 40, "T2": 35, "T3": 38, "T4": 42}
    n_cuyes = {"T1": 14, "T2": 18, "T3": 16, "T4": 20}
    
    for trat in ["T1", "T2", "T3", "T4"]:
        for i in range(n_cuyes[trat]):
            peso_inicial = np.random.normal(248, 22)
            ganancia = np.random.normal(medias[trat], desv[trat])
            datos.append({
                "Cuy": i+1,
                "Tratamiento": trat,
                "Peso_Inicial_g": round(peso_inicial, 1),
                "Peso_Final_g": round(peso_inicial + ganancia, 1),
                "Ganancia_Peso_g": round(ganancia, 1)
            })
    return pd.DataFrame(datos)

def generar_datos_modelo3():
    np.random.seed(456)
    datos = []
    medias = {"T1": 525, "T2": 575, "T3": 550, "T4": 588}
    desv_poza = {"T1": 22, "T2": 25, "T3": 20, "T4": 28}
    desv_cuy = 18
    
    for trat in ["T1", "T2", "T3", "T4"]:
        for poza in range(1, 6):
            media_poza = np.random.normal(medias[trat], desv_poza[trat])
            for cuy in range(1, 5):
                peso_inicial = np.random.normal(252, 18)
                ganancia = np.random.normal(media_poza, desv_cuy)
                datos.append({
                    "Poza": f"{trat}-P{poza}",
                    "Cuy": cuy,
                    "Tratamiento": trat,
                    "Peso_Inicial_g": round(peso_inicial, 1),
                    "Peso_Final_g": round(peso_inicial + ganancia, 1),
                    "Ganancia_Peso_g": round(ganancia, 1)
                })
    return pd.DataFrame(datos)

def generar_datos_modelo4():
    np.random.seed(789)
    datos = []
    medias = {"T1": 518, "T2": 585, "T3": 548, "T4": 600}
    desv_poza = {"T1": 28, "T2": 30, "T3": 24, "T4": 32}
    desv_cuy = 16
    n_cuyes_poza = {"T1": 3, "T2": 4, "T3": 5, "T4": 3}
    
    for trat in ["T1", "T2", "T3", "T4"]:
        for poza in range(1, 6):
            media_poza = np.random.normal(medias[trat], desv_poza[trat])
            for cuy in range(1, n_cuyes_poza[trat] + 1):
                peso_inicial = np.random.normal(249, 19)
                ganancia = np.random.normal(media_poza, desv_cuy)
                datos.append({
                    "Poza": f"{trat}-P{poza}",
                    "Cuy": cuy,
                    "Tratamiento": trat,
                    "Peso_Inicial_g": round(peso_inicial, 1),
                    "Peso_Final_g": round(peso_inicial + ganancia, 1),
                    "Ganancia_Peso_g": round(ganancia, 1)
                })
    return pd.DataFrame(datos)

def generar_datos_modelo5():
    np.random.seed(321)
    datos = []
    medias = {"T1": 522, "T2": 578, "T3": 552, "T4": 592}
    desv_poza = {"T1": 26, "T2": 29, "T3": 23, "T4": 31}
    desv_cuy = 17
    n_pozas = {"T1": 4, "T2": 6, "T3": 5, "T4": 7}
    
    for trat in ["T1", "T2", "T3", "T4"]:
        for poza in range(1, n_pozas[trat] + 1):
            media_poza = np.random.normal(medias[trat], desv_poza[trat])
            for cuy in range(1, 5):
                peso_inicial = np.random.normal(251, 17)
                ganancia = np.random.normal(media_poza, desv_cuy)
                datos.append({
                    "Poza": f"{trat}-P{poza}",
                    "Cuy": cuy,
                    "Tratamiento": trat,
                    "Peso_Inicial_g": round(peso_inicial, 1),
                    "Peso_Final_g": round(peso_inicial + ganancia, 1),
                    "Ganancia_Peso_g": round(ganancia, 1)
                })
    return pd.DataFrame(datos)

def generar_datos_modelo6():
    np.random.seed(654)
    datos = []
    medias = {"T1": 510, "T2": 595, "T3": 535, "T4": 610}
    desv_poza = {"T1": 30, "T2": 33, "T3": 27, "T4": 36}
    desv_cuy = 20
    n_pozas = {"T1": 4, "T2": 6, "T3": 5, "T4": 7}
    cuyes_por_poza = {
        "T1": [3, 4, 5, 4],
        "T2": [5, 4, 6, 4, 5, 4],
        "T3": [4, 5, 3, 6, 4],
        "T4": [6, 4, 5, 4, 6, 5, 4]
    }
    
    for trat in ["T1", "T2", "T3", "T4"]:
        for poza in range(n_pozas[trat]):
            media_poza = np.random.normal(medias[trat], desv_poza[trat])
            n_cuyes = cuyes_por_poza[trat][poza]
            for cuy in range(1, n_cuyes + 1):
                peso_inicial = np.random.normal(247, 21)
                ganancia = np.random.normal(media_poza, desv_cuy)
                datos.append({
                    "Poza": f"{trat}-P{poza+1}",
                    "Cuy": cuy,
                    "Tratamiento": trat,
                    "Peso_Inicial_g": round(peso_inicial, 1),
                    "Peso_Final_g": round(peso_inicial + ganancia, 1),
                    "Ganancia_Peso_g": round(ganancia, 1)
                })
    return pd.DataFrame(datos)

# PRUEBAS DE SUPUESTOS
def pruebas_supuestos(df):
    """Realiza pruebas de normalidad y homogeneidad de varianzas"""
    st.markdown("### 🔬 Verificación de Supuestos Estadísticos")
    
    st.info("""
    **Supuestos del ANOVA:**
    1. **Normalidad:** Los residuos deben seguir una distribución normal
    2. **Homogeneidad de varianzas:** Las varianzas entre grupos deben ser similares
    3. **Independencia:** Las observaciones deben ser independientes
    """)
    
    col1, col2 = st.columns(2)
    
    # Prueba de Normalidad (Shapiro-Wilk)
    with col1:
        st.markdown("#### 📊 Prueba de Normalidad (Shapiro-Wilk)")
        
        # Calcular residuos
        grand_mean = df['Ganancia_Peso_g'].mean()
        group_means = df.groupby('Tratamiento')['Ganancia_Peso_g'].transform('mean')
        residuos = df['Ganancia_Peso_g'] - group_means
        
        # Shapiro-Wilk
        stat_shapiro, p_shapiro = stats.shapiro(residuos)
        
        st.latex(r"H_0: \text{Los residuos siguen una distribución normal}")
        st.latex(r"H_1: \text{Los residuos NO siguen una distribución normal}")
        
        resultado_norm = pd.DataFrame({
            'Estadístico': ['W', 'P-valor', 'Decisión'],
            'Valor': [
                f"{stat_shapiro:.6f}",
                f"{p_shapiro:.6f}",
                f"{'✅ Normalidad aceptada' if p_shapiro > alpha_custom else '❌ Normalidad rechazada'}"
            ]
        })
        st.dataframe(resultado_norm, hide_index=True, use_container_width=True)
        
        if p_shapiro > alpha_custom:
            st.success(f"✅ p-valor ({p_shapiro:.4f}) > α ({alpha_custom}): Se cumple el supuesto de normalidad")
        else:
            st.warning(f"⚠️ p-valor ({p_shapiro:.4f}) ≤ α ({alpha_custom}): No se cumple normalidad. Considere transformaciones o pruebas no paramétricas")
    
    # Prueba de Homogeneidad de Varianzas (Levene)
    with col2:
        st.markdown("#### 📊 Prueba de Homogeneidad (Levene)")
        
        grupos = [df[df['Tratamiento'] == t]['Ganancia_Peso_g'].values 
                  for t in sorted(df['Tratamiento'].unique())]
        
        stat_levene, p_levene = stats.levene(*grupos)
        
        st.latex(r"H_0: \sigma_1^2 = \sigma_2^2 = ... = \sigma_k^2")
        st.latex(r"H_1: \text{Al menos una varianza es diferente}")
        
        resultado_homo = pd.DataFrame({
            'Estadístico': ['W', 'P-valor', 'Decisión'],
            'Valor': [
                f"{stat_levene:.6f}",
                f"{p_levene:.6f}",
                f"{'✅ Homogeneidad aceptada' if p_levene > alpha_custom else '❌ Homogeneidad rechazada'}"
            ]
        })
        st.dataframe(resultado_homo, hide_index=True, use_container_width=True)
        
        if p_levene > alpha_custom:
            st.success(f"✅ p-valor ({p_levene:.4f}) > α ({alpha_custom}): Se cumple homogeneidad de varianzas")
        else:
            st.warning(f"⚠️ p-valor ({p_levene:.4f}) ≤ α ({alpha_custom}): Varianzas heterogéneas. Considere prueba Welch ANOVA")
    
    # Gráficos de diagnóstico
    st.markdown("#### 📈 Gráficos de Diagnóstico")
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Q-Q Plot (Normalidad)', 'Residuos vs Valores Ajustados', 'Histograma de Residuos')
    )
    
    # Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuos)))
    sample_quantiles = np.sort(residuos)
    
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers',
                   marker=dict(color='blue', size=6), name='Datos'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                   y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                   mode='lines', line=dict(color='red', dash='dash'), name='Línea teórica'),
        row=1, col=1
    )
    
    # Residuos vs Ajustados
    fig.add_trace(
        go.Scatter(x=group_means, y=residuos, mode='markers',
                   marker=dict(color='green', size=6), name='Residuos'),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    # Histograma
    fig.add_trace(
        go.Histogram(x=residuos, nbinsx=20, marker=dict(color='purple'), name='Residuos'),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="Cuantiles Teóricos", row=1, col=1)
    fig.update_yaxes(title_text="Cuantiles Muestrales", row=1, col=1)
    fig.update_xaxes(title_text="Valores Ajustados", row=1, col=2)
    fig.update_yaxes(title_text="Residuos", row=1, col=2)
    fig.update_xaxes(title_text="Residuos", row=1, col=3)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=3)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de varianzas por grupo
    st.markdown("#### 📋 Varianzas por Tratamiento")
    var_table = df.groupby('Tratamiento')['Ganancia_Peso_g'].agg(['var', 'std']).round(2)
    var_table.columns = ['Varianza', 'Desv. Estándar']
    var_table['CV (%)'] = (var_table['Desv. Estándar'] / df.groupby('Tratamiento')['Ganancia_Peso_g'].mean() * 100).round(2)
    st.dataframe(var_table, use_container_width=True)
    
    return {
        'shapiro_stat': stat_shapiro,
        'shapiro_p': p_shapiro,
        'levene_stat': stat_levene,
        'levene_p': p_levene,
        'normalidad_ok': p_shapiro > alpha_custom,
        'homogeneidad_ok': p_levene > alpha_custom
    }

# CÁLCULO ANOVA UNIFACTORIAL
def calcular_anova_unifactorial_pasos(df):
    st.markdown("### 📐 Cálculos Paso a Paso - ANOVA Unifactorial")
    
    n_total = len(df)
    tratamientos = sorted(df['Tratamiento'].unique())
    k = len(tratamientos)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("N total", n_total)
    col2.metric("Tratamientos (k)", k)
    col3.metric("Grupos", k)
    
    st.markdown("#### Paso 2: Cálculo de medias")
    medias_df = df.groupby('Tratamiento').agg({
        'Ganancia_Peso_g': ['count', 'mean', 'sum']
    }).round(2)
    medias_df.columns = ['n', 'Media', 'Suma']
    st.dataframe(medias_df, use_container_width=True)
    
    grand_mean = df['Ganancia_Peso_g'].mean()
    st.info(f"**Media General (Ȳ..):** {grand_mean:.2f} g")
    
    st.markdown("#### Paso 3: Sumas de Cuadrados")
    
    ss_total = ((df['Ganancia_Peso_g'] - grand_mean) ** 2).sum()
    st.write(f"**SCT = {ss_total:.2f}**")
    
    ss_between = 0
    for trat in tratamientos:
        n_i = len(df[df['Tratamiento'] == trat])
        mean_i = df[df['Tratamiento'] == trat]['Ganancia_Peso_g'].mean()
        ss_between += n_i * (mean_i - grand_mean) ** 2
    
    st.write(f"**SC Trat = {ss_between:.2f}**")
    
    ss_within = ss_total - ss_between
    st.write(f"**SC Error = {ss_within:.2f}**")
    
    df_between = k - 1
    df_within = n_total - k
    df_total = n_total - 1
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    f_calc = ms_between / ms_within
    p_value = 1 - stats.f.cdf(f_calc, df_between, df_within)
    
    if p_value < alpha_custom:
        st.success(f"✅ p-valor < {alpha_custom}, rechazamos H₀")
    else:
        st.warning(f"⚠️ p-valor ≥ {alpha_custom}, no rechazamos H₀")
    
    return {
        'SS_Between': ss_between, 'SS_Within': ss_within, 'SS_Total': ss_total,
        'DF_Between': df_between, 'DF_Within': df_within, 'DF_Total': df_total,
        'MS_Between': ms_between, 'MS_Within': ms_within,
        'F_Statistic': f_calc, 'P_Value': p_value
    }

# ANOVA ANIDADO PARA SUBMUESTREO
def calcular_anova_submuestreo_pasos(df):
    """ANOVA anidado: Tratamiento > Poza(Tratamiento) > Cuy(Poza)"""
    
    st.markdown("### 🎯 ANOVA con Submuestreo (Modelo Anidado)")
    
    st.info("""
    **Modelo Anidado:** Yijk = μ + τi + β(i)j + εijk
    - τi = Efecto del tratamiento i
    - β(i)j = Efecto de la poza j anidada en tratamiento i  
    - εijk = Error (cuy k dentro de poza j)
    """)
    
    tratamientos = sorted(df['Tratamiento'].unique())
    t = len(tratamientos)
    n_total = len(df)
    
    pozas_info = df.groupby(['Tratamiento', 'Poza']).size().reset_index(name='n_cuyes')
    pozas_por_trat = df.groupby('Tratamiento')['Poza'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tratamientos (t)", t)
    col2.metric("Pozas totales", df['Poza'].nunique())
    col3.metric("Cuyes totales (N)", n_total)
    
    grand_mean = df['Ganancia_Peso_g'].mean()
    medias_trat = df.groupby('Tratamiento')['Ganancia_Peso_g'].mean()
    medias_poza = df.groupby(['Tratamiento', 'Poza'])['Ganancia_Peso_g'].mean()
    
    ss_total = ((df['Ganancia_Peso_g'] - grand_mean) ** 2).sum()
    
    ss_trat = 0
    for trat in tratamientos:
        n_trat = len(df[df['Tratamiento'] == trat])
        media_trat = medias_trat[trat]
        ss_trat += n_trat * (media_trat - grand_mean) ** 2
    
    ss_poza = 0
    for trat in tratamientos:
        pozas_trat = df[df['Tratamiento'] == trat]['Poza'].unique()
        media_trat = medias_trat[trat]
        for poza in pozas_trat:
            subset = df[(df['Tratamiento'] == trat) & (df['Poza'] == poza)]
            n_poza = len(subset)
            media_poza = subset['Ganancia_Peso_g'].mean()
            ss_poza += n_poza * (media_poza - media_trat) ** 2
    
    ss_error = ss_total - ss_trat - ss_poza
    
    df_trat = t - 1
    n_pozas_total = df['Poza'].nunique()
    df_poza = n_pozas_total - t
    df_error = n_total - n_pozas_total
    df_total = n_total - 1
    
    cm_trat = ss_trat / df_trat if df_trat > 0 else 0
    cm_poza = ss_poza / df_poza if df_poza > 0 else 0
    cm_error = ss_error / df_error if df_error > 0 else 1
    
    f_trat = cm_trat / cm_poza if cm_poza > 0 else 0
    f_poza = cm_poza / cm_error if cm_error > 0 else 0
    
    p_trat = 1 - stats.f.cdf(f_trat, df_trat, df_poza) if f_trat > 0 else 1
    p_poza = 1 - stats.f.cdf(f_poza, df_poza, df_error) if f_poza > 0 else 1
    
    return {
        'SS_Trat': ss_trat, 'SS_Poza': ss_poza, 'SS_Error': ss_error, 'SS_Total': ss_total,
        'DF_Trat': df_trat, 'DF_Poza': df_poza, 'DF_Error': df_error, 'DF_Total': df_total,
        'MS_Trat': cm_trat, 'MS_Poza': cm_poza, 'MS_Error': cm_error,
        'F_Trat': f_trat, 'F_Poza': f_poza,
        'P_Trat': p_trat, 'P_Poza': p_poza
    }

def mostrar_tabla_anova_anidado(result):
    """Muestra tabla ANOVA anidado final"""
    st.markdown("### 📊 Tabla ANOVA Anidado Final")
    
    anova_table = pd.DataFrame({
        'Fuente de Variación': ['Tratamientos', 'Pozas(Tratamiento)', 'Error (Cuyes)', 'Total'],
        'SC': [
            f"{result['SS_Trat']:.2f}",
            f"{result['SS_Poza']:.2f}",
            f"{result['SS_Error']:.2f}",
            f"{result['SS_Total']:.2f}"
        ],
        'GL': [
            result['DF_Trat'],
            result['DF_Poza'],
            result['DF_Error'],
            result['DF_Total']
        ],
        'CM': [
            f"{result['MS_Trat']:.2f}",
            f"{result['MS_Poza']:.2f}",
            f"{result['MS_Error']:.2f}",
            '-'
        ],
        'F': [
            f"{result['F_Trat']:.4f}",
            f"{result['F_Poza']:.4f}",
            '-',
            '-'
        ],
        'P-valor': [
            f"{result['P_Trat']:.6f}",
            f"{result['P_Poza']:.6f}",
            '-',
            '-'
        ]
    })
    
    st.dataframe(anova_table, use_container_width=True, hide_index=True)

# ANOVA BIFACTORIAL
def calcular_anova_bifactorial_pasos(df):
    st.markdown("### 📐 ANOVA Bifactorial")
    st.info("Factor A = Tratamiento, Factor B = Sexo simulado")
    
    df_bif = df.copy()
    df_bif['Factor_A'] = df_bif['Tratamiento']
    
    sexos = []
    for trat in sorted(df_bif['Tratamiento'].unique()):
        subset = df_bif[df_bif['Tratamiento'] == trat]
        n_trat = len(subset)
        n_machos = n_trat // 2
        sexos_trat = ['Macho'] * n_machos + ['Hembra'] * (n_trat - n_machos)
        sexos.extend(sexos_trat)
    
    df_bif['Factor_B'] = sexos
    
    factor_a_levels = sorted(df_bif['Factor_A'].unique())
    factor_b_levels = sorted(df_bif['Factor_B'].unique())
    a = len(factor_a_levels)
    b = len(factor_b_levels)
    n_total = len(df_bif)
    
    medias_a = df_bif.groupby('Factor_A')['Ganancia_Peso_g'].mean()
    medias_b = df_bif.groupby('Factor_B')['Ganancia_Peso_g'].mean()
    grand_mean = df_bif['Ganancia_Peso_g'].mean()
    
    ss_total = ((df_bif['Ganancia_Peso_g'] - grand_mean) ** 2).sum()
    
    ss_a = 0
    for nivel in factor_a_levels:
        n_nivel = len(df_bif[df_bif['Factor_A'] == nivel])
        ss_a += n_nivel * (medias_a[nivel] - grand_mean) ** 2
    
    ss_b = 0
    for nivel in factor_b_levels:
        n_nivel = len(df_bif[df_bif['Factor_B'] == nivel])
        ss_b += n_nivel * (medias_b[nivel] - grand_mean) ** 2
    
    ss_ab = 0
    for nivel_a in factor_a_levels:
        for nivel_b in factor_b_levels:
            subset = df_bif[(df_bif['Factor_A'] == nivel_a) & (df_bif['Factor_B'] == nivel_b)]
            if len(subset) > 0:
                n_cell = len(subset)
                mean_cell = subset['Ganancia_Peso_g'].mean()
                ss_ab += n_cell * (mean_cell - medias_a[nivel_a] - medias_b[nivel_b] + grand_mean) ** 2
    
    ss_error = ss_total - ss_a - ss_b - ss_ab
    
    df_a = a - 1
    df_b = b - 1
    df_ab = (a - 1) * (b - 1)
    df_error = n_total - (a * b)
    df_total = n_total - 1
    
    cm_a = ss_a / df_a if df_a > 0 else 0
    cm_b = ss_b / df_b if df_b > 0 else 0
    cm_ab = ss_ab / df_ab if df_ab > 0 else 0
    cm_error = ss_error / df_error if df_error > 0 else 1
    
    f_a = cm_a / cm_error if cm_error > 0 else 0
    f_b = cm_b / cm_error if cm_error > 0 else 0
    f_ab = cm_ab / cm_error if cm_error > 0 else 0
    
    p_a = 1 - stats.f.cdf(f_a, df_a, df_error) if f_a > 0 else 1
    p_b = 1 - stats.f.cdf(f_b, df_b, df_error) if f_b > 0 else 1
    p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_error) if f_ab > 0 else 1
    
    return {
        'SS_A': ss_a, 'SS_B': ss_b, 'SS_AB': ss_ab, 'SS_Error': ss_error, 'SS_Total': ss_total,
        'DF_A': df_a, 'DF_B': df_b, 'DF_AB': df_ab, 'DF_Error': df_error, 'DF_Total': df_total,
        'MS_A': cm_a, 'MS_B': cm_b, 'MS_AB': cm_ab, 'MS_Error': cm_error,
        'F_A': f_a, 'F_B': f_b, 'F_AB': f_ab,
        'P_A': p_a, 'P_B': p_b, 'P_AB': p_ab
    }

def mostrar_tabla_anova_unifactorial(result):
    st.markdown("### 📊 Tabla ANOVA Unifactorial Final")
    anova_table = pd.DataFrame({
        'Fuente': ['Entre Tratamientos', 'Error', 'Total'],
        'SC': [f"{result['SS_Between']:.2f}", f"{result['SS_Within']:.2f}", f"{result['SS_Total']:.2f}"],
        'GL': [result['DF_Between'], result['DF_Within'], result['DF_Total']],
        'CM': [f"{result['MS_Between']:.2f}", f"{result['MS_Within']:.2f}", '-'],
        'F': [f"{result['F_Statistic']:.4f}", '-', '-'],
        'P-valor': [f"{result['P_Value']:.6f}", '-', '-']
    })
    st.dataframe(anova_table, use_container_width=True, hide_index=True)

def mostrar_tabla_anova_bifactorial(result):
    st.markdown("### 📊 Tabla ANOVA Bifactorial Final")
    anova_table = pd.DataFrame({
        'Fuente': ['Factor A', 'Factor B', 'Interacción AB', 'Error', 'Total'],
        'SC': [f"{result['SS_A']:.2f}", f"{result['SS_B']:.2f}", f"{result['SS_AB']:.2f}", 
               f"{result['SS_Error']:.2f}", f"{result['SS_Total']:.2f}"],
        'GL': [result['DF_A'], result['DF_B'], result['DF_AB'], result['DF_Error'], result['DF_Total']],
        'CM': [f"{result['MS_A']:.2f}", f"{result['MS_B']:.2f}", f"{result['MS_AB']:.2f}", 
               f"{result['MS_Error']:.2f}", '-'],
        'F': [f"{result['F_A']:.4f}", f"{result['F_B']:.4f}", f"{result['F_AB']:.4f}", '-', '-'],
        'P-valor': [f"{result['P_A']:.6f}", f"{result['P_B']:.6f}", f"{result['P_AB']:.6f}", '-', '-']
    })
    st.dataframe(anova_table, use_container_width=True, hide_index=True)

def tukey_hsd(df):
    from scipy.stats import studentized_range
    medias = df.groupby('Tratamiento')['Ganancia_Peso_g'].mean().sort_values(ascending=False)
    n = df.groupby('Tratamiento')['Ganancia_Peso_g'].count()
    
    n_total = len(df)
    k = len(df['Tratamiento'].unique())
    grand_mean = df['Ganancia_Peso_g'].mean()
    
    ss_within = sum([(df[df['Tratamiento'] == t]['Ganancia_Peso_g'] - df[df['Tratamiento'] == t]['Ganancia_Peso_g'].mean()).pow(2).sum() 
                     for t in df['Tratamiento'].unique()])
    df_within = n_total - k
    mse = ss_within / df_within
    
    comparaciones = []
    tratamientos = list(medias.index)
    
    for i in range(len(tratamientos)):
        for j in range(i+1, len(tratamientos)):
            t1, t2 = tratamientos[i], tratamientos[j]
            diff = abs(medias[t1] - medias[t2])
            n_harmonic = 2 / (1/n[t1] + 1/n[t2])
            se = np.sqrt(mse / n_harmonic)
            q_crit = studentized_range.ppf(0.95, len(tratamientos), df_within)
            hsd = q_crit * se
            
            comparaciones.append({
                'Comparación': f"{t1} vs {t2}",
                'Diferencia': round(diff, 2),
                'HSD': round(hsd, 2),
                'Significativo': 'Sí' if diff > hsd else 'No'
            })
    
    return pd.DataFrame(comparaciones), medias

def crear_graficos(df, result_uni):
    st.markdown("## 📊 Visualización de Resultados")
    
    fig_box = px.box(df, x='Tratamiento', y='Ganancia_Peso_g',
                     title='Distribución por Tratamiento',
                     color='Tratamiento',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_box.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig_box, use_container_width=True)
    
    fig_violin = px.violin(df, x='Tratamiento', y='Ganancia_Peso_g',
                          title='Densidad de Distribución',
                          color='Tratamiento',
                          box=True,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_violin.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig_violin, use_container_width=True)
    
    stats_df = df.groupby('Tratamiento')['Ganancia_Peso_g'].agg(['mean', 'sem']).reset_index()
    stats_df['ci'] = stats_df['sem'] * 1.96
    
    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(
        x=stats_df['Tratamiento'],
        y=stats_df['mean'],
        error_y=dict(type='data', array=stats_df['ci']),
        mode='markers',
        marker=dict(size=12, color='rgb(31, 119, 180)'),
        name='Media ± IC95%'
    ))
    fig_ci.update_layout(title='Intervalos de Confianza 95%', height=500)
    st.plotly_chart(fig_ci, use_container_width=True)

def mostrar_interpretaciones_y_recomendaciones(df, result_uni, supuestos=None):
    st.markdown("## 💡 Interpretaciones y Recomendaciones")
    
    medias = df.groupby('Tratamiento')['Ganancia_Peso_g'].mean().sort_values(ascending=False)
    mejor_trat = medias.index[0]
    mejor_media = medias.iloc[0]
    peor_trat = medias.index[-1]
    peor_media = medias.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**🏆 Mejor Tratamiento: {mejor_trat}**")
        st.metric("Ganancia promedio", f"{mejor_media:.1f} g")
        
    with col2:
        st.error(f"**📉 Menor Rendimiento: {peor_trat}**")
        st.metric("Ganancia promedio", f"{peor_media:.1f} g")
    
    diferencia = mejor_media - peor_media
    porcentaje = (diferencia / peor_media) * 100
    
    st.info(f"**Diferencia:** {diferencia:.1f} g ({porcentaje:.1f}% más que {peor_trat})")
    
    # Interpretación estadística
    st.markdown("### 📊 Interpretación Estadística")
    
    if result_uni['P_Value'] < alpha_custom:
        st.success(f"""
        ✅ **Resultado Significativo** (p = {result_uni['P_Value']:.6f})
        
        Existe evidencia estadística suficiente para concluir que **al menos un tratamiento 
        difiere significativamente** de los demás en términos de ganancia de peso.
        """)
    else:
        st.warning(f"""
        ⚠️ **Resultado No Significativo** (p = {result_uni['P_Value']:.6f})
        
        No hay evidencia estadística suficiente para afirmar que existen diferencias 
        entre los tratamientos evaluados.
        """)
    
    # Recomendaciones prácticas
    st.markdown("### 🎯 Recomendaciones para la Implementación")
    
    st.markdown(f"""
    #### 1. **Recomendación Principal**
    
    Se recomienda implementar el **{mejor_trat}** como dieta principal para el engorde de cuyes, 
    ya que mostró la mayor ganancia de peso promedio ({mejor_media:.1f} g) durante el período experimental.
    
    #### 2. **Análisis Costo-Beneficio**
    
    Antes de la implementación a escala comercial, considere:
    
    - **Costo del alimento por kg**
    - **Conversión alimenticia** (kg alimento / kg ganancia peso)
    - **Disponibilidad y acceso** al tipo de alimento
    - **Facilidad de preparación** y suministro
    - **Aceptación por parte de los animales**
    
    #### 3. **Consideraciones Técnicas**
    
    - **Periodo de adaptación:** Implementar cambios dietarios gradualmente (7-10 días)
    - **Monitoreo constante:** Registrar peso semanalmente
    - **Condiciones ambientales:** Mantener temperatura (18-24°C) y ventilación adecuadas
    - **Agua limpia:** Disponibilidad ad libitum
    - **Densidad de población:** No exceder 8-10 cuyes/m²
    
    #### 4. **Indicadores de Éxito**
    
    Durante la implementación, monitorear:
    
    - Ganancia diaria de peso (GDP) esperada: **{(mejor_media/84):.1f} g/día** (asumiendo 12 semanas)
    - Tasa de mortalidad: objetivo < 5%
    - Índice de conversión alimenticia: objetivo < 4.0
    - Peso comercial objetivo: 900-1000 g a las 12-14 semanas
    
    #### 5. **Plan de Contingencia**
    
    Si el tratamiento óptimo ({mejor_trat}) no está disponible o es muy costoso:
    
    - **Alternativa 1:** {medias.index[1]} (ganancia: {medias.iloc[1]:.1f} g)
    - **Diferencia con óptimo:** {(mejor_media - medias.iloc[1]):.1f} g ({((mejor_media - medias.iloc[1])/mejor_media*100):.1f}% menos)
    
    #### 6. **Validación en Campo**
    
    Antes de implementación masiva:
    
    1. Realizar prueba piloto con 50-100 animales
    2. Periodo mínimo de validación: 4-6 semanas
    3. Comparar resultados con datos experimentales
    4. Ajustar protocolos según necesidad
    
    #### 7. **Aspectos Económicos**
    """)
    
    # Tabla de análisis económico estimado
    costos_estimados = {
        'T1': 2.5,
        'T2': 3.2,
        'T3': 2.8,
        'T4': 3.5
    }
    
    precio_cuy = 25.0  # soles por cuy comercial
    
    analisis_economico = []
    for trat in medias.index:
        ganancia = medias[trat]
        costo_alimento = costos_estimados.get(trat, 3.0)
        costo_total = costo_alimento * 84  # 12 semanas
        ingreso_estimado = (ganancia / 1000) * precio_cuy
        beneficio_neto = ingreso_estimado - costo_total
        roi = (beneficio_neto / costo_total) * 100 if costo_total > 0 else 0
        
        analisis_economico.append({
            'Tratamiento': trat,
            'Ganancia (g)': f"{ganancia:.1f}",
            'Costo/día (S/)': f"{costo_alimento:.2f}",
            'Costo Total (S/)': f"{costo_total:.2f}",
            'Ingreso Est. (S/)': f"{ingreso_estimado:.2f}",
            'Beneficio Neto (S/)': f"{beneficio_neto:.2f}",
            'ROI (%)': f"{roi:.1f}"
        })
    
    st.dataframe(pd.DataFrame(analisis_economico), hide_index=True, use_container_width=True)
    
    st.caption("*Valores estimados. Ajustar según precios locales y condiciones específicas.*")
    
    # Validación de supuestos
    if supuestos:
        st.markdown("#### 8. **Validez de las Conclusiones**")
        
        if supuestos['normalidad_ok'] and supuestos['homogeneidad_ok']:
            st.success("""
            ✅ Los supuestos estadísticos se cumplen adecuadamente:
            - Normalidad de residuos: ✓
            - Homogeneidad de varianzas: ✓
            
            Las conclusiones del análisis son **estadísticamente válidas y confiables**.
            """)
        else:
            st.warning("""
            ⚠️ Advertencia sobre supuestos:
            """)
            if not supuestos['normalidad_ok']:
                st.write("- ❌ Normalidad no cumplida completamente")
            if not supuestos['homogeneidad_ok']:
                st.write("- ❌ Homogeneidad de varianzas comprometida")
            
            st.info("""
            **Recomendación:** Las conclusiones deben tomarse con precaución. 
            Considere realizar análisis complementarios o aumentar tamaño muestral.
            """)

def exportar_excel(df, anova_uni, anova_sub, tukey_df, supuestos=None):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Datos', index=False)
        
        # ANOVA Unifactorial
        pd.DataFrame({
            'Fuente': ['Entre Tratamientos', 'Error', 'Total'],
            'SC': [anova_uni['SS_Between'], anova_uni['SS_Within'], anova_uni['SS_Total']],
            'GL': [anova_uni['DF_Between'], anova_uni['DF_Within'], anova_uni['DF_Total']],
            'CM': [anova_uni['MS_Between'], anova_uni['MS_Within'], ''],
            'F': [anova_uni['F_Statistic'], '', ''],
            'P-valor': [anova_uni['P_Value'], '', '']
        }).to_excel(writer, sheet_name='ANOVA Unifactorial', index=False)
        
        # ANOVA Submuestreo (si existe)
        if anova_sub and 'SS_Trat' in anova_sub:
            pd.DataFrame({
                'Fuente': ['Tratamientos', 'Pozas(Trat)', 'Error', 'Total'],
                'SC': [anova_sub['SS_Trat'], anova_sub['SS_Poza'], anova_sub['SS_Error'], anova_sub['SS_Total']],
                'GL': [anova_sub['DF_Trat'], anova_sub['DF_Poza'], anova_sub['DF_Error'], anova_sub['DF_Total']],
                'CM': [anova_sub['MS_Trat'], anova_sub['MS_Poza'], anova_sub['MS_Error'], ''],
                'F': [anova_sub['F_Trat'], anova_sub['F_Poza'], '', ''],
                'P-valor': [anova_sub['P_Trat'], anova_sub['P_Poza'], '', '']
            }).to_excel(writer, sheet_name='ANOVA Submuestreo', index=False)
        
        if not tukey_df.empty:
            tukey_df.to_excel(writer, sheet_name='Tukey HSD', index=False)
        
        df.groupby('Tratamiento')['Ganancia_Peso_g'].agg(['count', 'mean', 'std', 'min', 'max']).to_excel(writer, sheet_name='Estadísticas')
        
        # Supuestos
        if supuestos:
            pd.DataFrame({
                'Prueba': ['Shapiro-Wilk (Normalidad)', 'Levene (Homogeneidad)'],
                'Estadístico': [supuestos['shapiro_stat'], supuestos['levene_stat']],
                'P-valor': [supuestos['shapiro_p'], supuestos['levene_p']],
                'Cumple': [
                    'Sí' if supuestos['normalidad_ok'] else 'No',
                    'Sí' if supuestos['homogeneidad_ok'] else 'No'
                ]
            }).to_excel(writer, sheet_name='Supuestos', index=False)
    
    return output.getvalue()

# FUNCIÓN PARA DATOS PROPIOS
def analizar_datos_propios():
    st.header("📤 Analizar Mis Datos")
    
    st.info("**Columnas requeridas:** Tratamiento, Ganancia_Peso_g")
    
    plantilla = pd.DataFrame({
        'Cuy': range(1, 21),
        'Tratamiento': ['T1']*5 + ['T2']*5 + ['T3']*5 + ['T4']*5,
        'Peso_Inicial_g': [250]*20,
        'Peso_Final_g': [750]*20,
        'Ganancia_Peso_g': [500]*20
    })
    csv = plantilla.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Plantilla CSV", csv, "plantilla.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Subir archivo", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            df_usuario = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            if 'Tratamiento' not in df_usuario.columns or 'Ganancia_Peso_g' not in df_usuario.columns:
                st.error("❌ Faltan columnas requeridas: Tratamiento y Ganancia_Peso_g")
                return
            
            st.success(f"✅ {len(df_usuario)} observaciones cargadas correctamente")
            st.dataframe(df_usuario.head(10), use_container_width=True)
            
            if st.button("🔬 Analizar Datos", type="primary"):
                tab1, tab2, tab3, tab4 = st.tabs(["🔬 Supuestos", "📊 ANOVA", "📈 Gráficos", "💡 Recomendaciones"])
                
                with tab1:
                    supuestos = pruebas_supuestos(df_usuario)
                
                with tab2:
                    result = calcular_anova_unifactorial_pasos(df_usuario)
                    mostrar_tabla_anova_unifactorial(result)
                    
                    if result['P_Value'] < alpha_custom:
                        st.markdown("---")
                        tukey_df, _ = tukey_hsd(df_usuario)
                        st.markdown("### 🔍 Comparaciones Múltiples - Tukey HSD")
                        st.dataframe(tukey_df, use_container_width=True, hide_index=True)
                
                with tab3:
                    crear_graficos(df_usuario, result)
                
                with tab4:
                    mostrar_interpretaciones_y_recomendaciones(df_usuario, result, supuestos)
        
        except Exception as e:
            st.error(f"❌ Error al procesar archivo: {str(e)}")

# ==================== SECCIONES ====================

if seccion == "🏠 Inicio":
    st.markdown("## 📄 Problema de Investigación")
    
    st.markdown("""
    ### 🎯 Enunciado del Problema
    
    Un productor pecuario especializado en la crianza de cuyes (Cavia porcellus) desea **determinar 
    cuál es la dieta más eficiente** para el engorde de estos animales hasta alcanzar el peso comercial 
    óptimo (900-1000 gramos). El objetivo principal es **maximizar la ganancia de peso** minimizando 
    costos de alimentación y el tiempo hasta la comercialización.
    
    ### 🔬 Planteamiento Experimental
    
    Para resolver este problema, se diseñó un **experimento completamente aleatorizado (DCA)** 
    evaluando cuatro tipos diferentes de dietas alimenticias durante un período de **12 semanas**.
    
    #### Variables del Experimento:
    
    - **Variable Independiente (Factor):** Tipo de dieta alimenticia
    - **Variable Dependiente (Respuesta):** Ganancia de peso en gramos
    - **Unidad Experimental:** Cuy individual o poza (según el modelo)
    - **Periodo Experimental:** 12 semanas (84 días)
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Tratamientos Evaluados")
        
        tratamientos_df = pd.DataFrame({
            'Código': ['T1', 'T2', 'T3', 'T4'],
            'Descripción': [
                'Balanceado comercial estándar',
                'Forraje verde (alfalfa)',
                'Dieta mixta (balanceado + forraje)',
                'Dieta con probióticos'
            ],
            'Proteína': ['18%', '20%', '16%', '19%'],
            'Tipo': ['Concentrado', 'Natural', 'Mixto', 'Suplementado']
        })
        
        st.dataframe(tratamientos_df, hide_index=True, use_container_width=True)
        
        st.markdown("""
        **Características de cada tratamiento:**
        
        - **T1 (Balanceado Comercial):** Alimento concentrado comercial estándar con 18% de proteína. 
          Es la dieta control tradicional utilizada por la mayoría de productores.
        
        - **T2 (Forraje Verde):** Alimentación basada en alfalfa fresca con alto contenido proteico (20%). 
          Representa una alternativa natural y potencialmente más económica.
        
        - **T3 (Dieta Mixta):** Combinación estratégica de concentrado comercial con forraje verde, 
          buscando balancear costos y nutrición (16% proteína total).
        
        - **T4 (Con Probióticos):** Alimento balanceado suplementado con probióticos para mejorar 
          la digestión y absorción de nutrientes (19% proteína).
        """)
    
    with col2:
        st.markdown("### 📊 Información Experimental")
        
        st.metric("Animales por tratamiento", "15-20 cuyes")
        st.metric("Peso inicial promedio", "250 ± 20 g")
        st.metric("Duración", "12 semanas")
        st.metric("Edad inicial", "21 días (destete)")
        
        st.info("""
        **Condiciones Controladas:**
        
        - Temperatura: 18-24°C
        - Humedad: 60-70%
        - Ventilación: Adecuada
        - Agua: Ad libitum
        - Fotoperíodo: Natural
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Objetivos del Estudio
    
    #### Objetivo General:
    Determinar el efecto del tipo de dieta sobre la ganancia de peso en cuyes durante la fase de engorde.
    
    #### Objetivos Específicos:
    
    1. Evaluar la ganancia de peso promedio de cada tratamiento
    2. Identificar el tratamiento con mayor rendimiento productivo
    3. Determinar si existen diferencias estadísticamente significativas entre tratamientos
    4. Establecer recomendaciones técnicas para la producción comercial
    5. Analizar la relación costo-beneficio de cada dieta
    
    ### 📈 Hipótesis de Investigación
    
    **H₀ (Hipótesis Nula):** No existen diferencias significativas en la ganancia de peso entre 
    los diferentes tipos de dieta evaluados.
    
    **H₁ (Hipótesis Alternativa):** Al menos uno de los tratamientos dietéticos produce una 
    ganancia de peso significativamente diferente a los demás.
    
    ### 🔍 Diseños Experimentales Implementados
    
    En este estudio se implementaron **6 modelos diferentes** de Diseño Completamente al Azar (DCA):
    """)
    
    modelos_df = pd.DataFrame({
        'Modelo': ['1', '2', '3', '4', '5', '6'],
        'Tipo': [
            'Balanceado',
            'No Balanceado',
            'Bal-Bal (Submuestreo)',
            'Bal-NoBal (Submuestreo)',
            'NoBal-Bal (Submuestreo)',
            'NoBal-NoBal (Submuestreo)'
        ],
        'Descripción': [
            'Igual número de cuyes por tratamiento',
            'Diferente número de cuyes por tratamiento',
            'Igual pozas, igual cuyes por poza',
            'Igual pozas, diferente cuyes por poza',
            'Diferente pozas, igual cuyes por poza',
            'Diferente pozas, diferente cuyes por poza'
        ],
        'Análisis': [
            'ANOVA Unifactorial',
            'ANOVA Unifactorial',
            'ANOVA Anidado',
            'ANOVA Anidado',
            'ANOVA Anidado',
            'ANOVA Anidado'
        ]
    })
    
    st.dataframe(modelos_df, hide_index=True, use_container_width=True)
    
    st.success("""
    💡 **Nota Importante:** Los modelos 3-6 incluyen submuestreo (pozas), lo que permite 
    evaluar la variabilidad entre unidades experimentales (pozas) y dentro de ellas (cuyes individuales), 
    proporcionando un análisis más completo de las fuentes de variación.
    """)
