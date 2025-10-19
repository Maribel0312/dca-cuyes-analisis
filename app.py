import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
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

# NUEVA SECCIÓN: Configuración del experimento
st.sidebar.markdown("---")
st.sidebar.title("⚙️ Configuración")

with st.sidebar.expander("🔧 Personalizar Experimento", expanded=False):
    st.markdown("**Parámetros del Diseño:**")
    
    # Número de tratamientos
    n_tratamientos_custom = st.slider(
        "Número de tratamientos", 
        min_value=2, 
        max_value=6, 
        value=4,
        help="Cantidad de dietas diferentes a comparar"
    )
    
    # Repeticiones
    n_repeticiones_custom = st.number_input(
        "Repeticiones por tratamiento",
        min_value=5,
        max_value=30,
        value=15,
        help="Número de unidades experimentales por tratamiento"
    )
    
    # Nivel de significancia
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

# Si selecciona Modelos, mostrar submenu
modelo_seleccionado = None
if seccion == "📊 Modelos Experimentales":
    modelo_seleccionado = st.sidebar.selectbox(
        "Seleccione el Modelo:",
        ["Modelo 1: Balanceado", "Modelo 2: No Balanceado", 
         "Modelo 3: Bal-Bal (Sub)", "Modelo 4: Bal-NoBal (Sub)",
         "Modelo 5: NoBal-Bal (Sub)", "Modelo 6: NoBal-NoBal (Sub)"]
    )

# Funciones para generar datos - CADA MODELO CON SEMILLA DIFERENTE
def generar_datos_modelo1():
    """Modelo 1: Balanceado - Mayor uniformidad"""
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
    """Modelo 2: No Balanceado"""
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
    """Modelo 3: Bal-Bal Sub"""
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
    """Modelo 4: Bal-NoBal Sub"""
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
    """Modelo 5: NoBal-Bal Sub"""
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
    """Modelo 6: NoBal-NoBal Sub"""
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

# CÁLCULO ANOVA UNIFACTORIAL PASO A PASO
def calcular_anova_unifactorial_pasos(df):
    st.markdown("### 📐 Cálculos Paso a Paso - ANOVA Unifactorial")
    
    n_total = len(df)
    tratamientos = sorted(df['Tratamiento'].unique())
    k = len(tratamientos)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("N total", n_total)
    col2.metric("Tratamientos (k)", k)
    col3.metric("Grupos", k)
    
    st.markdown("#### Paso 2: Cálculo de medias por tratamiento")
    medias_df = df.groupby('Tratamiento').agg({
        'Ganancia_Peso_g': ['count', 'mean', 'sum']
    }).round(2)
    medias_df.columns = ['n', 'Media', 'Suma']
    st.dataframe(medias_df, use_container_width=True)
    
    grand_mean = df['Ganancia_Peso_g'].mean()
    st.info(f"**Media General (Ȳ..):** {grand_mean:.2f} g")
    
    st.markdown("#### Paso 3: Cálculo de Sumas de Cuadrados")
    
    st.markdown("**3.1. Suma de Cuadrados Total (SCT)**")
    st.latex(r"SCT = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(Y_{ij} - \bar{Y}_{..})^2")
    
    ss_total = ((df['Ganancia_Peso_g'] - grand_mean) ** 2).sum()
    st.write(f"SCT = {ss_total:.2f}")
    
    st.markdown("**3.2. Suma de Cuadrados Entre Tratamientos (SC Trat)**")
    st.latex(r"SC_{Trat} = \sum_{i=1}^{k}n_i(\bar{Y}_{i.} - \bar{Y}_{..})^2")
    
    ss_between = 0
    calc_between = []
    for trat in tratamientos:
        n_i = len(df[df['Tratamiento'] == trat])
        mean_i = df[df['Tratamiento'] == trat]['Ganancia_Peso_g'].mean()
        ss_i = n_i * (mean_i - grand_mean) ** 2
        ss_between += ss_i
        calc_between.append({
            'Tratamiento': trat,
            'n_i': n_i,
            'Media': f"{mean_i:.2f}",
            'Cálculo': f"{n_i} × ({mean_i:.2f} - {grand_mean:.2f})²",
            'SC_i': f"{ss_i:.2f}"
        })
    
    st.dataframe(pd.DataFrame(calc_between), use_container_width=True, hide_index=True)
    st.write(f"**SC Trat = {ss_between:.2f}**")
    
    st.markdown("**3.3. Suma de Cuadrados del Error (SC Error)**")
    st.latex(r"SC_{Error} = SCT - SC_{Trat}")
    
    ss_within = ss_total - ss_between
    st.write(f"SC Error = {ss_total:.2f} - {ss_between:.2f} = **{ss_within:.2f}**")
    
    st.markdown("#### Paso 4: Grados de Libertad")
    df_between = k - 1
    df_within = n_total - k
    df_total = n_total - 1
    
    gl_df = pd.DataFrame({
        'Fuente': ['Entre Tratamientos', 'Error', 'Total'],
        'Fórmula': [f'k - 1 = {k} - 1', f'N - k = {n_total} - {k}', f'N - 1 = {n_total} - 1'],
        'GL': [df_between, df_within, df_total]
    })
    st.dataframe(gl_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### Paso 5: Cuadrados Medios")
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    st.latex(r"CM_{Trat} = \frac{SC_{Trat}}{GL_{Trat}} = \frac{" + f"{ss_between:.2f}" + r"}{" + str(df_between) + r"} = " + f"{ms_between:.2f}")
    st.latex(r"CM_{Error} = \frac{SC_{Error}}{GL_{Error}} = \frac{" + f"{ss_within:.2f}" + r"}{" + str(df_within) + r"} = " + f"{ms_within:.2f}")
    
    st.markdown("#### Paso 6: Estadístico F")
    f_calc = ms_between / ms_within
    st.latex(r"F = \frac{CM_{Trat}}{CM_{Error}} = \frac{" + f"{ms_between:.2f}" + r"}{" + f"{ms_within:.2f}" + r"} = " + f"{f_calc:.4f}")
    
    st.markdown("#### Paso 7: P-valor")
    p_value = 1 - stats.f.cdf(f_calc, df_between, df_within)
    st.write(f"P-valor = P(F > {f_calc:.4f}) = **{p_value:.6f}**")
    
    if p_value < alpha_custom:
        st.success(f"✅ Como p-valor < {alpha_custom}, rechazamos H₀")
    else:
        st.warning(f"⚠️ Como p-valor ≥ {alpha_custom}, no rechazamos H₀")
    
    return {
        'SS_Between': ss_between,
        'SS_Within': ss_within,
        'SS_Total': ss_total,
        'DF_Between': df_between,
        'DF_Within': df_within,
        'DF_Total': df_total,
        'MS_Between': ms_between,
        'MS_Within': ms_within,
        'F_Statistic': f_calc,
        'P_Value': p_value
    }

# CÁLCULO ANOVA BIFACTORIAL
def calcular_anova_bifactorial_pasos(df):
    st.markdown("### 📐 Cálculos Paso a Paso - ANOVA Bifactorial")
    
    st.info("**Diseño Bifactorial:** Factor A = Tratamiento, Factor B = Sexo simulado")
    
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
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N total", n_total)
    col2.metric("Factor A (a)", a)
    col3.metric("Factor B (b)", b)
    col4.metric("Celdas (a×b)", a*b)
    
    tabla_medias = df_bif.pivot_table(values='Ganancia_Peso_g', 
                                       index='Factor_A', 
                                       columns='Factor_B', 
                                       aggfunc='mean')
    st.write("**Medias por Celda:**")
    st.dataframe(tabla_medias.round(2), use_container_width=True)
    
    medias_a = df_bif.groupby('Factor_A')['Ganancia_Peso_g'].mean()
    medias_b = df_bif.groupby('Factor_B')['Ganancia_Peso_g'].mean()
    grand_mean = df_bif['Ganancia_Peso_g'].mean()
    
    st.success(f"**Media General:** {grand_mean:.2f} g")
    
    ss_total = ((df_bif['Ganancia_Peso_g'] - grand_mean) ** 2).sum()
    
    ss_a = 0
    for nivel in factor_a_levels:
        n_nivel = len(df_bif[df_bif['Factor_A'] == nivel])
        media_nivel = medias_a[nivel]
        ss_a += n_nivel * (media_nivel - grand_mean) ** 2
    
    ss_b = 0
    for nivel in factor_b_levels:
        n_nivel = len(df_bif[df_bif['Factor_B'] == nivel])
        media_nivel = medias_b[nivel]
        ss_b += n_nivel * (media_nivel - grand_mean) ** 2
    
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
        'Fuente de Variación': ['Entre Tratamientos', 'Error', 'Total'],
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
        'Fuente': ['Factor A (Tratamiento)', 'Factor B (Sexo)', 'Interacción A×B', 'Error', 'Total'],
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
    
    grupos = [df[df['Tratamiento'] == t]['Ganancia_Peso_g'].values for t in df['Tratamiento'].unique()]
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
    
    # Boxplot
    st.markdown("### 1. Distribución de Datos por Tratamiento")
    fig_box = px.box(df, x='Tratamiento', y='Ganancia_Peso_g',
                     title='Distribución de Ganancia de Peso por Tratamiento',
                     labels={'Ganancia_Peso_g': 'Ganancia de Peso (g)'},
                     color='Tratamiento',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_box.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("""
    **📌 Interpretación:** Este gráfico muestra la distribución, variabilidad y valores atípicos de cada tratamiento.
    """)
    
    # Violin Plot
    st.markdown("### 2. Densidad y Distribución")
    fig_violin = px.violin(df, x='Tratamiento', y='Ganancia_Peso_g',
                          title='Gráfico de Violín - Densidad de Distribución',
                          labels={'Ganancia_Peso_g': 'Ganancia de Peso (g)'},
                          color='Tratamiento',
                          box=True,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_violin.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig_violin, use_container_width=True)
    
    # Intervalos de Confianza
    st.markdown("### 3. Medias e Intervalos de Confianza (95%)")
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
    fig_ci.update_layout(
        title='Medias con Intervalos de Confianza del 95%',
        xaxis_title='Tratamiento',
        yaxis_title='Ganancia de Peso (g)',
        height=500
    )
    st.plotly_chart(fig_ci, use_container_width=True)
    
    # QQ-Plot
    st.markdown("### 4. Verificación de Normalidad (QQ-Plot)")
    residuos = df['Ganancia_Peso_g'] - df.groupby('Tratamiento')['Ganancia_Peso_g'].transform('mean')
    
    fig_qq = go.Figure()
    sorted_residuos = np.sort(residuos)
    n = len(sorted_residuos)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
    
    fig_qq.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sorted_residuos,
        mode='markers',
        marker=dict(color='blue', size=6),
        name='Residuos'
    ))
    
    fig_qq.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=theoretical_quantiles,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Línea de referencia'
    ))
    
    fig_qq.update_layout(
        title='QQ-Plot - Verificación de Normalidad',
        xaxis_title='Cuantiles Teóricos',
        yaxis_title='Cuantiles Muestrales',
        height=500
    )
    st.plotly_chart(fig_qq, use_container_width=True)
    
    # Residuos vs Ajustados
    st.markdown("### 5. Diagnóstico de Residuos")
    valores_ajustados = df.groupby('Tratamiento')['Ganancia_Peso_g'].transform('mean')
    
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=valores_ajustados,
        y=residuos,
        mode='markers',
        marker=dict(color='green', size=6),
        name='Residuos'
    ))
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    fig_resid.update_layout(
        title='Residuos vs Valores Ajustados',
        xaxis_title='Valores Ajustados',
        yaxis_title='Residuos',
        height=500
    )
    st.plotly_chart(fig_resid, use_container_width=True)

def mostrar_interpretaciones(df, result_uni):
    st.markdown("## 💡 Interpretaciones y Conclusiones")
    
    medias = df.groupby('Tratamiento')['Ganancia_Peso_g'].mean().sort_values(ascending=False)
    mejor_trat = medias.index[0]
    mejor_media = medias.iloc[0]
    peor_trat = medias.index[-1]
    peor_media = medias.iloc[-1]
    
    descripciones = {
        'T1': 'Alimento balanceado comercial (18% proteína)',
        'T2': 'Alimento con forraje verde (20% proteína)',
        'T3': 'Dieta mixta (16% proteína)',
        'T4': 'Alimento con probióticos (19% proteína)'
    }
    
    st.markdown("### 🏆 El Mejor Tratamiento")
    st.success(f"""
    **Tratamiento {mejor_trat}** es el mejor con **{mejor_media:.1f} g** de ganancia promedio.
    
    **Descripción:** {descripciones.get(mejor_trat, 'Tratamiento experimental')}
    """)
    
    if result_uni['P_Value'] < alpha_custom:
        tukey_df, _ = tukey_hsd(df)
        comparaciones_mejor = tukey_df[tukey_df['Comparación'].str.contains(mejor_trat)]
        n_sig = comparaciones_mejor[comparaciones_mejor['Significativo'] == 'Sí'].shape[0]
        
        st.markdown("### 📊 ¿Por qué es el mejor?")
        st.write(f"""
        ✅ **Evidencia Estadística:**
        - ANOVA significativo (p = {result_uni['P_Value']:.6f})
        - Superior a {n_sig} tratamiento(s) según Tukey HSD
        - Diferencia vs peor: {mejor_media - peor_media:.1f} g ({((mejor_media - peor_media)/peor_media * 100):.1f}%)
        """)

def exportar_excel(df, anova_uni, anova_bif, tukey_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Datos', index=False)
        
        anova_uni_df = pd.DataFrame({
            'Fuente': ['Entre Tratamientos', 'Error', 'Total'],
            'SC': [anova_uni['SS_Between'], anova_uni['SS_Within'], anova_uni['SS_Total']],
            'GL': [anova_uni['DF_Between'], anova_uni['DF_Within'], anova_uni['DF_Total']],
            'CM': [anova_uni['MS_Between'], anova_uni['MS_Within'], ''],
            'F': [anova_uni['F_Statistic'], '', ''],
            'P-valor': [anova_uni['P_Value'], '', '']
        })
        anova_uni_df.to_excel(writer, sheet_name='ANOVA Unifactorial', index=False)
        
        anova_bif_df = pd.DataFrame({
            'Fuente': ['Factor A', 'Factor B', 'Interacción AB', 'Error', 'Total'],
            'SC': [anova_bif['SS_A'], anova_bif['SS_B'], anova_bif['SS_AB'], 
                   anova_bif['SS_Error'], anova_bif['SS_Total']],
            'GL': [anova_bif['DF_A'], anova_bif['DF_B'], anova_bif['DF_AB'], 
                   anova_bif['DF_Error'], anova_bif['DF_Total']],
            'CM': [anova_bif['MS_A'], anova_bif['MS_B'], anova_bif['MS_AB'], 
                   anova_bif['MS_Error'], ''],
            'F': [anova_bif['F_A'], anova_bif['F_B'], anova_bif['F_AB'], '', ''],
            'P-valor': [anova_bif['P_A'], anova_bif['P_B'], anova_bif['P_AB'], '', '']
        })
        anova_bif_df.to_excel(writer, sheet_name='ANOVA Bifactorial', index=False)
        
        if not tukey_df.empty:
            tukey_df.to_excel(writer, sheet_name='Tukey HSD', index=False)
        
        stats_df = df.groupby('Tratamiento')['Ganancia_Peso_g'].agg(['count', 'mean', 'std', 'min', 'max'])
        stats_df.to_excel(writer, sheet_name='Estadísticas')
    
    return output.getvalue()

# FUNCIÓN PARA ANALIZAR DATOS PROPIOS
def analizar_datos_propios():
    """Permite analizar datos cargados por el usuario"""
    st.header("📤 Analizar Mis Propios Datos")
    
    st.info("""
    **📋 Formato requerido del archivo:**
    - Columnas obligatorias: `Tratamiento`, `Ganancia_Peso_g`
    - Columnas opcionales: `Cuy`, `Poza`, `Peso_Inicial_g`, `Peso_Final_g`
    - Formatos aceptados: CSV, Excel (.xlsx, .xls)
    """)
    
    # Descarga plantilla
    col1, col2 = st.columns(2)
    with col1:
        plantilla = pd.DataFrame({
            'Cuy': range(1, 21),
            'Tratamiento': ['T1']*5 + ['T2']*5 + ['T3']*5 + ['T4']*5,
            'Peso_Inicial_g': [250]*20,
            'Peso_Final_g': [750]*20,
            'Ganancia_Peso_g': [500]*20
        })
        csv = plantilla.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Descargar Plantilla CSV",
            csv,
            "plantilla_dca_cuyes.csv",
            "text/csv",
            help="Descarga y llena con tus datos"
        )
    
    # Subir archivo
    st.markdown("### 📂 Cargar tus datos")
    uploaded_file = st.file_uploader(
        "Selecciona tu archivo",
        type=['csv', 'xlsx', 'xls'],
        help="Sube un archivo CSV o Excel"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_usuario = pd.read_csv(uploaded_file)
            else:
                df_usuario = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Archivo cargado: {len(df_usuario)} observaciones")
            
            columnas_requeridas = ['Tratamiento', 'Ganancia_Peso_g']
            columnas_faltantes = [col for col in columnas_requeridas if col not in df_usuario.columns]
            
            if columnas_faltantes:
                st.error(f"❌ Faltan columnas: {', '.join(columnas_faltantes)}")
                return
            
            st.markdown("### 👀 Vista Previa")
            st.dataframe(df_usuario.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Observaciones", len(df_usuario))
            col2.metric("Tratamientos", df_usuario['Tratamiento'].nunique())
            col3.metric("Media General", f"{df_usuario['Ganancia_Peso_g'].mean():.1f} g")
            
            if st.button("🔬 Analizar", type="primary"):
                st.markdown("---")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Estadísticas", "🔢 ANOVA", "📈 Gráficos", "📥 Exportar"])
                
                with tab1:
                    stats = df_usuario.groupby('Tratamiento')['Ganancia_Peso_g'].agg([
                        ('N', 'count'), ('Media', 'mean'), ('Desv.Est.', 'std'), ('Mín', 'min'), ('Máx', 'max')
                    ]).round(2)
                    st.dataframe(stats, use_container_width=True)
                
                with tab2:
                    result_uni = calcular_anova_unifactorial_pasos(df_usuario)
                    st.markdown("---")
                    mostrar_tabla_anova_unifactorial(result_uni)
                    
                    if result_uni['P_Value'] < alpha_custom:
                        tukey_df, _ = tukey_hsd(df_usuario)
                        st.markdown("### 🔍 Tukey HSD")
                        st.dataframe(tukey_df, use_container_width=True, hide_index=True)
                
                with tab3:
                    crear_graficos(df_usuario, result_uni)
                
                with tab4:
                    tukey_df, _ = tukey_hsd(df_usuario) if result_uni['P_Value'] < alpha_custom else (pd.DataFrame(), None)
                    result_bif = {'SS_A': 0, 'SS_B': 0, 'SS_AB': 0, 'SS_Error': 0, 'SS_Total': 0,
                                'DF_A': 0, 'DF_B': 0, 'DF_AB': 0, 'DF_Error': 0, 'DF_Total': 0,
                                'MS_A': 0, 'MS_B': 0, 'MS_AB': 0, 'MS_Error': 0,
                                'F_A': 0, 'F_B': 0, 'F_AB': 0, 'P_A': 1, 'P_B': 1, 'P_AB': 1}
                    excel_data = exportar_excel(df_usuario, result_uni, result_bif, tukey_df)
                    st.download_button("📥 Descargar Excel", excel_data, "analisis_propios.xlsx",
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ==================== SECCIÓN INICIO ====================
if seccion == "🏠 Inicio":
    st.markdown("---")
    st.markdown("## 📄 Contexto del Caso")
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px;'>
        <p style='font-size: 16px;'>
        Este estudio determina el <b>mejor alimento de engorde</b> para cuyes hasta alcanzar peso comercial óptimo.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔬 Factor Experimental")
        st.info("**Tipo de alimento o dieta de engorde**")
        
        st.markdown("### 📋 Tratamientos")
        tratamientos_df = pd.DataFrame({
            'Código': ['T1', 'T2', 'T3', 'T4'],
            'Descripción': [
                'Alimento balanceado comercial (18% proteína)',
                'Alimento con forraje verde (20% proteína)',
                'Dieta mixta (16% proteína)',
                'Alimento con probióticos (19% proteína)'
            ]
        })
        st.dataframe(tratamientos_df, use_container_width=True, hide_index=True)
        
        st.markdown("### 📈 Variable Respuesta")
        st.success("**Ganancia de peso (g)** después de 8 semanas")
        
    with col2:
        st.markdown("### 📚 Navegación")
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>
        <p>🏠 <b>Inicio</b></p>
        <p>📚 <b>Teoría</b></p>
        <p>📊 <b>Modelos (6)</b></p>
        <p>📤 <b>Mis Datos</b></p>
        <p>📈 <b>Comparación</b></p>
        </div>
        """, unsafe_allow_html=True)

# ==================== SECCIÓN TEORÍA ====================
elif seccion == "📚 Teoría":
    st.header("📚 Marco Teórico")
    
    tab1, tab2 = st.tabs(["🔢 Unifactorial", "🔢 Bifactorial"])
    
    with tab1:
        st.markdown("## DCA Unifactorial")
        st.latex(r"Y_{ij} = \mu + \tau_i + \varepsilon_{ij}")
        st.latex(r"SC_{Total} = \sum(Y_{ij} - \bar{Y})^2")
        st.latex(r"F = \frac{CM_{Trat}}{CM_{Error}}")
    
    with tab2:
        st.markdown("## DCA Bifactorial")
        st.latex(r"Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}")

# ==================== SECCIÓN MODELOS ====================
elif seccion == "📊 Modelos Experimentales":
    
    def mostrar_analisis_completo(df, titulo, descripcion):
        st.header(titulo)
        st.info(descripcion)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📊 Datos", "🔢 ANOVA Unifactorial", "🔢 ANOVA Bifactorial", 
             "📈 Gráficos", "💡 Interpretaciones", "📥 Exportar"]
        )
        
        with tab1:
            st.subheader("Datos Experimentales")
            st.dataframe(df, use_container_width=True, height=400)
            
            summary = df.groupby('Tratamiento')['Ganancia_Peso_g'].agg([
                ('N', 'count'), ('Media', 'mean'), ('Desv.Est.', 'std'), ('Mín', 'min'), ('Máx', 'max')
            ]).round(2)
            st.dataframe(summary, use_container_width=True)
        
        with tab2:
            result_uni = calcular_anova_unifactorial_pasos(df)
            st.markdown("---")
            mostrar_tabla_anova_unifactorial(result_uni)
            
            if result_uni['P_Value'] < alpha_custom:
                st.markdown("---")
                tukey_df, medias = tukey_hsd(df)
                st.markdown("### 🔍 Prueba de Tukey HSD")
                st.dataframe(tukey_df, use_container_width=True, hide_index=True)
        
        with tab3:
            result_bif = calcular_anova_bifactorial_pasos(df)
            st.markdown("---")
            mostrar_tabla_anova_bifactorial(result_bif)
        
        with tab4:
            crear_graficos(df, result_uni)
        
        with tab5:
            mostrar_interpretaciones(df, result_uni)
        
        with tab6:
            tukey_df, _ = tukey_hsd(df) if result_uni['P_Value'] < alpha_custom else (pd.DataFrame(), None)
            excel_data = exportar_excel(df, result_uni, result_bif, tukey_df)
            st.download_button("📥 Descargar Excel", excel_data, 
                             f"{titulo.lower().replace(' ', '_')}.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    if modelo_seleccionado == "Modelo 1: Balanceado":
        df = generar_datos_modelo1()
        mostrar_analisis_completo(df, "Modelo 1: DCA Balanceado", 
                                 "60 cuyes (15 por tratamiento)")
    
    elif modelo_seleccionado == "Modelo 2: No Balanceado":
        df = generar_datos_modelo2()
        mostrar_analisis_completo(df, "Modelo 2: DCA No Balanceado",
                                 "68 cuyes (14,18,16,20)")
    
    elif modelo_seleccionado == "Modelo 3: Bal-Bal (Sub)":
        df = generar_datos_modelo3()
        mostrar_analisis_completo(df, "Modelo 3: Bal-Bal",
                                 "20 pozas, 4 cuyes/poza")
    
    elif modelo_seleccionado == "Modelo 4: Bal-NoBal (Sub)":
        df = generar_datos_modelo4()
        mostrar_analisis_completo(df, "Modelo 4: Bal-NoBal",
                                 "20 pozas, 3-5 cuyes/poza")
    
    elif modelo_seleccionado == "Modelo 5: NoBal-Bal (Sub)":
        df = generar_datos_modelo5()
        mostrar_analisis_completo(df, "Modelo 5: NoBal-Bal",
                                 "4-7 pozas, 4 cuyes/poza")
    
    elif modelo_seleccionado == "Modelo 6: NoBal-NoBal (Sub)":
        df = generar_datos_modelo6()
        mostrar_analisis_completo(df, "Modelo 6: NoBal-NoBal",
                                 "Completamente desbalanceado")

# ==================== SECCIÓN MIS DATOS ====================
elif seccion == "📤 Mis Datos":
    analizar_datos_propios()

# ==================== COMPARACIÓN ====================
elif seccion == "📈 Comparación de Modelos":
    st.header("📈 Comparación entre Modelos")
    
    modelos_data = {
        "Modelo 1": generar_datos_modelo1(),
        "Modelo 2": generar_datos_modelo2(),
        "Modelo 3": generar_datos_modelo3(),
        "Modelo 4": generar_datos_modelo4(),
        "Modelo 5": generar_datos_modelo5(),
        "Modelo 6": generar_datos_modelo6()
    }
    
    comparacion = []
    for nombre, df in modelos_data.items():
        grupos = [df[df['Tratamiento'] == t]['Ganancia_Peso_g'].values for t in df['Tratamiento'].unique()]
        f_stat, p_value = stats.f_oneway(*grupos)
        
        comparacion.append({
            'Modelo': nombre,
            'n Total': len(df),
            'Media General': round(df['Ganancia_Peso_g'].mean(), 1),
            'F': round(f_stat, 4),
            'P-valor': round(p_value, 6),
            'Significativo': 'Sí ✓' if p_value < alpha_custom else 'No ✗'
        })
    
    comp_df = pd.DataFrame(comparacion)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(comp_df, x='Modelo', y='F', 
                     title='Comparación de Estadísticos F',
                     color='Significativo',
                     color_discrete_map={'Sí ✓': '#28a745', 'No ✗': '#dc3545'})
        fig1.update_layout(height=450)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(comp_df, x='n Total', y='P-valor',
                         title='P-valor vs Tamaño Muestral',
                         color='Significativo',
                         size='F',
                         color_discrete_map={'Sí ✓': '#28a745', 'No ✗': '#dc3545'})
        fig2.add_hline(y=alpha_custom, line_dash="dash", line_color="red")
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Dina Maribel Yana Yucra | Código: 221086</p>
    <p style='font-size: 12px;'>Aplicación Streamlit - Análisis Estadístico DCA</p>
</div>
""", unsafe_allow_html=True)
