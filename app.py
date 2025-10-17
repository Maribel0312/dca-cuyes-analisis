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
seccion = st.sidebar.radio(
    "Seleccione una sección:",
    ["🏠 Inicio", "📚 Teoría", "📊 Modelos Experimentales", "📈 Comparación de Modelos"]
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

# Funciones para generar datos - CADA MODELO CON SEMILLA DIFERENTE Y CARACTERÍSTICAS ÚNICAS
def generar_datos_modelo1():
    """Modelo 1: Balanceado - Mayor uniformidad"""
    np.random.seed(42)
    datos = []
    medias = {"T1": 520, "T2": 580, "T3": 545, "T4": 595}
    desv = {"T1": 28, "T2": 32, "T3": 25, "T4": 35}  # Baja variabilidad
    
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
    """Modelo 2: No Balanceado - Mayor variabilidad entre grupos"""
    np.random.seed(123)
    datos = []
    medias = {"T1": 515, "T2": 590, "T3": 540, "T4": 605}  # Diferencias más marcadas
    desv = {"T1": 40, "T2": 35, "T3": 38, "T4": 42}  # Mayor variabilidad
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
    """Modelo 3: Bal-Bal Sub - Efecto de poza moderado"""
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
    """Modelo 4: Bal-NoBal Sub - Submuestreo desigual"""
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
    """Modelo 5: NoBal-Bal Sub - Pozas desiguales"""
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
    """Modelo 6: NoBal-NoBal Sub - Completamente desbalanceado"""
    np.random.seed(654)
    datos = []
    medias = {"T1": 510, "T2": 595, "T3": 535, "T4": 610}  # Diferencias máximas
    desv_poza = {"T1": 30, "T2": 33, "T3": 27, "T4": 36}  # Mayor variabilidad
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
    
    # Paso 1: Datos básicos
    st.markdown("#### Paso 1: Identificación de datos básicos")
    n_total = len(df)
    tratamientos = sorted(df['Tratamiento'].unique())
    k = len(tratamientos)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("N total", n_total)
    col2.metric("Tratamientos (k)", k)
    col3.metric("Grupos", k)
    
    # Paso 2: Medias por tratamiento
    st.markdown("#### Paso 2: Cálculo de medias por tratamiento")
    medias_df = df.groupby('Tratamiento').agg({
        'Ganancia_Peso_g': ['count', 'mean', 'sum']
    }).round(2)
    medias_df.columns = ['n', 'Media', 'Suma']
    st.dataframe(medias_df, use_container_width=True)
    
    grand_mean = df['Ganancia_Peso_g'].mean()
    st.info(f"**Media General (Ȳ..):** {grand_mean:.2f} g")
    
    # Paso 3: Suma de Cuadrados
    st.markdown("#### Paso 3: Cálculo de Sumas de Cuadrados")
    
    # SC Total
    st.markdown("**3.1. Suma de Cuadrados Total (SCT)**")
    st.latex(r"SCT = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(Y_{ij} - \bar{Y}_{..})^2")
    
    ss_total = ((df['Ganancia_Peso_g'] - grand_mean) ** 2).sum()
    st.write(f"SCT = {ss_total:.2f}")
    
    # SC Entre Tratamientos
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
    
    # SC Error
    st.markdown("**3.3. Suma de Cuadrados del Error (SC Error)**")
    st.latex(r"SC_{Error} = SCT - SC_{Trat}")
    
    ss_within = ss_total - ss_between
    st.write(f"SC Error = {ss_total:.2f} - {ss_between:.2f} = **{ss_within:.2f}**")
    
    # Paso 4: Grados de Libertad
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
    
    # Paso 5: Cuadrados Medios
    st.markdown("#### Paso 5: Cuadrados Medios")
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    st.latex(r"CM_{Trat} = \frac{SC_{Trat}}{GL_{Trat}} = \frac{" + f"{ss_between:.2f}" + r"}{" + str(df_between) + r"} = " + f"{ms_between:.2f}")
    st.latex(r"CM_{Error} = \frac{SC_{Error}}{GL_{Error}} = \frac{" + f"{ss_within:.2f}" + r"}{" + str(df_within) + r"} = " + f"{ms_within:.2f}")
    
    # Paso 6: Estadístico F
    st.markdown("#### Paso 6: Estadístico F")
    f_calc = ms_between / ms_within
    st.latex(r"F = \frac{CM_{Trat}}{CM_{Error}} = \frac{" + f"{ms_between:.2f}" + r"}{" + f"{ms_within:.2f}" + r"} = " + f"{f_calc:.4f}")
    
    # Paso 7: P-valor
    st.markdown("#### Paso 7: P-valor")
    p_value = 1 - stats.f.cdf(f_calc, df_between, df_within)
    st.write(f"P-valor = P(F > {f_calc:.4f}) = **{p_value:.6f}**")
    
    if p_value < 0.05:
        st.success("✅ Como p-valor < 0.05, rechazamos H₀")
    else:
        st.warning("⚠️ Como p-valor ≥ 0.05, no rechazamos H₀")
    
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

# CÁLCULO ANOVA BIFACTORIAL COMPLETO PASO A PASO
def calcular_anova_bifactorial_pasos(df):
    st.markdown("### 📐 Cálculos Completos Paso a Paso - ANOVA Bifactorial")
    
    st.info("**Diseño Bifactorial:** Factor A = Tratamiento (4 niveles), Factor B = Sexo simulado (2 niveles: Macho/Hembra)")
    
    # Crear Factor B basado en índice de forma determinística
    df_bif = df.copy()
    df_bif['Factor_A'] = df_bif['Tratamiento']
    
    # Asignar sexo de forma balanceada por tratamiento
    sexos = []
    for trat in sorted(df_bif['Tratamiento'].unique()):
        subset = df_bif[df_bif['Tratamiento'] == trat]
        n_trat = len(subset)
        n_machos = n_trat // 2
        sexos_trat = ['Macho'] * n_machos + ['Hembra'] * (n_trat - n_machos)
        sexos.extend(sexos_trat)
    
    df_bif['Factor_B'] = sexos
    
    # PASO 1: Estructura del diseño
    st.markdown("#### Paso 1: Estructura del Diseño Bifactorial")
    
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
    
    st.write(f"**Factor A (Tratamiento):** {', '.join(factor_a_levels)}")
    st.write(f"**Factor B (Sexo):** {', '.join(factor_b_levels)}")
    
    # Conteo por celda
    st.markdown("**Frecuencias por Celda:**")
    freq_table = df_bif.pivot_table(values='Ganancia_Peso_g', 
                                     index='Factor_A', 
                                     columns='Factor_B', 
                                     aggfunc='count',
                                     fill_value=0)
    st.dataframe(freq_table, use_container_width=True)
    
    # PASO 2: Tabla de medias
    st.markdown("#### Paso 2: Cálculo de Medias")
    st.latex(r"\bar{Y}_{ij.} = \frac{\sum_{k=1}^{n_{ij}} Y_{ijk}}{n_{ij}}")
    
    tabla_medias = df_bif.pivot_table(values='Ganancia_Peso_g', 
                                       index='Factor_A', 
                                       columns='Factor_B', 
                                       aggfunc='mean')
    st.write("**Medias por Celda (Ȳᵢⱼ.):**")
    st.dataframe(tabla_medias.round(2), use_container_width=True)
    
    # Medias marginales
    medias_a = df_bif.groupby('Factor_A')['Ganancia_Peso_g'].mean()
    medias_b = df_bif.groupby('Factor_B')['Ganancia_Peso_g'].mean()
    grand_mean = df_bif['Ganancia_Peso_g'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Medias Marginales Factor A (Ȳᵢ..):**")
        medias_a_df = medias_a.round(2).to_frame('Media')
        st.dataframe(medias_a_df, use_container_width=True)
    with col2:
        st.write("**Medias Marginales Factor B (Ȳ.ⱼ.):**")
        medias_b_df = medias_b.round(2).to_frame('Media')
        st.dataframe(medias_b_df, use_container_width=True)
    
    st.success(f"**Media General (Ȳ...):** {grand_mean:.2f} g")
    
    # PASO 3: Suma de Cuadrados Total
    st.markdown("#### Paso 3: Suma de Cuadrados Total (SCT)")
    st.latex(r"SCT = \sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}(Y_{ijk} - \bar{Y}_{...})^2")
    
    ss_total = ((df_bif['Ganancia_Peso_g'] - grand_mean) ** 2).sum()
    st.write(f"**SCT = {ss_total:.2f}**")
    
    # PASO 4: Suma de Cuadrados Factor A
    st.markdown("#### Paso 4: Suma de Cuadrados Factor A (SC_A)")
    st.latex(r"SC_A = bn\sum_{i=1}^{a}(\bar{Y}_{i..} - \bar{Y}_{...})^2")
    st.write("Donde n es el promedio de observaciones por nivel de A")
    
    calc_a = []
    ss_a = 0
    for nivel_a in factor_a_levels:
        n_a = len(df_bif[df_bif['Factor_A'] == nivel_a])
        media_a = medias_a[nivel_a]
        ss_a_i = n_a * (media_a - grand_mean) ** 2
        ss_a += ss_a_i
        calc_a.append({
            'Nivel A': nivel_a,
            'n': n_a,
            'Media': f"{media_a:.2f}",
            'Cálculo': f"{n_a} × ({media_a:.2f} - {grand_mean:.2f})²",
            'SC': f"{ss_a_i:.2f}"
        })
    
    st.dataframe(pd.DataFrame(calc_a), use_container_width=True, hide_index=True)
    st.write(f"**SC_A = {ss_a:.2f}**")
    
    # PASO 5: Suma de Cuadrados Factor B
    st.markdown("#### Paso 5: Suma de Cuadrados Factor B (SC_B)")
    st.latex(r"SC_B = an\sum_{j=1}^{b}(\bar{Y}_{.j.} - \bar{Y}_{...})^2")
    
    calc_b = []
    ss_b = 0
    for nivel_b in factor_b_levels:
        n_b = len(df_bif[df_bif['Factor_B'] == nivel_b])
        media_b = medias_b[nivel_b]
        ss_b_j = n_b * (media_b - grand_mean) ** 2
        ss_b += ss_b_j
        calc_b.append({
            'Nivel B': nivel_b,
            'n': n_b,
            'Media': f"{media_b:.2f}",
            'Cálculo': f"{n_b} × ({media_b:.2f} - {grand_mean:.2f})²",
            'SC': f"{ss_b_j:.2f}"
        })
    
    st.dataframe(pd.DataFrame(calc_b), use_container_width=True, hide_index=True)
    st.write(f"**SC_B = {ss_b:.2f}**")
    
    # PASO 6: Suma de Cuadrados Interacción AB
    st.markdown("#### Paso 6: Suma de Cuadrados Interacción AB (SC_AB)")
    st.latex(r"SC_{AB} = n\sum_{i=1}^{a}\sum_{j=1}^{b}(\bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...})^2")
    
    calc_ab = []
    ss_ab = 0
    for nivel_a in factor_a_levels:
        for nivel_b in factor_b_levels:
            subset = df_bif[(df_bif['Factor_A'] == nivel_a) & (df_bif['Factor_B'] == nivel_b)]
            if len(subset) > 0:
                n_cell = len(subset)
                mean_cell = subset['Ganancia_Peso_g'].mean()
                interaction_effect = mean_cell - medias_a[nivel_a] - medias_b[nivel_b] + grand_mean
                ss_ab_ij = n_cell * (interaction_effect ** 2)
                ss_ab += ss_ab_ij
                calc_ab.append({
                    'Celda': f"{nivel_a}-{nivel_b}",
                    'n': n_cell,
                    'Ȳᵢⱼ': f"{mean_cell:.2f}",
                    'Efecto Int.': f"{interaction_effect:.2f}",
                    'SC': f"{ss_ab_ij:.2f}"
                })
    
    st.dataframe(pd.DataFrame(calc_ab), use_container_width=True, hide_index=True)
    st.write(f"**SC_AB = {ss_ab:.2f}**")
    
    # PASO 7: Suma de Cuadrados Error
    st.markdown("#### Paso 7: Suma de Cuadrados Error (SC_Error)")
    st.latex(r"SC_{Error} = SCT - SC_A - SC_B - SC_{AB}")
    
    ss_error = ss_total - ss_a - ss_b - ss_ab
    st.write(f"SC_Error = {ss_total:.2f} - {ss_a:.2f} - {ss_b:.2f} - {ss_ab:.2f}")
    st.write(f"**SC_Error = {ss_error:.2f}**")
    
    # PASO 8: Grados de Libertad
    st.markdown("#### Paso 8: Grados de Libertad")
    
    df_a = a - 1
    df_b = b - 1
    df_ab = (a - 1) * (b - 1)
    df_error = n_total - (a * b)
    df_total = n_total - 1
    
    gl_df = pd.DataFrame({
        'Fuente': ['Factor A', 'Factor B', 'Interacción AB', 'Error', 'Total'],
        'Fórmula': [
            f'a - 1 = {a} - 1',
            f'b - 1 = {b} - 1',
            f'(a-1)(b-1) = ({a}-1)({b}-1)',
            f'N - ab = {n_total} - ({a}×{b})',
            f'N - 1 = {n_total} - 1'
        ],
        'GL': [df_a, df_b, df_ab, df_error, df_total]
    })
    st.dataframe(gl_df, use_container_width=True, hide_index=True)
    
    # PASO 9: Cuadrados Medios
    st.markdown("#### Paso 9: Cuadrados Medios (CM)")
    st.latex(r"CM = \frac{SC}{GL}")
    
    cm_a = ss_a / df_a if df_a > 0 else 0
    cm_b = ss_b / df_b if df_b > 0 else 0
    cm_ab = ss_ab / df_ab if df_ab > 0 else 0
    cm_error = ss_error / df_error if df_error > 0 else 1
    
    cm_df = pd.DataFrame({
        'Fuente': ['Factor A', 'Factor B', 'Interacción AB', 'Error'],
        'Cálculo': [
            f'{ss_a:.2f} / {df_a}',
            f'{ss_b:.2f} / {df_b}',
            f'{ss_ab:.2f} / {df_ab}',
            f'{ss_error:.2f} / {df_error}'
        ],
        'CM': [f"{cm_a:.2f}", f"{cm_b:.2f}", f"{cm_ab:.2f}", f"{cm_error:.2f}"]
    })
    st.dataframe(cm_df, use_container_width=True, hide_index=True)
    
    # PASO 10: Estadísticos F
    st.markdown("#### Paso 10: Estadísticos F y P-valores")
    st.latex(r"F = \frac{CM_{Efecto}}{CM_{Error}}")
    
    f_a = cm_a / cm_error if cm_error > 0 else 0
    f_b = cm_b / cm_error if cm_error > 0 else 0
    f_ab = cm_ab / cm_error if cm_error > 0 else 0
    
    p_a = 1 - stats.f.cdf(f_a, df_a, df_error) if f_a > 0 else 1
    p_b = 1 - stats.f.cdf(f_b, df_b, df_error) if f_b > 0 else 1
    p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_error) if f_ab > 0 else 1
    
    f_result_df = pd.DataFrame({
        'Fuente': ['Factor A (Tratamiento)', 'Factor B (Sexo)', 'Interacción A×B'],
        'F calculado': [f"{f_a:.4f}", f"{f_b:.4f}", f"{f_ab:.4f}"],
        'P-valor': [f"{p_a:.6f}", f"{p_b:.6f}", f"{p_ab:.6f}"],
        'Significativo (α=0.05)': [
            '✅ Sí' if p_a < 0.05 else '❌ No',
            '✅ Sí' if p_b < 0.05 else '❌ No',
            '✅ Sí' if p_ab < 0.05 else '❌ No'
        ]
    })
    st.dataframe(f_result_df, use_container_width=True, hide_index=True)
    
    # Interpretación
    st.markdown("#### 📊 Interpretación de Resultados:")
    
    if p_a < 0.05:
        st.success(f"✅ **Factor A (Tratamiento):** Efecto significativo (p = {p_a:.6f}). Los tratamientos producen diferentes ganancias de peso.")
    else:
        st.warning(f"⚠️ **Factor A (Tratamiento):** No hay efecto significativo (p = {p_a:.6f}).")
    
    if p_b < 0.05:
        st.success(f"✅ **Factor B (Sexo):** Efecto significativo (p = {p_b:.6f}). El sexo influye en la ganancia de peso.")
    else:
        st.info(f"ℹ️ **Factor B (Sexo):** No hay efecto significativo (p = {p_b:.6f}).")
    
    if p_ab < 0.05:
        st.success(f"✅ **Interacción A×B:** Significativa (p = {p_ab:.6f}). El efecto del tratamiento depende del sexo del animal.")
    else:
        st.info(f"ℹ️ **Interacción A×B:** No significativa (p = {p_ab:.6f}). Los factores actúan independientemente.")
    
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
    """Crear gráficos necesarios con interpretaciones"""
    
    medias = df.groupby('Tratamiento')['Ganancia_Peso_g'].mean().sort_values(ascending=False)
    mejor_trat = medias.index[0]
    mejor_media = medias.iloc[0]
    
    st.markdown("## 📊 Visualización de Resultados")
    
    # Gráfico 1: Boxplot
    st.markdown("### 1. Distribución de Datos por Tratamiento")
    fig_box = px.box(df, x='Tratamiento', y='Ganancia_Peso_g',
                     title='Distribución de Ganancia de Peso por Tratamiento',
                     labels={'Ganancia_Peso_g': 'Ganancia de Peso (g)', 'Tratamiento': 'Tratamiento'},
                     color='Tratamiento',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_box.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("""
    **📌 Interpretación del Boxplot:**
    - Este gráfico muestra la distribución, variabilidad y valores atípicos de cada tratamiento
    - La **línea central** de cada caja representa la mediana
    - Los **bordes de la caja** representan el primer y tercer cuartil (Q1 y Q3)
    - Los **bigotes** muestran el rango de datos dentro de 1.5×IQR
    - Los **puntos** fuera de los bigotes son valores atípicos
    """)
    
    # Gráfico 2: Violin Plot
    st.markdown("### 2. Densidad y Distribución de Datos")
    fig_violin = px.violin(df, x='Tratamiento', y='Ganancia_Peso_g',
                          title='Gráfico de Violín - Densidad de Distribución',
                          labels={'Ganancia_Peso_g': 'Ganancia de Peso (g)', 'Tratamiento': 'Tratamiento'},
                          color='Tratamiento',
                          box=True,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_violin.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig_violin, use_container_width=True)
    
    st.markdown("""
    **📌 Interpretación del Gráfico de Violín:**
    - Combina la información del boxplot con la densidad de distribución
    - El **ancho** del violín indica la densidad de datos en ese valor
    - Permite comparar visualmente la forma de distribución entre tratamientos
    - Un violín más ancho indica mayor concentración de datos en ese rango
    """)
    
    # Gráfico 3: Intervalos de Confianza
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
    
    st.markdown("""
    **📌 Interpretación de Intervalos de Confianza:**
    - Los **puntos** representan la media de cada tratamiento
    - Las **barras de error** muestran el intervalo de confianza del 95%
    - Si dos intervalos **NO se solapan**, existe alta probabilidad de diferencia significativa
    - Tratamientos con intervalos que se solapan pueden no ser significativamente diferentes
    """)
    
    # Gráfico 4: QQ-Plot para normalidad
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
        title='QQ-Plot - Verificación de Normalidad de Residuos',
        xaxis_title='Cuantiles Teóricos',
        yaxis_title='Cuantiles Muestrales',
        height=500
    )
    st.plotly_chart(fig_qq, use_container_width=True)
    
    st.markdown("""
    **📌 Interpretación del QQ-Plot:**
    - Verifica el supuesto de **normalidad** de los datos
    - Si los puntos siguen aproximadamente la línea roja, los datos son normales
    - Desviaciones en los extremos son comunes y aceptables
    - Este supuesto es importante para la validez del ANOVA
    """)
    
    # Gráfico 5: Residuos vs Valores Ajustados
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
        xaxis_title='Valores Ajustados (Medias de Tratamiento)',
        yaxis_title='Residuos',
        height=500
    )
    st.plotly_chart(fig_resid, use_container_width=True)
    
    st.markdown("""
    **📌 Interpretación del Gráfico de Residuos:**
    - Verifica el supuesto de **homocedasticidad** (varianza constante)
    - Los residuos deben distribuirse aleatoriamente alrededor de cero
    - **No debe haber patrones** (forma de embudo, curvas, etc.)
    - Patrones indican violación de supuestos del ANOVA
    """)

def mostrar_interpretaciones(df, result_uni):
    """Mostrar interpretaciones detalladas"""
    
    st.markdown("## 💡 Interpretaciones y Conclusiones")
    
    medias = df.groupby('Tratamiento')['Ganancia_Peso_g'].mean().sort_values(ascending=False)
    mejor_trat = medias.index[0]
    mejor_media = medias.iloc[0]
    peor_trat = medias.index[-1]
    peor_media = medias.iloc[-1]
    
    descripciones = {
        'T1': 'Alimento balanceado comercial (18% proteína)',
        'T2': 'Alimento con forraje verde (alfalfa + concentrado 20% proteína)',
        'T3': 'Dieta mixta (forraje + subproductos agrícolas + 16% proteína)',
        'T4': 'Alimento suplementado con probióticos (19% proteína)'
    }
    
    st.markdown("### 🏆 El Mejor Tratamiento")
    st.success(f"""
    **Tratamiento {mejor_trat}** es el mejor tratamiento con una ganancia promedio de **{mejor_media:.1f} g**.
    
    **Descripción:** {descripciones[mejor_trat]}
    """)
    
    st.markdown("### 📊 ¿Por qué es el mejor tratamiento?")
    
    if result_uni['P_Value'] < 0.05:
        tukey_df, _ = tukey_hsd(df)
        comparaciones_mejor = tukey_df[tukey_df['Comparación'].str.contains(mejor_trat)]
        n_sig = comparaciones_mejor[comparaciones_mejor['Significativo'] == 'Sí'].shape[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Evidencia Estadística")
            st.write(f"""
            ✅ **Diferencias significativas detectadas:**
            - El ANOVA mostró p-valor = {result_uni['P_Value']:.6f} < 0.05
            - {mejor_trat} es superior a **{n_sig}** tratamiento(s) según Tukey HSD
            - La diferencia NO se debe al azar, sino al efecto real del tratamiento
            
            ✅ **Magnitud del efecto:**
            - Diferencia vs peor tratamiento: {mejor_media - peor_media:.1f} g
            - Representa un {((mejor_media - peor_media)/peor_media * 100):.1f}% más de ganancia
            """)
        
        with col2:
            st.markdown("#### 🔬 Explicación Biológica")
            
            if mejor_trat == 'T1':
                st.write("""
                **Alimento Balanceado Comercial:**
                - Formulación industrial optimizada
                - Balance perfecto de nutrientes
                - Digestibilidad constante
                - Control de calidad riguroso
                """)
            elif mejor_trat == 'T2':
                st.write("""
                **Alimento con Forraje Verde:**
                - Mayor contenido proteico (20%)
                - Mejor palatabilidad
                - Fibra de calidad (alfalfa)
                - Estimula consumo voluntario
                - Ácidos grasos omega-3 presentes
                """)
            elif mejor_trat == 'T3':
                st.write("""
                **Dieta Mixta:**
                - Aprovechamiento de subproductos
                - Diversidad de nutrientes
                - Costo-beneficio favorable
                - Adaptación local
                """)
            elif mejor_trat == 'T4':
                st.write("""
                **Alimento con Probióticos:**
                - Mejora salud intestinal
                - Mayor absorción de nutrientes
                - Fortalece sistema inmune
                - Reduce mortalidad
                - Optimiza conversión alimenticia
                """)
        
        st.markdown("#### 📋 Tabla Comparativa de Tratamientos")
        ranking_df = pd.DataFrame({
            'Posición': range(1, len(medias) + 1),
            'Tratamiento': medias.index,
            'Descripción': [descripciones[t] for t in medias.index],
            'Ganancia Media (g)': medias.values.round(1),
            'Diferencia vs Mejor (g)': [0] + [(mejor_media - m) for m in medias.values[1:]],
            'Eficiencia (%)': [100] + [(m/mejor_media * 100) for m in medias.values[1:]]
        })
        st.dataframe(ranking_df.round(2), use_container_width=True, hide_index=True)
        
    else:
        st.warning(f"""
        **⚠️ Consideración Importante:**
        
        Aunque **{mejor_trat}** presenta la mayor ganancia promedio ({mejor_media:.1f} g), 
        el ANOVA no encontró diferencias estadísticamente significativas (p = {result_uni['P_Value']:.4f}).
        
        **Esto significa:**
        - Las diferencias observadas podrían deberse a variación aleatoria
        - No hay evidencia suficiente para afirmar superioridad real
        - Cualquier tratamiento podría producir resultados similares
        
        **Recomendaciones:**
        1. Aumentar el tamaño de muestra
        2. Extender el período experimental
        3. Controlar mejor las fuentes de variación
        4. Considerar análisis económico complementario
        """)
    
    st.markdown("### 🎯 Recomendaciones Prácticas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Para Producción Comercial")
        if result_uni['P_Value'] < 0.05:
            st.success(f"""
            ✅ **Implementar {mejor_trat}** en producción:
            - Ganancia promedio esperada: {mejor_media:.1f} g en 8 semanas
            - Superioridad estadísticamente comprobada
            - Proyección de peso comercial: {250 + mejor_media:.1f} g
            """)
        else:
            st.info("""
            💡 **Análisis complementario necesario:**
            - Evaluar costo-beneficio de cada tratamiento
            - Considerar disponibilidad de insumos
            - Analizar preferencias del mercado local
            """)
    
    with col2:
        st.markdown("#### Para Investigación Futura")
        st.write("""
        🔬 **Líneas de investigación sugeridas:**
        - Evaluar conversión alimenticia
        - Medir calidad de carne obtenida
        - Analizar costos de producción
        - Estudiar viabilidad económica
        - Evaluar aceptación del consumidor
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

# ==================== SECCIÓN INICIO ====================
if seccion == "🏠 Inicio":
    st.markdown("---")
    st.markdown("## 📄 Contexto del Caso")
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;'>
        <p style='font-size: 16px; line-height: 1.8;'>
        Este estudio tiene como objetivo <b>determinar el mejor alimento o dieta de engorde</b> para cuyes 
        hasta alcanzar un peso comercial óptimo.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔬 Factor Experimental")
        st.info("**Tipo de alimento o dieta de engorde** para cuyes")
        
        st.markdown("### 📋 Tratamientos Evaluados")
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
        st.success("**Ganancia de peso en gramos** después de 8 semanas")
        
    with col2:
        st.markdown("### 📚 Modelos Disponibles")
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>
        <p><b>1️⃣</b> Balanceado</p>
        <p><b>2️⃣</b> No Balanceado</p>
        <p><b>3️⃣</b> Bal-Bal (Sub)</p>
        <p><b>4️⃣</b> Bal-NoBal (Sub)</p>
        <p><b>5️⃣</b> NoBal-Bal (Sub)</p>
        <p><b>6️⃣</b> NoBal-NoBal (Sub)</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== SECCIÓN TEORÍA ====================
elif seccion == "📚 Teoría":
    st.header("📚 Marco Teórico")
    
    tab1, tab2 = st.tabs(["🔢 Modelo Unifactorial", "🔢 Modelo Bifactorial"])
    
    with tab1:
        st.markdown("## Diseño Completamente al Azar (DCA) - Unifactorial")
        st.markdown("### 📐 Modelo Estadístico")
        st.latex(r"Y_{ij} = \mu + \tau_i + \varepsilon_{ij}")
        st.markdown("### 🧮 Fórmulas")
        st.latex(r"SC_{Total} = \sum_{i=1}^{t}\sum_{j=1}^{n_i}(Y_{ij} - \bar{Y}_{..})^2")
        st.latex(r"SC_{Trat} = \sum_{i=1}^{t}n_i(\bar{Y}_{i.} - \bar{Y}_{..})^2")
        st.latex(r"F = \frac{CM_{Trat}}{CM_{Error}}")
    
    with tab2:
        st.markdown("## Diseño Bifactorial")
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
            
            # Resumen estadístico
            st.markdown("### 📊 Resumen Estadístico")
            summary = df.groupby('Tratamiento')['Ganancia_Peso_g'].agg([
                ('N', 'count'),
                ('Media', 'mean'),
                ('Desv.Est.', 'std'),
                ('Mín', 'min'),
                ('Máx', 'max')
            ]).round(2)
            st.dataframe(summary, use_container_width=True)
        
        with tab2:
            result_uni = calcular_anova_unifactorial_pasos(df)
            st.markdown("---")
            mostrar_tabla_anova_unifactorial(result_uni)
            
            # Tukey HSD si es significativo
            if result_uni['P_Value'] < 0.05:
                st.markdown("---")
                st.markdown("### 🔍 Prueba de Tukey HSD")
                tukey_df, medias = tukey_hsd(df)
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
            st.subheader("📥 Exportar Resultados")
            st.write("Descargue un archivo Excel completo con:")
            st.write("- ✅ Datos experimentales")
            st.write("- ✅ ANOVA Unifactorial")
            st.write("- ✅ ANOVA Bifactorial")
            st.write("- ✅ Prueba de Tukey HSD")
            st.write("- ✅ Estadísticas descriptivas")
            
            tukey_df, _ = tukey_hsd(df) if result_uni['P_Value'] < 0.05 else (pd.DataFrame(), None)
            excel_data = exportar_excel(df, result_uni, result_bif, tukey_df)
            st.download_button("📥 Descargar Excel Completo", excel_data, 
                             f"{titulo.lower().replace(' ', '_').replace(':', '')}.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    if modelo_seleccionado == "Modelo 1: Balanceado":
        df = generar_datos_modelo1()
        mostrar_analisis_completo(df, "Modelo 1: DCA Balanceado", 
                                 "📌 Estructura: 60 cuyes distribuidos equitativamente (15 cuyes por tratamiento)")
    
    elif modelo_seleccionado == "Modelo 2: No Balanceado":
        df = generar_datos_modelo2()
        mostrar_analisis_completo(df, "Modelo 2: DCA No Balanceado",
                                 "📌 Estructura: 68 cuyes con distribución desigual (T1:14, T2:18, T3:16, T4:20)")
    
    elif modelo_seleccionado == "Modelo 3: Bal-Bal (Sub)":
        df = generar_datos_modelo3()
        mostrar_analisis_completo(df, "Modelo 3: Muestreo Bal-Bal con Submuestreo",
                                 "📌 Estructura: 20 pozas (5 por tratamiento), 4 cuyes por poza = 80 cuyes")
    
    elif modelo_seleccionado == "Modelo 4: Bal-NoBal (Sub)":
        df = generar_datos_modelo4()
        mostrar_analisis_completo(df, "Modelo 4: Muestreo Bal-NoBal con Submuestreo",
                                 "📌 Estructura: 20 pozas (5 por tratamiento), cuyes variables por poza (T1:3, T2:4, T3:5, T4:3)")
    
    elif modelo_seleccionado == "Modelo 5: NoBal-Bal (Sub)":
        df = generar_datos_modelo5()
        mostrar_analisis_completo(df, "Modelo 5: Muestreo NoBal-Bal con Submuestreo",
                                 "📌 Estructura: Pozas desiguales (T1:4, T2:6, T3:5, T4:7), 4 cuyes por poza = 88 cuyes")
    
    elif modelo_seleccionado == "Modelo 6: NoBal-NoBal (Sub)":
        df = generar_datos_modelo6()
        mostrar_analisis_completo(df, "Modelo 6: Muestreo Completamente Desbalanceado",
                                 "📌 Estructura: Pozas y cuyes completamente desiguales (100 cuyes totales)")

# ==================== COMPARACIÓN DE MODELOS ====================
elif seccion == "📈 Comparación de Modelos":
    st.header("📈 Comparación entre Modelos Experimentales")
    
    st.info("Esta sección compara los resultados del ANOVA entre todos los modelos experimentales")
    
    modelos_data = {
        "Modelo 1\nBalanceado": generar_datos_modelo1(),
        "Modelo 2\nNo Balanceado": generar_datos_modelo2(),
        "Modelo 3\nBal-Bal (Sub)": generar_datos_modelo3(),
        "Modelo 4\nBal-NoBal (Sub)": generar_datos_modelo4(),
        "Modelo 5\nNoBal-Bal (Sub)": generar_datos_modelo5(),
        "Modelo 6\nNoBal-NoBal (Sub)": generar_datos_modelo6()
    }
    
    comparacion = []
    for nombre, df in modelos_data.items():
        grupos = [df[df['Tratamiento'] == t]['Ganancia_Peso_g'].values for t in df['Tratamiento'].unique()]
        f_stat, p_value = stats.f_oneway(*grupos)
        
        # Media general
        media_general = df['Ganancia_Peso_g'].mean()
        
        comparacion.append({
            'Modelo': nombre.replace('\n', ' '),
            'n Total': len(df),
            'Media General': round(media_general, 1),
            'F-statistic': round(f_stat, 4),
            'P-valor': round(p_value, 6),
            'Significativo': 'Sí ✓' if p_value < 0.05 else 'No ✗'
        })
    
    comp_df = pd.DataFrame(comparacion)
    
    st.markdown("### 📊 Tabla Comparativa de Resultados")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # Análisis comparativo
    st.markdown("### 📈 Análisis Comparativo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Estadísticos F por Modelo")
        fig1 = px.bar(comp_df, x='Modelo', y='F-statistic', 
                     title='Comparación de Estadísticos F entre Modelos',
                     color='Significativo',
                     labels={'F-statistic': 'Valor F'},
                     color_discrete_map={'Sí ✓': '#28a745', 'No ✗': '#dc3545'})
        fig1.update_layout(height=450, xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        **Interpretación:**
        - Valores F más altos indican mayor diferencia entre tratamientos
        - Modelos en verde tienen diferencias significativas (p < 0.05)
        - Modelos en rojo no detectaron diferencias significativas
        """)
    
    with col2:
        st.markdown("#### P-valores vs Tamaño Muestral")
        fig2 = px.scatter(comp_df, x='n Total', y='P-valor',
                         title='Relación entre Tamaño Muestral y P-valor',
                         color='Significativo',
                         size='F-statistic',
                         hover_data=['Modelo'],
                         color_discrete_map={'Sí ✓': '#28a745', 'No ✗': '#dc3545'})
        fig2.add_hline(y=0.05, line_dash="dash", line_color="red", 
                      annotation_text="α = 0.05")
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        **Interpretación:**
        - La línea roja representa el umbral de significancia (α = 0.05)
        - Puntos debajo de la línea tienen diferencias significativas
        - El tamaño de los puntos representa la magnitud del estadístico F
        """)
    
    # Resumen ejecutivo
    st.markdown("### 📋 Resumen Ejecutivo")
    
    n_significativos = comp_df[comp_df['Significativo'] == 'Sí ✓'].shape[0]
    mejor_f = comp_df.loc[comp_df['F-statistic'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modelos Significativos", f"{n_significativos}/6")
    
    with col2:
        st.metric("Mejor F-statistic", f"{mejor_f['F-statistic']:.4f}")
    
    with col3:
        st.metric("Modelo con Mayor F", mejor_f['Modelo'])
    
    st.markdown("---")
    
    # Comparación de medias por tratamiento
    st.markdown("### 🎯 Comparación de Medias por Tratamiento")
    
    medias_comparacion = []
    for nombre, df in modelos_data.items():
        medias_trat = df.groupby('Tratamiento')['Ganancia_Peso_g'].mean()
        for trat in medias_trat.index:
            medias_comparacion.append({
                'Modelo': nombre.replace('\n', ' '),
                'Tratamiento': trat,
                'Media': medias_trat[trat]
            })
    
    medias_df = pd.DataFrame(medias_comparacion)
    
    fig3 = px.line(medias_df, x='Tratamiento', y='Media', color='Modelo',
                   title='Medias de Ganancia de Peso por Tratamiento en Cada Modelo',
                   labels={'Media': 'Ganancia de Peso (g)'},
                   markers=True)
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("""
    **📌 Interpretación:**
    - Este gráfico muestra cómo varían las medias entre modelos
    - Cada línea representa un modelo experimental diferente
    - Permite identificar consistencia de resultados entre modelos
    - Tratamientos con líneas paralelas indican efectos consistentes
    """)
    
    # Tabla de medias detallada
    st.markdown("### 📊 Tabla Detallada de Medias por Tratamiento y Modelo")
    medias_pivot = medias_df.pivot_table(values='Media', 
                                         index='Tratamiento', 
                                         columns='Modelo')
    st.dataframe(medias_pivot.round(2), use_container_width=True)
    
    # Conclusiones
    st.markdown("### 💡 Conclusiones de la Comparación")
    
    if n_significativos >= 4:
        st.success(f"""
        ✅ **Resultados Robustos:**
        - {n_significativos} de 6 modelos mostraron diferencias significativas
        - Esto indica que los efectos de los tratamientos son consistentes
        - Los resultados son confiables independientemente del diseño usado
        """)
    elif n_significativos >= 2:
        st.info(f"""
        ℹ️ **Resultados Moderados:**
        - {n_significativos} de 6 modelos mostraron diferencias significativas
        - La detección de efectos puede depender del diseño experimental
        - Se recomienda enfocarse en los modelos con mayor poder estadístico
        """)
    else:
        st.warning(f"""
        ⚠️ **Resultados Limitados:**
        - Solo {n_significativos} modelo(s) mostraron diferencias significativas
        - Los efectos de los tratamientos pueden ser pequeños
        - Se recomienda aumentar tamaños muestrales o duración del experimento
        """)
    
    # Recomendaciones
    st.markdown("### 🎯 Recomendaciones Metodológicas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Para Futuros Experimentos:")
        st.write("""
        1. **Diseño balanceado** (Modelo 1) ofrece mayor poder estadístico
        2. **Submuestreo** (Modelos 3-6) es útil cuando hay estructuras jerárquicas
        3. El tamaño muestral influye directamente en la capacidad de detección
        4. Considerar costos vs beneficios de cada diseño
        """)
    
    with col2:
        st.markdown("#### 🔬 Análisis de Resultados:")
        st.write("""
        1. Verificar consistencia de resultados entre modelos
        2. Priorizar modelos con diseños más robustos
        3. Interpretar con cautela resultados no significativos
        4. Complementar con análisis económico
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 10px;'>
    <p><b>Desarrollado para análisis estadístico de diseños experimentales en producción animal</b> 🐹</p>
    <p>Dina Maribel Yana Yucra | Código: 221086</p>
    <p style='font-size: 12px; color: #999;'>Aplicación Streamlit - Análisis Estadístico Completo</p>
</div>
""", unsafe_allow_html=True)