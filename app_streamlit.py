"""
APLICACIÓN STREAMLIT V3 - SISTEMA DE PREDICCIÓN DE ABANDONO ACADÉMICO
Proyecto: Predicción de Abandono Académico  

Versión 3: Enfoque en Árbol de Decisión, sin asistencia
Variables clave: NO. VEZ, NIVEL, PROMEDIO, TASA_APROBACION
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Sistema de Predicción de Abandono Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

@st.cache_data
def cargar_datos():
    """Carga todos los datos necesarios"""
    df_estudiantes = pd.read_csv('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/data/estudiantes_procesados_v3.csv')
    df_importancias = pd.read_csv('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/importancia_features_v3.csv')
    df_originales = pd.read_csv('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/data/datos_originales_v3.csv')
    return df_estudiantes, df_importancias, df_originales

@st.cache_resource
def cargar_modelos():
    """Carga el modelo y componentes"""
    with open('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/modelo_arbol_decision_v3.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/features_v3.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/metricas_v3.pkl', 'rb') as f:
        metricas = pickle.load(f)
    with open('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/jornada_encoding_v3.pkl', 'rb') as f:
        jornada_map = pickle.load(f)
    return modelo, features, metricas, jornada_map

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def predecir_abandono(datos_estudiante, modelo, features):
    """Predice el riesgo de abandono"""
    df_input = pd.DataFrame([datos_estudiante])
    
    for feature in features:
        if feature not in df_input.columns:
            df_input[feature] = 0
    
    X = df_input[features]
    prediccion = modelo.predict(X)[0]
    probabilidad = modelo.predict_proba(X)[0][1]
    
    if probabilidad < 0.3:
        nivel_riesgo, color = "BAJO", "green"
    elif probabilidad < 0.6:
        nivel_riesgo, color = "MEDIO", "orange"
    else:
        nivel_riesgo, color = "ALTO", "red"
    
    return {
        'prediccion': int(prediccion),
        'probabilidad': probabilidad,
        'nivel_riesgo': nivel_riesgo,
        'color': color
    }

def generar_recomendaciones(datos_estudiante, resultado):
    """Genera recomendaciones personalizadas"""
    recomendaciones = []
    
    if datos_estudiante.get('PROMEDIO_GENERAL', 10) < 7.0:
        recomendaciones.append({
            'icon': '📚',
            'area': 'Rendimiento Académico',
            'problema': 'Promedio general bajo (< 7.0)',
            'accion': 'Asignar tutor académico y plan de nivelación inmediata',
            'prioridad': 'Alta'
        })
    
    if datos_estudiante.get('TASA_APROBACION', 100) < 70:
        recomendaciones.append({
            'icon': '✅',
            'area': 'Tasa de Aprobación',
            'problema': f'Baja tasa de aprobación ({datos_estudiante.get("TASA_APROBACION", 0):.1f}%)',
            'accion': 'Implementar estrategias de estudio y apoyo psicopedagógico',
            'prioridad': 'Alta'
        })
    
    if datos_estudiante.get('TIENE_REPETICIONES', 0) == 1:
        n_rep = int(datos_estudiante.get('MATERIAS_REPETIDAS', 0))
        recomendaciones.append({
            'icon': '🔄',
            'area': 'Persistencia/Reincidencia',
            'problema': f'Tiene {n_rep} materia(s) repetida(s) - Señal crítica según NO. VEZ',
            'accion': 'Evaluación individualizada de dificultades específicas y refuerzo',
            'prioridad': 'Alta'
        })
    
    if datos_estudiante.get('NIVEL_MAXIMO', 5) == 1:
        recomendaciones.append({
            'icon': '⚠️',
            'area': 'Vulnerabilidad por Etapa',
            'problema': 'Estudiante en NIVEL 1 - Mayor riesgo de abandono',
            'accion': 'Programa de adaptación y acompañamiento intensivo',
            'prioridad': 'Alta'
        })
    
    if not recomendaciones and resultado['prediccion'] == 1:
        recomendaciones.append({
            'icon': '👁️',
            'area': 'Seguimiento General',
            'problema': 'Perfil de riesgo detectado',
            'accion': 'Mantener monitoreo cercano y ofrecer recursos de apoyo',
            'prioridad': 'Media'
        })
    
    return recomendaciones

# ============================================================================
# CARGAR DATOS
# ============================================================================

try:
    df_estudiantes, df_importancias, df_originales = cargar_datos()
    modelo, features, metricas, jornada_map = cargar_modelos()
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# ============================================================================
# SIDEBAR - NAVEGACIÓN
# ============================================================================

st.sidebar.markdown("# 🎓 Navegación")
pagina = st.sidebar.radio(
    "Seleccione una sección:",
    ["🏠 Inicio",
     "📊 Análisis Exploratorio",
     "🤖 Evaluación del Modelo",
     "🔮 Predicción por Estudiante",
     "📈 Importancia de Variables"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información del Sistema")
st.sidebar.info(f"""
**Modelo:** Árbol de Decisión  
**Accuracy:** {metricas['accuracy']:.2%}  
**Recall:** {metricas['recall']:.2%}  
**F1-Score:** {metricas['f1_score']:.4f}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Variables Clave")
st.sidebar.success("""
✓ **NO. VEZ** (Reincidencia)  
✓ **NIVEL** (Vulnerabilidad)  
✓ **PROMEDIO** (Desempeño)  
✓ **TASA_APROBACION**

❌ Asistencia (no usada)  
❌ Carrera (no usada)  
❌ Facultad (no usada)
""")

# ============================================================================
# PÁGINA: INICIO
# ============================================================================

if pagina == "🏠 Inicio":
    st.markdown('<p class="main-header">🎓 Sistema de Predicción de Abandono Académico</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Sistema Inteligente para Identificación Temprana de Riesgo de Abandono
    
    Este sistema utiliza un **modelo de Árbol de Decisión** con variables clave para predecir 
    el riesgo de abandono/desvinculación académica.
    
    #### 🎯 Variables Predictivas Principales:
    - **NO. VEZ**: Detecta persistencia y dificultad acumulada (reincidencia en materias)
    - **NIVEL**: Identifica vulnerabilidad por etapa académica
    - **PROMEDIO**: Mide desempeño académico general
    - **TASA_APROBACION**: Indicador de éxito académico
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_estudiantes = len(df_estudiantes)
    persistencia = (df_estudiantes['RIESGO_ABANDONO'] == 0).sum()
    riesgo = (df_estudiantes['RIESGO_ABANDONO'] == 1).sum()
    tasa_riesgo = (riesgo / total_estudiantes * 100)
    
    with col1:
        st.metric("Total Estudiantes", total_estudiantes)
    with col2:
        st.metric("Persistencia", persistencia,
                  delta=f"{persistencia/total_estudiantes*100:.1f}%")
    with col3:
        st.metric("En Riesgo", riesgo,
                  delta=f"{tasa_riesgo:.1f}%",
                  delta_color="inverse")
    with col4:
        st.metric("Recall del Modelo", f"{metricas['recall']*100:.0f}%",
                  delta="Perfecto" if metricas['recall'] == 1.0 else "Bueno")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Distribución de Riesgo")
        fig = px.pie(
            values=[persistencia, riesgo],
            names=['Persistencia', 'Riesgo de Abandono'],
            color_discrete_sequence=['#4CAF50', '#F44336'],
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Riesgo por Nivel Académico")
        riesgo_nivel = df_estudiantes.groupby('NIVEL_MAXIMO')['RIESGO_ABANDONO'].apply(
            lambda x: (x == 1).sum() / len(x) * 100
        ).reset_index()
        riesgo_nivel.columns = ['NIVEL', 'Porcentaje_Riesgo']
        
        fig = px.bar(
            riesgo_nivel,
            x='NIVEL',
            y='Porcentaje_Riesgo',
            color='Porcentaje_Riesgo',
            color_continuous_scale='Reds',
            labels={'NIVEL': 'Nivel Académico', 'Porcentaje_Riesgo': 'Riesgo (%)'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔍 Hallazgos Críticos del Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **NO. VEZ - Persistencia/Reincidencia**
        - Primera vez (NO.VEZ=1): ~15.7% reprobación
        - Segunda vez (NO.VEZ=2): ~45.0% reprobación
        - **Diferencia:** 29.2 puntos porcentuales
        
        ➡️ La reincidencia es un predictor crítico
        """)
    
    with col2:
        st.warning("""
        **NIVEL - Vulnerabilidad por Etapa**
        - Nivel 1: ~23.3% reprobación (MAYOR RIESGO)
        - Nivel 3-4: ~5-7% reprobación
        
        ➡️ Los primeros niveles son más vulnerables
        """)

# ============================================================================
# PÁGINA: ANÁLISIS EXPLORATORIO
# ============================================================================

elif pagina == "📊 Análisis Exploratorio":
    st.title("📊 Análisis Exploratorio de Datos")
    
    tab1, tab2 = st.tabs(["📈 Métricas por Nivel", "🔍 Comparaciones"])
    
    with tab1:
        st.subheader("Métricas Académicas por Nivel")
        
        # Agregar por nivel
        metricas_nivel = df_estudiantes.groupby('NIVEL_MAXIMO').agg({
            'PROMEDIO_GENERAL': 'mean',
            'TASA_APROBACION': 'mean',
            'MATERIAS_REPETIDAS': 'mean',
            'RIESGO_ABANDONO': lambda x: (x == 1).sum() / len(x) * 100,
            'ESTUDIANTE': 'count'
        }).round(2)
        metricas_nivel.columns = ['Promedio_Medio', 'Tasa_Aprobacion_%', 
                                   'Repeticiones_Media', 'Riesgo_%', 'Total_Estudiantes']
        
        st.dataframe(metricas_nivel, use_container_width=True)
        
        # Gráficos por nivel
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                metricas_nivel.reset_index(),
                x='NIVEL_MAXIMO',
                y='Promedio_Medio',
                title='Promedio Académico por Nivel',
                color='Promedio_Medio',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                metricas_nivel.reset_index(),
                x='NIVEL_MAXIMO',
                y='Riesgo_%',
                title='Porcentaje en Riesgo por Nivel',
                color='Riesgo_%',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Comparación: Persistencia vs Riesgo de Abandono")
        
        comparacion_vars = ['PROMEDIO_GENERAL', 'TASA_APROBACION', 
                           'MATERIAS_REPETIDAS', 'NIVEL_MAXIMO']
        
        datos_comp = []
        for var in comparacion_vars:
            pers = df_estudiantes[df_estudiantes['RIESGO_ABANDONO'] == 0][var].mean()
            riesg = df_estudiantes[df_estudiantes['RIESGO_ABANDONO'] == 1][var].mean()
            datos_comp.append({
                'Variable': var,
                'Persistencia': pers,
                'Riesgo': riesg,
                'Diferencia': riesg - pers
            })
        
        df_comp = pd.DataFrame(datos_comp)
        
        fig = go.Figure(data=[
            go.Bar(name='Persistencia', x=df_comp['Variable'], y=df_comp['Persistencia'],
                   marker_color='#4CAF50'),
            go.Bar(name='Riesgo', x=df_comp['Variable'], y=df_comp['Riesgo'],
                   marker_color='#F44336')
        ])
        fig.update_layout(barmode='group', height=500, title='Comparación de Métricas')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_comp.round(2), use_container_width=True)

# ============================================================================
# PÁGINA: EVALUACIÓN DEL MODELO
# ============================================================================

elif pagina == "🤖 Evaluación del Modelo":
    st.title("🤖 Evaluación del Modelo de Árbol de Decisión")
    
    st.markdown("""
    Se entrenó un **modelo de Árbol de Decisión** optimizado para predicción de abandono académico.
    Este modelo fue seleccionado por su interpretabilidad y capacidad de capturar reglas claras.
    """)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metricas['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metricas['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metricas['recall']:.2%}",
                  delta="100%" if metricas['recall'] == 1.0 else None)
    with col4:
        st.metric("F1-Score", f"{metricas['f1_score']:.4f}")
    
    st.markdown("---")
    
    # Matriz de confusión
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Matriz de Confusión")
        cm = np.array(metricas['confusion_matrix'])
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicho: Persistencia', 'Predicho: Riesgo'],
            y=['Real: Persistencia', 'Real: Riesgo'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Métricas Detalladas")
        
        st.markdown(f"""
        **Verdaderos Positivos (TP):** {metricas['tp']}  
        Casos en riesgo correctamente identificados
        
        **Verdaderos Negativos (TN):** {metricas['tn']}  
        Persistencia correctamente identificada
        
        **Falsos Positivos (FP):** {metricas['fp']}  
        Falsa alarma de riesgo
        
        **Falsos Negativos (FN):** {metricas['fn']}  
        Riesgo NO detectado (crítico)
        
        ---
        
        **Sensibilidad:** {metricas['sensibilidad']:.2%}  
        Detecta el {metricas['sensibilidad']*100:.1f}% de casos en riesgo
        
        **Especificidad:** {metricas['especificidad']:.2%}  
        Identifica el {metricas['especificidad']*100:.1f}% de persistencia
        """)
    
    # Destacar si hay 100% recall
    if metricas['recall'] == 1.0:
        st.success("""
        ### 🏆 ¡Rendimiento Excepcional!
        
        El modelo alcanza un **Recall del 100%**, lo que significa que identifica 
        correctamente **TODOS** los estudiantes en riesgo de abandono. Esto es crítico 
        para una intervención temprana efectiva.
        """)

# ============================================================================
# PÁGINA: PREDICCIÓN INDIVIDUAL
# ============================================================================

elif pagina == "🔮 Predicción por Estudiante":
    st.title("🔮 Predicción Individual de Riesgo de Abandono")
    
    st.markdown("""
    Busque un estudiante por su ID o ingrese datos manualmente para obtener una predicción.
    """)
    
    tab1, tab2 = st.tabs(["🔍 Buscar Estudiante", "✍️ Entrada Manual"])
    
    with tab1:
        st.subheader("Buscar por ID de Estudiante")
        
        estudiante_id = st.selectbox(
            "Seleccione ID del estudiante:",
            options=sorted(df_estudiantes['ESTUDIANTE'].unique())
        )
        
        if st.button("🔮 Predecir Riesgo", type="primary", use_container_width=True):
            # Obtener datos del estudiante
            est_data = df_estudiantes[df_estudiantes['ESTUDIANTE'] == estudiante_id].iloc[0]
            
            # Preparar datos para predicción
            datos_input = {
                'PERIODOS_CURSADOS': est_data['PERIODOS_CURSADOS'],
                'TOTAL_MATERIAS': est_data['TOTAL_MATERIAS'],
                'PROMEDIO_GENERAL': est_data['PROMEDIO_GENERAL'],
                'MATERIAS_REPETIDAS': est_data['MATERIAS_REPETIDAS'],
                'MATERIAS_REPROBADAS': est_data['MATERIAS_REPROBADAS'],
                'TASA_APROBACION': est_data['TASA_APROBACION'],
                'NIVEL_MAXIMO': est_data['NIVEL_MAXIMO'],
                'TIENE_REPETICIONES': est_data['TIENE_REPETICIONES'],
                'JORNADA_ENCODED': jornada_map.get(est_data['JORNADA_PRINCIPAL'], 2)
            }
            
            # Mostrar perfil
            st.markdown("### 📋 Perfil del Estudiante")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Nivel Académico", int(est_data['NIVEL_MAXIMO']))
                st.metric("Promedio General", f"{est_data['PROMEDIO_GENERAL']:.2f}")
            with col2:
                st.metric("Tasa de Aprobación", f"{est_data['TASA_APROBACION']:.1f}%")
                st.metric("Materias Repetidas", int(est_data['MATERIAS_REPETIDAS']))
            with col3:
                st.metric("Total Materias", int(est_data['TOTAL_MATERIAS']))
                st.metric("Jornada", est_data['JORNADA_PRINCIPAL'])
            
            # Predicción
            resultado = predecir_abandono(datos_input, modelo, features)
            recomendaciones = generar_recomendaciones(datos_input, resultado)
            
            st.markdown("---")
            st.markdown("## 🎯 Resultado de la Predicción")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if resultado['prediccion'] == 1:
                    st.error("### ⚠️ RIESGO DE ABANDONO")
                else:
                    st.success("### ✅ PERSISTENCIA")
            
            with col2:
                st.metric("Probabilidad de Abandono", f"{resultado['probabilidad']*100:.1f}%")
            
            with col3:
                color_class = f"risk-{resultado['color']}"
                st.markdown(f"### Nivel: <span class='{color_class}'>{resultado['nivel_riesgo']}</span>", 
                            unsafe_allow_html=True)
            
            # Medidor
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = resultado['probabilidad'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Riesgo de Abandono (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': resultado['color']},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendaciones
            if recomendaciones:
                st.markdown("---")
                st.markdown("## 💡 Recomendaciones de Intervención")
                
                for rec in recomendaciones:
                    with st.expander(f"{rec['icon']} {rec['area']} - Prioridad: {rec['prioridad']}", 
                                    expanded=True):
                        st.markdown(f"**Problema:** {rec['problema']}")
                        st.markdown(f"**Acción:** {rec['accion']}")
    
    with tab2:
        st.subheader("Ingreso Manual de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            periodos = st.number_input("Períodos Cursados", 1, 10, 2)
            total_materias = st.number_input("Total de Materias", 1, 50, 10)
            promedio = st.slider("Promedio General (0-10)", 0.0, 10.0, 7.0, 0.1)
            nivel = st.selectbox("Nivel Actual", [1, 2, 3, 4])
        
        with col2:
            materias_aprobadas = st.number_input("Materias Aprobadas", 0, 50, 8)
            materias_repetidas = st.number_input("Materias Repetidas", 0, 10, 0)
            jornada = st.selectbox("Jornada", ['MATUTINA', 'NOCTURNA', 'OTRA'])
        
        if st.button("🔮 Realizar Predicción", type="primary", use_container_width=True):
            # Preparar datos
            materias_reprobadas = total_materias - materias_aprobadas
            tasa_aprobacion = (materias_aprobadas / total_materias * 100) if total_materias > 0 else 0
            tiene_repeticiones = 1 if materias_repetidas > 0 else 0
            
            datos_input = {
                'PERIODOS_CURSADOS': periodos,
                'TOTAL_MATERIAS': total_materias,
                'PROMEDIO_GENERAL': promedio,
                'MATERIAS_REPETIDAS': materias_repetidas,
                'MATERIAS_REPROBADAS': materias_reprobadas,
                'TASA_APROBACION': tasa_aprobacion,
                'NIVEL_MAXIMO': nivel,
                'TIENE_REPETICIONES': tiene_repeticiones,
                'JORNADA_ENCODED': jornada_map.get(jornada, 2)
            }
            
            resultado = predecir_abandono(datos_input, modelo, features)
            
            st.markdown("## 🎯 Resultado")
            
            if resultado['prediccion'] == 1:
                st.error(f"### ⚠️ RIESGO DE ABANDONO - Nivel: {resultado['nivel_riesgo']}")
            else:
                st.success(f"### ✅ PERSISTENCIA - Nivel: {resultado['nivel_riesgo']}")
            
            st.metric("Probabilidad", f"{resultado['probabilidad']*100:.1f}%")

# ============================================================================
# PÁGINA: IMPORTANCIA DE VARIABLES
# ============================================================================

elif pagina == "📈 Importancia de Variables":
    st.title("📈 Importancia de Características en el Modelo")
    
    st.markdown("""
    Las siguientes características son las más importantes para predecir el abandono académico,
    según el **modelo de Árbol de Decisión** entrenado.
    """)
    
    # Gráfico de importancia
    fig = px.bar(
        df_importancias.head(9),
        x='Importancia',
        y='Caracteristica',
        orientation='h',
        color='Importancia',
        color_continuous_scale='Viridis',
        title='Importancia de Variables'
    )
    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla
    st.subheader("📋 Detalle de Importancias")
    st.dataframe(df_importancias, use_container_width=True)
    
    # Interpretación
    st.markdown("---")
    st.markdown("### 🔍 Interpretación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **TASA_APROBACION** es el predictor más fuerte (93.94%)
        
        Esta variable captura directamente el éxito académico del estudiante.
        Una tasa baja es señal crítica de riesgo de abandono.
        """)
    
    with col2:
        st.warning("""
        **MATERIAS_REPETIDAS** y **PROMEDIO_GENERAL** complementan
        
        Las repeticiones (NO. VEZ) indican dificultad acumulada.
        El promedio refleja el desempeño general.
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Sistema de Predicción de Abandono Académico | Metodología CRISP-DM</p>
    <p>Modelo: Árbol de Decisión | Variables clave: NO. VEZ, NIVEL, PROMEDIO</p>
</div>
""", unsafe_allow_html=True)
