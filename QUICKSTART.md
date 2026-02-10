# 🚀 GUÍA RÁPIDA - VERSIÓN 3

## ⚡ Ejecución Inmediata

```bash
# 1. Instalar dependencias
pip install streamlit pandas numpy scikit-learn plotly

# 2. Navegar al directorio
cd PROYECTO_ABANDONO_ACADEMICO_V3/

# 3. Ejecutar aplicación
streamlit run app_streamlit.py
```

## 📊 Características Principales - V3

### ✅ Lo que SÍ incluye
- ✅ Modelo de Árbol de Decisión únicamente
- ✅ Variables: NO. VEZ, NIVEL, PROMEDIO, TASA_APROBACION
- ✅ Jornada (ELNO/ELMA) incluida
- ✅ Métricas por NIVEL académico
- ✅ Análisis de reincidencia (NO. VEZ)
- ✅ Predicción por ID de estudiante

### ❌ Lo que NO incluye
- ❌ Asistencia (excluida)
- ❌ Carrera (excluida)
- ❌ Facultad (excluida)
- ❌ Comparación de múltiples modelos
- ❌ Métricas no relacionadas con árbol de decisión

## 🎯 Resultados del Modelo

- **Accuracy:** 95.92%
- **Recall:** 100% ⭐ (¡Detecta TODOS los casos en riesgo!)
- **F1-Score:** 0.9394
- **Falsos Negativos:** 0 (ningún estudiante en riesgo pasa desapercibido)

## 📱 Secciones de la Aplicación

### 1. 🏠 Inicio
- Métricas generales
- Gráfico de distribución
- Riesgo por nivel
- Hallazgos de NO. VEZ y NIVEL

### 2. 📊 Análisis Exploratorio
- **Tab 1:** Métricas por NIVEL
- **Tab 2:** Comparaciones Persistencia vs Riesgo

### 3. 🤖 Evaluación del Modelo
- Métricas del Árbol de Decisión
- Matriz de confusión
- Interpretación de resultados

### 4. 🔮 Predicción por Estudiante
- **Tab 1:** Buscar por ID de estudiante
- **Tab 2:** Entrada manual de datos

### 5. 📈 Importancia de Variables
- Ranking de variables
- TASA_APROBACION = 93.94% de importancia

## 🔍 Hallazgos Clave

### NO. VEZ (Reincidencia)
- Primera vez: 15.7% reprobación
- Segunda vez: 45.0% reprobación
- **Diferencia: +29.2 pp**

### NIVEL (Vulnerabilidad)
- Nivel 1: 45.6% en riesgo
- Nivel 2: 30.3% en riesgo
- Nivel 3: 18.1% en riesgo
- Nivel 4: 13.6% en riesgo

## 📂 Archivos Clave

- `app_streamlit.py` - Aplicación principal
- `notebooks/01_analisis_exploratorio_v3.py` - EDA
- `notebooks/02_modelado_v3.py` - Entrenamiento
- `models/modelo_arbol_decision_v3.pkl` - Modelo guardado
- `data/estudiantes_procesados_v3.csv` - Datos finales

## 💡 Ejemplo de Predicción

**Estudiante en Riesgo:**
- Nivel: 1
- Promedio: 5.5
- Tasa Aprobación: 40%
- Materias Repetidas: 2
- → **RESULTADO: RIESGO DE ABANDONO**

**Recomendaciones generadas:**
1. Tutor académico inmediato
2. Programa de nivelación
3. Monitoreo intensivo (Nivel 1)

## 🆘 Solución de Problemas

**Si falta algún paquete:**
```bash
pip install -r requirements.txt
# o instalar individualmente:
pip install streamlit pandas numpy scikit-learn plotly
```

**Si no encuentra archivos:**
Verificar que estás en el directorio correcto:
```bash
pwd  # Debe mostrar: .../PROYECTO_ABANDONO_ACADEMICO_V3
ls   # Debe mostrar: app_streamlit.py, data/, models/, notebooks/
```

## 📞 Más Información

- Ver `README.md` para documentación completa
- Código comentado en cada script
- Ayuda integrada en la aplicación

---

**¡Listo para usar!** 🎉
