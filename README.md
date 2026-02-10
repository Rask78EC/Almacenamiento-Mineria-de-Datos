# 🎓 Sistema de Predicción de Abandono Académico - Versión 3

Sistema de predicción basado en **Árbol de Decisión** para identificar estudiantes en riesgo de abandono/desvinculación académica.

## 🎯 Cambios Principales - Versión 3

### ✅ Variables Utilizadas (Enfoque Específico)
- ✓ **NO. VEZ** - Persistencia/Reincidencia en materias (predictor crítico)
- ✓ **NIVEL** - Vulnerabilidad por etapa académica
- ✓ **PROMEDIO** - Desempeño académico general
- ✓ **TASA_APROBACION** - Indicador principal de éxito
- ✓ **MATERIAS_REPETIDAS** - Dificultad acumulada
- ✓ **JORNADA** (ELNO/ELMA) - Jornada nocturna/matutina

### ❌ Variables Excluidas
- ❌ **ASISTENCIA** - No utilizada según especificaciones
- ❌ **CARRERA** - No utilizada
- ❌ **FACULTAD** - No utilizada

### 📊 Modelo Único
- **Algoritmo:** Árbol de Decisión únicamente
- **Enfoque:** Reglas interpretables basadas en variables clave
- **Sin comparación:** Se presenta solo el árbol de decisión

## 🏆 Resultados del Modelo

### Métricas de Rendimiento
- **Accuracy:** 95.92%
- **Precision:** 88.57%
- **Recall:** 100.00% ⭐ (¡Detecta TODOS los casos en riesgo!)
- **F1-Score:** 0.9394
- **ROC-AUC:** 0.9911

### Matriz de Confusión
```
                    Predicho:        Predicho:
                    Persistencia     Riesgo
Real: Persistencia        63              4
Real: Riesgo               0             31
```

**Interpretación:**
- ✅ **0 Falsos Negativos** - No pierde ningún estudiante en riesgo
- ⚠️ **4 Falsos Positivos** - 4 estudiantes con falsa alarma (aceptable)

## 📊 Hallazgos Críticos del Análisis

### NO. VEZ - Reincidencia
- **Primera vez (NO.VEZ=1):** 15.7% de reprobación
- **Segunda vez (NO.VEZ=2):** 45.0% de reprobación
- **Diferencia:** +29.2 puntos porcentuales

**Conclusión:** La reincidencia es un predictor crítico de abandono.

### NIVEL - Vulnerabilidad por Etapa
- **Nivel 1:** 23.3% de reprobación (MAYOR RIESGO)
- **Nivel 2:** 12.1% de reprobación
- **Nivel 3-4:** ~5-7% de reprobación

**Conclusión:** Los estudiantes de primer nivel son más vulnerables.

## 📈 Importancia de Variables

Ranking de variables según el árbol de decisión:

1. **TASA_APROBACION** - 93.94% 🥇
2. **MATERIAS_REPETIDAS** - 2.70%
3. **PROMEDIO_GENERAL** - 2.62%
4. Otras variables - <1%

**Hallazgo:** La tasa de aprobación domina la predicción, capturando el éxito académico directamente.

## 🚀 Instalación y Ejecución

### Requisitos
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### Ejecutar Aplicación
```bash
# 1. Navegar al directorio
cd PROYECTO_ABANDONO_ACADEMICO_V3/

# 2. Ejecutar Streamlit
streamlit run app_streamlit.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📂 Estructura del Proyecto

```
PROYECTO_ABANDONO_ACADEMICO_V3/
│
├── app_streamlit.py                      # Aplicación web
│
├── notebooks/
│   ├── 01_analisis_exploratorio_v3.py   # EDA actualizado
│   └── 02_modelado_v3.py                # Árbol de decisión
│
├── data/
│   ├── estudiantes_procesados_v3.csv    # Dataset final
│   └── datos_originales_v3.csv          # Datos base
│
└── models/
    ├── modelo_arbol_decision_v3.pkl     # Modelo entrenado
    ├── features_v3.pkl                  # Lista de features
    ├── metricas_v3.pkl                  # Métricas del modelo
    ├── importancia_features_v3.csv      # Importancias
    └── jornada_encoding_v3.pkl          # Encoding de jornada
```

## 🎮 Funcionalidades de la Aplicación

### 1. 🏠 Página de Inicio
- Métricas clave del sistema
- Distribución de riesgo vs persistencia
- Gráfico de riesgo por nivel académico
- Hallazgos críticos destacados

### 2. 📊 Análisis Exploratorio

#### Tab 1: Métricas por Nivel
- Tabla con estadísticas por nivel académico
- Gráfico de promedio por nivel
- Gráfico de porcentaje en riesgo por nivel

#### Tab 2: Comparaciones
- Comparación entre estudiantes con persistencia vs en riesgo
- Gráficos de barras agrupados
- Tabla comparativa de métricas

### 3. 🤖 Evaluación del Modelo
- Métricas principales (Accuracy, Precision, Recall, F1)
- Matriz de confusión visual
- Métricas detalladas (TP, TN, FP, FN)
- Sensibilidad y especificidad
- Destacado de Recall 100%

### 4. 🔮 Predicción por Estudiante

#### Opción A: Buscar Estudiante
- Seleccionar estudiante por ID
- Ver perfil completo
- Obtener predicción automática
- Recibir recomendaciones personalizadas

#### Opción B: Entrada Manual
- Ingresar datos manualmente
- Predicción inmediata
- Nivel de riesgo calculado

**Recomendaciones incluyen:**
- 📚 Rendimiento académico
- ✅ Tasa de aprobación
- 🔄 Persistencia/Reincidencia (NO. VEZ)
- ⚠️ Vulnerabilidad por etapa (NIVEL)

### 5. 📈 Importancia de Variables
- Gráfico de barras de importancia
- Tabla detallada
- Interpretación de variables clave

## 🎯 Terminología Utilizada

### Términos Principales
- **Abandono Académico** - Retiro o desvinculación del estudiante
- **Persistencia** - Estudiante activo que continúa
- **Riesgo de Abandono** - Probabilidad de desvinculación
- **Reincidencia** - Detectada por NO. VEZ > 1
- **Vulnerabilidad** - Mayor en niveles iniciales (NIVEL 1)

### Clasificación
| Clase | Valor | Significado |
|-------|-------|-------------|
| 0 | Persistencia | Estudiante activo |
| 1 | Riesgo de Abandono | Estudiante en riesgo |

## 🔬 Metodología CRISP-DM

### 1. Comprensión del Negocio
- Problema: Abandono académico en educación superior
- Objetivo: Identificación temprana de riesgo
- Variables clave: NO. VEZ, NIVEL, PROMEDIO, TASA_APROBACION

### 2. Comprensión de Datos
- 4,448 registros de materias
- 488 estudiantes únicos
- 9 períodos académicos
- Variables categóricas y numéricas

### 3. Preparación de Datos
- Agregación por estudiante
- Cálculo de tasa de aprobación
- Identificación de jornada (ELNO/ELMA)
- Creación de variable TIENE_REPETICIONES
- Definición de RIESGO_ABANDONO

### 4. Modelado
- Algoritmo: Árbol de Decisión
- Parámetros:
  - max_depth=6
  - min_samples_split=10
  - min_samples_leaf=5
  - class_weight='balanced'
- División: 80% train, 20% test
- Validación cruzada: 5-fold

### 5. Evaluación
- Accuracy: 95.92%
- Recall: 100% (crítico para el problema)
- 0 Falsos Negativos
- Validación cruzada exitosa

### 6. Despliegue
- Aplicación web Streamlit
- Sistema de recomendaciones
- Búsqueda por estudiante

## 💡 Reglas del Árbol de Decisión

El árbol aprendió reglas como:

```
Si TASA_APROBACION <= 82.97%:
    Si PROMEDIO_GENERAL <= 7.31:
        → RIESGO DE ABANDONO
    Si PROMEDIO_GENERAL > 7.31:
        → PERSISTENCIA
Si TASA_APROBACION > 82.97%:
    Si MATERIAS_REPETIDAS <= 1.5:
        → PERSISTENCIA
    Si MATERIAS_REPETIDAS > 1.5:
        → RIESGO DE ABANDONO
```

## 📊 Distribución de Datos

- **Total estudiantes:** 488
- **En Persistencia:** 332 (68.0%)
- **En Riesgo:** 156 (32.0%)

**Riesgo por Nivel:**
- Nivel 1: 45.6% en riesgo
- Nivel 2: 30.3% en riesgo
- Nivel 3: 18.1% en riesgo
- Nivel 4: 13.6% en riesgo

## 🔄 Actualización del Modelo

Para reentrenar con nuevos datos:

```bash
# 1. Actualizar datos en /data/
# 2. Ejecutar análisis
python notebooks/01_analisis_exploratorio_v3.py

# 3. Entrenar modelo
python notebooks/02_modelado_v3.py

# 4. Listo para usar
streamlit run app_streamlit.py
```

## ✅ Ventajas de esta Versión

1. **Enfoque Específico:** Solo variables relevantes y validadas
2. **Recall Perfecto:** No pierde ningún estudiante en riesgo
3. **Interpretabilidad:** Árbol de decisión = reglas claras
4. **Variables Clave:** NO. VEZ y NIVEL como predictores críticos
5. **Sin Ruido:** Eliminadas variables no relevantes

## 📝 Ejemplo de Uso

```python
# Ejemplo de predicción
estudiante = {
    'NIVEL_MAXIMO': 1,
    'PROMEDIO_GENERAL': 5.5,
    'TASA_APROBACION': 40.0,
    'MATERIAS_REPETIDAS': 2,
    'TIENE_REPETICIONES': 1,
    # ... otros campos
}

# El modelo predice:
# → RIESGO DE ABANDONO (Alta probabilidad)
# 
# Recomendaciones:
# 1. Tutor académico inmediato
# 2. Programa de nivelación
# 3. Apoyo psicopedagógico
# 4. Monitoreo intensivo (Nivel 1)
```

## 📞 Soporte

Para más información, consultar:
- Código fuente comentado
- Notebooks de análisis
- Documentación en la aplicación

---

## 🎉 Conclusiones Clave

1. **Modelo Efectivo:** 95.92% accuracy, 100% recall
2. **NO Falsos Negativos:** Detecta TODOS los casos en riesgo
3. **Variables Críticas:** TASA_APROBACION (94%), NO. VEZ, NIVEL
4. **Nivel 1 Vulnerable:** 45.6% de estudiantes en riesgo
5. **Reincidencia Crítica:** 45% reprobación en segunda matrícula

---

**Desarrollado con Python, Scikit-learn y Streamlit**
