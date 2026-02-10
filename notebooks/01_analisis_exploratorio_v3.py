"""
ANÁLISIS EXPLORATORIO DE DATOS - VERSIÓN 3
Proyecto: Predicción de Abandono Académico
Metodología: CRISP-DM - Fase de Comprensión de Datos

Cambios hechos
- No usar: Asistencia, Carrera, Facultad
- Variables clave: NO. VEZ (reincidencia), NIVEL (vulnerabilidad)
- Análisis por NIVEL como métrica principal
- Terminología: Abandono/Desvinculación en lugar de Deserción
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ANÁLISIS EXPLORATORIO - PREDICCIÓN DE ABANDONO ACADÉMICO")
print("=" * 80)

# ============================================================================
# 1. CARGA Y LIMPIEZA INICIAL
# ============================================================================
print("\n1. CARGA Y LIMPIEZA DE DATOS")
print("-" * 80)

df = pd.read_excel('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx')
print(f"✓ Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

# Convertir promedios con comas a puntos decimales
df['PROMEDIO_NUM'] = df['PROMEDIO'].astype(str).str.replace(',', '.').astype(float)

print(f"\n✓ Promedios corregidos")
print(f"  Rango: {df['PROMEDIO_NUM'].min():.2f} - {df['PROMEDIO_NUM'].max():.2f}")

# Extraer jornada de GRUPO/PARALELO
df['JORNADA'] = df['GRUPO/PARALELO'].apply(
    lambda x: 'MATUTINA' if 'ELMA' in str(x).upper() else 
             ('NOCTURNA' if 'ELNO' in str(x).upper() else 'OTRA')
)

print("\nInformación del dataset:")
print(f"  - Periodos académicos: {df['PERIODO'].nunique()}")
print(f"  - Estudiantes únicos: {df['ESTUDIANTE'].nunique()}")
print(f"  - Materias únicas: {df['MATERIA'].nunique()}")
print(f"  - Niveles académicos: {sorted(df['NIVEL'].unique())}")

print("\nJornadas identificadas:")
print(df['JORNADA'].value_counts())

print("\nEstados de materias:")
estados = df['ESTADO'].value_counts()
for estado, count in estados.items():
    print(f"  - {estado}: {count} ({count/len(df)*100:.1f}%)")

# ============================================================================
# 2. ANÁLISIS DE VARIABLES CLAVE
# ============================================================================
print("\n2. ANÁLISIS DE VARIABLES CLAVE")
print("-" * 80)

# NO. VEZ - Persistencia/Reincidencia
print("\nNO. VEZ (Persistencia en la materia):")
vez_analysis = df.groupby('NO. VEZ').agg({
    'ESTADO': lambda x: (x == 'REPROBADA').sum() / len(x) * 100,
    'PROMEDIO_NUM': 'mean',
    'COD_MATERIA': 'count'
}).round(2)
vez_analysis.columns = ['Tasa_Reprobacion_%', 'Promedio_Medio', 'Total_Registros']
print(vez_analysis)

print("\n✓ HALLAZGO CRÍTICO:")
if 2 in vez_analysis.index:
    tasa_vez1 = vez_analysis.loc[1, 'Tasa_Reprobacion_%'] if 1 in vez_analysis.index else 0
    tasa_vez2 = vez_analysis.loc[2, 'Tasa_Reprobacion_%']
    print(f"  - Primera vez (NO.VEZ=1): {tasa_vez1:.1f}% de reprobación")
    print(f"  - Segunda vez (NO.VEZ=2): {tasa_vez2:.1f}% de reprobación")
    print(f"  - Diferencia: {tasa_vez2 - tasa_vez1:.1f} puntos porcentuales")

# NIVEL - Vulnerabilidad por etapa
print("\nNIVEL (Vulnerabilidad académica):")
nivel_analysis = df.groupby('NIVEL').agg({
    'ESTADO': lambda x: (x == 'REPROBADA').sum() / len(x) * 100,
    'PROMEDIO_NUM': 'mean',
    'COD_MATERIA': 'count'
}).round(2)
nivel_analysis.columns = ['Tasa_Reprobacion_%', 'Promedio_Medio', 'Total_Registros']
print(nivel_analysis)

print("\n✓ HALLAZGO CRÍTICO:")
print(f"  - Nivel 1 (Mayor riesgo): {nivel_analysis.loc[1, 'Tasa_Reprobacion_%']:.1f}% reprobación")
if 3 in nivel_analysis.index:
    print(f"  - Nivel 3-4 (Menor riesgo): ~{nivel_analysis.loc[3, 'Tasa_Reprobacion_%']:.1f}% reprobación")

# ============================================================================
# 3. AGREGACIÓN POR ESTUDIANTE Y NIVEL
# ============================================================================
print("\n3. AGREGACIÓN POR ESTUDIANTE Y NIVEL")
print("-" * 80)

# Agregar por estudiante
estudiantes_agg = df.groupby('ESTUDIANTE').agg({
    'PERIODO': 'nunique',
    'MATERIA': 'count',
    'PROMEDIO_NUM': 'mean',
    'NO. VEZ': lambda x: (x > 1).sum(),
    'ESTADO': lambda x: (x == 'APROBADA').sum(),
    'NIVEL': 'max',
    'JORNADA': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'OTRA'
}).reset_index()

estudiantes_agg.columns = ['ESTUDIANTE', 'PERIODOS_CURSADOS', 'TOTAL_MATERIAS',
                            'PROMEDIO_GENERAL', 'MATERIAS_REPETIDAS',
                            'MATERIAS_APROBADAS', 'NIVEL_MAXIMO', 'JORNADA_PRINCIPAL']

# Calcular métricas derivadas
estudiantes_agg['MATERIAS_REPROBADAS'] = (
    estudiantes_agg['TOTAL_MATERIAS'] - estudiantes_agg['MATERIAS_APROBADAS']
)
estudiantes_agg['TASA_APROBACION'] = (
    estudiantes_agg['MATERIAS_APROBADAS'] / estudiantes_agg['TOTAL_MATERIAS'] * 100
)
estudiantes_agg['TIENE_REPETICIONES'] = (estudiantes_agg['MATERIAS_REPETIDAS'] > 0).astype(int)

# Estadísticas por nivel del estudiante
print("\nMétricas por NIVEL del estudiante:")
metricas_nivel = estudiantes_agg.groupby('NIVEL_MAXIMO').agg({
    'PROMEDIO_GENERAL': 'mean',
    'TASA_APROBACION': 'mean',
    'MATERIAS_REPETIDAS': 'mean',
    'ESTUDIANTE': 'count'
}).round(2)
metricas_nivel.columns = ['Promedio_Medio', 'Tasa_Aprobacion_Media_%', 
                          'Repeticiones_Media', 'Total_Estudiantes']
print(metricas_nivel)

print(f"\nTotal de estudiantes: {len(estudiantes_agg)}")
print("\nEstadísticas generales:")
print(estudiantes_agg[['PROMEDIO_GENERAL', 'TASA_APROBACION', 
                        'MATERIAS_REPETIDAS']].describe().round(2))

# ============================================================================
# 4. DEFINICIÓN DE ABANDONO ACADÉMICO (VARIABLE OBJETIVO)
# ============================================================================
print("\n4. DEFINICIÓN DE ABANDONO ACADÉMICO")
print("-" * 80)

"""
Criterios para ABANDONO/DESVINCULACIÓN basados en:
1. Desempeño (PROMEDIO < 7.0) - Filtro principal
2. Reincidencia (NO.VEZ > 1) - Señal crítica de dificultad
3. Vulnerabilidad (NIVEL bajo) - Mayor riesgo en primeros niveles
4. Tasa de aprobación baja

Un estudiante se clasifica como EN RIESGO DE ABANDONO si cumple:
- PROMEDIO < 7.0 Y (TIENE_REPETICIONES O TASA_APROBACION < 70%)
O
- NIVEL_MAXIMO == 1 Y TASA_APROBACION < 70%
"""

estudiantes_agg['RIESGO_ABANDONO'] = 0

# Criterio 1: Bajo desempeño con reincidencia
criterio1 = (estudiantes_agg['PROMEDIO_GENERAL'] < 7.0) & \
            (estudiantes_agg['TIENE_REPETICIONES'] == 1)
estudiantes_agg.loc[criterio1, 'RIESGO_ABANDONO'] = 1

# Criterio 2: Bajo desempeño con baja aprobación
criterio2 = (estudiantes_agg['PROMEDIO_GENERAL'] < 7.0) & \
            (estudiantes_agg['TASA_APROBACION'] < 70)
estudiantes_agg.loc[criterio2, 'RIESGO_ABANDONO'] = 1

# Criterio 3: Vulnerabilidad en Nivel 1
criterio3 = (estudiantes_agg['NIVEL_MAXIMO'] == 1) & \
            (estudiantes_agg['TASA_APROBACION'] < 70)
estudiantes_agg.loc[criterio3, 'RIESGO_ABANDONO'] = 1

# Criterio 4: Múltiples repeticiones
criterio4 = estudiantes_agg['MATERIAS_REPETIDAS'] >= 3
estudiantes_agg.loc[criterio4, 'RIESGO_ABANDONO'] = 1

print("Distribución de RIESGO DE ABANDONO:")
riesgo_dist = estudiantes_agg['RIESGO_ABANDONO'].value_counts()
for valor in [0, 1]:
    if valor in riesgo_dist.index:
        count = riesgo_dist[valor]
        label = "PERSISTENCIA (Activo)" if valor == 0 else "RIESGO DE ABANDONO"
        print(f"  - {label}: {count} ({count/len(estudiantes_agg)*100:.1f}%)")

# ============================================================================
# 5. ANÁLISIS COMPARATIVO
# ============================================================================
print("\n5. ANÁLISIS COMPARATIVO: RIESGO VS PERSISTENCIA")
print("-" * 80)

variables_comparar = ['PROMEDIO_GENERAL', 'TASA_APROBACION', 'MATERIAS_REPETIDAS',
                      'NIVEL_MAXIMO']

print(f"\n{'Variable':<25} {'Persistencia':>15} {'Riesgo':>15} {'Diferencia':>15}")
print("-" * 70)

for var in variables_comparar:
    persistencia = estudiantes_agg[estudiantes_agg['RIESGO_ABANDONO'] == 0][var].mean()
    riesgo = estudiantes_agg[estudiantes_agg['RIESGO_ABANDONO'] == 1][var].mean()
    diff = riesgo - persistencia
    print(f"{var:<25} {persistencia:>15.2f} {riesgo:>15.2f} {diff:>15.2f}")

# Análisis por Jornada
print("\n\nDistribución por JORNADA:")
jornada_riesgo = pd.crosstab(estudiantes_agg['JORNADA_PRINCIPAL'], 
                              estudiantes_agg['RIESGO_ABANDONO'], 
                              normalize='index') * 100
print(jornada_riesgo.round(1))

# ============================================================================
# 6. ESTADÍSTICAS POR NIVEL
# ============================================================================
print("\n6. MÉTRICAS DETALLADAS POR NIVEL")
print("-" * 80)

for nivel in sorted(estudiantes_agg['NIVEL_MAXIMO'].unique()):
    est_nivel = estudiantes_agg[estudiantes_agg['NIVEL_MAXIMO'] == nivel]
    riesgo_pct = (est_nivel['RIESGO_ABANDONO'] == 1).sum() / len(est_nivel) * 100
    
    print(f"\nNIVEL {nivel}:")
    print(f"  Total estudiantes: {len(est_nivel)}")
    print(f"  Riesgo de abandono: {riesgo_pct:.1f}%")
    print(f"  Promedio medio: {est_nivel['PROMEDIO_GENERAL'].mean():.2f}")
    print(f"  Tasa aprobación media: {est_nivel['TASA_APROBACION'].mean():.1f}%")
    print(f"  Con repeticiones: {(est_nivel['TIENE_REPETICIONES'] == 1).sum()} estudiantes")

# ============================================================================
# 7. GUARDAR DATOS PROCESADOS
# ============================================================================
print("\n7. GUARDADO DE DATOS PROCESADOS")
print("-" * 80)

# Seleccionar características para el modelo
features_modelo = [
    'ESTUDIANTE', 'PERIODOS_CURSADOS', 'TOTAL_MATERIAS', 'PROMEDIO_GENERAL',
    'MATERIAS_REPETIDAS', 'MATERIAS_APROBADAS', 'MATERIAS_REPROBADAS',
    'TASA_APROBACION', 'NIVEL_MAXIMO', 'TIENE_REPETICIONES',
    'JORNADA_PRINCIPAL', 'RIESGO_ABANDONO'
]

estudiantes_final = estudiantes_agg[features_modelo].copy()

# Guardar
estudiantes_final.to_csv('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/data/estudiantes_procesados_v3.csv', 
                         index=False)
print("✓ Datos procesados guardados")

df.to_csv('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/data/datos_originales_v3.csv', index=False)
print("✓ Datos originales guardados")

# Resumen final
print("\n" + "=" * 80)
print("RESUMEN DEL ANÁLISIS")
print("=" * 80)
print(f"Dataset final: {len(estudiantes_final)} estudiantes")
print(f"Características: {len(features_modelo) - 1} (+ variable objetivo)")
print(f"En Riesgo: {(estudiantes_final['RIESGO_ABANDONO'] == 1).sum()} "
      f"({(estudiantes_final['RIESGO_ABANDONO'] == 1).sum()/len(estudiantes_final)*100:.1f}%)")
print(f"Persistencia: {(estudiantes_final['RIESGO_ABANDONO'] == 0).sum()} "
      f"({(estudiantes_final['RIESGO_ABANDONO'] == 0).sum()/len(estudiantes_final)*100:.1f}%)")

print("\n✓ Variables clave utilizadas:")
print("  - NO. VEZ: Detecta reincidencia/dificultad acumulada")
print("  - NIVEL: Identifica vulnerabilidad por etapa académica")
print("  - PROMEDIO: Filtro principal de desempeño")
print("  - TASA_APROBACION: Indicador de éxito académico")

print("\n✓ Variables excluidas según especificaciones:")
print("  - ASISTENCIA (no utilizada)")
print("  - CARRERA (no utilizada)")
print("  - FACULTAD (no utilizada)")

print("\n✓ Análisis exploratorio completado exitosamente")
print("=" * 80)
