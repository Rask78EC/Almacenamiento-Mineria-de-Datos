"""
MODELADO CON ÁRBOL DE DECISIÓN
Proyecto: Predicción de Abandono Académico

Enfoque: Modelo único de Árbol de Decisión
Variables clave: NO. VEZ (reincidencia), NIVEL (vulnerabilidad)
Sin usar: Asistencia, Carrera, Facultad
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (classification_report, confusion_matrix, 
                              accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MODELADO - ÁRBOL DE DECISIÓN PARA PREDICCIÓN DE ABANDONO ACADÉMICO")
print("=" * 80)

# ============================================================================
# 1. CARGA DE DATOS PROCESADOS
# ============================================================================
print("\n1. CARGA DE DATOS")
print("-" * 80)

df = pd.read_csv('data/estudiantes_procesados_v3.csv')
print(f"✓ Datos cargados: {df.shape[0]} estudiantes, {df.shape[1]} columnas")

# ============================================================================
# 2. PREPARACIÓN DE DATOS PARA MODELADO
# ============================================================================
print("\n2. PREPARACIÓN DE DATOS")
print("-" * 80)

# Características para el modelo (SIN ASISTENCIA, CARRERA, FACULTAD)
features_numericas = [
    'PERIODOS_CURSADOS',
    'TOTAL_MATERIAS',
    'PROMEDIO_GENERAL',
    'MATERIAS_REPETIDAS',
    'MATERIAS_REPROBADAS',
    'TASA_APROBACION',
    'NIVEL_MAXIMO',
    'TIENE_REPETICIONES'
]

# Codificar jornada
df['JORNADA_ENCODED'] = df['JORNADA_PRINCIPAL'].map({
    'MATUTINA': 0,
    'NOCTURNA': 1,
    'OTRA': 2
}).fillna(2)

features_numericas.append('JORNADA_ENCODED')

print(f"Características seleccionadas: {len(features_numericas)}")
print("\nVariables del modelo:")
for i, feat in enumerate(features_numericas, 1):
    print(f"  {i:2d}. {feat}")

# Preparar X (características) y y (objetivo)
X = df[features_numericas].fillna(0)
y = df['RIESGO_ABANDONO']

print(f"\nDistribución de clases:")
print(f"  - Clase 0 (Persistencia): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"  - Clase 1 (Riesgo Abandono): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")

# División train-test (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDivisión de datos:")
print(f"  - Conjunto de entrenamiento: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
print(f"  - Conjunto de prueba: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 3. ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN
# ============================================================================
print("\n3. ENTRENAMIENTO DEL MODELO")
print("-" * 80)

print("\nConfigurando Árbol de Decisión...")
print("  Parámetros:")
print("  - Profundidad máxima: 6")
print("  - Min muestras para split: 10")
print("  - Min muestras en hoja: 5")
print("  - Balance de clases: Activado")

# Entrenar modelo
modelo = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)

modelo.fit(X_train, y_train)
print("\n✓ Modelo entrenado exitosamente")

# ============================================================================
# 4. EVALUACIÓN DEL MODELO
# ============================================================================
print("\n4. EVALUACIÓN DEL MODELO")
print("-" * 80)

# Predicciones
y_train_pred = modelo.predict(X_train)
y_test_pred = modelo.predict(X_test)
y_test_proba = modelo.predict_proba(X_test)[:, 1]

# Métricas de entrenamiento
print("\nRENDIMIENTO EN ENTRENAMIENTO:")
print(f"  Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")

# Métricas de prueba
print("\nRENDIMIENTO EN PRUEBA:")
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

# Matriz de confusión
print("\n" + "=" * 80)
print("MATRIZ DE CONFUSIÓN")
print("=" * 80)
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n                    Predicho:        Predicho:")
print(f"                    Persistencia     Riesgo")
print(f"Real: Persistencia  {cm[0, 0]:8d}         {cm[0, 1]:8d}")
print(f"Real: Riesgo        {cm[1, 0]:8d}         {cm[1, 1]:8d}")

# Métricas derivadas
tn, fp, fn, tp = cm.ravel()
especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nMétricas Detalladas:")
print(f"  Verdaderos Positivos (TP):  {tp} - Riesgo correctamente identificado")
print(f"  Verdaderos Negativos (TN):  {tn} - Persistencia correctamente identificada")
print(f"  Falsos Positivos (FP):      {fp} - Falsa alarma de riesgo")
print(f"  Falsos Negativos (FN):      {fn} - Riesgo NO detectado (crítico)")
print(f"\n  Sensibilidad (Recall):      {sensibilidad:.4f} - Detecta {sensibilidad*100:.1f}% de casos en riesgo")
print(f"  Especificidad:              {especificidad:.4f} - Identifica {especificidad*100:.1f}% de persistencia")

# Reporte de clasificación
print("\n" + "=" * 80)
print("REPORTE DE CLASIFICACIÓN DETALLADO")
print("=" * 80)
print(classification_report(y_test, y_test_pred, 
                           target_names=['Persistencia', 'Riesgo de Abandono'],
                           digits=4))

# ============================================================================
# 5. VALIDACIÓN CRUZADA
# ============================================================================
print("\n5. VALIDACIÓN CRUZADA (5-FOLD)")
print("-" * 80)

cv_scores_accuracy = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
cv_scores_f1 = cross_val_score(modelo, X_train, y_train, cv=5, scoring='f1')
cv_scores_recall = cross_val_score(modelo, X_train, y_train, cv=5, scoring='recall')

print(f"Accuracy promedio:  {cv_scores_accuracy.mean():.4f} (+/- {cv_scores_accuracy.std():.4f})")
print(f"F1-Score promedio:  {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std():.4f})")
print(f"Recall promedio:    {cv_scores_recall.mean():.4f} (+/- {cv_scores_recall.std():.4f})")

# ============================================================================
# 6. IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================
print("\n6. IMPORTANCIA DE CARACTERÍSTICAS")
print("=" * 80)

importancias = pd.DataFrame({
    'Caracteristica': features_numericas,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\nRanking de Importancia:")
print("-" * 80)
for idx, row in importancias.iterrows():
    bar_length = int(row['Importancia'] * 50)
    bar = '█' * bar_length
    print(f"{row['Caracteristica']:25s} {bar} {row['Importancia']:.4f}")

print("\n✓ Variables más predictivas:")
top_3 = importancias.head(3)
for i, (_, row) in enumerate(top_3.iterrows(), 1):
    print(f"  {i}. {row['Caracteristica']} ({row['Importancia']:.2%})")

# ============================================================================
# 7. REGLAS DEL ÁRBOL
# ============================================================================
print("\n7. REGLAS DEL ÁRBOL DE DECISIÓN")
print("=" * 80)

# Exportar reglas textuales
tree_rules = export_text(modelo, feature_names=features_numericas, max_depth=3)
print("\nPrimeras reglas del árbol (profundidad 3):")
print(tree_rules)

# ============================================================================
# 8. ANÁLISIS DE CASOS CRÍTICOS
# ============================================================================
print("\n8. ANÁLISIS DE CASOS CRÍTICOS")
print("=" * 80)

# Identificar falsos negativos (estudiantes en riesgo NO detectados)
if fn > 0:
    indices_test = X_test.index
    fn_indices = indices_test[(y_test == 1) & (y_test_pred == 0)]
    
    print(f"\n⚠ FALSOS NEGATIVOS: {fn} casos")
    print("Estudiantes en riesgo que NO fueron detectados:")
    print("-" * 80)
    
    for idx in fn_indices[:3]:  # Mostrar primeros 3
        estudiante = df.loc[idx]
        print(f"\nEstudiante ID: {estudiante['ESTUDIANTE']}")
        print(f"  Nivel: {estudiante['NIVEL_MAXIMO']}")
        print(f"  Promedio: {estudiante['PROMEDIO_GENERAL']:.2f}")
        print(f"  Tasa Aprobación: {estudiante['TASA_APROBACION']:.1f}%")
        print(f"  Materias Repetidas: {int(estudiante['MATERIAS_REPETIDAS'])}")
else:
    print("\n✓ ¡Excelente! No hay falsos negativos.")
    print("  El modelo detecta TODOS los casos en riesgo.")

# ============================================================================
# 9. GUARDAR MODELO Y RESULTADOS
# ============================================================================
print("\n9. GUARDADO DE MODELO Y RESULTADOS")
print("=" * 80)

# Guardar modelo
with open('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/modelo_arbol_decision_v3.pkl', 'wb') as f:
    pickle.dump(modelo, f)
print("✓ Modelo guardado")

# Guardar features
with open('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/features_v3.pkl', 'wb') as f:
    pickle.dump(features_numericas, f)
print("✓ Lista de características guardada")

# Guardar importancias
importancias.to_csv('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/importancia_features_v3.csv', index=False)
print("✓ Importancia de características guardada")

# Guardar métricas
metricas = {
    'modelo': 'Árbol de Decisión',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'confusion_matrix': cm.tolist(),
    'tp': int(tp),
    'tn': int(tn),
    'fp': int(fp),
    'fn': int(fn),
    'sensibilidad': sensibilidad,
    'especificidad': especificidad
}

with open('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/metricas_v3.pkl', 'wb') as f:
    pickle.dump(metricas, f)
print("✓ Métricas guardadas")

# Guardar codificación de jornada
jornada_map = {'MATUTINA': 0, 'NOCTURNA': 1, 'OTRA': 2}
with open('C:/Users/Magno/Documents/U GUAYAQUIL/NIVEL 5/ALMACENAMIENTO DE DATOS Y MINERIA/Proyecto Final/2/PROYECTO_ABANDONO_ACADEMICO_V3/models/jornada_encoding_v3.pkl', 'wb') as f:
    pickle.dump(jornada_map, f)
print("✓ Codificación de jornada guardada")

print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print(f"\nModelo: Árbol de Decisión")
print(f"Accuracy: {accuracy:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.4f}")
print(f"\nVariables más importantes:")
for i, (_, row) in enumerate(importancias.head(3).iterrows(), 1):
    print(f"  {i}. {row['Caracteristica']}")
print(f"\nFalsos Negativos: {fn} (casos en riesgo no detectados)")
print(f"Falsos Positivos: {fp} (falsas alarmas)")
print("\n✓ Modelado completado exitosamente")
print("=" * 80)
