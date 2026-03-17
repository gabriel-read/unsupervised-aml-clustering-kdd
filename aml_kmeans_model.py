"""
Modelo de Detección de Anomalías Transaccionales (AML) - K-Means Clustering
Metodología KDD aplicada al Riesgo Actuarial y Cumplimiento Normativo.
"""

# ==============================================================================
# PASO 0: CONFIGURACIÓN DEL ENTORNO Y MITIGACIÓN DE ERRORES
# ==============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Mitigación del error de 'joblib/loky' en entornos Windows para el recuento de núcleos
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning)

sns.set_theme(style="whitegrid", palette="muted")

# ==============================================================================
# PASO 1: GENERACIÓN DE DATOS SINTÉTICOS Y EXPORTACIÓN (DATA PREPARATION)
# ==============================================================================
np.random.seed(42)

def generate_aml_data(n_samples: int = 1000) -> pd.DataFrame:
    # 1.1 Generar 995 perfiles de comportamiento "Normal"
    n_normal = n_samples - 5
    monto_promedio = np.random.lognormal(mean=7, sigma=1, size=n_normal)
    frecuencia = np.random.poisson(lam=5, size=n_normal)
    indice_riesgo = np.random.normal(loc=30, scale=10, size=n_normal) 
    
    df_normal = pd.DataFrame({
        'Cliente_ID': range(1, n_normal + 1),
        'Monto_Promedio_Transaccion': monto_promedio,
        'Frecuencia_Mensual_Transacciones': np.clip(frecuencia, 1, None),
        'Indice_Riesgo_Jurisdiccion': np.clip(indice_riesgo, 1, 100),
        'Es_Outlier_Inyectado': False
    })
    
    # 1.2 Inyección de Anomalías: 5 Outliers de Alto Riesgo (Tipologías de Lavado)
    outliers_data = {
        'Cliente_ID': range(n_normal + 1, n_samples + 1),
        'Monto_Promedio_Transaccion': [500000, 750000, 950000, 100000, 850000],
        'Frecuencia_Mensual_Transacciones': [1, 2, 1, 95, 3],
        'Indice_Riesgo_Jurisdiccion': [95, 99, 92, 85, 98],
        'Es_Outlier_Inyectado': [True]*5
    }
    df_outliers = pd.DataFrame(outliers_data)
    
    # 1.3 Combinar y mezclar el dataset
    df_final = pd.concat([df_normal, df_outliers]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_final

df = generate_aml_data()

# Exportar datos crudos para posterior validación cruzada en SPSS
df.to_csv("aml_synthetic_data_raw.csv", index=False)
print("✅ Dataset crudo exportado como 'aml_synthetic_data_raw.csv'.")

# ==============================================================================
# PASO 2: TRANSFORMACIÓN Y ESCALADO (PREPROCESSING)
# ==============================================================================
features = ['Monto_Promedio_Transaccion', 'Frecuencia_Mensual_Transacciones', 'Indice_Riesgo_Jurisdiccion']
X = df[features]

# La estandarización es vital para calcular distancias espaciales equitativas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# PASO 3: MODELADO K-MEANS Y OPTIMIZACIÓN
# ==============================================================================
# 3.1 Método del Codo (Descomentado para uso interactivo, opcional para ejecución en batch)
'''
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    wcss.append(kmeans_temp.inertia_)
plt.figure(figsize=(8, 4))
plt.plot(K_range, wcss, marker='o', linestyle='-', color='#2c3e50')
plt.title('Método del Codo (Optimización de K)')
plt.show()
'''

# 3.2 Entrenamiento del Modelo Final con K=3
k_optimo = 3
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 3.3 Validación Matemática de Densidad
sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f"✅ Silhouette Score para K={k_optimo}: {sil_score:.4f}")

# ==============================================================================
# PASO 4: LÓGICA DE DETECCIÓN DE ANOMALÍAS (EVALUATION)
# ==============================================================================
centroides = kmeans.cluster_centers_
distancias = [np.linalg.norm(x - centroides[cluster]) for x, cluster in zip(X_scaled, df['Cluster'])]
df['Distancia_Centroide'] = distancias

# Establecer umbral de riesgo estricto (Percentil 99 de las distancias euclidianas)
umbral_anomalia = np.percentile(distancias, 99)
df['Es_Anomalia_Modelo'] = df['Distancia_Centroide'] > umbral_anomalia

# Cálculo de Eficacia (Recall)
anomalias_detectadas = df[df['Es_Anomalia_Modelo']]
outliers_reales_detectados = anomalias_detectadas[anomalias_detectadas['Es_Outlier_Inyectado']]
recall_count = len(outliers_reales_detectados)

print(f"\n--- REPORTE DE EFICACIA DEL MODELO ---")
print(f"Umbral de distancia (P99): {umbral_anomalia:.4f}")
print(f"Total de anomalías detectadas: {len(anomalias_detectadas)}")
print(f"Anomalías sintéticas capturadas: {recall_count} de 5.")

# ==============================================================================
# PASO 5: VISUALIZACIÓN INTERACTIVA Y EXPORTACIÓN FINAL
# ==============================================================================
fig = px.scatter_3d(
    df, x='Monto_Promedio_Transaccion', y='Frecuencia_Mensual_Transacciones', z='Indice_Riesgo_Jurisdiccion',
    color='Cluster', symbol='Es_Outlier_Inyectado', color_continuous_scale=px.colors.sequential.Viridis, opacity=0.8
)
fig.update_layout(title=dict(text='Proyección K-Means 3D: Topología del Riesgo', x=0.5), margin=dict(l=0, r=0, b=0, t=50))
fig.show()

# Exportación de resultados para SPSS (Validación ANOVA)
df.to_csv("aml_results_clustered.csv", index=False)
print("\n✅ Dataset final exportado como 'aml_results_clustered.csv'.")
