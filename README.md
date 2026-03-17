# 🛡️ Detección de Anomalías Transaccionales (AML) con K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24+-orange.svg)
![SPSS](https://img.shields.io/badge/IBM_SPSS-Statistics-blue.svg)

## 📌 Business Understanding e Introducción Teórica

El **Lavado de Dinero (ML - Money Laundering)** representa un riesgo crítico tanto normativo como macroeconómico para las instituciones financieras. Tradicionalmente, los enfoques de cumplimiento se han basado en reglas estáticas (ej. alertas para transacciones mayores a $10,000). Sin embargo, las tipologías delictivas evolucionan mediante la estructuración de operaciones complejas para evadir estos umbrales rígidos.

El **Aprendizaje No Supervisado** resulta vital desde una perspectiva actuarial y de ciencia de datos. Al carecer de datos etiquetados sobre modalidades de fraude emergentes, el análisis de conglomerados (*clustering*) mediante metodologías KDD nos permite descubrir patrones ocultos de comportamiento transaccional y aislar las entidades que presentan desviaciones significativas.

## 🔬 Fundamentos Matemáticos: K-Means en Riesgo Espacial

El algoritmo K-Means evalúa la similitud midiendo la distancia en un espacio n-dimensional de variables estandarizadas mediante la **Distancia Euclidiana**. Una gran distancia respecto al centroide normativo representa una alta desviación estándar y, por tanto, un elevado riesgo transaccional.

Además, el modelo optimiza la posición de sus centroides minimizando la **Función de Costo de Inercia (WCSS)**. Un WCSS bajo indica grupos compactos, aislando eficazmente el "ruido" o valores atípicos (*outliers*).

**Importancia de la Estandarización:**
En modelos basados en distancias, la estandarización mediante `StandardScaler` (Puntuaciones Z) es innegociable. Si omitimos este paso, las variables de escala mayor (como los montos financieros) dominarían aritméticamente la distancia geométrica frente a variables de menor escala (como la frecuencia operativa), sesgando completamente la asignación de riesgo.

## 📊 Arquitectura de Datos
El modelo analiza un dataset multivariable evaluando:
1. `Monto_Promedio_Transaccion`
2. `Frecuencia_Mensual_Transacciones`
3. `Indice_Riesgo_Jurisdiccion` (Escala basada en directrices GAFI).

## 🚀 Resultados del Modelo y Salidas (Outputs)

El algoritmo logró converger exitosamente y segmentar el comportamiento transaccional base del riesgo extremo. Se inyectaron 5 anomalías simulando tipologías reales de lavado, las cuales fueron detectadas rigurosamente estableciendo el **Percentil 99** de las distancias euclidianas como umbral de alerta.

### Validación Paramétrica Cruzada (IBM SPSS Statistics)
Para garantizar la robustez topológica y evitar sobreajustes locales, se realizó una validación cruzada exportando la matriz de datos a IBM SPSS Statistics, logrando los siguientes resultados espaciales:

* **Conglomerado 2 (Comportamiento Base):** Agrupó a los 995 clientes regulares, ubicando su centroide prácticamente en el origen del plano estandarizado (Puntuaciones Z cercanas a 0).
* **Conglomerado 1 (Riesgo de Montos Extremos y Alta Jurisdicción):** Aisló 4 entidades anómalas situadas a más de 15 desviaciones estándar de la media poblacional.
* **Conglomerado 3 (Hiper-transaccionalidad / Pitufeo):** Aisló un caso extremo con una frecuencia operativa ubicada a más de 25 desviaciones estándar de la media.

**Validación ANOVA:**
Las pruebas F de Análisis de Varianza en SPSS arrojaron un p-valor (Sig.) de `.000` para todas las dimensiones operativas. Esto confirma estadísticamente que la separación multivariable posee estricta validez y que la varianza entre los clústeres es significativamente mayor a su dispersión interna.

## ⚖️ Veredicto de Cumplimiento

Las entidades localizadas en el umbral de anomalía máxima exhiben un comportamiento multidimensional que carece de justificación económica aparente bajo el perfil de su conglomerado. 

Acorde a las normativas de Cumplimiento, el estatus actuarial asignado a estas cuentas es: **"Sujeto a Investigación Profunda (EDD - Enhanced Due Diligence)"**, recomendando la inmovilización preventiva y la emisión de un Reporte de Operación Sospechosa (ROS) si la debida diligencia resulta insatisfactoria.

## 📂 Estructura del Repositorio
* `aml_kmeans_model.py`: Script principal ejecutable con la parametrización de K-Means.
* `/data`: (Ignorado en Git) Carpeta de destino para los datasets `aml_synthetic_data_raw.csv` y `aml_results_clustered.csv`.
