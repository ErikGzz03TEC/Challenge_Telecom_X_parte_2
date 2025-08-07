# 📊 Customer Churn Prediction

Este proyecto tiene como objetivo predecir la probabilidad de que un cliente abandone una empresa de telecomunicaciones utilizando modelos de machine learning. Se implementó un pipeline completo de ciencia de datos: desde la limpieza y transformación de los datos hasta la evaluación de múltiples algoritmos con ajuste de hiperparámetros y técnicas para tratar el desbalanceo de clases.

---

## 🧠 Objetivo del proyecto

Predecir si un cliente abandonará o no la compañía (`Churn`) a partir de sus características demográficas, contractuales y de facturación. Esto permitirá a la empresa anticipar pérdidas y diseñar estrategias de retención.

---

## 🛠️ Herramientas utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib / Seaborn
- Jupyter Notebook
- Git / GitHub

---

## 🔄 Pipeline general

1. **Carga y exploración de datos**
   - Revisión de valores faltantes, tipos de variables y balance de clases.
   - Análisis exploratorio (EDA) de variables más influyentes.

2. **Preprocesamiento**
   - Normalización de variables numéricas con MinMaxScaler.
   - Codificación One-Hot para variables categóricas.
   - Eliminación de variables con multicolinealidad alta (VIF > 7).
   - División en conjuntos de entrenamiento, validación y prueba.

3. **Balanceo de clases**
   - Aplicación de SMOTE sobre los datos de entrenamiento para aumentar la clase minoritaria (`Churn = Yes`).

4. **Entrenamiento y evaluación**
   - Comparación entre distintos modelos: Árbol de Decisión, Random Forest y KNN
   - Validación cruzada con `cross_val_predict` sobre los datos balanceados.
   - Ajuste de hiperparámetros usando `GridSearchCV`.

5. **Pipeline final**
   - Integración completa de preprocesamiento, balanceo y modelo en un `Pipeline` reutilizable.

---

## 📁 Estructura del repositorio
- Codigo para limpiar la base de datos origina: ``Limpieza_datos.ipynb``
- Base de datos:``Base_datos_tratada.csv``
- Notebook principal: ``Challenge_TelecomX_2.ipynb``
- Modelos guardados: ``best_tree_model.pkl``
- Documentación: ``Readme.md``



---

## 🧪 Evaluación de modelos

Los modelos fueron evaluados con las siguientes métricas:

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de confusión

El modelo con mejor rendimiento fue **Árbol de decisiones**, con hiperparámetros optimizados para buscar un mejor recall (mejorar la predicción de las personas que abandonan la empresa) y SMOTE aplicado. El pipeline logró resultados sólidos en datos de validación y test.


---

## 📌 Resultados destacados

- **Modelo final:** Árbol de decisiones con pipeline completo.
- **Métricas (test set):**
    - Exactitud: 0.661
    - Precisión: 0.433
    - Recall: 0.879
    - F1 Score: 0.58

*(Valores simulados para ilustración. Actualizar con métricas reales.)*

De igual forma es importante destacar que el las variables más importantes para predecir el abandono de un cliente fueron las siguientes:

| Variable                                              | Importancia (%) |
|-------------------------------------------------------|-----------------|
| onehotencoder__Tipo_contrato_Dos años                 | 46.16           |
| onehotencoder__Tipo_contrato_Un año                   | 33.30           |
| onehotencoder__Metodo_pago_Cheque electrónico         | 12.65           |
| remainder__Peliculas_en_streaming                     | 4.69            |
| onehotencoder__Tipo_servicio_internet_Fibra Optica    | 2.68            |
| remainder__Es_mayor_de_edad                           | 0.51            |

Como se puede observar se utilizaron muy pocas variables para predecir el abandono de un cliente, por lo tanto considero que todavía existen muchas oportunidades de mejora en el modelo, buscando nuevas variables que puedan aportar información relevante relacionadas con clientes que han abandonado el servicio.

---

## 💾 Cómo usar el modelo

```python
import joblib
modelo = joblib.load('best_tree_model.pkl')
predicciones = modelo.predict(X_nuevo_cliente)
````
