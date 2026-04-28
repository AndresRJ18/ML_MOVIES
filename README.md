<div align="center">
  <h1>🎬 Movie Recommender System (MOVIE_ML)</h1>
  <p><em>Construyendo un Sistema de Recomendación desde Cero (sin librerías de Machine Learning)</em></p>

  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
  ![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
</div>

---

## 📖 Descripción del Proyecto

En la era de la economía digital, la sobreabundancia de información en las plataformas de streaming genera una severa fatiga de decisión en los usuarios. **MOVIE_ML** nace como una solución integral a este problema, utilizando el dataset de TMDB para filtrar contenido de manera eficiente y ajustar matemáticamente las recomendaciones a la identidad de cada usuario.

El proyecto implementa un enfoque híbrido que combina la precisión de los atributos intrínsecos de las películas con el descubrimiento de patrones ocultos de comportamiento, tratando el gusto cinematográfico como una variable estrictamente cuantificable.

---

## 👥 Equipo de Desarrollo

Este proyecto fue desarrollado colaborativamente por los estudiantes:
* Andrés Rodas
* Patricia Rebatta
* Marcelo Rodriguez

---

## ⚙️ Tecnologías y Algoritmos Implementados

Para resolver el problema de recomendación se construyeron 3 motores algorítmicos:

1. **Content-Based Filtering (TF-IDF + Similitud Coseno)**
   * Construcción de "Tag Soups" uniendo directores, keywords, y cast.
   * Vectorización de textos usando la frecuencia de términos inversa matemáticamente pura.
2. **Matrix Factorization con SGD (Stochastic Gradient Descent)**
   * Algoritmo de optimización que descompone una matriz de ratings empírica en matrices latentes P y Q.
   * Entrenamiento iterativo minimizando el Error Cuadrático Medio (RMSE) con regularización $\lambda$.
3. **K-Nearest Neighbors (KNN Colaborativo)**
   * Basado en memoria mediante promedios ponderados por distancia coseno entre ítems.

---

## 🚀 Instrucciones de Instalación y Uso

Sigue estos pasos para ejecutar el proyecto en tu máquina local:

### 1. Clonar el repositorio y preparar el entorno
```bash
git clone [https://github.com/AndresRJ18/ML_MOVIES.git](https://github.com/AndresRJ18/ML_MOVIES.git)
cd ML_MOVIES

# Instalar los requerimientos (solo Numpy, Pandas, Matplotlib y Jupyter)
pip install -r requirements.txt

---

## Instrucciones de Instalación y Uso

Sigue estos pasos para ejecutar el proyecto en tu máquina local:

### 1. Clonar el repositorio y preparar el entorno
```bash
git clone https://github.com/AndresRJ18/ML_MOVIES.git
cd ML_MOVIES

# Instalar los requerimientos (solo Numpy, Pandas, Matplotlib y Jupyter)
pip install -r requirements.txt
```

### 2. Descargar el Dataset
Utilizamos el dataset **TMDB Movie Metadata** de Kaggle.
1. Ingresa a [Kaggle: TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
2. Descarga el archivo comprimido y extrae los CSVs.
3. Coloca los archivos `tmdb_5000_movies.csv` y `tmdb_5000_credits.csv` dentro de la ruta `data/raw/` (debes crear la carpeta si clonaste el repo vacío).

### 3. Ejecutar el flujo de trabajo
El proyecto está dividido en notebooks interactivos que deben ejecutarse en orden secuencial. 
Inicia Jupyter Notebook:
```bash
jupyter notebook
```
Y ejecuta los notebooks en el siguiente orden estricto dentro de la carpeta `/notebooks`:
1. `01_data_exploration.ipynb` *(Análisis exploratorio)*
2. `02_preprocessing.ipynb` *(Limpieza, Tag Soups y división Train/Test)*
3. `03_content_based.ipynb` *(Ejecución del modelo de Andres)*
4. `04_matrix_factorization.ipynb` *(Ejecución del modelo de Patricia)*
5. `05_knn.ipynb` *(Ejecución del modelo de Marcelo)*
6. `06_evaluation_comparison.ipynb` *(Métricas finales y tabla de conclusiones)*

---

## Estructura del Directorio

```text
ML_MOVIES/
├── data/
│   ├── raw/             # (Vacía en GitHub) → colocar CSVs de Kaggle aquí
│   └── processed/       # Matrices y datasets procesados por el notebook 02
├── notebooks/           # Notebooks enumerados con la secuencia de ejecución
├── src/
│   ├── preprocessing/   # Funciones de parseo JSON y división de datos
│   ├── models/          # Implementación matemática de TF-IDF, SGD y KNN
│   └── evaluation/      # Métricas (Precision@K, Recall@K, RMSE, Coverage)
├── reports/             # Figuras generadas y reportes en PDF del proyecto
├── presentation/        # Diapositivas para la defensa (20 min)
├── requirements.txt     # Dependencias necesarias
└── README.md            # Este archivo
```

---
*Proyecto Universitario - Creado con fines académicos y de investigación profunda sobre los fundamentos del Machine Learning.*
