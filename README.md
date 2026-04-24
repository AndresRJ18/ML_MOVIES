# Sistema de Recomendación de Películas

Proyecto universitario de Machine Learning — Grupo 3 personas.

## Descripción

Sistema de recomendación de películas implementado **desde cero** usando solo NumPy y Pandas, sin frameworks de modelamiento (sin scikit-learn, PyTorch ni TensorFlow).

## Dataset

**TMDB Movie Metadata** — Kaggle  
Descargar desde: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata  
Colocar los dos archivos en `data/raw/`:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

## Algoritmos Implementados

| Algoritmo | Tipo | Optimización |
|---|---|---|
| TF-IDF + Similitud Coseno | Content-Based Filtering | Fórmula cerrada |
| Matrix Factorization + SGD | Collaborative Filtering | Gradient Descent |
| KNN Colaborativo | Collaborative Filtering | Basado en distancia |

## Estructura del Proyecto

```
MOVIE_ML/
├── data/
│   ├── raw/           ← colocar CSVs de Kaggle aquí
│   └── processed/     ← generado por notebook 02
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_matrix_factorization.ipynb
│   ├── 05_knn.ipynb
│   └── 06_evaluation_comparison.ipynb
├── src/
│   ├── preprocessing/   ← json_parser, feature_engineer, train_test_split
│   ├── models/          ← tfidf, cosine_similarity, matrix_factorization, knn
│   └── evaluation/      ← metrics (Precision@K, Recall@K, RMSE, Coverage)
├── reports/figures/
└── presentation/
```

## Instalación

```bash
pip install -r requirements.txt
```

Solo requiere: `numpy`, `pandas`, `matplotlib`, `jupyter`.

## Ejecución

Correr los notebooks en orden: `01 → 02 → 03 → 04 → 05 → 06`

## División del Trabajo

- **Estudiante A:** EDA, Preprocesamiento, Content-Based Filtering  
  → Informe: Introducción, descripción del dataset, metodología Content-Based
- **Estudiante B:** Matrix Factorization + SGD  
  → Informe: Estado del Arte, derivación matemática del algoritmo de optimización
- **Estudiante C:** KNN, Evaluación comparativa, Slides  
  → Informe: Experimentación y Resultados, Discusión, Conclusiones

## Entregables

- `reports/report_final.pdf` — Informe (Introducción, Estado del Arte, Metodología, Resultados, Discusión, Conclusiones)
- `notebooks/` — Código Python comentado, sin frameworks de ML
- `presentation/slides.pdf` — Presentación 20 minutos + 5 min preguntas
