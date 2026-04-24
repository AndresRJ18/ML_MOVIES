# Resumen del Proyecto — Sistema de Recomendación de Películas

---

## ¿Qué vamos a construir?

Un sistema que dado una película (o un usuario), recomienda otras películas. Implementado **desde cero** con solo NumPy y Pandas — sin usar sklearn, PyTorch ni TensorFlow.

---

## El Dataset

**TMDB Movie Metadata** de Kaggle — son **dos archivos CSV**:

| Archivo | Contenido |
|---|---|
| `tmdb_5000_movies.csv` | ~4,800 películas: título, géneros, keywords, presupuesto, rating |
| `tmdb_5000_credits.csv` | reparto y equipo por película |

**Problema clave:** el dataset NO tiene ratings por usuario (no sabe qué usuarios vieron qué películas). Solo tiene `vote_average` (ej: 7.8/10) y `vote_count` (ej: 14,000 votos). Esto lo resolvemos simulando usuarios sintéticos en el preprocesamiento.

**Otro problema:** las columnas de géneros, actores, etc. están en formato JSON dentro de strings:
```
genres = "[{'id': 18, 'name': 'Drama'}, {'id': 35, 'name': 'Comedy'}]"
```
→ Hay que parsearlas con `ast.literal_eval()` (no `json.loads`).

---

## Los 3 Algoritmos

### Algoritmo 1 — Content-Based Filtering (TF-IDF + Similitud Coseno)
**Responsable: Estudiante A**

**Idea:** Cada película se describe con un "tag soup" — una cadena de texto con sus géneros, keywords, actores principales y director. Luego se compara qué tan parecidos son esos textos.

**Cómo funciona:**
1. Construir el tag soup por película:
   ```
   Avatar → "action adventure sciencefiction iceplanet samworthington zoesaldana jamescameron jamescameron"
   ```
2. Convertir cada película en un vector numérico con **TF-IDF** (peso de cada palabra)
3. Medir similitud entre películas con **similitud coseno**:
   ```
   sim(a, b) = (a · b) / (||a|| × ||b||)
   ```
4. Las N películas con mayor similitud → recomendación

**No necesita ratings de usuarios.** Puede recomendar películas nuevas (no hay cold-start para ítems).

---

### Algoritmo 2 — Matrix Factorization con SGD ⭐ ALGORITMO CENTRAL
**Responsable: Estudiante B**

**Este es el algoritmo de optimización requerido por el curso.**

**Idea:** Hay una matriz R (usuarios × películas) con los ratings. La mayoría son desconocidos (0). Queremos "factorizarla" en dos matrices más pequeñas que capturan patrones ocultos:

```
R  ≈  P  ×  Qᵀ
(500×200) = (500×k) × (k×200)
```

- **P**: cada usuario como vector de k características latentes
- **Q**: cada película como vector de k características latentes
- La predicción del rating de usuario u para película i = `P[u] · Q[i]`

**Optimización (SGD):**

Función de pérdida a minimizar:
```
L = Σ (r_ui - p_u · q_i)² + λ(||p_u||² + ||q_i||²)
```

Reglas de actualización en cada paso:
```
error = r_ui - p_u · q_i
p_u ← p_u + α × (error × q_i  -  λ × p_u)
q_i ← q_i + α × (error × p_u  -  λ × q_i)
```

**Resultado clave del informe:** gráfico de **curva de convergencia** (pérdida disminuye por época) → prueba que SGD funciona.

**Hiperparámetros a experimentar:** k (dimensión latente), α (learning rate), λ (regularización).

---

### Algoritmo 3 — KNN Colaborativo Item-Item
**Responsable: Estudiante C**

**Idea:** Para recomendar película i al usuario u, busca las k películas más similares a i que u ya calificó, y predice el rating como promedio ponderado.

```
r̂_ui = Σ sim(i,j) × r_uj  /  Σ |sim(i,j)|
       (j vecinos de i que u calificó)
```

No hay entrenamiento con gradientes — la "preparación" es precalcular la matriz de similitudes entre todas las películas.

**Contraste con MF+SGD:**

| | MF+SGD | KNN |
|---|---|---|
| Tipo | Model-based | Memory-based |
| Optimización | Gradient Descent | Ninguna |
| Interpretabilidad | Baja | Alta ("porque viste X") |
| Escala | Bien | Mal (O(n²)) |

---

## División del Trabajo

### Estudiante A — Data + Content-Based
**Archivos:** `01_data_exploration.ipynb`, `02_preprocessing.ipynb`, `03_content_based.ipynb`  
**src:** `json_parser.py`, `feature_engineer.py`, `train_test_split.py`, `tfidf.py`, `cosine_similarity.py`  
**Informe:** Introducción, descripción del dataset, metodología Content-Based

**Tareas concretas:**
- Explorar el dataset: distribuciones, valores nulos, géneros más frecuentes
- Parsear columnas JSON con `ast.literal_eval()`
- Construir el tag soup
- Implementar TF-IDF desde cero (numpy)
- Implementar similitud coseno desde cero (numpy)
- Simular la matriz de ratings (para que B y C tengan datos)

---

### Estudiante B — Algoritmo de Optimización (MF+SGD)
**Archivos:** `04_matrix_factorization.ipynb`  
**src:** `matrix_factorization.py`  
**Informe:** Estado del Arte (8-10 papers), derivación matemática del SGD

**Tareas concretas:**
- Implementar la clase `MatrixFactorizationSGD` con numpy
- Loop de entrenamiento SGD: inicializar P y Q aleatorio, iterar sobre ratings conocidos, actualizar con las fórmulas de gradiente
- Graficar curva de convergencia (Train RMSE vs Test RMSE por época)
- Grid search: k ∈ {5,10,20,50}, α ∈ {0.001,0.005,0.01}
- Buscar 8-10 papers sobre CF y Matrix Factorization (2018-2024) para el Estado del Arte

---

### Estudiante C — KNN + Evaluación + Slides
**Archivos:** `05_knn.ipynb`, `06_evaluation_comparison.ipynb`  
**src:** `knn.py`, `metrics.py`  
**Informe:** Experimentación y Resultados, Discusión, Conclusiones  
**Slides de presentación**

**Tareas concretas:**
- Implementar `ItemKNN` con numpy (similitud coseno item-item, predicción ponderada)
- Experimentar con k ∈ {5,10,20,30,50}
- Implementar desde cero: RMSE, MAE, Precision@K, Recall@K, F1@K, Coverage
- Crear tabla comparativa de los 3 modelos
- Graficar Precision@K y Recall@K para diferentes valores de K

---

## Métricas de Evaluación

| Métrica | Fórmula | Aplica a |
|---|---|---|
| RMSE | √(Σ(real−pred)²/N) | MF, KNN |
| Precision@K | \|relevantes en top-K\| / K | Todos |
| Recall@K | \|relevantes en top-K\| / \|todos relevantes\| | Todos |
| F1@K | 2×P×R/(P+R) | Todos |
| Coverage | % del catálogo recomendado alguna vez | Todos |

> "Relevante" = película con rating ≥ 7.0

---

## Flujo de Ejecución

```
Descargar CSVs de Kaggle → data/raw/
         ↓
  01_data_exploration   (entender el dataset)
         ↓
  02_preprocessing      (parsear JSON, tag soup, simular ratings, split 80/20)
         ↓
  03_content_based  04_matrix_factorization  05_knn   ← correr en paralelo
         ↓
  06_evaluation_comparison  (tabla final + gráficos)
```

---

## Para el Informe — ¿Qué va en cada sección?

| Sección | Quién | Contenido |
|---|---|---|
| Introducción | A | Por qué es útil recomendar películas, descripción del dataset, objetivos |
| Estado del Arte | B | 8-10 papers: Netflix Prize, SVD (Koren 2009), NCF, etc. |
| Metodología | A+B+C | Derivación matemática de TF-IDF, SGD (con ecuaciones), KNN, métricas |
| Experimentación | C | Tablas, gráficos de convergencia, grid search, resultados numéricos |
| Discusión | C | Cuál modelo es mejor y por qué, limitaciones (ratings simulados) |
| Conclusiones | C | Hallazgos principales, trabajo futuro |

---

## Para la Presentación (20 min)

| Slides | Tiempo | Quién | Contenido |
|---|---|---|---|
| 1-2 | 3 min | A | Introducción + dataset |
| 3-4 | 4 min | A | Preprocesamiento + tag soup |
| 5-7 | 7 min | B | MF+SGD: ecuaciones + curva de convergencia + resultados |
| 8-9 | 3 min | A o C | Content-Based + KNN |
| 10-12 | 3 min | C | Tabla comparativa + conclusiones |

> **El bloque de B (MF+SGD) es el más importante** — hay que mostrar las ecuaciones del gradiente y la curva de convergencia. Eso es lo que demuestra que entendieron el algoritmo de optimización.

---

## Lo único que falta hacer

1. Descargar los CSVs de Kaggle y ponerlos en `data/raw/`
2. `pip install -r requirements.txt` (solo 4 paquetes: numpy, pandas, matplotlib, jupyter)
3. Correr los notebooks en orden: `01 → 02 → 03 → 04 → 05 → 06`
