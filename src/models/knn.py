"""
knn.py — KNN Colaborativo (item-item) implementado desde cero con NumPy.

Algoritmo basado en memoria: predice el rating de un usuario u para
una película i como el promedio ponderado de los ratings del usuario u
en las k películas más similares a i que él sí ha calificado.

Fórmula de predicción:
    r̂_ui = Σ_{j ∈ N_k(i)} sim(i, j) * r_uj / Σ_{j ∈ N_k(i)} |sim(i, j)|

donde N_k(i) son las k películas más similares a i que el usuario u calificó.

La similitud entre películas se calcula con similitud coseno sobre
los vectores de ratings de todos los usuarios.

Responsable: Estudiante C
"""
import numpy as np


class ItemKNN:
    """
    KNN Colaborativo item-item.

    Parámetros
    ----------
    k : int — número de vecinos a considerar para la predicción
    """

    def __init__(self, k=20):
        self.k = k
        self.similarity_matrix = None  # (n_movies, n_movies)
        self.ratings_matrix    = None  # referencia a la matriz de train

    def fit(self, ratings_matrix):
        """
        Precalcula la matriz de similitud coseno item-item.

        Parámetros
        ----------
        ratings_matrix : np.ndarray, shape (n_users, n_movies)
            0 = rating desconocido.
        """
        self.ratings_matrix = ratings_matrix
        # Transponer: cada fila = vector de ratings de una película
        item_vectors = ratings_matrix.T.astype(float)  # (n_movies, n_users)

        # Normalizar por norma L2 (ignorar ceros en la normalización)
        norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = item_vectors / norms

        # Similitud coseno: producto de puntos de vectores normalizados
        self.similarity_matrix = normalized @ normalized.T  # (n_movies, n_movies)

        return self

    def predict(self, user_idx, movie_idx):
        """
        Predice el rating del usuario user_idx para la película movie_idx.

        Parámetros
        ----------
        user_idx  : int
        movie_idx : int

        Retorna
        -------
        float — predicción del rating (0 si no hay vecinos suficientes)
        """
        # Ratings del usuario para todas las películas
        user_ratings = self.ratings_matrix[user_idx]

        # Similitudes de movie_idx con todas las películas
        sims = self.similarity_matrix[movie_idx].copy()

        # Excluir la misma película y las no calificadas por el usuario
        sims[movie_idx] = 0
        sims[user_ratings == 0] = 0

        # Seleccionar los k vecinos más similares
        top_k_idx = np.argsort(sims)[::-1][:self.k]
        top_k_sims = sims[top_k_idx]
        top_k_ratings = user_ratings[top_k_idx]

        # Solo vecinos con similitud positiva y rating conocido
        mask = (top_k_sims > 0) & (top_k_ratings > 0)
        if not np.any(mask):
            return 0.0

        numerator   = np.sum(top_k_sims[mask] * top_k_ratings[mask])
        denominator = np.sum(np.abs(top_k_sims[mask]))

        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    def recommend(self, user_idx, n=10):
        """
        Genera las N mejores recomendaciones para un usuario.
        Excluye películas ya calificadas.

        Parámetros
        ----------
        user_idx : int
        n        : int

        Retorna
        -------
        list[int] — índices de películas recomendadas
        """
        n_movies = self.ratings_matrix.shape[1]
        unrated  = np.where(self.ratings_matrix[user_idx] == 0)[0]

        predictions = []
        for movie_idx in unrated:
            pred = self.predict(user_idx, movie_idx)
            predictions.append((movie_idx, pred))

        # Ordenar por predicción descendente y retornar top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in predictions[:n]]
