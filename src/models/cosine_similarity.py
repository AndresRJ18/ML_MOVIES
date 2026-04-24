"""
cosine_similarity.py — Similitud coseno implementada desde cero con NumPy.

La similitud coseno entre dos vectores a y b es:

    sim(a, b) = (a · b) / (||a|| * ||b||)

Dado que la matriz TF-IDF ya está normalizada (||fila|| = 1),
la similitud coseno se reduce a un simple producto de puntos: A @ A.T

Responsable: Estudiante A
"""
import numpy as np


def cosine_similarity_matrix(matrix):
    """
    Calcula la matriz de similitud coseno entre todas las filas.

    Si la matriz ya está normalizada (norma L2 = 1 por fila), la similitud
    coseno equivale al producto de puntos: sim = matrix @ matrix.T

    Parámetros
    ----------
    matrix : np.ndarray, shape (n, d)
        Matriz de vectores (e.g., TF-IDF normalizada).

    Retorna
    -------
    np.ndarray, shape (n, n)
        sim[i, j] = similitud coseno entre el documento i y el documento j.
    """
    # Normalizar por las normas (por si la matriz no está pre-normalizada)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = matrix / norms
    return normalized @ normalized.T


def cosine_similarity_vector(query_vec, matrix):
    """
    Calcula la similitud coseno entre un vector consulta y todas las filas.

    Parámetros
    ----------
    query_vec : np.ndarray, shape (d,)
    matrix    : np.ndarray, shape (n, d)

    Retorna
    -------
    np.ndarray, shape (n,)
        Similitudes del query con cada fila de matrix.
    """
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return np.zeros(matrix.shape[0])
    q = query_vec / q_norm

    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1
    m = matrix / norms[:, np.newaxis]

    return m @ q


def get_top_n(similarity_scores, movie_idx, n=10, exclude_self=True):
    """
    Retorna los índices de las N películas más similares.

    Parámetros
    ----------
    similarity_scores : np.ndarray, shape (n_movies,)
        Similitudes de una película con todas las demás.
    movie_idx : int
        Índice de la película consulta (para excluirse a sí misma).
    n : int
        Número de recomendaciones.
    exclude_self : bool

    Retorna
    -------
    list[int]
    """
    scores = similarity_scores.copy()
    if exclude_self:
        scores[movie_idx] = -1.0  # excluir la película misma
    top_indices = np.argsort(scores)[::-1][:n]
    return top_indices.tolist()
