"""
train_test_split.py — División del conjunto de ratings en train/test.

Implementado desde cero: sin sklearn ni librerías externas.

Responsable: Estudiante A
"""
import numpy as np


def split_ratings(ratings_matrix, test_ratio=0.2, seed=42):
    """
    Divide la matriz de ratings en conjuntos de entrenamiento y prueba.

    Para cada usuario, enmascara aleatoriamente test_ratio de sus ratings
    conocidos. Los ratings enmascarados van al conjunto de prueba.

    Parámetros
    ----------
    ratings_matrix : np.ndarray, shape (n_users, n_movies)
        Matriz de ratings. 0 indica rating desconocido.
    test_ratio : float
        Proporción de ratings por usuario que van al conjunto de prueba.
    seed : int

    Retorna
    -------
    train : np.ndarray — misma forma, ratings de prueba puestos a 0
    test  : np.ndarray — misma forma, solo los ratings de prueba
    """
    rng = np.random.default_rng(seed)
    train = ratings_matrix.copy().astype(float)
    test  = np.zeros_like(ratings_matrix, dtype=float)

    for u in range(ratings_matrix.shape[0]):
        # Índices de películas con rating conocido para este usuario
        known_indices = np.where(ratings_matrix[u] > 0)[0]
        if len(known_indices) == 0:
            continue
        n_test = max(1, int(len(known_indices) * test_ratio))
        test_indices = rng.choice(known_indices, size=n_test, replace=False)
        test[u, test_indices]  = ratings_matrix[u, test_indices]
        train[u, test_indices] = 0.0  # enmascarar en train

    return train, test
