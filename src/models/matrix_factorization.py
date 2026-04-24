"""
matrix_factorization.py — Matrix Factorization con SGD implementado desde cero.

ALGORITMO CENTRAL del proyecto. Satisface el requisito de algoritmo de optimización.

=============================================================================
FUNDAMENTO MATEMÁTICO
=============================================================================

Dado una matriz de ratings R de forma (n_users, n_movies) donde R[u,i] es
el rating del usuario u para la película i (0 = desconocido), buscamos
factorizarla como:

    R ≈ P · Q^T

donde:
    P: matriz latente de usuarios, shape (n_users, k)
    Q: matriz latente de películas, shape (n_movies, k)
    k: dimensión del espacio latente (hiperparámetro)

La predicción del rating del usuario u para la película i es:
    r̂_ui = p_u · q_i  (producto punto de sus vectores latentes)

FUNCIÓN DE PÉRDIDA (con regularización L2):
    L = Σ_{(u,i) conocidos} (r_ui - p_u · q_i)² + λ(||p_u||² + ||q_i||²)

GRADIENTES:
    ∂L/∂p_u = -2 * error_ui * q_i + 2λ * p_u
    ∂L/∂q_i = -2 * error_ui * p_u + 2λ * q_i

    donde error_ui = r_ui - p_u · q_i

REGLAS DE ACTUALIZACIÓN SGD:
    p_u ← p_u + α * (error_ui * q_i - λ * p_u)
    q_i ← q_i + α * (error_ui * p_u - λ * q_i)

HIPERPARÁMETROS:
    k      : dimensión latente (ej. 10, 20, 50)
    alpha  : tasa de aprendizaje (ej. 0.001, 0.005, 0.01)
    lambda_: regularización L2 (ej. 0.01, 0.1)
    epochs : número de épocas de entrenamiento

Responsable: Estudiante B
=============================================================================
"""
import numpy as np


class MatrixFactorizationSGD:
    """
    Factorización de matrices con Stochastic Gradient Descent.

    Parámetros
    ----------
    k      : int     — dimensión del espacio latente
    alpha  : float   — tasa de aprendizaje
    lambda_: float   — coeficiente de regularización L2
    epochs : int     — número de épocas de entrenamiento
    seed   : int     — semilla para reproducibilidad
    """

    def __init__(self, k=20, alpha=0.005, lambda_=0.02, epochs=100, seed=42):
        self.k       = k
        self.alpha   = alpha
        self.lambda_ = lambda_
        self.epochs  = epochs
        self.seed    = seed

        # Se inicializan en fit()
        self.P = None  # (n_users, k)
        self.Q = None  # (n_movies, k)
        self.train_rmse_history = []  # pérdida por época (para graficar)
        self.test_rmse_history  = []

    def fit(self, train_matrix, test_matrix=None, verbose=True):
        """
        Entrena el modelo sobre la matriz de ratings.

        Parámetros
        ----------
        train_matrix : np.ndarray, shape (n_users, n_movies)
            Matriz de entrenamiento. 0 = rating desconocido.
        test_matrix  : np.ndarray o None
            Si se proporciona, se calcula RMSE de prueba por época.
        verbose : bool
            Si True, imprime progreso cada 10 épocas.
        """
        rng = np.random.default_rng(self.seed)
        n_users, n_movies = train_matrix.shape

        # Inicialización aleatoria con escala pequeña
        self.P = rng.normal(0, 0.1, (n_users, self.k))
        self.Q = rng.normal(0, 0.1, (n_movies, self.k))

        # Obtener índices de ratings conocidos una sola vez
        users_idx, movies_idx = np.where(train_matrix > 0)
        n_ratings = len(users_idx)

        for epoch in range(1, self.epochs + 1):
            # Barajar los ratings en cada época (SGD estocástico)
            perm = rng.permutation(n_ratings)
            users_shuffled  = users_idx[perm]
            movies_shuffled = movies_idx[perm]

            # Iterar sobre cada rating conocido
            for u, i in zip(users_shuffled, movies_shuffled):
                # Error de predicción
                error_ui = train_matrix[u, i] - np.dot(self.P[u], self.Q[i])

                # Guardar copias para actualización simultánea
                p_u_old = self.P[u].copy()
                q_i_old = self.Q[i].copy()

                # Actualización SGD
                self.P[u] += self.alpha * (error_ui * q_i_old - self.lambda_ * p_u_old)
                self.Q[i] += self.alpha * (error_ui * p_u_old - self.lambda_ * q_i_old)

            # Calcular RMSE en train
            train_rmse = self._rmse(train_matrix, users_idx, movies_idx)
            self.train_rmse_history.append(train_rmse)

            # Calcular RMSE en test (opcional)
            if test_matrix is not None:
                t_users, t_movies = np.where(test_matrix > 0)
                test_rmse = self._rmse(test_matrix, t_users, t_movies)
                self.test_rmse_history.append(test_rmse)

            if verbose and epoch % 10 == 0:
                msg = f"Época {epoch:3d}/{self.epochs} | Train RMSE: {train_rmse:.4f}"
                if test_matrix is not None:
                    msg += f" | Test RMSE: {test_rmse:.4f}"
                print(msg)

        return self

    def _rmse(self, matrix, users_idx, movies_idx):
        """Calcula RMSE sobre los índices dados."""
        predictions = np.sum(self.P[users_idx] * self.Q[movies_idx], axis=1)
        actual      = matrix[users_idx, movies_idx]
        return np.sqrt(np.mean((actual - predictions) ** 2))

    def predict(self, user_idx, movie_idx):
        """
        Predice el rating de un usuario para una película.

        Parámetros
        ----------
        user_idx  : int
        movie_idx : int

        Retorna
        -------
        float
        """
        return float(np.dot(self.P[user_idx], self.Q[movie_idx]))

    def predict_all(self):
        """
        Genera la matriz completa de predicciones R̂ ≈ P · Q^T.

        Retorna
        -------
        np.ndarray, shape (n_users, n_movies)
        """
        return self.P @ self.Q.T

    def recommend(self, user_idx, ratings_matrix, n=10):
        """
        Genera las N mejores recomendaciones para un usuario.
        Excluye películas que el usuario ya calificó.

        Parámetros
        ----------
        user_idx       : int
        ratings_matrix : np.ndarray — para excluir películas ya vistas
        n              : int

        Retorna
        -------
        list[int] — índices de películas recomendadas
        """
        predictions = self.P[user_idx] @ self.Q.T
        # Excluir películas ya calificadas
        already_rated = ratings_matrix[user_idx] > 0
        predictions[already_rated] = -np.inf
        top_n = np.argsort(predictions)[::-1][:n]
        return top_n.tolist()
