"""
metrics.py — Métricas de evaluación para sistemas de recomendación.

Todas implementadas desde cero con NumPy. Sin scikit-learn.

MÉTRICAS IMPLEMENTADAS:
    - RMSE: error cuadrático medio (para predicción de ratings)
    - MAE:  error absoluto medio
    - Precision@K: fracción de recomendaciones relevantes en el top-K
    - Recall@K:    fracción de ítems relevantes recuperados en el top-K
    - F1@K:        media armónica de Precision@K y Recall@K
    - Coverage:    % del catálogo que aparece en alguna recomendación

Responsable: Estudiante C
"""
import numpy as np


# =============================================================================
# Métricas de predicción de ratings
# =============================================================================

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.

        RMSE = sqrt( (1/N) * Σ (r_true - r_pred)² )

    Parámetros
    ----------
    y_true : array-like — ratings reales
    y_pred : array-like — ratings predichos

    Retorna
    -------
    float
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.

        MAE = (1/N) * Σ |r_true - r_pred|

    Parámetros
    ----------
    y_true : array-like
    y_pred : array-like

    Retorna
    -------
    float
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# =============================================================================
# Métricas de ranking (Top-N)
# =============================================================================

def precision_at_k(recommended, relevant, k):
    """
    Precision@K: fracción de ítems recomendados en el top-K que son relevantes.

        Precision@K = |{relevant} ∩ {top-K recommended}| / K

    Parámetros
    ----------
    recommended : list[int] — lista ordenada de ítems recomendados
    relevant    : set[int]  — conjunto de ítems relevantes (ej. rating >= umbral)
    k           : int

    Retorna
    -------
    float ∈ [0, 1]
    """
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended, relevant, k):
    """
    Recall@K: fracción de ítems relevantes recuperados en el top-K.

        Recall@K = |{relevant} ∩ {top-K recommended}| / |relevant|

    Parámetros
    ----------
    recommended : list[int]
    relevant    : set[int]
    k           : int

    Retorna
    -------
    float ∈ [0, 1]
    """
    if len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def f1_at_k(recommended, relevant, k):
    """
    F1@K: media armónica de Precision@K y Recall@K.

        F1@K = 2 * (P@K * R@K) / (P@K + R@K)

    Parámetros
    ----------
    recommended : list[int]
    relevant    : set[int]
    k           : int

    Retorna
    -------
    float ∈ [0, 1]
    """
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mean_average_precision_at_k(all_recommended, all_relevant, k):
    """
    MAP@K: promedio de Average Precision@K sobre todos los usuarios.

    AP@K para un usuario = promedio de Precision@i para i=1..K
    donde solo se cuenta si el ítem en posición i es relevante.

    Parámetros
    ----------
    all_recommended : list[list[int]] — recomendaciones por usuario
    all_relevant    : list[set[int]]  — ítems relevantes por usuario
    k               : int

    Retorna
    -------
    float
    """
    ap_scores = []
    for recommended, relevant in zip(all_recommended, all_relevant):
        if len(relevant) == 0:
            continue
        relevant_set = set(relevant)
        score = 0.0
        hits  = 0
        for i, item in enumerate(recommended[:k], start=1):
            if item in relevant_set:
                hits += 1
                score += hits / i
        ap_scores.append(score / min(len(relevant_set), k))

    return float(np.mean(ap_scores)) if ap_scores else 0.0


# =============================================================================
# Métricas de sistema
# =============================================================================

def coverage(all_recommended, n_items):
    """
    Coverage: porcentaje del catálogo que aparece en al menos una recomendación.

        Coverage = |items únicos recomendados| / n_items

    Una cobertura alta indica que el sistema recomienda diversidad de películas
    en lugar de concentrarse siempre en las mismas populares.

    Parámetros
    ----------
    all_recommended : list[list[int]] — recomendaciones de todos los usuarios
    n_items         : int — tamaño total del catálogo

    Retorna
    -------
    float ∈ [0, 1]
    """
    recommended_set = set()
    for recs in all_recommended:
        recommended_set.update(recs)
    return len(recommended_set) / n_items


# =============================================================================
# Evaluación batch (para el notebook de comparación)
# =============================================================================

def evaluate_model(all_recommended, all_relevant, n_items, k=10):
    """
    Calcula todas las métricas de ranking para un modelo dado.

    Parámetros
    ----------
    all_recommended : list[list[int]]
    all_relevant    : list[set[int]]
    n_items         : int
    k               : int

    Retorna
    -------
    dict con claves: precision@k, recall@k, f1@k, map@k, coverage
    """
    precisions = [precision_at_k(r, rel, k) for r, rel in zip(all_recommended, all_relevant)]
    recalls    = [recall_at_k(r, rel, k)    for r, rel in zip(all_recommended, all_relevant)]
    f1s        = [f1_at_k(r, rel, k)        for r, rel in zip(all_recommended, all_relevant)]

    return {
        f'precision@{k}': float(np.mean(precisions)),
        f'recall@{k}'   : float(np.mean(recalls)),
        f'f1@{k}'       : float(np.mean(f1s)),
        f'map@{k}'      : mean_average_precision_at_k(all_recommended, all_relevant, k),
        'coverage'      : coverage(all_recommended, n_items),
    }
