"""
feature_engineer.py — Ingeniería de features para el sistema de recomendación.

Construye el "tag soup": una cadena de texto por película que combina
géneros, keywords, actores y director. Esta cadena es la entrada para TF-IDF.

Responsable: Estudiante A
"""
import numpy as np
import pandas as pd


def build_tag_soup(row):
    """
    Combina todos los campos de texto en una cadena de tags por película.

    Se eliminan espacios dentro de nombres para evitar que "Sam Worthington"
    y "Samuel L. Jackson" compartan el token "Sam". Ej: "samworthington".

    Parámetros
    ----------
    row : pd.Series
        Fila del DataFrame con columnas genres_list, keywords_list,
        cast_list y director.

    Retorna
    -------
    str
        Tags concatenados en minúsculas y sin espacios internos.
    """
    parts = []

    # Géneros (x2 para darles más peso en TF-IDF)
    for g in row.get('genres_list', []):
        token = g.lower().replace(' ', '')
        parts.extend([token, token])

    # Keywords
    for k in row.get('keywords_list', []):
        parts.append(k.lower().replace(' ', ''))

    # Actores (top 3)
    for a in row.get('cast_list', []):
        parts.append(a.lower().replace(' ', ''))

    # Director (x2 para darle más peso)
    director = row.get('director')
    if director and isinstance(director, str):
        token = director.lower().replace(' ', '')
        parts.extend([token, token])

    return ' '.join(parts)


def add_tag_soup(df):
    """
    Agrega la columna 'tag_soup' al DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        Debe tener columns genres_list, keywords_list, cast_list, director.

    Retorna
    -------
    pd.DataFrame
    """
    df = df.copy()
    df['tag_soup'] = df.apply(build_tag_soup, axis=1)
    return df


def normalize_numeric(df, columns):
    """
    Aplica normalización log1p a columnas numéricas sesgadas (budget, revenue).
    Devuelve el DataFrame con nuevas columnas '{col}_log'.

    Parámetros
    ----------
    df : pd.DataFrame
    columns : list[str]

    Retorna
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        df[f'{col}_log'] = np.log1p(df[col].fillna(0))
    return df
