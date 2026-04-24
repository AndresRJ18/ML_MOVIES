"""
json_parser.py — Parseo de columnas JSON del dataset TMDB.

Las columnas genres, keywords, cast y crew del dataset TMDB contienen
strings con formato de diccionarios Python (comillas simples), NO JSON válido.
Por eso usamos ast.literal_eval() en lugar de json.loads().

Responsable: Estudiante A
"""
import ast


def parse_names(value, max_items=None):
    """
    Convierte una columna JSON-like en lista de nombres.

    Parámetros
    ----------
    value : str
        String tipo "[{'id': 18, 'name': 'Drama'}, ...]"
    max_items : int o None
        Si se especifica, retorna solo los primeros N nombres.

    Retorna
    -------
    list[str]
        Lista de nombres extraídos.
    """
    try:
        items = ast.literal_eval(value)
        names = [item['name'] for item in items if 'name' in item]
        if max_items is not None:
            names = names[:max_items]
        return names
    except (ValueError, SyntaxError):
        return []


def parse_director(crew_value):
    """
    Extrae el nombre del director desde la columna 'crew'.

    El director es la primera entrada con job == 'Director'.

    Parámetros
    ----------
    crew_value : str
        String de la columna crew.

    Retorna
    -------
    str o None
    """
    try:
        crew = ast.literal_eval(crew_value)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
    except (ValueError, SyntaxError):
        pass
    return None


def extract_all_features(df):
    """
    Aplica el parseo de columnas JSON al DataFrame completo.

    Espera columnas: 'genres', 'keywords' (de movies),
    'cast', 'crew' (de credits tras el join).

    Parámetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame con columnas nuevas:
        genres_list, keywords_list, cast_list, director
    """
    df = df.copy()
    df['genres_list']   = df['genres'].apply(lambda x: parse_names(x))
    df['keywords_list'] = df['keywords'].apply(lambda x: parse_names(x))
    df['cast_list']     = df['cast'].apply(lambda x: parse_names(x, max_items=3))
    df['director']      = df['crew'].apply(parse_director)
    return df
