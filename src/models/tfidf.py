"""
tfidf.py — Vectorizador TF-IDF implementado desde cero con NumPy.

TF-IDF (Term Frequency - Inverse Document Frequency) representa cada
documento como un vector numérico donde el peso de cada término refleja
su importancia relativa dentro del documento y en el corpus.

Fórmulas:
    TF(t, d)  = count(t en d) / total_tokens(d)
    IDF(t)    = log((1 + N) / (1 + df(t))) + 1   [suavizado]
    TF-IDF    = TF * IDF
    Vector normalizado: v / ||v||_2

Responsable: Estudiante A
"""
import numpy as np


class TFIDFVectorizer:
    """
    Transforma una lista de documentos en una matriz TF-IDF.

    Uso:
        vectorizer = TFIDFVectorizer()
        matrix = vectorizer.fit_transform(corpus)      # shape: (n_docs, vocab)
        vec    = vectorizer.transform(["nuevo texto"])
    """

    def __init__(self):
        self.vocabulary_ = {}    # token → índice de columna
        self.idf_        = None  # vector IDF, shape (vocab_size,)

    def _tokenize(self, text):
        """Divide el texto en tokens (ya están en minúsculas y sin espacios)."""
        return text.lower().split()

    def fit(self, corpus):
        """
        Construye el vocabulario y calcula IDF sobre el corpus.

        Parámetros
        ----------
        corpus : list[str]
            Lista de documentos (tag soups).
        """
        N = len(corpus)
        document_frequency = {}  # token → nº de documentos que lo contienen

        # Construir vocabulario y contar document frequency
        for doc in corpus:
            tokens = set(self._tokenize(doc))  # set: cada token cuenta una vez por doc
            for token in tokens:
                if token not in self.vocabulary_:
                    self.vocabulary_[token] = len(self.vocabulary_)
                document_frequency[token] = document_frequency.get(token, 0) + 1

        # Calcular IDF con suavizado para evitar división por cero
        vocab_size = len(self.vocabulary_)
        self.idf_ = np.zeros(vocab_size)
        for token, idx in self.vocabulary_.items():
            df = document_frequency.get(token, 0)
            self.idf_[idx] = np.log((1 + N) / (1 + df)) + 1

        return self

    def transform(self, corpus):
        """
        Transforma documentos en matriz TF-IDF normalizada.

        Parámetros
        ----------
        corpus : list[str]

        Retorna
        -------
        np.ndarray, shape (n_docs, vocab_size)
        """
        n_docs = len(corpus)
        vocab_size = len(self.vocabulary_)
        matrix = np.zeros((n_docs, vocab_size))

        for i, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            if not tokens:
                continue
            # TF: frecuencia relativa de cada token en el documento
            for token in tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    matrix[i, idx] += 1
            matrix[i] /= len(tokens)  # normalizar por longitud del doc

        # Multiplicar TF por IDF
        matrix = matrix * self.idf_

        # Normalización L2 por fila (cada vector queda con norma = 1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # evitar división por cero
        matrix = matrix / norms

        return matrix

    def fit_transform(self, corpus):
        """Ajusta el vectorizador y transforma el corpus en un solo paso."""
        return self.fit(corpus).transform(corpus)
