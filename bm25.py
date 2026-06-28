import re
import math
from typing import List, Dict, Set, Union, Tuple

class BM25Retriever:
    """
    Recuperador clasico basado en el algoritmo de puntuacion Okapi BM25.
    
    Analiza la relevancia de un texto de consulta frente a un corpus entrenado,
    penalizando la redundancia de palabras hiper-frecuentes mediante IDF
    y compensando la longitud de los documentos.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Args:
            k1: Parametro de saturacion de frecuencia de termino (tipicamente entre 1.2 y 2.0).
            b: Parametro de normalizacion de longitud del documento (tipicamente 0.75).
        """
        self.k1 = k1
        self.b = b
        
        # Stop-words basicas en español e ingles para reducir ruido
        self.stopwords: Set[str] = {
            "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "si", "no",
            "de", "del", "a", "al", "en", "con", "por", "para", "como", "que", "es", "son",
            "the", "a", "an", "and", "or", "but", "if", "not", "of", "to", "in", "with", "for", "is", "are"
        }
        
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        
        # Estructuras de datos para calculo de BM25
        self.doc_lengths: Dict[Union[str, int], int] = {}
        # Frecuencia de terminos por documento: {doc_id: {term: frequency}}
        self.doc_term_freqs: Dict[Union[str, int], Dict[str, int]] = {}
        # Frecuencia de documentos por termino: {term: numero_de_documentos_que_lo_contienen}
        self.doc_freqs: Dict[str, int] = {}
        # Inverse Document Frequency de cada termino
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokeniza un string: convierte a minusculas, limpia signos de puntuacion,
        normaliza acentos, genera stems y filtra palabras vacias (stop-words).
        """
        accents = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
        text_norm = text.lower()
        for a, b in accents.items():
            text_norm = text_norm.replace(a, b)
        # Filtramos caracteres no alfanumericos usando expresiones regulares
        words = re.findall(r'\b\w+\b', text_norm)
        
        tokens = []
        for w in words:
            if w in self.stopwords:
                continue
            tokens.append(w)
            if len(w) > 4:
                tokens.append(w[:4])
        return tokens

    def fit(self, corpus: Dict[Union[str, int], str]) -> None:
        """
        Entrena el recuperador BM25 calculando las estadisticas globales sobre el corpus.
        
        Args:
            corpus: Diccionario mapeando doc_id a su contenido de texto plano.
        """
        self.corpus = corpus
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return
            
        total_len = 0
        self.doc_lengths.clear()
        self.doc_term_freqs.clear()
        self.doc_freqs.clear()
        
        # 1. Contamos frecuencias de terminos y longitudes por documento
        for doc_id, text in corpus.items():
            tokens = self._tokenize(text)
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_len += doc_len
            
            # Frecuencias locales en este documento
            freqs: Dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_term_freqs[doc_id] = freqs
            
            # Frecuencias globales de documentos que contienen cada termino
            for token in freqs.keys():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                
        self.avg_doc_len = total_len / self.corpus_size
        
        # 2. Calculamos el IDF (Inverse Document Frequency) de cada termino
        # Usamos la formula estandar de BM25 con suavizado
        for term, df in self.doc_freqs.items():
            numerator = self.corpus_size - df + 0.5
            denominator = df + 0.5
            # Evitamos IDFs negativos para terminos hiper-comunes aplicando un maximo inferior
            self.idf[term] = max(0.0001, math.log(1.0 + (numerator / denominator)))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, Union[str, int]]]:
        """
        Calcula la puntuacion BM25 de todos los documentos frente a una consulta dada.
        Retorna una lista ordenada de tuplas (score, doc_id).
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []
            
        scores: List[Tuple[float, Union[str, int]]] = []
        
        for doc_id, doc_len in self.doc_lengths.items():
            score = 0.0
            term_freqs = self.doc_term_freqs[doc_id]
            
            for token in query_tokens:
                if token in term_freqs:
                    tf = term_freqs[token]
                    # Aplicamos formula de Okapi BM25
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += self.idf.get(token, 0.0) * ((tf * (self.k1 + 1.0)) / denom)
                    
            if score > 0.0:
                # Penalizar paginas de bibliografia, indices o anexos (sin importar acentos)
                doc_text = self.corpus.get(doc_id, "")
                doc_lower = doc_text.lower()
                accents = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
                doc_norm = doc_lower
                for a, b in accents.items():
                    doc_norm = doc_norm.replace(a, b)
                    
                penalty_keywords = ["bibliografia", "webgrafia", "indice de", "anexo", "referencias"]
                if any(kw in doc_norm[:120] for kw in penalty_keywords):
                    score *= 0.1
                scores.append((score, doc_id))
                
        # Ordenamos descendente
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]
