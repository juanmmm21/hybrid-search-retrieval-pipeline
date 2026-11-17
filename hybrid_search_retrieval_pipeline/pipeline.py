import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

import numpy as np

from .bm25 import BM25Retriever
from .fusion import reciprocal_rank_fusion, score_normalization_fusion

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Lo que el pipeline necesita de la rama densa.

    Cualquier base vectorial que lo cumpla sirve (``nano_vector_db.NanoVectorDB`` lo cumple tal
    cual). El pipeline no importa ninguna: quien lo monta (p. ej. el SDK ``aicore`` del hub) le
    pasa la instancia, de modo que los componentes del ecosistema no se importan entre sí.
    """

    dimension: int
    metric: str
    metadata: Dict[Any, Dict[str, Any]]
    vectors: Dict[Any, Any]

    def query(
        self, vector: Any, top_k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...


EmbedFn = Callable[[str], Union[List[float], np.ndarray]]


def hash_embedding(text: str, dimension: int) -> np.ndarray:
    """Vector pseudoaleatorio estable entre procesos (md5, no ``hash()``, que cambia con
    ``PYTHONHASHSEED``). Solo para pruebas y demos: no captura ningún significado."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    vec = np.random.RandomState(seed).normal(0.0, 1.0, dimension)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


class HybridSearchPipeline:
    """
    Orquestador del pipeline de busqueda hibrida (Hybrid Search).
    
    Combina las puntuaciones de busqueda lexica clasica (BM25) con la busqueda
    vectorial semantica (NanoVectorDB) aplicando metodos de fusion de rankings.
    """
    
    def __init__(
        self,
        vector_db: VectorStore,
        k1: float = 1.5,
        b: float = 0.75,
        embed_fn: Optional[EmbedFn] = None,
    ) -> None:
        """
        Args:
            vector_db: Base vectorial ya poblada que cumpla ``VectorStore``.
            k1: Parametro k1 de saturacion en BM25.
            b: Parametro b de normalizacion de longitud en BM25.
            embed_fn: Codificador de la consulta (texto -> vector). Sin el, se usa
                ``hash_embedding``, que solo sirve para pruebas.
        """
        self.vector_db = vector_db
        self.embed_fn = embed_fn
        self.bm25 = BM25Retriever(k1=k1, b=b)
        
    def fit_sparse(self, corpus: Dict[Union[str, int], str]) -> None:
        """
        Indexa y entrena el corpus clasico para la busqueda BM25.
        """
        self.bm25.fit(corpus)
        logger.info(f"Corpus de busqueda lexica BM25 entrenado exitosamente con {len(corpus)} documentos.")
        
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Embedding normalizado de la consulta con ``embed_fn``; sin el, ``hash_embedding``."""
        if self.embed_fn is not None:
            vec = np.asarray(self.embed_fn(query), dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-9 else vec
        return hash_embedding(query, self.vector_db.dimension)

    def search(
        self,
        query: str,
        query_vector: Optional[Union[List[float], np.ndarray]] = None,
        top_k: int = 5,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Realiza una busqueda hibrida unificando Dense y Sparse.
        
        Args:
            query: La consulta en texto plano.
            query_vector: El vector correspondiente a la consulta (opcional). Si es None, se genera localmente.
            top_k: Numero de resultados finales a retornar.
            fusion_method: Metodo de fusion ('rrf' o 'score').
            alpha: Peso para Score Fusion (peso asignado a la rama densa).
            rrf_k: Constante de suavizado para el algoritmo RRF.
            filter: Filtro de metadatos opcional para aplicar a la rama vectorial.
            
        Returns:
            Lista de resultados ordenados por relevancia hibrida final.
        """
        # 1. Ejecucion de la busqueda Dispersa (BM25)
        # Consultamos un colchon mayor de elementos para evitar que el RRF pierda candidatos de corte
        sparse_res_raw = self.bm25.retrieve(query, top_k=top_k * 3)
        
        # 2. Ejecucion de la busqueda Densa (NanoVectorDB)
        if query_vector is None:
            query_vector = self._get_query_embedding(query)
        else:
            query_vector = np.asarray(query_vector, dtype=np.float32)
            
        dense_res_raw = self.vector_db.query(query_vector, top_k=top_k * 3, filter=filter)
        dense_res_tuples = [(r["distance"], r["id"]) for r in dense_res_raw]
        
        # 3. Fusion de rankings
        fusion_method_lower = fusion_method.lower()
        if fusion_method_lower == "rrf":
            fused_results = reciprocal_rank_fusion(
                dense_results=dense_res_tuples,
                sparse_results=sparse_res_raw,
                k=rrf_k
            )
        elif fusion_method_lower == "score":
            fused_results = score_normalization_fusion(
                dense_results=dense_res_tuples,
                sparse_results=sparse_res_raw,
                alpha=alpha,
                metric=self.vector_db.metric
            )
        else:
            raise ValueError(f"Metodo de fusion '{fusion_method}' invalido. Usar 'rrf' o 'score'.")
            
        # 4. Formatear y construir la respuesta final enriquecida
        sparse_lookup = {doc_id: score for score, doc_id in sparse_res_raw}
        dense_lookup = {r["id"]: r["distance"] for r in dense_res_raw}
        
        final_results = []
        for rank_score, doc_id in fused_results[:top_k]:
            final_results.append({
                "id": doc_id,
                "score": rank_score,
                "metadata": self.vector_db.metadata.get(doc_id, {}),
                "sparse_score": sparse_lookup.get(doc_id, 0.0),
                "dense_distance": dense_lookup.get(doc_id, None),
                "vector": self.vector_db.vectors[doc_id].tolist()
            })
            
        return final_results
