import os
import sys
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np

# Añadimos dinámicamente el directorio del nano-vector-db al path del sistema (Interlinking)
sys.path.append(os.path.abspath("../nano-vector-db"))

try:
    from database import NanoVectorDB
except ImportError:
    # Definimos un placeholder para tipado estricto si no compila de forma preliminar
    class NanoVectorDB:
        pass

from bm25 import BM25Retriever
from fusion import reciprocal_rank_fusion, score_normalization_fusion

logger = logging.getLogger(__name__)

# Intentamos habilitar codificación local para queries de texto plano mediante el contrastive model
MODEL_PATH = "../contrastive-embedding-trainer/model_output"
EMBEDDING_MODEL_AVAILABLE = False
tokenizer = None
model = None
torch = None

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH)
        model.eval()
        EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    pass


class HybridSearchPipeline:
    """
    Orquestador del pipeline de busqueda hibrida (Hybrid Search).
    
    Combina las puntuaciones de busqueda lexica clasica (BM25) con la busqueda
    vectorial semantica (NanoVectorDB) aplicando metodos de fusion de rankings.
    """
    
    def __init__(
        self,
        vector_db: NanoVectorDB,
        k1: float = 1.5,
        b: float = 0.75
    ) -> None:
        """
        Args:
            vector_db: Instancia iniciada y poblada de NanoVectorDB.
            k1: Parametro k1 de saturacion en BM25.
            b: Parametro b de normalizacion de longitud en BM25.
        """
        self.vector_db = vector_db
        self.bm25 = BM25Retriever(k1=k1, b=b)
        
    def fit_sparse(self, corpus: Dict[Union[str, int], str]) -> None:
        """
        Indexa y entrena el corpus clasico para la busqueda BM25.
        """
        self.bm25.fit(corpus)
        logger.info(f"Corpus de busqueda lexica BM25 entrenado exitosamente con {len(corpus)} documentos.")
        
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Obtiene el embedding normalizado de la query. Usa el modelo contrastivo local si esta.
        Si no, realiza una simulacion vectorial determinista mediante hash de palabras.
        """
        if EMBEDDING_MODEL_AVAILABLE and tokenizer is not None and model is not None and torch is not None:
            with torch.no_grad():
                inputs = tokenizer(
                    query,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                )
                outputs = model(**inputs)
                
                # Mean Pooling + Normalizacion L2
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                embedding = sum_embeddings / sum_mask
                normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
                return normalized.squeeze(0).cpu().numpy()
        else:
            # Simulacion vectorial determinista
            state = np.random.RandomState(abs(hash(query)) % (2**32))
            vec = state.normal(0.0, 1.0, self.vector_db.dimension)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-9 else vec

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
