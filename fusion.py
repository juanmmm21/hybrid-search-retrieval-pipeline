from typing import List, Dict, Tuple, Union

def reciprocal_rank_fusion(
    dense_results: List[Tuple[float, Union[str, int]]],
    sparse_results: List[Tuple[float, Union[str, int]]],
    k: int = 60
) -> List[Tuple[float, Union[str, int]]]:
    """
    Combina dos rankings ordenados usando el algoritmo Reciprocal Rank Fusion (RRF).
    
    Las listas de entrada deben estar pre-ordenadas (el mas relevante primero).
    RRF suma las posiciones reciprocas en ambos rankings, independientemente de la escala
    de las puntuaciones de origen, utilizando un factor de suavizado k (por defecto 60).
    """
    rrf_scores: Dict[Union[str, int], float] = {}
    
    # Procesamos ranking denso (vectores)
    for rank, (_, doc_id) in enumerate(dense_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    # Procesamos ranking disperso (BM25)
    for rank, (_, doc_id) in enumerate(sparse_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    # Convertimos a lista de tuplas y ordenamos de mayor a menor puntuacion RRF
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Devolvemos tuplas ordenadas (rrf_score, doc_id)
    return [(score, doc_id) for doc_id, score in sorted_rrf]
