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


def score_normalization_fusion(
    dense_results: List[Tuple[float, Union[str, int]]],
    sparse_results: List[Tuple[float, Union[str, int]]],
    alpha: float = 0.5,
    metric: str = "cosine"
) -> List[Tuple[float, Union[str, int]]]:
    """
    Combina resultados densos y dispersos normalizando linealmente sus puntuaciones.
    
    Aplica una normalizacion Min-Max a los scores de BM25 y a las distancias vectoriales
    (tras convertirlas a similitudes relativas). Posteriormente calcula:
    Score = alpha * Sim_densa + (1.0 - alpha) * Sim_dispersa
    """
    if not dense_results and not sparse_results:
        return []
    if not dense_results:
        # Si no hay resultados densos, normalizamos solo los dispersos y los devolvemos
        return _normalize_single_ranking(sparse_results)
    if not sparse_results:
        # Si no hay resultados dispersos, normalizamos solo los densos y los devolvemos
        return _normalize_single_ranking(dense_results, is_distance=True, metric=metric)

    # 1. Convertimos distancias vectoriales a similitudes
    dense_sims: Dict[Union[str, int], float] = {}
    for dist, doc_id in dense_results:
        if metric == "cosine":
            dense_sims[doc_id] = 1.0 - dist
        elif metric == "l2":
            dense_sims[doc_id] = 1.0 / (1.0 + dist)
        else:  # dot_product
            # En dot_product_distance guardamos -dot_product, por ende sim = -dist
            dense_sims[doc_id] = -dist

    # 2. Min-Max de similitudes densas
    dense_vals = list(dense_sims.values())
    min_d, max_d = min(dense_vals), max(dense_vals)
    range_d = max_d - min_d
    
    dense_norm: Dict[Union[str, int], float] = {}
    for doc_id, sim in dense_sims.items():
        dense_norm[doc_id] = (sim - min_d) / range_d if range_d > 1e-9 else 1.0

    # 3. Min-Max de puntuaciones dispersas (BM25)
    sparse_scores = {doc_id: score for score, doc_id in sparse_results}
    sparse_vals = list(sparse_scores.values())
    min_s, max_s = min(sparse_vals), max(sparse_vals)
    range_s = max_s - min_s
    
    sparse_norm: Dict[Union[str, int], float] = {}
    for doc_id, score in sparse_scores.items():
        sparse_norm[doc_id] = (score - min_s) / range_s if range_s > 1e-9 else 1.0

    # 4. Fusion lineal ponderada
    all_keys = set(dense_norm.keys()).union(sparse_norm.keys())
    combined_scores: List[Tuple[float, Union[str, int]]] = []
    
    for doc_id in all_keys:
        d_score = dense_norm.get(doc_id, 0.0)
        s_score = sparse_norm.get(doc_id, 0.0)
        final_score = alpha * d_score + (1.0 - alpha) * s_score
        combined_scores.append((final_score, doc_id))

    # Ordenamos de mayor a menor similitud combinada resultante
    combined_scores.sort(key=lambda x: x[0], reverse=True)
    return combined_scores


def _normalize_single_ranking(
    results: List[Tuple[float, Union[str, int]]],
    is_distance: bool = False,
    metric: str = "cosine"
) -> List[Tuple[float, Union[str, int]]]:
    """
    Helper para normalizar una sola lista de resultados al rango [0, 1].
    """
    if not results:
        return []
        
    scores_dict: Dict[Union[str, int], float] = {}
    for score, doc_id in results:
        if is_distance:
            if metric == "cosine":
                scores_dict[doc_id] = 1.0 - score
            elif metric == "l2":
                scores_dict[doc_id] = 1.0 / (1.0 + score)
            else:
                scores_dict[doc_id] = -score
        else:
            scores_dict[doc_id] = score
            
    vals = list(scores_dict.values())
    min_v, max_v = min(vals), max(vals)
    r_v = max_v - min_v
    
    norm_results = []
    for doc_id, val in scores_dict.items():
        norm_val = (val - min_v) / r_v if r_v > 1e-9 else 1.0
        norm_results.append((norm_val, doc_id))
        
    norm_results.sort(key=lambda x: x[0], reverse=True)
    return norm_results
