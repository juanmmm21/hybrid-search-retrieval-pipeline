"""Compatibilidad: el código vive en el paquete `hybrid_search_retrieval_pipeline`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `fusion`
directamente."""

from hybrid_search_retrieval_pipeline.fusion import *  # noqa: F401,F403
from hybrid_search_retrieval_pipeline.fusion import reciprocal_rank_fusion, score_normalization_fusion  # noqa: F401
