"""Compatibilidad: el código vive en el paquete `hybrid_search_retrieval_pipeline`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `pipeline`
directamente."""

from hybrid_search_retrieval_pipeline.pipeline import *  # noqa: F401,F403
from hybrid_search_retrieval_pipeline.pipeline import HybridSearchPipeline, VectorStore, hash_embedding  # noqa: F401
