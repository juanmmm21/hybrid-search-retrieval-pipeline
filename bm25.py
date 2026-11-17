"""Compatibilidad: el código vive en el paquete `hybrid_search_retrieval_pipeline`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `bm25`
directamente."""

from hybrid_search_retrieval_pipeline.bm25 import *  # noqa: F401,F403
from hybrid_search_retrieval_pipeline.bm25 import BM25Retriever  # noqa: F401
