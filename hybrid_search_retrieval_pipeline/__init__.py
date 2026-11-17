"""Hybrid retrieval: BM25 plus any dense vector store, fused with RRF or score normalization."""

from .bm25 import BM25Retriever
from .fusion import reciprocal_rank_fusion, score_normalization_fusion
from .pipeline import HybridSearchPipeline, VectorStore, hash_embedding

__version__ = "0.1.0"

__all__ = ['BM25Retriever', 'reciprocal_rank_fusion', 'score_normalization_fusion', 'HybridSearchPipeline', 'VectorStore', 'hash_embedding']
