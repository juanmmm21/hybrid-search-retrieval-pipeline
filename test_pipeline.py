import unittest
import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Union

# Añadimos rutas locales para cargar tanto el modulo actual como el de nano-vector-db (Interlinking)
sys.path.append(os.path.abspath("../nano-vector-db"))

from bm25 import BM25Retriever
from fusion import reciprocal_rank_fusion, score_normalization_fusion

class TestHybridSearchPipeline(unittest.TestCase):
    """
    Suite de pruebas unitarias para validar el BM25 y los algoritmos
    de fusion de rankings en el pipeline de busqueda hibrida.
    """
    
    def test_bm25_retriever(self) -> None:
        """
        Verifica el analisis de frecuencia y calculo de relevancia de BM25.
        """
        corpus = {
            1: "el gato negro come pescado y leche",
            2: "el perro corre detras del gato en el parque",
            3: "la receta para cocinar pescado fresco al horno"
        }
        retriever = BM25Retriever(k1=1.2, b=0.75)
        retriever.fit(corpus)
        
        self.assertEqual(retriever.corpus_size, 3)
        self.assertEqual(retriever.doc_freqs.get("gato", 0), 2)
        self.assertEqual(retriever.doc_freqs.get("pescado", 0), 2)
        self.assertEqual(retriever.doc_freqs.get("perro", 0), 1)
        
        # Realizamos consulta por palabra clave
        res = retriever.retrieve("gato", top_k=2)
        self.assertEqual(len(res), 2)
        # Los documentos que contienen la palabra son 1 y 2
        doc_ids = [r[1] for r in res]
        self.assertIn(1, doc_ids)
        self.assertIn(2, doc_ids)

    def test_reciprocal_rank_fusion(self) -> None:
        """
        Verifica el calculo del algoritmo Reciprocal Rank Fusion (RRF).
        """
        # Dense rank: A (1), B (2), C (3)
        dense = [(0.1, "A"), (0.2, "B"), (0.3, "C")]
        # Sparse rank: B (1), D (2), A (3)
        sparse = [(10.0, "B"), (5.0, "D"), (1.0, "A")]
        
        # Calculos manuales con k=60:
        # A: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
        # B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # El elemento B debe quedar por encima de A
        res = reciprocal_rank_fusion(dense, sparse, k=60)
        self.assertEqual(res[0][1], "B")
        self.assertEqual(res[1][1], "A")
        self.assertTrue(res[0][0] > res[1][0])

    def test_score_normalization_fusion(self) -> None:
        """
        Verifica la fusion lineal normalizada Min-Max.
        """
        # Dense (distance, doc_id) -> distancias
        # A=0.1, B=0.9 -> sim(A)=0.9, sim(B)=0.1 -> norm: A=1.0, B=0.0
        dense = [(0.1, "A"), (0.9, "B")]
        # Sparse (score, doc_id) -> scores
        # A=1.0, B=10.0 -> norm: A=0.0, B=1.0
        sparse = [(1.0, "A"), (10.0, "B")]
        
        # Con alpha=0.5, el score combinado debe ser igual para ambos (0.5 * 1.0 + 0.5 * 0.0 = 0.5)
        res = score_normalization_fusion(dense, sparse, alpha=0.5, metric="cosine")
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0][0], 0.5)
        self.assertAlmostEqual(res[1][0], 0.5)

    def test_pipeline_integration(self) -> None:
        """
        Verifica la orquestacion completa del pipeline de busqueda hibrida.
        """
        from database import NanoVectorDB
        from pipeline import HybridSearchPipeline
        
        # 1. Poblamos la base de datos vectorial
        db = NanoVectorDB(dimension=3, index_type="flat", metric="cosine")
        db.insert(id="doc1", vector=[1.0, 0.0, 0.0], metadata={"category": "nature"})
        db.insert(id="doc2", vector=[0.0, 1.0, 0.0], metadata={"category": "tech"})
        
        # 2. Inicializamos el pipeline y entrenamos BM25
        pipeline = HybridSearchPipeline(vector_db=db)
        corpus = {
            "doc1": "el bosque y los arboles de la selva tropical",
            "doc2": "programacion de software y servidores en la nube"
        }
        pipeline.fit_sparse(corpus)
        
        # 3. Busqueda hibrida: query "selva" + vector apuntando a nature
        res = pipeline.search(
            query="selva",
            query_vector=[1.0, 0.0, 0.0],
            top_k=2,
            fusion_method="rrf"
        )
        
        self.assertEqual(len(res), 2)
        # El primero debe ser doc1 (mayor coincidencia semantica y lexica)
        self.assertEqual(res[0]["id"], "doc1")
        self.assertEqual(res[0]["metadata"]["category"], "nature")
        self.assertTrue(res[0]["sparse_score"] > 0.0)
        self.assertIsNotNone(res[0]["dense_distance"])

if __name__ == "__main__":
    unittest.main()
