import os
import sys
import logging
import numpy as np
from typing import List, Dict, Any

# Añadimos rutas locales para cargar el modulo de nano-vector-db (Interlinking)
sys.path.append(os.path.abspath("../nano-vector-db"))
from database import NanoVectorDB
from pipeline import HybridSearchPipeline

# Habilitamos logs basicos
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Intentamos importar Deep Learning para la generacion de embeddings semanticos reales
SEMAN_SEARCH_ENABLED = False
tokenizer = None
model = None
torch = None

MODEL_PATH = "../contrastive-embedding-trainer/model_output"

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH)
        model.eval()
        SEMAN_SEARCH_ENABLED = True
except ImportError:
    pass


def get_embedding(text: str) -> np.ndarray:
    """
    Genera el embedding L2 normalizado de una frase.
    Usa el modelo local si esta disponible, o cae a simulacion vectorial determinista.
    """
    if SEMAN_SEARCH_ENABLED and tokenizer is not None and model is not None and torch is not None:
        with torch.no_grad():
            inputs = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt")
            outputs = model(**inputs)
            
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            embedding = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return normalized.squeeze(0).cpu().numpy()
    else:
        # Generador de embeddings de simulacion determinista basado en hash
        state = np.random.RandomState(abs(hash(text)) % (2**32))
        vec = state.normal(0.0, 1.0, 768)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec


def main() -> None:
    print("==================================================")
    print("    Demostracion de Hybrid Search Retrieval      ")
    print("==================================================")
    
    # 1. Definicion y poblacion de la base de datos vectorial
    dimension = 768
    metric = "cosine"
    
    print("\n1. Inicializando base de datos de vectores (HNSW)...")
    db = NanoVectorDB(
        dimension=dimension,
        metric=metric,
        index_type="hnsw",
        M=16,
        efConstruction=64,
        efSearch=32
    )
    
    # Corpus de oraciones de prueba
    documents = {
        1: "La astronomia nos permite estudiar las estrellas y galaxias distantes.",
        2: "El cosmos es un lugar misterioso y en constante expansion acelerada.",
        3: "La exploracion espacial por satelites aporta datos clave del sistema solar.",
        4: "Aprender a programar en Python abre muchas puertas en desarrollo web y ciencia de datos.",
        5: "La optimizacion de consultas SQL mejora de forma critica el rendimiento de las aplicaciones.",
        6: "El codigo limpio y refactorizado reduce la deuda tecnica de los proyectos de software.",
        7: "Una receta tradicional de paella requiere ingredientes frescos de mar o tierra.",
        8: "El pan de masa madre se fermenta de forma natural y requiere paciencia.",
        9: "Para freir patatas perfectas se recomienda usar aceite de oliva a fuego medio."
    }
    
    # Metadatos para agregar a cada documento
    categories = {
        1: "astronomia", 2: "astronomia", 3: "astronomia",
        4: "programacion", 5: "programacion", 6: "programacion",
        7: "cocina", 8: "cocina", 9: "cocina"
    }
    
    print("Vectorizando e insertando documentos en HNSW...")
    for doc_id, text in documents.items():
        vec = get_embedding(text)
        metadata = {"category": categories[doc_id], "text": text}
        db.insert(id=doc_id, vector=vec, metadata=metadata)
        
    # 2. Inicializacion del pipeline hibrido
    print("\n2. Inicializando el HybridSearchPipeline...")
    pipeline = HybridSearchPipeline(vector_db=db, k1=1.5, b=0.75)
    # Alimentamos el motor de busqueda lexica BM25 con el corpus de textos
    pipeline.fit_sparse(documents)
    
    # 3. Realizacion de consultas de prueba comparativas
    queries = [
        "receta de paella tradicional",
        "programacion limpia en Python",
        "estudiar estrellas y cosmos profundo"
    ]
    
    print("\n" + "="*50)
    print(" Ejecutando busquedas comparativas (Sparse, Dense, Hibrida) ")
    print("="*50)
    
    for q in queries:
        print(f"\nConsulta: '{q}'")
        q_vec = get_embedding(q)
        
        # A. Solo Dispersa (Okapi BM25)
        print("\n  [A] Solo Sparse (BM25):")
        sparse_res = pipeline.bm25.retrieve(q, top_k=2)
        for score, doc_id in sparse_res:
            print(f"    - ID: {doc_id} | Score: {score:.4f} | '{documents[doc_id]}'")
            
        # B. Solo Densa (Vector HNSW)
        print("\n  [B] Solo Dense (Vectores HNSW):")
        dense_res = db.query(q_vec, top_k=2)
        for r in dense_res:
            print(f"    - ID: {r['id']} | Distancia: {r['distance']:.4f} | '{documents[r['id']]}'")
            
        # C. Hibrida (Fusion RRF)
        print("\n  [C] Hibrido (Reciprocal Rank Fusion - RRF):")
        hybrid_rrf = pipeline.search(query=q, query_vector=q_vec, top_k=2, fusion_method="rrf")
        for r in hybrid_rrf:
            dense_dist = r['dense_distance'] if r['dense_distance'] is not None else 0.0
            print(f"    - ID: {r['id']} | Score RRF: {r['score']:.4f} | Sparse: {r['sparse_score']:.2f} | Dense Dist: {dense_dist:.4f} | '{documents[r['id']]}'")
            
        # D. Hibrida (Score Normalization Fusion - Ponderada)
        alpha = 0.6  # Asignamos 60% peso a dense y 40% a sparse
        print(f"\n  [D] Hibrido (Score Fusion - pesos alpha={alpha}):")
        hybrid_score = pipeline.search(query=q, query_vector=q_vec, top_k=2, fusion_method="score", alpha=alpha)
        for r in hybrid_score:
            dense_dist = r['dense_distance'] if r['dense_distance'] is not None else 0.0
            print(f"    - ID: {r['id']} | Score Normalizado: {r['score']:.4f} | Sparse: {r['sparse_score']:.2f} | Dense Dist: {dense_dist:.4f} | '{documents[r['id']]}'")
            
        print("-"*50)
        
    print("\nDemostracion de busqueda hibrida finalizada exitosamente.")


if __name__ == "__main__":
    main()
