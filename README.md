# hybrid-search-retrieval-pipeline

Pipeline de recuperacion hibrida que combina busqueda lexica clásica (Okapi BM25) y busqueda vectorial semantica (NanoVectorDB) utilizando fusiones avanzadas de rankings (RRF y Score Normalization).

Este modulo es clave en arquitecturas de produccion de Generacion Aumentada por Recuperacion (RAG), garantizando que las consultas recuperen tanto coincidencias exactas por palabras clave (ej. acronimos, codigos, IDs de error) como conceptos semanticamente equivalentes o sinónimos.

## Arquitectura y Fundamentos Teoricos

El pipeline se compone de tres piezas fundamentales:

### 1. Búsqueda Dispersa (Okapi BM25)
BM25 es el algoritmo del estado del arte para busquedas basadas en terminos. Para una consulta $Q$ que contiene los terminos $q_1, q_2, ..., q_n$ y un documento $D$, la puntuacion es:

$$\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Donde:
*   $f(q_i, D)$ es la frecuencia del termino en el documento.
*   $|D|$ y $\text{avgdl}$ representan la longitud del documento y la longitud promedio del corpus.
*   $k_1$ (saturacion de termino, por defecto `1.5`) y $b$ (normalizacion de longitud, por defecto `0.75`) son hiperparametros ajustables.
*   El $\text{IDF}$ se calcula con suavizado para prevenir valores negativos en palabras hiper-frecuentes:

$$\text{IDF}(q_i) = \ln\left(1.0 + \frac{N - df(q_i) + 0.5}{df(q_i) + 0.5}\right)$$

### 2. Reciprocal Rank Fusion (RRF)
RRF combina rankings ordenados basándose en la posicion del documento en cada lista de resultados en lugar de comparar sus scores brutos. Esto es ideal para integrar sistemas con escalas de puntuacion dispares:

$$\text{RRF\_score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Donde $r_m(d)$ es el rango del documento $d$ en la lista del recuperador $m$ (1-indexed), y $k$ es una constante de suavizado (por defecto `60`) que mitiga el impacto de posiciones bajas.

### 3. Fusión Lineal Ponderada (Score Fusion)
Normaliza las puntuaciones mediante escala Min-Max:

$$\text{Score}_{\text{norm}} = \frac{\text{Score} - \text{Score}_{\text{min}}}{\text{Score}_{\text{max}} - \text{Score}_{\text{min}}}$$

Las distancias vectoriales se convierten previamente a similitudes relativas:
*   Similitud de Coseno: $1.0 - \text{distancia}$
*   Similitud Euclidea L2: $1.0 / (1.0 + \text{distancia})$

El score final es la combinacion lineal:

$$\text{Score}_{\text{final}} = \alpha \cdot \text{Score}_{\text{norm\_dense}} + (1.0 - \alpha) \cdot \text{Score}_{\text{norm\_sparse}}$$

donde $\alpha$ (rango $[0, 1]$) regula el peso asignado a la rama vectorial densa.

## Conexión con el Ecosistema

Este modulo coordina de forma interconectada los siguientes componentes:
*   **nano-vector-db:** Actua como almacen semantico de entrada. El pipeline delega en esta base de datos para recuperar candidatos vectoriales densos aplicando filtros de metadatos.
*   **contrastive-embedding-trainer:** Si existen pesos en este modulo vecino, el pipeline los carga mediante PyTorch/Transformers para codificar la query del usuario al vuelo. En caso de ausencia, realiza una simulacion de hash de palabras determinista.

## Estructura de Archivos

*   **bm25.py:** Clase `BM25Retriever` con tokenizacion y formulas de Okapi BM25.
*   **fusion.py:** Modulo que implementa los algoritmos `reciprocal_rank_fusion` y `score_normalization_fusion`.
*   **pipeline.py:** Clase `HybridSearchPipeline` unificando y coordinando las busquedas.
*   **test_pipeline.py:** Pruebas unitarias de las formulas e integraciones del sistema.
*   **example.py:** Demostracion interactiva de busquedas cruzadas Sparse vs Dense vs RRF vs Score Fusion.

## Instalacion y Uso

### 1. Activar Entorno Virtual e Instalar Dependencias
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar Pruebas Unitarias
```bash
python -m unittest test_pipeline.py
```

### 3. Ejecutar Demostración
```bash
python example.py
```
El script poblará el grafo HNSW vectorial de [NanoVectorDB](file:///Users/golfeno/Desarrollo/ai-core-infra/nano-vector-db), indexara el corpus lexicamente en BM25, ejecutara consultas y contrastara las diferencias de recuperacion entre tecnicas lexicas, semanticas e hibridas.
