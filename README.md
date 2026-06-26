# hybrid-search-retrieval-pipeline

Pipeline de recuperacion hibrida que combina busqueda lexica clasica (Okapi BM25) y busqueda vectorial semantica (NanoVectorDB) utilizando algoritmos avanzados de fusion de rankings (RRF y Score Normalization).

Este modulo es una pieza central en arquitecturas de produccion de Generacion Aumentada por Recuperacion (RAG), resolviendo la limitacion de buscar unicamente por similitud semantica (que suele ignorar codigos de error, acronimos, numeros de serie o IDs de producto) o buscar unicamente por palabras clave (que carece de comprension contextual y sinonimia).

## Arquitectura y Fundamentos Teoricos

El pipeline implementa una estrategia de busqueda paralela de dos vias (Dense + Sparse) y consolida sus resultados mediante tecnicas de alineacion de rankings.

```mermaid
graph TD
    Query[Consulta del Usuario] --> Tokenizer[Tokenizador BM25]
    Query --> Embedder[Codificador de Embeddings]
    
    subgraph Rama Dispersa (Sparse)
        Tokenizer --> BM25Engine[BM25 Retriever]
        BM25Engine --> SparseScores[Rankings Lexicos]
    end
    
    subgraph Rama Densa (Dense)
        Embedder --> LocalModel[Modelo Contrastivo Local]
        Embedder -. Fallback .-> HashSim[Hash Vectorizer determinista]
        LocalModel --> VectorQuery[Query Vector]
        HashSim --> VectorQuery
        VectorQuery --> NanoVectorDB[HNSW NanoVectorDB]
        NanoVectorDB --> DenseScores[Rankings Vectoriales]
    end
    
    SparseScores --> FusionManager[Gestor de Fusion]
    DenseScores --> FusionManager
    
    FusionManager --> RRFAlg[Reciprocal Rank Fusion]
    FusionManager --> ScoreNormAlg[Score Normalization & Fusion]
    
    RRFAlg --> FinalRank[Top-K Documentos Recuperados]
    ScoreNormAlg --> FinalRank
```

### 1. Busqueda Dispersa (Okapi BM25)

El modulo `BM25Retriever` implementa el algoritmo Okapi BM25 no ponderado sobre un corpus estatico indexado localmente. La puntuacion de un documento $D$ respecto a una consulta $Q$ compuesta por terminos $q_1, q_2, \dots, q_n$ se define formalmente como:

$$\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Donde:
*   $f(q_i, D)$ representa la frecuencia absoluta del termino $q_i$ en el documento $D$.
*   $|D|$ es la longitud en tokens del documento analizado.
*   $\text{avgdl}$ es la longitud promedio de todos los documentos dentro del corpus indexado.
*   $k_1$ es el parametro de saturacion del termino (configurado por defecto en `1.5`), que limita la influencia del crecimiento no lineal de frecuencias del mismo termino.
*   $b$ es el coeficiente de normalizacion de longitud del documento (por defecto `0.75`), penalizando documentos excesivamente largos si no aportan informacion relevante.

El calculo de la frecuencia inversa de documento ($\text{IDF}$) aplica suavizado para prevenir valores indeterminados o negativos ante terminos excesivamente comunes en el corpus:

$$\text{IDF}(q_i) = \ln\left(1.0 + \frac{N - df(q_i) + 0.5}{df(q_i) + 0.5}\right)$$

Donde $N$ es el tamano del corpus y $df(q_i)$ es el numero de documentos que contienen al menos una ocurrencia del termino $q_i$. Si un termino genera un IDF teorico negativo, se establece un suelo inferior de `0.0001` para asegurar la estabilidad numerica del ranking.

El pipeline pre-procesa la consulta y el corpus aplicando una tokenizacion que convierte el texto a minusculas, extrae palabras mediante expresiones regulares a nivel de palabra `\b\w+\b` y remueve stop-words de una lista integrada en espanol e ingles para mitigar el ruido estadistico.

### 2. Reciprocal Rank Fusion (RRF)

RRF es un metodo empirico de fusion de rankings basado en el orden relativo de los elementos en lugar de la magnitud absoluta de sus puntuaciones. Permite combinar distribuciones de probabilidad y distancias metricas distintas sin requerir calibracion o ajuste de escalas:

$$\text{RRF\_score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Donde:
*   $M$ es el conjunto de recuperadores (en este caso, Sparse y Dense).
*   $r_m(d)$ es la posicion (rango) del documento $d$ en la lista devuelta por el recuperador $m$ (1-indexed). Si un documento no aparece en el ranking de un recuperador, su rango se considera infinito ($\frac{1}{\infty} = 0$).
*   $k$ es una constante de suavizado (por defecto `60`), encargada de reducir la influencia desproporcionada de los primeros puestos del ranking y estabilizar la fusion frente a variaciones menores en posiciones bajas.

### 3. Fusion Lineal Ponderada (Score Normalization Fusion)

Cuando las puntuaciones de origen son necesarias para su analisis posterior, este metodo convierte las distancias de la busqueda densa y las puntuaciones de BM25 a un intervalo normalizado $[0, 1]$ usando la tecnica Min-Max:

$$\text{Score}_{\text{norm}} = \frac{\text{Score} - \text{Score}_{\text{min}}}{\text{Score}_{\text{max}} - \text{Score}_{\text{min}}}$$

Previamente, las distancias devueltas por `NanoVectorDB` (que dependen de la metrica configurada) se proyectan a indices de similitud:
*   **Similitud de Coseno:** Se asume $\text{Similitud} = 1.0 - \text{distancia\_coseno}$.
*   **Similitud Euclidea (L2):** Se transforma mediante la funcion monotonica $\text{Similitud} = \frac{1.0}{1.0 + \text{distancia\_L2}}$.
*   **Similitud Dot Product:** Dado que `NanoVectorDB` almacena el negativo del producto escalar para minimizarlo como distancia, se establece $\text{Similitud} = -\text{distancia\_dot}$.

Una vez obtenidas las similitudes normalizadas para ambas vias, el score combinado se calcula mediante una ponderacion parametrizable:

$$\text{Score}_{\text{final}}(d) = \alpha \cdot \text{Score}_{\text{norm\_dense}}(d) + (1.0 - \alpha) \cdot \text{Score}_{\text{norm\_sparse}}(d)$$

Donde $\alpha \in [0, 1]$ es el coeficiente de ponderacion densa (por defecto `0.5`).

## Conexión con el Ecosistema

Este modulo actua como un orquestador que conecta los siguientes subproyectos:
1.  **nano-vector-db:** Recupera candidatos vectoriales densos a traves de busquedas en grafos HNSW multicapa. El pipeline le pasa vectores de consulta y filtros de metadatos opcionales, extrayendo las distancias y los IDs de los documentos.
2.  **contrastive-embedding-trainer:** El pipeline comprueba si existe un modelo PyTorch/Transformers entrenado en la ruta relativa `../contrastive-embedding-trainer/model_output`. De estar disponible, lo carga automaticamente en memoria para codificar las queries de entrada mediante tecnicas de *Mean Pooling* y normalizacion L2. En caso de ausencia, utiliza un generador determinista pseudo-aleatorio basado en el hash de la cadena de texto para evitar la interrupcion del flujo de ejecucion.

## Estructura de Archivos

*   `bm25.py`: Contiene la clase `BM25Retriever`, encargada de la tokenizacion, extraccion de frecuencias y calculo de puntuaciones Okapi BM25 e IDF.
*   `fusion.py`: Modulo con las implementaciones de los algoritmos `reciprocal_rank_fusion` y `score_normalization_fusion`.
*   `pipeline.py`: Clase `HybridSearchPipeline` que unifica los recuperadores y coordina la ejecucion de busquedas paralelas.
*   `test_pipeline.py`: Conjunto de pruebas unitarias que validan el comportamiento y precision de las formulas matematicas de BM25, RRF y normalizacion.
*   `example.py`: Codigo de demostracion interactiva que ilustra la diferencia de resultados al realizar busquedas lexicas pura, vectorial pura, RRF e hibrida por Score Normalization.

## Instalacion y Uso

### 1. Configurar Entorno e Instalar Dependencias

Se requiere Python 3.9 o superior. Inicialice el entorno virtual y asegure la presencia de las dependencias requeridas:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar las Pruebas Unitarias

El conjunto de pruebas valida la correccion matematica de las formulas de fusion de rankings y la coherencia de la busqueda BM25:

```bash
.venv/bin/python -m unittest test_pipeline.py
```

### 3. Ejecutar Demostración de Búsqueda Híbrida

Para ver el pipeline en accion, ejecute:

```bash
.venv/bin/python example.py
```

El script pobla una base de datos vectorial `NanoVectorDB`, indexa el corpus de texto en el motor BM25 de forma concurrente, genera busquedas de ejemplo y muestra detalladamente la transicion de scores y posiciones resultantes bajo RRF y Score Normalization.
