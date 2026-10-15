# VECTA: Intelligent Auto-Tuning Framework for Approximate Nearest Neighbor (ANN) Search

## 1. Executive Summary & Vision

Approximate Nearest Neighbor (ANN) search is the bedrock of contemporary AI systems—powering **Retrieval-Augmented Generation (RAG)**, large-scale semantic text and image search, recommendation engines, and multimodal vector databases (FAISS, Milvus, Qdrant, Weaviate).

Modern vector search libraries offer high-performance indexing families (notably **HNSW**, **IVF-Flat**, **IVF-PQ**, and composite variants), but they expose delicate, non-intuitive hyper-parameters:
- **HNSW**: $M$ (bi-directional links per node), $efConstruction$ (build exploration factor), $efSearch$ (runtime query exploration factor).
- **IVF-Flat / IVF-PQ**: $nlist$ (number of Voronoi partition centroids), $nprobe$ (number of neighboring centroids inspected per query), $M_{pq}$ (sub-quantization dimension), $nbits$ (bits per code).

### The Core Problem
Selecting and tuning these parameters is presently an ad-hoc, manual trial-and-error procedure requiring vector systems expertise:
1. **Under-parameterization**: Low $efSearch$ or $nprobe$ causes silent recall collapse ($< 70\%$ Recall@10), hurting RAG context fidelity and recommendation precision.
2. **Over-parameterization**: High $M$, high $efConstruction$, or high $nprobe$ triggers 10x-100x query latency spikes, prohibitive indexing build times, and explosive RAM consumption.
3. **Workload Mismatch**: As dataset volume ($N$), dimensionality ($D$), query arrival rate (QPS), and hardware constraints fluctuate, static default configurations degrade rapidly.

### The Vecta Solution
**Vecta** is an end-to-end, intelligent auto-tuning framework built on top of **FAISS**. Vecta autonomously monitors workload features, leverages an ensemble Machine Learning surrogate cost model to predict (Recall@k, Latency, Memory, Build Time), and deploys an adaptive multi-objective tuning engine (Bayesian & Pareto optimization) to select, configure, and adaptively tune ANN indexes without manual trial-and-error.

---

## 2. System Architecture & Component Breakdown

```
                       +------------------------------------------+
                       |              Client Workload             |
                       |  (Vector Datasets, Query Traffic, SLAs)  |
                       +------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------+
|                              VECTA CORE FRAMEWORK                                 |
|                                                                                   |
|  +------------------------+   +-----------------------+   +--------------------+  |
|  |   Workload Profiler    |   |  ML-Based Cost Model  |   |  Adaptive Tuner    |  |
|  | - Dataset N, D, Norms  |-->| - Multi-target Regr.  |<->| - Constrained      |  |
|  | - QPS, Batch, Target k |   | - Latency, Recall, RAM|   |   Pareto Search    |  |
|  | - RAM / CPU Profiling  |   | - Uncertainty Bounds  |   | - Bayesian Opt.    |  |
|  +------------------------+   +-----------------------+   +--------------------+  |
|                                                                     |             |
|                                           +-------------------------+             |
|                                           v                                       |
|                             +---------------------------+                         |
|                             |    FAISS Engine Wrapper   |                         |
|                             | - Flat (L2 / IP Baseline) |                         |
|                             | - HNSW (M, efC, efS)      |                         |
|                             | - IVF-Flat (nlist, nprobe)|                         |
|                             | - IVF-PQ (nlist, M_pq, nP)|                         |
|                             +---------------------------+                         |
|                                           |                                       |
+-----------------------------------------------------------------------------------+
                                            |
                                            v
                       +------------------------------------------+
                       |      Evaluation & Benchmarking Suite     |
                       | - Recall@k vs Latency Trade-off curves   |
                       | - Ground Truth Exact Search Baseline     |
                       | - Head-to-Head: Vecta vs FAISS Defaults  |
                       +------------------------------------------+
                                            |
                                            v
                       +------------------------------------------+
                       |   Interactive Web UI & REST API Engine   |
                       | - Modern Glassmorphism Dashboard         |
                       | - Live Workload Simulator & Visualizer   |
                       | - Real-time Query Playground             |
                       +------------------------------------------+
```

---

## 3. Detailed Component Specifications

### 3.1 Workload Profiler (`vecta.profiler`)
Captures static vector features and dynamic runtime query characteristics:
1. **Dataset Properties**:
   - $N$: Number of database vectors.
   - $D$: Vector dimensionality (e.g. 64, 128, 384, 768, 1536).
   - Metric type: Euclidean ($L_2$) or Inner Product / Cosine ($IP$).
   - Statistical distribution: Mean norm, variance of norms, approximate intrinsic dimensionality (Two-NN estimator / sample covariance eigenvalue decay).
2. **Workload Demands & SLAs**:
   - Target Recall@k (e.g. $\ge 0.95$ at $k=10$).
   - Maximum tolerable latency constraint (e.g. $\le 2.5\text{ ms}$ at $p95$).
   - Maximum memory footprint budget (e.g. $\le 512\text{ MB}$).
   - Query dynamics: Expected QPS, burstiness factor, batch query size.
3. **Hardware & Resource Monitor**:
   - Available CPU cores, thread pool capacity (`threadpoolctl`), available system RAM (`psutil`).

### 3.2 FAISS Engine Wrapper & Index Space (`vecta.engine`)
Encapsulates FAISS index construction, vector addition, runtime parameter adjustment, and metric extraction:
1. **Index Types**:
   - `Flat`: Exact $L_2$ / $IP$ search (Ground Truth generator & baseline).
   - `HNSW`: `IndexHNSWFlat`, parameters: $M \in [4, 8, 16, 32, 64]$, $efConstruction \in [16, 40, 64, 128, 200]$, $efSearch \in [8, 16, 32, 64, 128, 256]$.
   - `IVF-Flat`: `IndexIVFFlat`, parameters: $nlist \in [16, 32, 64, 128, 256, 512, 1024]$, $nprobe \in [1, 2, 4, 8, 16, 32, 64, 128]$.
   - `IVF-PQ`: `IndexIVFPQ` for memory conservation, parameters: $nlist$, $M_{pq} \in [4, 8, 16, 32]$, $nbits=8$, $nprobe$.
2. **Dynamic Runtime Reconfiguration**:
   - Non-destructive query parameter modifications: Updating `index.hnsw.efSearch` or `index.nprobe` instantaneously without rebuilding the index.
   - Structural transitions: Asynchronous re-indexing worker when switching index families or structural parameters ($M$, $nlist$).

### 3.3 Machine Learning Cost Model (`vecta.cost_model`)
Surrogate predictor that maps workload properties + index hyper-parameters to predicted performance:
1. **Predicted Metrics**:
   - $\hat{R}$: Expected Recall@k $\in [0.0, 1.0]$.
   - $\hat{L}_{p50}, \hat{L}_{p95}, \hat{L}_{p99}$: Expected query latencies (milliseconds).
   - $\hat{M}_{rss}$: Expected memory footprint (MB).
   - $\hat{T}_{build}$: Expected index construction time (seconds).
2. **Modeling Technique**:
   - Gradient-boosted and Random Forest regressors with physics-guided analytical bounds:
     - HNSW theoretical query complexity: $O(D \cdot efSearch \cdot \log N)$.
     - IVF theoretical query complexity: $O(D \cdot \frac{N}{nlist} \cdot nprobe)$.
     - Quantization memory bound: $N \cdot M_{pq} \text{ bytes} + \text{codebook overhead}$.
   - Meta-training on curated synthetic and benchmark datasets.
   - **Online Calibration / Few-shot Tuning**: Executes micro-benchmarks on a tiny stratified sample (e.g. 1,000 vectors) to shift/scale predictions to current host hardware.

### 3.4 Adaptive Tuning Engine (`vecta.tuner`)
The optimizer that chooses the winning configuration:
1. **Optimization Strategies**:
   - **Cost-Model Zero-Shot Selection**: Evaluates all candidate configurations across the index family grid in $< 5\text{ ms}$, filtering out those violating memory or recall constraints, and selecting the minimum latency configuration.
   - **Constrained Pareto Optimizer**: Identifies the non-dominated set across $(Recall, Latency, Memory)$.
   - **Bayesian Optimization with Constraint Handling**: For workloads requiring hyper-fine tuning, utilizes acquisition function (Expected Improvement subject to Probability of Feasibility) against micro-benchmarks.
   - **Runtime Dynamic Adaptation**: Auto-adjusts $efSearch$ / $nprobe$ when runtime query latency or traffic surges above SLA threshold.

### 3.5 Benchmarking & Validation Suite (`vecta.benchmark`)
Automated test suite verifying Vecta against standard ANN datasets and default settings:
- Datasets:
  - Synthetic Gaussian Clusters (high contrast, tunable $N, D$).
  - GloVe/Dense Semantic Embedding distributions.
  - Uniform Random Hypersphere distribution.
- Metrics measured:
  - Exact Ground Truth Recall@k ($k=1, 10, 50, 100$).
  - p50, p95, p99 query latency over thousands of query trials.
  - Memory consumption delta before/after index build.
  - Tuning convergence time (seconds taken to find optimal configuration).
  - Head-to-head comparison against standard FAISS defaults:
    - Default HNSW: $M=32, efSearch=16$
    - Default IVF: $nlist=100, nprobe=1$

### 3.6 Interactive Web Application & REST API (`vecta.ui` & `vecta.api`)
A state-of-the-art interactive web dashboard built with modern aesthetic styling:
- **Glassmorphism Dark-Mode Design**: Sleek purple/cyan gradient accents, responsive cards, micro-interactions, responsive typography.
- **Interactive Workload Studio**: Sliders for $N, D$, Target Recall (0.80 - 0.99), Max Latency (ms), and Memory Budget (MB).
- **Auto-Tuning Visualizer**:
  - Live Pareto Frontier plot (Recall vs Latency vs Memory).
  - Step-by-step optimization logs and recommendation breakdown.
- **Head-to-Head Arena**:
  - Side-by-side comparison cards: FAISS Default vs Vecta Auto-Tuned.
  - Latency speedup multiplier (e.g. $3.8\times$ faster), recall improvement ($+14.2\%$), memory savings.
- **Interactive Vector Search Playground**:
  - Query generator, real-time nearest neighbor retrieval, distance inspection, and ground-truth verification display.
- **FastAPI Endpoints**:
  - `POST /api/profile`: Extract dataset & workload features.
  - `POST /api/tune`: Run tuning engine and return optimal configuration + Pareto candidates.
  - `POST /api/benchmark`: Execute real FAISS benchmark and return empirical metrics.
  - `POST /api/search`: Run query against active index.
  - `GET /api/status`: System hardware resources and active index telemetry.

---

## 4. Implementation Structure

```
COE/
├── VECTA_ARCHITECTURE_AND_PLAN.md   <- Master technical specification
├── README.md                        <- Project overview and quickstart
├── pyproject.toml / requirements.txt<- Dependencies
├── vecta/
│   ├── __init__.py                  <- Public API exports
│   ├── config.py                    <- Configuration dataclasses & search spaces
│   ├── profiler/
│   │   ├── __init__.py
│   │   ├── workload_profiler.py     <- Dataset & query profiling
│   │   └── hardware_profiler.py     <- CPU & RAM profiling
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── faiss_wrapper.py         <- FAISS Flat, HNSW, IVF-Flat, IVF-PQ wrappers
│   │   └── dynamic_index.py         <- Runtime parameter reconfigurator
│   ├── cost_model/
│   │   ├── __init__.py
│   │   ├── surrogate_model.py       <- ML Regressors (Recall, Latency, Memory)
│   │   ├── feature_extractor.py     <- Feature vectorization
│   │   └── precomputed_meta.py      <- Warm-start pre-trained knowledge base
│   ├── tuner/
│   │   ├── __init__.py
│   │   ├── optimizer.py             <- Multi-objective Pareto & Bayesian tuner
│   │   └── runtime_adapter.py       <- Real-time QPS/SLA dynamic controller
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── dataset_generator.py     <- Synthetic & benchmark dataset generators
│   │   ├── evaluator.py             <- Ground truth, Recall@k, p99 latency runner
│   │   └── baseline_comparer.py     <- Vecta vs FAISS Defaults comparison engine
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                   <- FastAPI REST API server
│   └── ui/
│       ├── static/
│       │   ├── index.html           <- Premium Glassmorphism UI
│       │   ├── css/
│       │   │   └── style.css        <- Custom modern CSS design system
│       │   └── js/
│       │       ├── app.js           <- Client dashboard logic & charts
│       │       └── charts.js        <- Interactive Pareto & Benchmark visualizers
│       └── launch.py                <- Web UI launcher
├── tests/
│   ├── test_profiler.py
│   ├── test_engine.py
│   ├── test_cost_model.py
│   ├── test_tuner.py
│   └── test_benchmark.py
└── demo.py                          <- End-to-end CLI & script demonstration
```

---

## 5. Verification & Validation Roadmap

1. **Unit & Integration Testing**:
   - `test_profiler.py`: Validate dataset feature extraction, norm variance, intrinsic dimension estimation.
   - `test_engine.py`: Test FAISS Flat, HNSW, IVF-Flat, IVF-PQ index creation, search correctness, and dynamic parameter updates.
   - `test_cost_model.py`: Test feature vectorization, surrogate model predictions, and monotonic bounds.
   - `test_tuner.py`: Validate constrained Pareto selection and parameter optimization under latency/recall budgets.
   - `test_benchmark.py`: Verify ground truth computation, Recall@10 accuracy calculation, and latency percentiles.

2. **End-to-End System Evaluation**:
   - Run `demo.py` benchmarking across different workloads:
     - Workload A: High-recall critical (Target Recall $\ge 0.98$, Latency $\le 5\text{ ms}$).
     - Workload B: Ultra-low latency RAG (Target Recall $\ge 0.90$, Latency $\le 1.0\text{ ms}$).
     - Workload C: Memory-constrained (Target Recall $\ge 0.85$, Memory $\le 30\text{ MB}$, Large $N$).
   - Confirm Vecta auto-tunes configuration to outperform FAISS default configurations in each scenario.

3. **Web Application Verification**:
   - Start FastAPI server and web dashboard.
   - Test UI interactions: Workload profiling, Auto-tune execution, Pareto chart rendering, Head-to-Head benchmark comparisons, and live vector queries.
