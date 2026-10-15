# VECTA: Intelligent Auto-Tuning Framework for Approximate Nearest Neighbor (ANN) Search

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%20v1.15-purple.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Status](https://img.shields.io/badge/Research-ACM%2FIEEE%20Ready-cyan.svg)](#)

**Vecta** is an intelligent, autonomous auto-tuning framework for Approximate Nearest Neighbor (ANN) search built on top of the **FAISS** vector search engine. It replaces manual, error-prone index configuration with workload geometry profiling, physics-informed machine learning surrogate cost models, and multi-objective constrained Pareto optimization.

---

## 1. Research Motivation & Problem Statement

Modern AI applications—including **Retrieval-Augmented Generation (RAG)**, semantic document retrieval, and recommender systems—rely heavily on vector indexes. However, vector database engines expose sensitive hyper-parameters:
- **HNSW (Graph-based)**: $M$ (graph degree), $efConstruction$ (build exploration), $efSearch$ (runtime query exploration).
- **IVF-Flat / IVF-PQ (Cluster-based)**: $nlist$ (Voronoi centroids), $nprobe$ (inspected centroids per query), $M_{pq}$ (sub-space quantizers).

Misconfiguring these parameters causes:
1. **Recall Collapse**: Under-parameterized settings (e.g. FAISS default IVF $nprobe=1$) achieve only $\approx 22.9\%$ Recall@10, poisoning RAG context fidelity.
2. **Latency & Memory Explosions**: Over-parameterization causes $10\times$ tail latency spikes and prohibitive RAM footprint.
3. **Static Default Failures**: Fixed configurations degrade rapidly as vector count ($N$), dimensionality ($D$), and QPS change.

---

## 2. Mathematical Foundations

### 2.1 Constrained Multi-Objective Optimization
Given vector database $\mathcal{X} \subset \mathbb{R}^D$ and operational Service Level Agreements (SLAs), Vecta solves:

$$\min_{\theta \in \Theta} \mathcal{L}_{p95}(\theta; \mathcal{W}) \quad \text{subject to} \quad 
\begin{cases}
  \mathcal{R}@k(\theta; \mathcal{W}) \ge R_{\min} \\
  \mathcal{M}_{rss}(\theta; \mathcal{W}) \le M_{\max} \\
  \mathcal{T}_{build}(\theta) \le T_{\max}
\end{cases}$$

Where $\Theta$ is the candidate index hyper-parameter lattice, $\mathcal{L}_{p95}$ is tail query latency, $\mathcal{R}@k$ is Recall@k, and $\mathcal{M}_{rss}$ is Resident Set Size memory consumption.

### 2.2 Two-NN Intrinsic Dimensionality Estimator
Under a local Poisson point process on a Riemannian manifold of dimension $d$, the ratio $\mu_i = r_{i,2} / r_{i,1}$ of Euclidean distances to the 2nd vs 1st nearest neighbor follows the cumulative distribution $P(\mu \le x) = 1 - x^{-d}$. Vecta computes the Maximum Likelihood Estimator (MLE):

$$\hat{d} = \frac{N}{\sum_{i=1}^N \ln(r_{i,2} / r_{i,1})}$$

### 2.3 Hopkins Statistic for Spatial Clustering Tendency
Evaluates whether vector distribution exhibits natural geometric clusters ($H \to 1.0$, optimal for Voronoi inverted files) or uniform hyperspherical dispersion ($H \approx 0.5$, optimal for HNSW graphs):

$$H = \frac{\sum_{i=1}^m w_i}{\sum_{i=1}^m w_i + \sum_{i=1}^m u_i}$$

---

## 3. Empirical Results: Live Head-to-Head Arena

Benchmarked on $N=15,000$ dense 128-dimensional clustered vectors ($k=10$):

| Configuration | Recall@10 | p95 Latency | p99 Latency | Throughput (QPS) | RAM Footprint | Build Time | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exact Brute-Force (Flat L2)** | 100.0% | 0.07 ms | - | 16,378 | 7.3 MB | < 0.01 s | Ground Truth Baseline |
| **FAISS Default IVF ($nlist=100, nprobe=1$)** | 22.9% | 0.03 ms | 0.09 ms | 41,256 | 7.5 MB | 0.06 s | Severe Recall Collapse |
| **FAISS Default HNSW ($M=32, efS=16$)** | 55.3% | 0.06 ms | 0.07 ms | 25,997 | 11.2 MB | 0.10 s | Insufficient RAG Recall |
| **VECTA Auto-Tuned HNSW($M=48, efC=32, efS=128$)** | **92.9%** | **0.22 ms** | **0.26 ms** | **5,079** | **13.1 MB** | **0.08 s** | **+37.6% Recall Delta** |

---

## 4. Quickstart: Running for the Panel Presentation

### Option 1: Visual Research Dashboard (Web UI)
Launch the interactive glassmorphism mission control dashboard:
```powershell
python run_dashboard.py
```
Open **`http://localhost:8000`** in your browser to inspect:
- Interactive Workload Studio (Sliders for $N, D$, Target Recall, Latency SLA)
- One-Click Industry Presets (RAG Semantic Search, High-QPS Recommender, Edge Memory Budget)
- Real-time 2D/3D Spectral Manifold Canvas Visualizer
- Multi-Objective Pareto Frontier Chart
- Live Head-to-Head Arena
- Sub-millisecond Vector Query Playground with Ground-Truth verification

### Option 2: Rich Console CLI Demo
Run the end-to-end benchmark in your terminal with live tables:
```powershell
python demo.py
```

### Option 3: Run the Automated Test Suite
```powershell
python tests/test_all.py
```

---

## 5. Repository Structure

```
COE/
├── VECTA_ARCHITECTURE_AND_PLAN.md   <- Master technical specification
├── README.md                        <- Project overview and guide
├── run_dashboard.py                 <- Web Dashboard & API launcher
├── demo.py                          <- Rich terminal demo script
├── requirements.txt                 <- Dependencies
├── vecta/
│   ├── __init__.py                  <- Public API exports
│   ├── config.py                    <- Data classes & search spaces
│   ├── profiler/                    <- Workload & hardware profilers
│   │   ├── workload_profiler.py     <- Two-NN & Hopkins feature extraction
│   │   └── hardware_profiler.py     <- CPU & RAM detection
│   ├── engine/                      <- High-performance FAISS engine wrappers
│   │   ├── faiss_wrapper.py         <- Flat, HNSW, IVF-Flat, IVF-PQ wrappers
│   │   └── dynamic_index.py         <- Moving query telemetry & dynamic updates
│   ├── cost_model/                  <- Machine Learning surrogate models
│   │   ├── feature_extractor.py     <- Physics-informed feature vectors
│   │   └── surrogate_model.py       <- Multi-objective RF regressors & calibration
│   ├── tuner/                       <- Multi-objective adaptive optimization
│   │   ├── optimizer.py             <- Constrained Pareto frontier finder
│   │   └── runtime_adapter.py       <- Closed-loop runtime SLA controller
│   ├── benchmark/                   <- Scientific benchmarking suite
│   │   ├── dataset_generator.py     <- Clustered & manifold vector generators
│   │   ├── evaluator.py             <- Ground-truth Recall@k & tail latency
│   │   └── baseline_comparer.py     <- Head-to-head evaluation arena
│   ├── api/                         <- FastAPI REST endpoints
│   │   └── app.py                   <- Serving layer
│   └── ui/                          <- Modern frontend
│       └── static/
│           ├── index.html           <- Glassmorphism interface
│           ├── css/style.css        <- Custom stylesheet
│           └── js/app.js            <- Interactive client logic & charts
└── tests/
    └── test_all.py                  <- Unit and integration tests
```
