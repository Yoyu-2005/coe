"""
VECTA Configuration, Schema Definitions, and Parameter Search Spaces.
Formalized for research-grade Approximate Nearest Neighbor (ANN) index auto-tuning.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class IndexFamily(str, Enum):
    FLAT = "FLAT"
    HNSW = "HNSW"
    IVF_FLAT = "IVF_FLAT"
    IVF_PQ = "IVF_PQ"


class MetricType(str, Enum):
    L2 = "L2"
    IP = "IP"  # Inner Product
    COSINE = "COSINE"


@dataclass
class WorkloadProfile:
    """Workload representation capturing static geometry and runtime constraints."""
    num_vectors: int = 20000
    dimension: int = 128
    metric: MetricType = MetricType.L2
    target_k: int = 10
    
    # SLA Constraints
    target_recall: float = 0.95
    max_latency_ms: float = 2.0
    memory_budget_mb: float = 256.0
    expected_qps: float = 500.0
    
    # Empirical Geometric Features
    intrinsic_dimension: float = 18.4
    norm_mean: float = 1.0
    norm_variance: float = 0.08
    sparsity_ratio: float = 0.0
    clustering_tendency_hopkins: float = 0.72
    
    # Host Hardware Snapshot
    cpu_count: int = 8
    available_ram_gb: float = 16.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['metric'] = self.metric.value
        return d


@dataclass
class IndexConfig:
    """ANN Index configuration hyper-parameter specification."""
    family: IndexFamily = IndexFamily.HNSW
    
    # HNSW Hyperparameters
    M: int = 16
    efConstruction: int = 64
    efSearch: int = 32
    
    # IVF Hyperparameters
    nlist: int = 128
    nprobe: int = 8
    
    # Product Quantization (IVF-PQ) Hyperparameters
    m_pq: int = 16          # Number of sub-quantizers (must divide dimension)
    nbits_per_idx: int = 8  # Typically 8 bits (256 centroids per sub-space)
    
    name_override: Optional[str] = None

    @property
    def id_string(self) -> str:
        if self.family == IndexFamily.FLAT:
            return "Flat-Exact"
        elif self.family == IndexFamily.HNSW:
            return f"HNSW(M={self.M}, efC={self.efConstruction}, efS={self.efSearch})"
        elif self.family == IndexFamily.IVF_FLAT:
            return f"IVF-Flat(nlist={self.nlist}, nprobe={self.nprobe})"
        elif self.family == IndexFamily.IVF_PQ:
            return f"IVF-PQ(nlist={self.nlist}, nprobe={self.nprobe}, M_pq={self.m_pq})"
        return str(self.family.value)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['family'] = self.family.value
        d['id_string'] = self.id_string
        return d


@dataclass
class PerformancePrediction:
    """Predicted metrics produced by the ML Surrogate Cost Model with conformal uncertainty."""
    predicted_recall: float
    predicted_latency_p50: float
    predicted_latency_p95: float
    predicted_latency_p99: float
    predicted_memory_mb: float
    predicted_build_time_s: float
    
    # Epistemic & Aleatoric Uncertainty Bounds (90% Confidence Interval half-widths)
    recall_uncertainty: float = 0.02
    latency_uncertainty_p95: float = 0.15
    memory_uncertainty: float = 1.2
    
    # Feasibility indicator
    is_feasible: bool = True
    constraint_violation_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Empirical benchmarking output comparing ground-truth to ANN configurations."""
    config: IndexConfig
    recall_at_k: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    throughput_qps: float
    memory_footprint_mb: float
    index_build_time_s: float
    tuning_time_ms: float = 0.0
    
    # Relative Comparisons
    speedup_vs_baseline: float = 1.0
    recall_delta_vs_baseline: float = 0.0
    memory_reduction_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['config'] = self.config.to_dict()
        return d


# Search Space Grid for Auto-Tuning Optimization
INDEX_SEARCH_SPACE = {
    IndexFamily.HNSW: {
        "M": [8, 16, 32, 48, 64],
        "efConstruction": [32, 64, 128, 200],
        "efSearch": [8, 16, 32, 64, 128, 256]
    },
    IndexFamily.IVF_FLAT: {
        "nlist": [32, 64, 128, 256, 512, 1024],
        "nprobe": [1, 2, 4, 8, 16, 32, 64]
    },
    IndexFamily.IVF_PQ: {
        "nlist": [32, 64, 128, 256, 512],
        "nprobe": [1, 2, 4, 8, 16, 32],
        "m_pq_divisors": [4, 8, 16, 32]  # Divisors of D
    }
}

# Standard Reference Baselines
FAISS_DEFAULT_HNSW = IndexConfig(
    family=IndexFamily.HNSW,
    M=32,
    efConstruction=40,
    efSearch=16,
    name_override="FAISS Default HNSW"
)

FAISS_DEFAULT_IVF = IndexConfig(
    family=IndexFamily.IVF_FLAT,
    nlist=100,
    nprobe=1,
    name_override="FAISS Default IVF"
)
