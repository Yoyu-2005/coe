"""
VECTA: Intelligent Auto-Tuning Framework for Approximate Nearest Neighbor (ANN) Search.
"""

from vecta.config import (
    IndexFamily,
    MetricType,
    WorkloadProfile,
    IndexConfig,
    PerformancePrediction,
    BenchmarkResult,
    INDEX_SEARCH_SPACE,
    FAISS_DEFAULT_HNSW,
    FAISS_DEFAULT_IVF
)
from vecta.profiler.workload_profiler import WorkloadProfiler
from vecta.profiler.hardware_profiler import HardwareProfiler
from vecta.engine.faiss_wrapper import FaissIndexWrapper
from vecta.engine.dynamic_index import DynamicIndexManager
from vecta.cost_model.surrogate_model import SurrogateCostModel
from vecta.tuner.optimizer import AutoTuningOptimizer
from vecta.tuner.runtime_adapter import RuntimeAdapter
from vecta.benchmark.dataset_generator import BenchmarkDatasetGenerator
from vecta.benchmark.evaluator import BenchmarkEvaluator
from vecta.benchmark.baseline_comparer import BaselineComparer

__version__ = "1.0.0"
__all__ = [
    "IndexFamily",
    "MetricType",
    "WorkloadProfile",
    "IndexConfig",
    "PerformancePrediction",
    "BenchmarkResult",
    "INDEX_SEARCH_SPACE",
    "FAISS_DEFAULT_HNSW",
    "FAISS_DEFAULT_IVF",
    "WorkloadProfiler",
    "HardwareProfiler",
    "FaissIndexWrapper",
    "DynamicIndexManager",
    "SurrogateCostModel",
    "AutoTuningOptimizer",
    "RuntimeAdapter",
    "BenchmarkDatasetGenerator",
    "BenchmarkEvaluator",
    "BaselineComparer"
]
