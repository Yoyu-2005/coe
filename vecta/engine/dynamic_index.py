"""
VECTA Dynamic Runtime Vector Search Manager.
Manages active serving index, query telemetry logging, and runtime parameter modulation.
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List
from collections import deque
from vecta.config import IndexConfig, IndexFamily, MetricType
from vecta.engine.faiss_wrapper import FaissIndexWrapper


class DynamicIndexManager:
    """Provides hot-swapping, runtime parameter tuning, and moving query latency telemetry."""

    def __init__(self, dimension: int, metric: MetricType = MetricType.L2):
        self.dimension = dimension
        self.metric = metric
        self.active_wrapper: Optional[FaissIndexWrapper] = None
        self.recent_latencies_ms: deque = deque(maxlen=200)
        self.total_queries_served: int = 0
        self.creation_timestamp: float = time.time()

    def set_active_index(self, wrapper: FaissIndexWrapper):
        """Set or swap the active index."""
        self.active_wrapper = wrapper

    def query(self, queries: np.ndarray, k: int = 10) -> Dict[str, Any]:
        """Execute query with fine-grained latency telemetry."""
        if self.active_wrapper is None or self.active_wrapper.index is None:
            raise RuntimeError("No active FAISS index is loaded.")

        t0 = time.perf_counter()
        distances, indices, raw_latency_ms = self.active_wrapper.search(queries, k=k)
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        self.recent_latencies_ms.append(total_latency_ms)
        self.total_queries_served += queries.shape[0]

        lat_arr = np.array(self.recent_latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))

        return {
            "distances": distances.tolist(),
            "indices": indices.tolist(),
            "query_latency_ms": round(total_latency_ms, 3),
            "recent_p50_ms": round(p50, 3),
            "recent_p95_ms": round(p95, 3),
            "recent_p99_ms": round(p99, 3),
            "total_queries_served": self.total_queries_served,
            "active_config": self.active_wrapper.config.to_dict()
        }

    def adapt_runtime_parameter(self, param_name: str, value: int):
        """Instantaneously modify runtime query parameters without index rebuild."""
        if self.active_wrapper is None:
            return
        if param_name == "efSearch":
            self.active_wrapper.set_runtime_parameters(efSearch=value)
        elif param_name == "nprobe":
            self.active_wrapper.set_runtime_parameters(nprobe=value)
