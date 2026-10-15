"""
VECTA Dynamic Runtime Closed-Loop Adapter.
Continuously monitors runtime query latency and dynamically modulates search parameters
(efSearch and nprobe) to guarantee SLA compliance under traffic surges.
"""

from typing import Dict, Any, Optional
import numpy as np
from vecta.config import IndexConfig, IndexFamily
from vecta.engine.dynamic_index import DynamicIndexManager


class RuntimeAdapter:
    """Feedback control loop adjusting search precision and throughput in real time."""

    def __init__(self, manager: DynamicIndexManager, target_latency_ms: float = 2.0):
        self.manager = manager
        self.target_latency_ms = target_latency_ms
        self.adaptation_history = []

    def evaluate_and_adapt(self) -> Dict[str, Any]:
        """Inspect moving window telemetry and modulate index parameters if necessary."""
        if self.manager.active_wrapper is None or len(self.manager.recent_latencies_ms) < 5:
            return {"status": "insufficient_data", "adapted": False}

        cfg = self.manager.active_wrapper.config
        lat_arr = list(self.manager.recent_latencies_ms)
        recent_p95 = float(np.percentile(lat_arr, 95))
        
        action_taken = None
        old_val = None
        new_val = None

        if cfg.family == IndexFamily.HNSW:
            old_val = cfg.efSearch
            if recent_p95 > self.target_latency_ms * 1.15:
                # Latency SLA breach: throttle efSearch down
                new_val = max(8, int(old_val * 0.75))
                if new_val != old_val:
                    self.manager.adapt_runtime_parameter("efSearch", new_val)
                    action_taken = f"Throttled efSearch from {old_val} -> {new_val} to defend {self.target_latency_ms}ms SLA"
            elif recent_p95 < self.target_latency_ms * 0.40:
                # Ample headroom: raise efSearch to maximize recall
                new_val = min(256, int(old_val * 1.35))
                if new_val != old_val:
                    self.manager.adapt_runtime_parameter("efSearch", new_val)
                    action_taken = f"Boosted efSearch from {old_val} -> {new_val} (headroom available, improving recall)"

        elif cfg.family in (IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ):
            old_val = cfg.nprobe
            if recent_p95 > self.target_latency_ms * 1.15:
                new_val = max(1, int(old_val // 2))
                if new_val != old_val:
                    self.manager.adapt_runtime_parameter("nprobe", new_val)
                    action_taken = f"Reduced nprobe from {old_val} -> {new_val} to curtail latency spike"
            elif recent_p95 < self.target_latency_ms * 0.40:
                new_val = min(cfg.nlist, int(old_val * 2))
                if new_val != old_val:
                    self.manager.adapt_runtime_parameter("nprobe", new_val)
                    action_taken = f"Increased nprobe from {old_val} -> {new_val} for superior recall"

        record = {
            "p95_latency_ms": round(recent_p95, 3),
            "target_latency_ms": self.target_latency_ms,
            "action": action_taken,
            "adapted": action_taken is not None
        }
        self.adaptation_history.append(record)
        return record
