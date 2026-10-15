"""
VECTA High-Performance FastAPI REST Server.
Exposes endpoints for workload profiling, ML surrogate cost modeling,
Pareto auto-tuning, live FAISS empirical benchmarking, and vector search playground.
"""

import os
import sys
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# Ensure project root is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vecta.config import (
    WorkloadProfile,
    IndexConfig,
    IndexFamily,
    MetricType,
    PerformancePrediction,
    BenchmarkResult
)
from vecta.profiler import WorkloadProfiler, HardwareProfiler
from vecta.cost_model import SurrogateCostModel
from vecta.tuner import AutoTuningOptimizer, RuntimeAdapter
from vecta.engine import FaissIndexWrapper, DynamicIndexManager
from vecta.benchmark import BenchmarkDatasetGenerator, BenchmarkEvaluator, BaselineComparer

app = FastAPI(
    title="Vecta ANN Auto-Tuning Kernel",
    description="Intelligent Autonomous Auto-Tuning Engine for Approximate Nearest Neighbor Search",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container for fast in-memory serving & playground
class AppState:
    def __init__(self):
        self.cost_model = SurrogateCostModel()
        self.optimizer = AutoTuningOptimizer(self.cost_model)
        self.current_database: Optional[np.ndarray] = None
        self.current_queries: Optional[np.ndarray] = None
        self.current_evaluator: Optional[BenchmarkEvaluator] = None
        self.dynamic_manager: Optional[DynamicIndexManager] = None
        self.current_workload: Optional[WorkloadProfile] = None
        self.active_config: Optional[IndexConfig] = None
        self.last_tuning_result: Optional[Dict[str, Any]] = None

state = AppState()


# Request / Response Schemas
class WorkloadRequest(BaseModel):
    num_vectors: int = 20000
    dimension: int = 128
    metric: str = "L2"
    target_recall: float = 0.95
    max_latency_ms: float = 2.0
    memory_budget_mb: float = 128.0
    expected_qps: float = 500.0
    target_k: int = 10
    preset_name: Optional[str] = None


class SearchRequest(BaseModel):
    query_vector: Optional[List[float]] = None
    k: int = 10
    use_random_query: bool = True


@app.on_event("startup")
async def startup_event():
    """Initialize default benchmark dataset and warm-start FAISS active index."""
    # Initialize with default clustered dataset
    n, d, q = 15000, 128, 50
    db, queries = BenchmarkDatasetGenerator.generate_gaussian_clusters(
        num_vectors=n, dimension=d, num_clusters=20, num_queries=q, random_seed=42
    )
    state.current_database = db
    state.current_queries = queries
    state.current_evaluator = BenchmarkEvaluator(db, queries, metric=MetricType.L2, k=10)
    
    # Profile default workload
    state.current_workload = WorkloadProfiler.profile(
        db, metric=MetricType.L2, target_recall=0.95, max_latency_ms=1.5, memory_budget_mb=128.0
    )
    
    # Initial Auto-tune
    state.last_tuning_result = state.optimizer.tune(state.current_workload)
    rec_cfg = state.last_tuning_result["recommended_config"]
    
    winner_cfg = IndexConfig(
        family=IndexFamily(rec_cfg["family"]),
        M=rec_cfg.get("M", 16),
        efConstruction=rec_cfg.get("efConstruction", 64),
        efSearch=rec_cfg.get("efSearch", 32),
        nlist=rec_cfg.get("nlist", 128),
        nprobe=rec_cfg.get("nprobe", 8),
        m_pq=rec_cfg.get("m_pq", 16)
    )
    state.active_config = winner_cfg
    
    # Active Dynamic Manager
    state.dynamic_manager = DynamicIndexManager(d, MetricType.L2)
    wrapper = FaissIndexWrapper(winner_cfg, d, MetricType.L2)
    wrapper.build_and_add(db)
    state.dynamic_manager.set_active_index(wrapper)


@app.get("/api/system")
def get_system_telemetry():
    """Return host hardware specifications and real-time resource utilization."""
    hw = HardwareProfiler.get_hardware_profile()
    process_rss_mb = round(HardwareProfiler.current_rss_memory_mb(), 2)
    
    active_info = None
    if state.dynamic_manager and state.dynamic_manager.active_wrapper:
        w = state.dynamic_manager.active_wrapper
        active_info = {
            "total_vectors": w.total_vectors,
            "dimension": w.dimension,
            "config": w.config.to_dict(),
            "memory_mb": w.estimated_memory_mb,
            "queries_served": state.dynamic_manager.total_queries_served
        }

    return {
        "hardware": hw,
        "process_rss_mb": process_rss_mb,
        "active_index": active_info
    }


@app.get("/api/presets")
def get_presets():
    """Return list of research & industry scenario presets for panel demo."""
    presets = [
        WorkloadProfiler.get_preset("rag_semantic_search"),
        WorkloadProfiler.get_preset("recommendation_high_qps"),
        WorkloadProfiler.get_preset("edge_memory_constrained"),
        WorkloadProfiler.get_preset("sift_benchmark")
    ]
    return {"presets": presets}


@app.post("/api/profile")
def profile_workload(req: WorkloadRequest):
    """Profile vector manifold geometry and register SLA targets."""
    # Synthesize dataset corresponding to requested scale
    n = min(req.num_vectors, 30000)  # Keep interactive response < 1s
    d = req.dimension
    
    db, queries = BenchmarkDatasetGenerator.generate_gaussian_clusters(
        num_vectors=n, dimension=d, num_clusters=max(4, n // 500), num_queries=50, random_seed=42
    )
    state.current_database = db
    state.current_queries = queries
    
    metric = MetricType.IP if req.metric.upper() == "IP" else MetricType.L2
    state.current_evaluator = BenchmarkEvaluator(db, queries, metric=metric, k=req.target_k)
    
    wp = WorkloadProfiler.profile(
        db,
        metric=metric,
        target_recall=req.target_recall,
        max_latency_ms=req.max_latency_ms,
        memory_budget_mb=req.memory_budget_mb,
        expected_qps=req.expected_qps,
        target_k=req.target_k
    )
    state.current_workload = wp
    
    return {
        "workload_profile": wp.to_dict(),
        "vector_sample_2d": db[:300, :2].tolist()  # Sample 2D slice for canvas rendering
    }


@app.post("/api/tune")
def run_auto_tuning(req: Optional[WorkloadRequest] = None):
    """Run surrogate model prediction, compute Pareto frontier, and select optimal configuration."""
    if req is not None:
        profile_workload(req)
        
    wp = state.current_workload
    if wp is None:
        raise HTTPException(status_code=400, detail="No active workload profiled.")
        
    tune_res = state.optimizer.tune(wp)
    state.last_tuning_result = tune_res
    
    rec_cfg = tune_res["recommended_config"]
    winner_cfg = IndexConfig(
        family=IndexFamily(rec_cfg["family"]),
        M=rec_cfg.get("M", 16),
        efConstruction=rec_cfg.get("efConstruction", 64),
        efSearch=rec_cfg.get("efSearch", 32),
        nlist=rec_cfg.get("nlist", 128),
        nprobe=rec_cfg.get("nprobe", 8),
        m_pq=rec_cfg.get("m_pq", 16)
    )
    state.active_config = winner_cfg
    
    # Rebuild serving index with winner
    if state.current_database is not None:
        wrapper = FaissIndexWrapper(winner_cfg, wp.dimension, wp.metric)
        wrapper.build_and_add(state.current_database)
        if state.dynamic_manager is None:
            state.dynamic_manager = DynamicIndexManager(wp.dimension, wp.metric)
        state.dynamic_manager.set_active_index(wrapper)

    return tune_res


@app.post("/api/benchmark")
def run_arena_benchmark():
    """Execute live empirical comparison: FAISS Defaults vs Vecta Auto-Tuned."""
    if state.current_evaluator is None or state.active_config is None:
        raise HTTPException(status_code=400, detail="Benchmark evaluator or active config not ready.")

    report = BaselineComparer.run_head_to_head_arena(state.current_evaluator, state.active_config)
    return report


@app.post("/api/search")
def search_vector(req: SearchRequest):
    """Query active index in real time and return nearest neighbors with distance & GT accuracy."""
    if state.dynamic_manager is None or state.dynamic_manager.active_wrapper is None:
        raise HTTPException(status_code=400, detail="No active index available.")

    if req.use_random_query or req.query_vector is None:
        # Sample random query from query pool
        if state.current_queries is not None and len(state.current_queries) > 0:
            rand_idx = np.random.randint(0, len(state.current_queries))
            q_vec = state.current_queries[rand_idx:rand_idx + 1]
        else:
            q_vec = np.random.randn(1, state.dynamic_manager.dimension).astype(np.float32)
            q_vec /= np.linalg.norm(q_vec)
    else:
        q_vec = np.array([req.query_vector], dtype=np.float32)

    res = state.dynamic_manager.query(q_vec, k=req.k)
    
    # Check ground truth accuracy for this specific query
    exact_match_count = req.k
    if state.current_evaluator and state.current_evaluator.gt_indices is not None:
        # Compute exact ground truth for this single query
        gt_dist, gt_idx, _ = state.current_evaluator.gt_distances, state.current_evaluator.gt_indices, 0
        if req.use_random_query and rand_idx < len(state.current_evaluator.gt_indices):
            true_top_k = set(state.current_evaluator.gt_indices[rand_idx, :req.k])
            retrieved = set(res["indices"][0])
            exact_match_count = len(retrieved.intersection(true_top_k))

    res["query_vector_sample"] = q_vec[0, :8].tolist()
    res["exact_matches_in_top_k"] = exact_match_count
    res["accuracy_pct"] = round((exact_match_count / req.k) * 100.0, 1)

    return res


# Mount static directory for modern Web UI
static_path = os.path.join(os.path.dirname(__file__), "..", "ui", "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(static_path, "index.html"))
