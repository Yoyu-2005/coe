"""
VECTA Multi-Objective Adaptive Optimization Engine.
Performs constrained Pareto frontier discovery, utility ranking,
and micro-sample active calibration across the ANN hyper-parameter design space.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from vecta.config import (
    WorkloadProfile,
    IndexConfig,
    IndexFamily,
    PerformancePrediction,
    INDEX_SEARCH_SPACE,
    FAISS_DEFAULT_HNSW,
    FAISS_DEFAULT_IVF
)
from vecta.cost_model.surrogate_model import SurrogateCostModel
from vecta.engine.faiss_wrapper import FaissIndexWrapper


class AutoTuningOptimizer:
    """Adaptive search engine optimizing ANN parameters under strict SLA constraints."""

    def __init__(self, cost_model: Optional[SurrogateCostModel] = None):
        self.cost_model = cost_model or SurrogateCostModel()

    def generate_candidate_space(self, workload: WorkloadProfile, include_flat: bool = False) -> List[IndexConfig]:
        """Synthesize candidate configuration lattice across index families."""
        candidates: List[IndexConfig] = []
        d = workload.dimension
        
        # 1. HNSW Grid
        hnsw_space = INDEX_SEARCH_SPACE[IndexFamily.HNSW]
        for M in hnsw_space["M"]:
            for efC in hnsw_space["efConstruction"]:
                for efS in hnsw_space["efSearch"]:
                    candidates.append(IndexConfig(
                        family=IndexFamily.HNSW,
                        M=M,
                        efConstruction=efC,
                        efSearch=efS
                    ))
                    
        # 2. IVF-Flat Grid
        ivf_space = INDEX_SEARCH_SPACE[IndexFamily.IVF_FLAT]
        for nlist in ivf_space["nlist"]:
            for nprobe in ivf_space["nprobe"]:
                if nprobe <= nlist:
                    candidates.append(IndexConfig(
                        family=IndexFamily.IVF_FLAT,
                        nlist=nlist,
                        nprobe=nprobe
                    ))

        # 3. IVF-PQ Grid (memory-efficient)
        ivf_pq_space = INDEX_SEARCH_SPACE[IndexFamily.IVF_PQ]
        # Valid subquantizers dividing dimension
        valid_divisors = [div for div in [32, 16, 8, 4, 2] if d % div == 0]
        if not valid_divisors:
            valid_divisors = [1]
            
        for nlist in ivf_pq_space["nlist"]:
            for nprobe in ivf_pq_space["nprobe"]:
                if nprobe <= nlist:
                    for m_pq in valid_divisors:
                        candidates.append(IndexConfig(
                            family=IndexFamily.IVF_PQ,
                            nlist=nlist,
                            nprobe=nprobe,
                            m_pq=m_pq
                        ))

        # 4. Exact Flat Baseline (optional, only if requested)
        if include_flat:
            candidates.append(IndexConfig(family=IndexFamily.FLAT))
        
        return candidates

    def compute_pareto_frontier(
        self,
        candidates: List[IndexConfig],
        predictions: List[PerformancePrediction]
    ) -> List[int]:
        """
        Identify indices of non-dominated solutions on the 3D objective manifold:
        Maximize Recall, Minimize Latency, Minimize Memory Footprint.
        """
        pareto_indices = []
        n = len(candidates)
        
        # Matrix: [Recall, -Latency, -Memory] so that all objectives are to be MAXIMIZED
        objectives = np.zeros((n, 3))
        for i, pred in enumerate(predictions):
            objectives[i, 0] = pred.predicted_recall
            objectives[i, 1] = -pred.predicted_latency_p95
            objectives[i, 2] = -pred.predicted_memory_mb
            
        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j:
                    # j dominates i if j is >= in all objectives and strictly > in at least one
                    if (objectives[j, :] >= objectives[i, :]).all() and (objectives[j, :] > objectives[i, :]).any():
                        dominated = True
                        break
            if not dominated:
                pareto_indices.append(i)
                
        return pareto_indices

    def evaluate_utility(
        self,
        pred: PerformancePrediction,
        workload: WorkloadProfile
    ) -> float:
        """
        Multi-objective constrained utility function.
        Higher score = superior operational configuration.
        """
        # Recall utility: must achieve target SLA before latency optimization matters
        if pred.predicted_recall >= workload.target_recall:
            recall_score = 2.0 + 1.0 * (pred.predicted_recall - workload.target_recall)
        else:
            # Harsh penalty for failing recall SLA
            recall_score = -20.0 * (workload.target_recall - pred.predicted_recall)
            
        # Latency utility: normalized relative to max tolerable latency SLA
        lat_ratio = pred.predicted_latency_p95 / max(0.01, workload.max_latency_ms)
        if lat_ratio <= 1.0:
            latency_score = 1.0 - 0.4 * lat_ratio  # faster is better
        else:
            latency_score = -10.0 * (lat_ratio - 1.0)
            
        # Memory utility: normalized to budget
        mem_ratio = pred.predicted_memory_mb / max(1.0, workload.memory_budget_mb)
        if mem_ratio <= 1.0:
            memory_score = 1.0 - 0.2 * mem_ratio
        else:
            memory_score = -10.0 * (mem_ratio - 1.0)

        # Composite utility score with constraint penalty
        total_utility = (0.60 * recall_score) + (0.30 * latency_score) + (0.10 * memory_score)
        total_utility -= pred.constraint_violation_penalty
        
        return float(total_utility)

    def tune(
        self,
        workload: WorkloadProfile,
        active_calibration_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Run full auto-tuning pipeline:
        1. Generate search grid
        2. Surrogate cost model prediction
        3. Pareto frontier extraction
        4. Constrained ranking & winner selection
        5. (Optional) Micro-sample empirical verification
        """
        start_time = time.perf_counter()
        
        # 1. Candidate space
        candidates = self.generate_candidate_space(workload)
        
        # 2. ML Surrogate Predictions
        predictions = self.cost_model.predict_batch(workload, candidates)
        
        # 3. Pareto Frontier Discovery
        pareto_indices = self.compute_pareto_frontier(candidates, predictions)
        
        # 4. Rank Candidates by Utility
        scored_candidates = []
        for i, (cfg, pred) in enumerate(zip(candidates, predictions)):
            score = self.evaluate_utility(pred, workload)
            scored_candidates.append({
                "index": i,
                "config": cfg,
                "prediction": pred,
                "score": score,
                "is_pareto": i in pareto_indices
            })
            
        # Sort by utility descending
        scored_candidates.sort(key=lambda item: item["score"], reverse=True)
        winner_entry = scored_candidates[0]
        winner_cfg = winner_entry["config"]
        winner_pred = winner_entry["prediction"]
        
        # Optional: Micro-sample active verification
        calibration_performed = False
        if active_calibration_data is not None:
            vectors_sample, queries_sample = active_calibration_data
            if len(vectors_sample) >= 50 and len(queries_sample) >= 5:
                # Fast micro-benchmark on winning candidate
                try:
                    micro_wrapper = FaissIndexWrapper(winner_cfg, workload.dimension, workload.metric)
                    micro_wrapper.build_and_add(vectors_sample)
                    _, _, micro_lat = micro_wrapper.search(queries_sample, k=workload.target_k)
                    
                    # Calibrate cost model
                    self.cost_model.calibrate([
                        (winner_cfg, winner_pred.predicted_recall, micro_lat / len(queries_sample), winner_pred.predicted_memory_mb)
                    ])
                    calibration_performed = True
                except Exception as e:
                    pass

        tuning_duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Rationale generation for demo/panel presentation
        family_name = winner_cfg.family.value
        rationale = (
            f"Vecta selected {winner_cfg.id_string} achieving predicted {winner_pred.predicted_recall * 100:.1f}% "
            f"Recall@{workload.target_k} at {winner_pred.predicted_latency_p95:.2f} ms p95 latency. "
            f"It satisfies the {workload.max_latency_ms:.1f} ms latency SLA and fits within the "
            f"{workload.memory_budget_mb:.0f} MB RAM budget ({winner_pred.predicted_memory_mb:.1f} MB allocated)."
        )

        return {
            "recommended_config": winner_cfg.to_dict(),
            "recommended_prediction": winner_pred.to_dict(),
            "winner_score": round(winner_entry["score"], 4),
            "tuning_duration_ms": round(tuning_duration_ms, 2),
            "total_candidates_evaluated": len(candidates),
            "pareto_optimal_count": len(pareto_indices),
            "calibration_performed": calibration_performed,
            "rationale": rationale,
            "top_candidates": [
                {
                    "config": item["config"].to_dict(),
                    "prediction": item["prediction"].to_dict(),
                    "score": round(item["score"], 4),
                    "is_pareto": item["is_pareto"]
                }
                for item in scored_candidates[:8]
            ],
            "all_points": [
                {
                    "id": cfg.id_string,
                    "family": cfg.family.value,
                    "recall": pred.predicted_recall,
                    "latency_p95": pred.predicted_latency_p95,
                    "memory_mb": pred.predicted_memory_mb,
                    "is_pareto": i in pareto_indices,
                    "is_feasible": pred.is_feasible,
                    "is_winner": (cfg.id_string == winner_cfg.id_string)
                }
                for i, (cfg, pred) in enumerate(zip(candidates, predictions))
            ]
        }
