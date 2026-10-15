"""
VECTA Mathematical Workload Profiler.
Analyzes geometric manifold properties, intrinsic dimensionality, clustering tendency,
and operational SLA parameters for high-dimensional vector search.
"""

import numpy as np
from typing import Optional, Dict, Any
from vecta.config import WorkloadProfile, MetricType
from vecta.profiler.hardware_profiler import HardwareProfiler


class WorkloadProfiler:
    """Rigorous mathematical feature extraction from vector collections and query traffic."""

    @staticmethod
    def estimate_intrinsic_dimension_two_nn(vectors: np.ndarray, max_samples: int = 500) -> float:
        """
        Estimate Intrinsic Dimensionality using the Two-NN algorithm (Facco et al., 2017).
        Calculates ratio mu_i = r_{i,2} / r_{i,1} of distance to 2nd vs 1st nearest neighbor.
        Under a local Poisson process on a manifold of dimension d,
        d = N / sum(ln(mu_i)).
        """
        n_vectors = vectors.shape[0]
        if n_vectors < 10:
            return float(vectors.shape[1])
        
        # Subsample to keep profiling latency sub-millisecond (< 5ms)
        sample_size = min(n_vectors, max_samples)
        indices = np.random.choice(n_vectors, size=sample_size, replace=False)
        sample = vectors[indices].astype(np.float32)
        
        # Compute pairwise Euclidean distance matrix for the sample
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        sq_norms = np.sum(sample ** 2, axis=1, keepdims=True)
        dist_sq = sq_norms + sq_norms.T - 2.0 * np.dot(sample, sample.T)
        np.maximum(dist_sq, 0, out=dist_sq)
        dists = np.sqrt(dist_sq)
        
        # Set diagonal to infinity so points don't match themselves
        np.fill_diagonal(dists, np.inf)
        
        # Sort distances along each row
        sorted_dists = np.sort(dists, axis=1)
        r1 = sorted_dists[:, 0]
        r2 = sorted_dists[:, 1]
        
        # Filter near-zero duplicates to prevent division by zero
        valid = (r1 > 1e-7) & (r2 > r1)
        if np.sum(valid) < 5:
            return float(vectors.shape[1])
        
        mu = r2[valid] / r1[valid]
        log_mu = np.log(mu)
        sum_log_mu = np.sum(log_mu)
        
        if sum_log_mu <= 1e-7:
            return float(vectors.shape[1])
        
        id_est = float(len(mu) / sum_log_mu)
        # Intrinsic dimension cannot exceed embedding dimension
        return float(np.clip(id_est, 1.0, float(vectors.shape[1])))

    @staticmethod
    def estimate_hopkins_statistic(vectors: np.ndarray, num_samples: int = 150) -> float:
        """
        Hopkins Statistic (H) measures clustering tendency in the vector space.
        H close to 0.5 indicates a uniform random distribution (better for HNSW graph traversal).
        H close to 1.0 indicates strong spatial cluster formations (highly favorable for IVF centroids).
        """
        n, d = vectors.shape
        m = min(num_samples, n // 3)
        if m < 5:
            return 0.5
        
        # Subsample real points
        real_idx = np.random.choice(n, size=m, replace=False)
        real_sample = vectors[real_idx]
        
        # Generate artificial uniform points across bounding box
        mins = np.min(vectors, axis=0)
        maxs = np.max(vectors, axis=0)
        synthetic_sample = np.random.uniform(mins, maxs, size=(m, d)).astype(np.float32)
        
        # Compute distances from synthetic points to nearest real points (w)
        # and from real sample to nearest real points (u)
        # Using a representative batch
        bg_idx = np.random.choice(n, size=min(1000, n), replace=False)
        bg = vectors[bg_idx]
        
        # Distance synthetic -> bg
        synth_norm = np.sum(synthetic_sample ** 2, axis=1, keepdims=True)
        bg_norm = np.sum(bg ** 2, axis=1, keepdims=True)
        dist_w_sq = synth_norm + bg_norm.T - 2 * np.dot(synthetic_sample, bg.T)
        np.maximum(dist_w_sq, 0, out=dist_w_sq)
        w = np.min(np.sqrt(dist_w_sq), axis=1)
        
        # Distance real_sample -> bg (exclude self if in bg)
        real_norm = np.sum(real_sample ** 2, axis=1, keepdims=True)
        dist_u_sq = real_norm + bg_norm.T - 2 * np.dot(real_sample, bg.T)
        np.maximum(dist_u_sq, 0, out=dist_u_sq)
        dist_u = np.sqrt(dist_u_sq)
        dist_u[dist_u < 1e-7] = np.inf
        u = np.min(dist_u, axis=1)
        
        sum_w = np.sum(w)
        sum_u = np.sum(u)
        if sum_w + sum_u == 0:
            return 0.5
        
        h = float(sum_w / (sum_w + sum_u))
        return float(np.clip(h, 0.0, 1.0))

    @staticmethod
    def profile(
        vectors: np.ndarray,
        metric: MetricType = MetricType.L2,
        target_recall: float = 0.95,
        max_latency_ms: float = 2.0,
        memory_budget_mb: float = 256.0,
        expected_qps: float = 500.0,
        target_k: int = 10
    ) -> WorkloadProfile:
        """Conduct comprehensive geometric profiling of dataset and SLAs."""
        num_vectors, dimension = vectors.shape
        norms = np.linalg.norm(vectors, axis=1)
        norm_mean = float(np.mean(norms))
        norm_variance = float(np.var(norms))
        sparsity_ratio = float(np.mean(vectors == 0))
        
        # Calculate geometric manifold indicators
        intrinsic_dim = WorkloadProfiler.estimate_intrinsic_dimension_two_nn(vectors)
        hopkins = WorkloadProfiler.estimate_hopkins_statistic(vectors)
        
        hw = HardwareProfiler.get_hardware_profile()
        
        return WorkloadProfile(
            num_vectors=num_vectors,
            dimension=dimension,
            metric=metric,
            target_k=target_k,
            target_recall=target_recall,
            max_latency_ms=max_latency_ms,
            memory_budget_mb=memory_budget_mb,
            expected_qps=expected_qps,
            intrinsic_dimension=round(intrinsic_dim, 2),
            norm_mean=round(norm_mean, 4),
            norm_variance=round(norm_variance, 4),
            sparsity_ratio=round(sparsity_ratio, 4),
            clustering_tendency_hopkins=round(hopkins, 4),
            cpu_count=hw['logical_cpus'],
            available_ram_gb=hw['available_ram_gb']
        )

    @staticmethod
    def get_preset(preset_name: str) -> Dict[str, Any]:
        """Predefined industry and research benchmark workloads for quick demo presentations."""
        presets = {
            "rag_semantic_search": {
                "name": "RAG Semantic Search (Dense Manifold)",
                "description": "High-fidelity text embeddings (e.g. BGE, OpenAI). Strict recall SLA, moderate latency tolerance.",
                "num_vectors": 30000,
                "dimension": 128,
                "metric": "L2",
                "target_recall": 0.98,
                "max_latency_ms": 2.5,
                "memory_budget_mb": 180.0,
                "expected_qps": 350.0,
                "target_k": 10,
                "preset_key": "rag_semantic_search"
            },
            "recommendation_high_qps": {
                "name": "High-Throughput E-Commerce Recommender",
                "description": "Extreme query traffic (QPS > 1000). Needs sub-millisecond p95 latency; 90% recall is acceptable.",
                "num_vectors": 50000,
                "dimension": 64,
                "metric": "IP",
                "target_recall": 0.90,
                "max_latency_ms": 0.8,
                "memory_budget_mb": 120.0,
                "expected_qps": 1200.0,
                "target_k": 20,
                "preset_key": "recommendation_high_qps"
            },
            "edge_memory_constrained": {
                "name": "Edge / Resource-Constrained Vector Search",
                "description": "Memory footprint strictly budgeted (< 40 MB) for on-device or micro-node deployment. Quantization critical.",
                "num_vectors": 40000,
                "dimension": 128,
                "metric": "L2",
                "target_recall": 0.88,
                "max_latency_ms": 3.0,
                "memory_budget_mb": 35.0,
                "expected_qps": 200.0,
                "target_k": 10,
                "preset_key": "edge_memory_constrained"
            },
            "sift_benchmark": {
                "name": "SIFT100K Computer Vision Benchmark",
                "description": "Standard 128-dimensional SIFT feature descriptors. Clustered visual bag-of-words geometry.",
                "num_vectors": 25000,
                "dimension": 128,
                "metric": "L2",
                "target_recall": 0.95,
                "max_latency_ms": 1.5,
                "memory_budget_mb": 150.0,
                "expected_qps": 500.0,
                "target_k": 10,
                "preset_key": "sift_benchmark"
            }
        }
        return presets.get(preset_name, presets["rag_semantic_search"])
