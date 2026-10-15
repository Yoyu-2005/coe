"""
VECTA Physics-Informed Feature Extractor.
Transforms workload geometry, hardware telemetry, and index hyper-parameters
into vectorized representations augmented with asymptotic computational complexity terms.
"""

import numpy as np
from typing import Dict, Any, List
from vecta.config import WorkloadProfile, IndexConfig, IndexFamily


class FeatureExtractor:
    """Vectorizes workload specifications and index candidates with theoretical complexity terms."""

    FEATURE_NAMES = [
        # Workload Geometry & Scale
        "log_num_vectors",
        "dimension",
        "log_dimension",
        "target_k",
        "intrinsic_dimension_ratio",  # intrinsic_dim / dimension
        "norm_variance",
        "clustering_tendency_hopkins",
        
        # Index Family One-Hot
        "is_flat",
        "is_hnsw",
        "is_ivf_flat",
        "is_ivf_pq",
        
        # HNSW Parameter Coordinates
        "hnsw_M",
        "hnsw_log_efConstruction",
        "hnsw_log_efSearch",
        
        # IVF Parameter Coordinates
        "ivf_log_nlist",
        "ivf_log_nprobe",
        "ivf_scan_fraction",  # nprobe / nlist
        
        # Product Quantization Features
        "pq_subquantizers",
        "pq_compression_ratio",
        
        # Physics-Informed Analytical Cost Terms
        "theoretical_complexity_hnsw",  # D * efSearch * log2(N)
        "theoretical_complexity_ivf",   # D * (N / nlist) * nprobe
        "theoretical_complexity_flat",  # D * N
        "theoretical_memory_bytes_per_vec"
    ]

    @staticmethod
    def extract_features(workload: WorkloadProfile, config: IndexConfig) -> np.ndarray:
        """Extract dense 1D feature array for surrogate ML prediction."""
        N = max(1, workload.num_vectors)
        D = max(1, workload.dimension)
        k = max(1, workload.target_k)
        
        log_N = np.log2(N)
        log_D = np.log2(D)
        id_ratio = workload.intrinsic_dimension / max(1.0, float(D))
        
        # One-hot encoding (support both Enum and string forms)
        fam_val = config.family.value if hasattr(config.family, 'value') else str(config.family)
        is_flat = 1.0 if fam_val == "FLAT" else 0.0
        is_hnsw = 1.0 if fam_val == "HNSW" else 0.0
        is_ivf_flat = 1.0 if fam_val == "IVF_FLAT" else 0.0
        is_ivf_pq = 1.0 if fam_val == "IVF_PQ" else 0.0
        
        # HNSW params
        M = float(config.M) if is_hnsw else 0.0
        log_efC = np.log2(max(1, config.efConstruction)) if is_hnsw else 0.0
        log_efS = np.log2(max(1, config.efSearch)) if is_hnsw else 0.0
        
        # IVF params
        nlist = max(1, config.nlist)
        nprobe = max(1, config.nprobe)
        log_nlist = np.log2(nlist) if (is_ivf_flat or is_ivf_pq) else 0.0
        log_nprobe = np.log2(nprobe) if (is_ivf_flat or is_ivf_pq) else 0.0
        scan_fraction = float(nprobe / nlist) if (is_ivf_flat or is_ivf_pq) else 0.0
        
        # PQ params
        m_pq = float(config.m_pq) if is_ivf_pq else 0.0
        compression_ratio = float((D * 4.0) / max(1.0, m_pq)) if is_ivf_pq else 1.0
        
        # Physics-informed analytical terms
        # Asymptotic search complexity for HNSW: O(D * efSearch * log2(N))
        th_hnsw = (D * config.efSearch * log_N) / 1000.0 if is_hnsw else 0.0
        # Asymptotic search complexity for IVF: O(D * (N / nlist) * nprobe)
        th_ivf = (D * (N / nlist) * nprobe) / 1000.0 if (is_ivf_flat or is_ivf_pq) else 0.0
        # Brute force Flat complexity: O(D * N)
        th_flat = (D * N) / 1000.0 if is_flat else 0.0
        
        # Bytes per vector memory theoretical
        if is_flat:
            th_mem_bytes = D * 4.0
        elif is_hnsw:
            th_mem_bytes = (D * 4.0) + (M * 8.0)
        elif is_ivf_flat:
            th_mem_bytes = (D * 4.0) + 8.0 + (nlist * D * 4.0 / N)
        elif is_ivf_pq:
            th_mem_bytes = m_pq + 8.0 + (nlist * D * 4.0 / N)
        else:
            th_mem_bytes = D * 4.0
            
        features = [
            log_N,
            float(D),
            log_D,
            float(k),
            id_ratio,
            workload.norm_variance,
            workload.clustering_tendency_hopkins,
            is_flat,
            is_hnsw,
            is_ivf_flat,
            is_ivf_pq,
            M,
            log_efC,
            log_efS,
            log_nlist,
            log_nprobe,
            scan_fraction,
            m_pq,
            compression_ratio,
            th_hnsw,
            th_ivf,
            th_flat,
            th_mem_bytes
        ]
        
        return np.array(features, dtype=np.float32)

    @staticmethod
    def extract_batch(workload: WorkloadProfile, configs: List[IndexConfig]) -> np.ndarray:
        """Extract features for multiple configurations in a single vectorized matrix."""
        matrix = [FeatureExtractor.extract_features(workload, cfg) for cfg in configs]
        return np.vstack(matrix)
