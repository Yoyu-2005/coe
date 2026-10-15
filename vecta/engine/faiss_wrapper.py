"""
VECTA High-Performance FAISS Engine Wrapper.
Unifies IndexFlat, IndexHNSWFlat, IndexIVFFlat, and IndexIVFPQ with precise telemetry,
memory inspection, and ground-truth generation.
"""

import time
import tempfile
import os
import numpy as np
import faiss
from typing import Tuple, Optional, Dict, Any
from vecta.config import IndexConfig, IndexFamily, MetricType


class FaissIndexWrapper:
    """Encapsulates FAISS index construction, vector indexing, query execution, and serialization."""

    def __init__(self, config: IndexConfig, dimension: int, metric: MetricType = MetricType.L2):
        self.config = config
        self.dimension = dimension
        self.metric = metric
        self.faiss_metric = faiss.METRIC_L2 if metric == MetricType.L2 else faiss.METRIC_INNER_PRODUCT
        self.index: Optional[faiss.Index] = None
        self.is_trained: bool = False
        self.total_vectors: int = 0
        self.build_time_s: float = 0.0
        self.estimated_memory_mb: float = 0.0

    def build_and_add(self, vectors: np.ndarray) -> float:
        """Instantiate index, train if necessary, add vectors, and record build time."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = vectors.shape
        assert d == self.dimension, f"Vector dimension {d} does not match configured {self.dimension}"

        start_time = time.perf_counter()

        if self.config.family == IndexFamily.FLAT:
            if self.metric == MetricType.L2:
                self.index = faiss.IndexFlatL2(d)
            else:
                self.index = faiss.IndexFlatIP(d)
            self.index.add(vectors)
            self.is_trained = True

        elif self.config.family == IndexFamily.HNSW:
            self.index = faiss.IndexHNSWFlat(d, self.config.M, self.faiss_metric)
            self.index.hnsw.efConstruction = int(self.config.efConstruction)
            self.index.hnsw.efSearch = int(self.config.efSearch)
            self.index.add(vectors)
            self.is_trained = True

        elif self.config.family == IndexFamily.IVF_FLAT:
            quantizer = faiss.IndexFlatL2(d) if self.metric == MetricType.L2 else faiss.IndexFlatIP(d)
            # Ensure nlist does not exceed vector count
            effective_nlist = min(int(self.config.nlist), max(1, n // 8))
            self.index = faiss.IndexIVFFlat(quantizer, d, effective_nlist, self.faiss_metric)
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = int(min(self.config.nprobe, effective_nlist))
            self.is_trained = True

        elif self.config.family == IndexFamily.IVF_PQ:
            quantizer = faiss.IndexFlatL2(d) if self.metric == MetricType.L2 else faiss.IndexFlatIP(d)
            effective_nlist = min(int(self.config.nlist), max(1, n // 8))
            
            # Find largest valid divisor for m_pq that divides dimension
            valid_m = self.config.m_pq
            if d % valid_m != 0:
                for candidate in [32, 16, 8, 4, 2, 1]:
                    if d % candidate == 0 and candidate <= d:
                        valid_m = candidate
                        break
            
            self.index = faiss.IndexIVFPQ(
                quantizer, d, effective_nlist, valid_m, int(self.config.nbits_per_idx), self.faiss_metric
            )
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = int(min(self.config.nprobe, effective_nlist))
            self.is_trained = True

        else:
            raise ValueError(f"Unsupported IndexFamily: {self.config.family}")

        self.build_time_s = time.perf_counter() - start_time
        self.total_vectors = self.index.ntotal
        self.estimated_memory_mb = self._compute_memory_footprint_mb(n, d)
        return self.build_time_s

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Execute batch ANN search.
        Returns: (distances, indices, query_latency_ms)
        """
        if self.index is None:
            raise RuntimeError("Index has not been constructed yet.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        
        # Apply runtime query parameter settings if supported
        if self.config.family == IndexFamily.HNSW:
            self.index.hnsw.efSearch = int(self.config.efSearch)
        elif self.config.family in (IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ):
            self.index.nprobe = int(self.config.nprobe)

        t0 = time.perf_counter()
        distances, indices = self.index.search(queries, k)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return distances, indices, t_elapsed_ms

    def set_runtime_parameters(self, efSearch: Optional[int] = None, nprobe: Optional[int] = None):
        """Dynamic runtime parameter adjustment without index rebuild."""
        if efSearch is not None and self.config.family == IndexFamily.HNSW and self.index is not None:
            self.config.efSearch = int(efSearch)
            self.index.hnsw.efSearch = int(efSearch)
        if nprobe is not None and self.config.family in (IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ) and self.index is not None:
            self.config.nprobe = int(nprobe)
            self.index.nprobe = int(nprobe)

    def _compute_memory_footprint_mb(self, n: int, d: int) -> float:
        """
        Accurate calculation of index RAM consumption:
        Combines serialized byte size with structural pointer overhead.
        """
        if self.index is None:
            return 0.0
        
        # Method 1: Serialized disk footprint
        temp_file = None
        try:
            fd, temp_file = tempfile.mkstemp(suffix=".faiss")
            os.close(fd)
            faiss.write_index(self.index, temp_file)
            file_bytes = os.path.getsize(temp_file)
            serialized_mb = file_bytes / (1024 * 1024)
        except Exception:
            serialized_mb = 0.0
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

        # Method 2: Analytical verification
        # Raw float32 vectors = N * D * 4 bytes
        raw_vec_mb = (n * d * 4.0) / (1024 * 1024)
        
        if self.config.family == IndexFamily.FLAT:
            analytical_mb = raw_vec_mb
        elif self.config.family == IndexFamily.HNSW:
            # HNSW overhead: node connections = N * M * 2 * 4 bytes + raw vector table
            graph_overhead_mb = (n * self.config.M * 8.0) / (1024 * 1024)
            analytical_mb = raw_vec_mb + graph_overhead_mb
        elif self.config.family == IndexFamily.IVF_FLAT:
            # Centroid table + inverted list pointer metadata
            centroid_mb = (self.config.nlist * d * 4.0) / (1024 * 1024)
            analytical_mb = raw_vec_mb + centroid_mb + (n * 8.0) / (1024 * 1024)
        elif self.config.family == IndexFamily.IVF_PQ:
            # Quantized codes: N * m_pq bytes + codebook
            code_mb = (n * self.config.m_pq * 1.0) / (1024 * 1024)
            codebook_mb = (self.config.m_pq * 256 * (d // max(1, self.config.m_pq)) * 4.0) / (1024 * 1024)
            analytical_mb = code_mb + codebook_mb + (n * 8.0) / (1024 * 1024)
        else:
            analytical_mb = raw_vec_mb

        if serialized_mb > 0:
            return round(max(serialized_mb, analytical_mb), 2)
        return round(analytical_mb, 2)
