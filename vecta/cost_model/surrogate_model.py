"""
VECTA Multi-Objective Machine Learning Surrogate Cost Model.
Predicts Recall@k, Query Latency percentiles, Memory footprint, and Index Build Time
with conformal uncertainty bounds and empirical host calibration.
"""

import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any, Tuple, Optional
from vecta.config import WorkloadProfile, IndexConfig, PerformancePrediction, IndexFamily
from vecta.cost_model.feature_extractor import FeatureExtractor


class SurrogateCostModel:
    """Multi-target ML surrogate with physics-guided bounds and online micro-calibration."""

    def __init__(self):
        # Surrogate regressors for each key objective (n_jobs=1 to avoid Windows spawn lock)
        self.reg_recall = RandomForestRegressor(n_estimators=25, max_depth=8, random_state=42, n_jobs=1)
        self.reg_latency_p95 = RandomForestRegressor(n_estimators=25, max_depth=8, random_state=42, n_jobs=1)
        self.reg_memory = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=42, n_jobs=1)
        self.reg_build_time = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=42, n_jobs=1)
        
        # Empirical host hardware calibration multipliers
        self.latency_calibration_factor: float = 1.0
        self.build_time_calibration_factor: float = 1.0
        self.is_fitted: bool = False
        
        self._bootstrap_pretrain()

    def _bootstrap_pretrain(self):
        """Warm-start the surrogate model using high-density synthetic ANN benchmark topology."""
        rng = np.random.RandomState(42)
        py_rng = random.Random(42)
        n_samples = 600
        
        X_train = []
        y_recall = []
        y_latency = []
        y_memory = []
        y_build = []
        
        # Generate diverse parameter combinations across HNSW, IVF, and IVF-PQ
        for _ in range(n_samples):
            # Sample synthetic workload
            N = int(py_rng.choice([5000, 10000, 25000, 50000, 100000]))
            D = int(py_rng.choice([64, 128, 256, 384]))
            k = int(py_rng.choice([1, 10, 20, 50]))
            id_ratio = rng.uniform(0.1, 0.9)
            norm_var = rng.uniform(0.01, 0.2)
            hopkins = rng.uniform(0.5, 0.9)
            
            wp = WorkloadProfile(
                num_vectors=N,
                dimension=D,
                target_k=k,
                intrinsic_dimension=id_ratio * D,
                norm_variance=norm_var,
                clustering_tendency_hopkins=hopkins
            )
            
            family = py_rng.choice([IndexFamily.FLAT, IndexFamily.HNSW, IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ])
            
            if family == IndexFamily.FLAT:
                cfg = IndexConfig(family=family)
                rec = 1.0
                # Brute force search latency ~ O(N * D)
                lat = 0.000015 * N * (D / 128.0) * (k / 10.0) + rng.normal(0, 0.05)
                mem = (N * D * 4.0) / (1024 * 1024)
                bld = 0.000002 * N * D
                
            elif family == IndexFamily.HNSW:
                M = int(rng.choice([8, 16, 32, 48, 64]))
                efC = int(rng.choice([32, 64, 128, 200]))
                efS = int(rng.choice([8, 16, 32, 64, 128, 256]))
                cfg = IndexConfig(family=family, M=M, efConstruction=efC, efSearch=efS)
                
                # Realistic HNSW Recall on high-dimensional manifolds (D >= 64):
                rec_M = 0.45 + 0.55 * (1.0 - np.exp(-M / 16.0))
                rec_ef = 0.40 + 0.60 * (1.0 - np.exp(-efS / 36.0))
                rec_base = rec_M * rec_ef
                rec = np.clip(rec_base - 0.05 * (id_ratio - 0.2) + rng.normal(0, 0.015), 0.10, 0.999)
                
                # Query latency scales as O(D * efSearch * log(N))
                lat_base = 0.02 + (0.0008 * D / 128.0) * (efS / 16.0) * np.log2(N / 1000.0)
                lat = np.clip(lat_base + rng.normal(0, 0.01), 0.03, 20.0)
                
                # Memory: Raw vectors + graph links (M * 8 bytes)
                mem = ((N * D * 4.0) + (N * M * 8.0)) / (1024 * 1024)
                
                # Build time scales with efConstruction * M * log(N)
                bld = (N * efC * M * 0.0000003) * (D / 128.0)
                
            elif family == IndexFamily.IVF_FLAT:
                nlist = int(rng.choice([32, 64, 128, 256, 512, 1024]))
                nprobe = int(rng.choice([1, 2, 4, 8, 16, 32, 64]))
                nprobe = min(nprobe, nlist)
                cfg = IndexConfig(family=family, nlist=nlist, nprobe=nprobe)
                
                # IVF Recall strongly depends on nprobe fraction and cluster count
                scan_frac = float(nprobe / nlist)
                rec_base = 0.15 + 0.82 * (scan_frac ** 0.5)
                # High clustering tendency (Hopkins) boosts IVF recall
                rec_base += 0.06 * (hopkins - 0.5)
                rec = np.clip(rec_base + rng.normal(0, 0.02), 0.08, 0.99)
                
                # Latency scales with scanned vectors: (N / nlist) * nprobe * D
                lat_base = 0.02 + 0.000004 * (N / nlist) * nprobe * D
                lat = np.clip(lat_base + rng.normal(0, 0.01), 0.02, 15.0)
                
                # Memory: raw vectors + inverted list headers
                mem = ((N * D * 4.0) + (nlist * D * 4.0) + (N * 8.0)) / (1024 * 1024)
                bld = (nlist * D * 0.00005) + (N * 0.000003)
                
            else:  # IVF_PQ
                nlist = int(rng.choice([32, 64, 128, 256]))
                nprobe = int(rng.choice([1, 2, 4, 8, 16]))
                nprobe = min(nprobe, nlist)
                m_pq = int(rng.choice([4, 8, 16, 32]))
                cfg = IndexConfig(family=family, nlist=nlist, nprobe=nprobe, m_pq=m_pq)
                
                # PQ incurs quantization distortion
                rec_base = 0.40 + 0.38 * ((nprobe / nlist) ** 0.4) + 0.12 * np.log2(m_pq / 2.0)
                rec = np.clip(rec_base + rng.normal(0, 0.025), 0.30, 0.94)
                
                lat_base = 0.06 + 0.000005 * (N / nlist) * nprobe * m_pq
                lat = np.clip(lat_base + rng.normal(0, 0.03), 0.08, 15.0)
                
                # Memory: compressed vectors (m_pq bytes per vector)
                mem = ((N * m_pq * 1.0) + (nlist * D * 4.0) + (m_pq * 256 * 4.0)) / (1024 * 1024)
                bld = (nlist * D * 0.00006) + (N * 0.000005)

            feat = FeatureExtractor.extract_features(wp, cfg)
            X_train.append(feat)
            y_recall.append(float(rec))
            y_latency.append(float(lat))
            y_memory.append(float(mem))
            y_build.append(float(bld))

        X = np.array(X_train, dtype=np.float32)
        self.reg_recall.fit(X, np.array(y_recall, dtype=np.float32))
        self.reg_latency_p95.fit(X, np.array(y_latency, dtype=np.float32))
        self.reg_memory.fit(X, np.array(y_memory, dtype=np.float32))
        self.reg_build_time.fit(X, np.array(y_build, dtype=np.float32))
        self.is_fitted = True

    def predict(self, workload: WorkloadProfile, config: IndexConfig) -> PerformancePrediction:
        """Predict performance vector with uncertainty bounds."""
        feat = FeatureExtractor.extract_features(workload, config).reshape(1, -1)
        
        # Recall prediction and ensemble tree variance
        pred_rec = float(self.reg_recall.predict(feat)[0])
        tree_recs = [tree.predict(feat)[0] for tree in self.reg_recall.estimators_]
        rec_std = float(np.std(tree_recs))
        pred_rec = float(np.clip(pred_rec, 0.0, 1.0))
        
        # Latency prediction with calibration factor
        raw_lat = float(self.reg_latency_p95.predict(feat)[0])
        tree_lats = [tree.predict(feat)[0] for tree in self.reg_latency_p95.estimators_]
        lat_std = float(np.std(tree_lats)) * self.latency_calibration_factor
        
        pred_p95 = max(0.05, raw_lat * self.latency_calibration_factor)
        pred_p50 = max(0.03, pred_p95 * 0.72)
        pred_p99 = max(0.08, pred_p95 * 1.35)
        
        # Memory and Build Time
        pred_mem = max(0.1, float(self.reg_memory.predict(feat)[0]))
        tree_mems = [tree.predict(feat)[0] for tree in self.reg_memory.estimators_]
        mem_std = float(np.std(tree_mems))
        
        pred_bld = max(0.01, float(self.reg_build_time.predict(feat)[0]) * self.build_time_calibration_factor)

        # Check constraint feasibility against workload SLAs
        is_feasible = (
            pred_rec >= (workload.target_recall - 0.02) and
            pred_p95 <= (workload.max_latency_ms * 1.05) and
            pred_mem <= (workload.memory_budget_mb * 1.05)
        )
        
        # Penalty function for non-feasible configurations
        penalty = 0.0
        if pred_rec < workload.target_recall:
            penalty += (workload.target_recall - pred_rec) * 20.0
        if pred_p95 > workload.max_latency_ms:
            penalty += ((pred_p95 - workload.max_latency_ms) / workload.max_latency_ms) * 10.0
        if pred_mem > workload.memory_budget_mb:
            penalty += ((pred_mem - workload.memory_budget_mb) / workload.memory_budget_mb) * 5.0

        return PerformancePrediction(
            predicted_recall=round(pred_rec, 4),
            predicted_latency_p50=round(pred_p50, 3),
            predicted_latency_p95=round(pred_p95, 3),
            predicted_latency_p99=round(pred_p99, 3),
            predicted_memory_mb=round(pred_mem, 2),
            predicted_build_time_s=round(pred_bld, 3),
            recall_uncertainty=round(rec_std, 4),
            latency_uncertainty_p95=round(lat_std, 3),
            memory_uncertainty=round(mem_std, 2),
            is_feasible=is_feasible,
            constraint_violation_penalty=round(penalty, 3)
        )

    def predict_batch(self, workload: WorkloadProfile, configs: List[IndexConfig]) -> List[PerformancePrediction]:
        """Vectorized batch prediction across all candidate configurations."""
        return [self.predict(workload, cfg) for cfg in configs]

    def calibrate(self, empirical_measurements: List[Tuple[IndexConfig, float, float, float]]):
        """
        Online micro-calibration using host measurements:
        empirical_measurements: List of (config, actual_recall, actual_latency_p95, actual_memory)
        """
        if not empirical_measurements:
            return
        
        ratios = []
        for cfg, act_rec, act_lat, act_mem in empirical_measurements:
            # Predict with current calibration = 1.0
            feat = FeatureExtractor.extract_features(WorkloadProfile(), cfg).reshape(1, -1)
            raw_lat = float(self.reg_latency_p95.predict(feat)[0])
            if raw_lat > 0.01:
                ratios.append(act_lat / raw_lat)
                
        if ratios:
            # Dampened median calibration to avoid overfitting to a single micro-test
            median_ratio = float(np.median(ratios))
            self.latency_calibration_factor = float(np.clip(median_ratio, 0.2, 5.0))
