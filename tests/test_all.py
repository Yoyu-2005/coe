import os
import sys
import unittest
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vecta.config import (
    WorkloadProfile,
    IndexConfig,
    IndexFamily,
    MetricType
)
from vecta.profiler.workload_profiler import WorkloadProfiler
from vecta.profiler.hardware_profiler import HardwareProfiler
from vecta.engine.faiss_wrapper import FaissIndexWrapper
from vecta.engine.dynamic_index import DynamicIndexManager
from vecta.cost_model.surrogate_model import SurrogateCostModel
from vecta.tuner.optimizer import AutoTuningOptimizer
from vecta.benchmark.dataset_generator import BenchmarkDatasetGenerator
from vecta.benchmark.evaluator import BenchmarkEvaluator
from vecta.benchmark.baseline_comparer import BaselineComparer


class TestVecta(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = 64
        cls.n = 1000
        cls.q = 20
        cls.db, cls.queries = BenchmarkDatasetGenerator.generate_gaussian_clusters(
            num_vectors=cls.n,
            dimension=cls.d,
            num_clusters=8,
            num_queries=cls.q,
            random_seed=42
        )

    def test_hardware_profiler(self):
        hw = HardwareProfiler.get_hardware_profile()
        self.assertIn("logical_cpus", hw)
        self.assertIn("available_ram_gb", hw)
        self.assertGreater(hw["logical_cpus"], 0)

    def test_workload_profiler(self):
        wp = WorkloadProfiler.profile(
            self.db,
            metric=MetricType.L2,
            target_recall=0.95,
            max_latency_ms=2.0
        )
        self.assertEqual(wp.num_vectors, self.n)
        self.assertEqual(wp.dimension, self.d)
        self.assertGreater(wp.intrinsic_dimension, 0.0)
        self.assertLessEqual(wp.intrinsic_dimension, self.d)
        self.assertGreaterEqual(wp.clustering_tendency_hopkins, 0.0)

    def test_faiss_hnsw_and_dynamic_update(self):
        cfg = IndexConfig(family=IndexFamily.HNSW, M=16, efConstruction=32, efSearch=16)
        wrapper = FaissIndexWrapper(cfg, self.d, MetricType.L2)
        build_time = wrapper.build_and_add(self.db)
        self.assertGreater(build_time, 0.0)
        self.assertEqual(wrapper.total_vectors, self.n)

        # Query
        dists, idxs, lat = wrapper.search(self.queries[:5], k=10)
        self.assertEqual(dists.shape, (5, 10))
        self.assertEqual(idxs.shape, (5, 10))

        # Dynamic parameter adjustment
        wrapper.set_runtime_parameters(efSearch=64)
        self.assertEqual(wrapper.config.efSearch, 64)

    def test_faiss_ivf_flat_and_pq(self):
        # IVF Flat
        ivf_cfg = IndexConfig(family=IndexFamily.IVF_FLAT, nlist=32, nprobe=4)
        ivf_wrap = FaissIndexWrapper(ivf_cfg, self.d, MetricType.L2)
        ivf_wrap.build_and_add(self.db)
        dists, idxs, _ = ivf_wrap.search(self.queries[:5], k=10)
        self.assertEqual(idxs.shape, (5, 10))

        # IVF PQ
        pq_cfg = IndexConfig(family=IndexFamily.IVF_PQ, nlist=32, nprobe=4, m_pq=8)
        pq_wrap = FaissIndexWrapper(pq_cfg, self.d, MetricType.L2)
        pq_wrap.build_and_add(self.db)
        dists, idxs, _ = pq_wrap.search(self.queries[:5], k=10)
        self.assertEqual(idxs.shape, (5, 10))
        self.assertLess(pq_wrap.estimated_memory_mb, ivf_wrap.estimated_memory_mb)

    def test_surrogate_cost_model(self):
        model = SurrogateCostModel()
        wp = WorkloadProfile(num_vectors=10000, dimension=64)
        cfg = IndexConfig(family=IndexFamily.HNSW, M=16, efConstruction=64, efSearch=32)
        pred = model.predict(wp, cfg)
        
        self.assertGreaterEqual(pred.predicted_recall, 0.0)
        self.assertLessEqual(pred.predicted_recall, 1.0)
        self.assertGreater(pred.predicted_latency_p95, 0.0)
        self.assertGreater(pred.predicted_memory_mb, 0.0)

    def test_auto_tuning_optimizer(self):
        optimizer = AutoTuningOptimizer()
        wp = WorkloadProfile(
            num_vectors=10000,
            dimension=64,
            target_recall=0.92,
            max_latency_ms=1.5,
            memory_budget_mb=50.0
        )
        tune_res = optimizer.tune(wp)
        
        self.assertIn("recommended_config", tune_res)
        self.assertIn("all_points", tune_res)
        self.assertGreater(tune_res["pareto_optimal_count"], 0)
        self.assertGreater(len(tune_res["all_points"]), 20)

    def test_benchmark_evaluator_and_baseline_comparer(self):
        evaluator = BenchmarkEvaluator(self.db, self.queries, metric=MetricType.L2, k=10)
        self.assertIsNotNone(evaluator.gt_indices)
        self.assertEqual(evaluator.gt_indices.shape, (self.q, 10))

        test_cfg = IndexConfig(family=IndexFamily.HNSW, M=16, efConstruction=32, efSearch=16)
        arena_res = BaselineComparer.run_head_to_head_arena(evaluator, test_cfg)
        
        self.assertIn("speedup_factor", arena_res)
        self.assertIn("summary_verdict", arena_res)
        self.assertIn("results", arena_res)
        self.assertIn("vecta_auto_tuned", arena_res["results"])


if __name__ == "__main__":
    unittest.main()
