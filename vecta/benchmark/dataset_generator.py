"""
VECTA Synthetic and Standard Benchmark Dataset Generators.
Generates clustered Gaussian manifolds, anisotropic dense embeddings,
and uniform hypersphere distributions with deterministic repeatability.
"""

import numpy as np
from typing import Tuple, Optional


class BenchmarkDatasetGenerator:
    """Generates synthetic high-dimensional vector spaces replicating real-world workloads."""

    @staticmethod
    def generate_gaussian_clusters(
        num_vectors: int = 20000,
        dimension: int = 128,
        num_clusters: int = 32,
        cluster_std: float = 0.25,
        num_queries: int = 100,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multimodal clustered vector space (similar to SIFT/GIST visual features).
        Returns: (database_vectors, query_vectors)
        """
        rng = np.random.RandomState(random_seed)
        
        # Centroids uniformly placed on unit hypersphere
        centroids = rng.randn(num_clusters, dimension).astype(np.float32)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        
        # Distribute vectors across clusters
        cluster_assignments = rng.choice(num_clusters, size=num_vectors)
        offsets = rng.randn(num_vectors, dimension).astype(np.float32) * cluster_std
        database = centroids[cluster_assignments] + offsets
        
        # Normalize to unit length for stability
        database /= np.linalg.norm(database, axis=1, keepdims=True)
        
        # Sample queries from perturbed cluster centers
        query_assignments = rng.choice(num_clusters, size=num_queries)
        q_offsets = rng.randn(num_queries, dimension).astype(np.float32) * (cluster_std * 1.2)
        queries = centroids[query_assignments] + q_offsets
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        
        return database.astype(np.float32), queries.astype(np.float32)

    @staticmethod
    def generate_semantic_embeddings(
        num_vectors: int = 20000,
        dimension: int = 128,
        intrinsic_dim: int = 16,
        num_queries: int = 100,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate anisotropic dense embeddings typical of LLM/Transformer models (e.g. OpenAI, BERT).
        Vectors lie predominantly on a lower-dimensional latent subspace with exponential eigenvalue decay.
        """
        rng = np.random.RandomState(random_seed)
        
        # Latent subspace projection matrix
        latent = rng.randn(num_vectors, intrinsic_dim).astype(np.float32)
        # Decay eigenvalues: lambda_i = exp(-i / 4)
        weights = np.exp(-np.arange(intrinsic_dim) / 4.0).astype(np.float32)
        latent *= weights
        
        # Projection to full dimension
        proj_matrix = rng.randn(intrinsic_dim, dimension).astype(np.float32)
        proj_matrix /= np.linalg.norm(proj_matrix, axis=1, keepdims=True)
        
        database = np.dot(latent, proj_matrix)
        # Add slight ambient isotropic noise
        database += 0.05 * rng.randn(num_vectors, dimension).astype(np.float32)
        database /= np.linalg.norm(database, axis=1, keepdims=True)
        
        # Generate queries from the same manifold
        q_latent = rng.randn(num_queries, intrinsic_dim).astype(np.float32) * weights
        queries = np.dot(q_latent, proj_matrix) + 0.05 * rng.randn(num_queries, dimension).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        
        return database.astype(np.float32), queries.astype(np.float32)
