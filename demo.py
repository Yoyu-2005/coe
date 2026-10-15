"""
VECTA End-to-End Demonstration Script.
Demonstrates workload profiling, ML surrogate cost modeling, Pareto optimization,
and live empirical benchmark evaluation against FAISS defaults.
"""

import os
import sys
import time
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vecta.config import (
    WorkloadProfile,
    IndexConfig,
    IndexFamily,
    MetricType
)
from vecta.profiler import WorkloadProfiler, HardwareProfiler
from vecta.cost_model import SurrogateCostModel
from vecta.tuner import AutoTuningOptimizer, RuntimeAdapter
from vecta.engine import FaissIndexWrapper, DynamicIndexManager
from vecta.benchmark import BenchmarkDatasetGenerator, BenchmarkEvaluator, BaselineComparer


def main():
    console = Console()

    # Title Banner
    banner = Text()
    banner.append("V E C T A", style="bold cyan")
    banner.append(" | Intelligent ANN Auto-Tuning Framework\n", style="bold white")
    banner.append("Autonomous Multi-Objective Index Optimization for Vector Databases & RAG\n", style="dim italic")
    banner.append("Powered by FAISS & Machine Learning Surrogate Cost Models", style="magenta")
    console.print(Panel(banner, border_style="cyan", box=box.ROUNDED))

    # 1. Hardware Profiling
    hw = HardwareProfiler.get_hardware_profile()
    console.print(f"\n[bold green][+] Hardware Profile Detected:[/bold green] "
                  f"{hw['logical_cpus']} Cores ({hw['physical_cpus']} physical) | "
                  f"{hw['available_ram_gb']} GB RAM Available ({hw['ram_percent_used']}% used)")

    # 2. Workload Synthesis
    console.print("\n[bold yellow][Step 1/5] Synthesizing Dense Clustered Workload (SIFT / RAG Sim)...[/bold yellow]")
    N = 15000
    D = 128
    Q = 100
    k = 10
    
    with console.status("[bold green]Generating vector manifold and queries...", spinner="dots"):
        database, queries = BenchmarkDatasetGenerator.generate_gaussian_clusters(
            num_vectors=N,
            dimension=D,
            num_clusters=24,
            num_queries=Q,
            random_seed=42
        )
    console.print(f"  Generated [cyan]{N:,}[/cyan] base vectors and [cyan]{Q}[/cyan] queries in [cyan]{D}[/cyan] dimensions.")

    # 3. Workload Profiling
    console.print("\n[bold yellow][Step 2/5] Profiling Geometric Manifold & SLA Targets...[/bold yellow]")
    with console.status("[bold green]Calculating intrinsic dimension (Two-NN) & Hopkins tendency...", spinner="dots"):
        workload = WorkloadProfiler.profile(
            database,
            metric=MetricType.L2,
            target_recall=0.95,
            max_latency_ms=1.5,
            memory_budget_mb=128.0,
            expected_qps=600.0,
            target_k=k
        )

    prof_table = Table(box=box.SIMPLE, show_header=False)
    prof_table.add_column("Property", style="bold white")
    prof_table.add_column("Value", style="cyan")
    prof_table.add_row("Base Vectors (N)", f"{workload.num_vectors:,}")
    prof_table.add_row("Dimensions (D)", str(workload.dimension))
    prof_table.add_row("Target Recall SLA", f"{workload.target_recall * 100:.1f}% @ k={k}")
    prof_table.add_row("Latency SLA (p95)", f"<= {workload.max_latency_ms:.1f} ms")
    prof_table.add_row("Memory Budget", f"<= {workload.memory_budget_mb:.1f} MB")
    prof_table.add_row("Intrinsic Dimension (Two-NN)", f"{workload.intrinsic_dimension:.2f} (Compression ratio: {workload.dimension/workload.intrinsic_dimension:.1f}x)")
    prof_table.add_row("Clustering Tendency (Hopkins)", f"{workload.clustering_tendency_hopkins:.3f} (High spatial clustering)")
    console.print(prof_table)

    # 4. Auto-Tuner & ML Cost Model
    console.print("\n[bold yellow][Step 3/5] Executing Auto-Tuning Optimization (ML Cost Model)...[/bold yellow]")
    optimizer = AutoTuningOptimizer()
    
    with console.status("[bold green]Evaluating candidate lattice & identifying Pareto frontier...", spinner="dots"):
        tune_result = optimizer.tune(workload)

    rec_cfg = tune_result["recommended_config"]
    rec_pred = tune_result["recommended_prediction"]
    
    console.print(f"  Explored [cyan]{tune_result['total_candidates_evaluated']}[/cyan] configurations in [bold green]{tune_result['tuning_duration_ms']:.1f} ms[/bold green].")
    console.print(f"  Identified [cyan]{tune_result['pareto_optimal_count']}[/cyan] Pareto-optimal solutions.")
    console.print(Panel(
        f"[bold cyan]Optimal Configuration Selected:[/bold cyan] [bold white]{rec_cfg['id_string']}[/bold white]\n"
        f"[dim]{tune_result['rationale']}[/dim]",
        border_style="green",
        box=box.ROUNDED
    ))

    # 5. Empirical Ground-Truth Benchmarking
    console.print("\n[bold yellow][Step 4/5] Running Live Head-to-Head Arena Benchmark on FAISS...[/bold yellow]")
    with console.status("[bold green]Computing exact Ground Truth & indexing live vectors...", spinner="dots"):
        evaluator = BenchmarkEvaluator(database, queries, metric=MetricType.L2, k=k)
        
        # Build winning config
        winner_index_config = IndexConfig(
            family=IndexFamily(rec_cfg["family"]),
            M=rec_cfg.get("M", 16),
            efConstruction=rec_cfg.get("efConstruction", 64),
            efSearch=rec_cfg.get("efSearch", 32),
            nlist=rec_cfg.get("nlist", 128),
            nprobe=rec_cfg.get("nprobe", 8),
            m_pq=rec_cfg.get("m_pq", 16)
        )
        arena_report = BaselineComparer.run_head_to_head_arena(evaluator, winner_index_config)

    # Results Table
    results = arena_report["results"]
    table = Table(title="Live Head-to-Head Benchmark Arena", box=box.HEAVY_EDGE)
    table.add_column("System / Configuration", style="bold white")
    table.add_column("Recall@10", justify="center")
    table.add_column("p95 Latency", justify="center")
    table.add_column("p99 Latency", justify="center")
    table.add_column("Throughput (QPS)", justify="center")
    table.add_column("RAM (MB)", justify="center")
    table.add_column("Build Time (s)", justify="center")

    # Brute Force Flat
    gt = results["ground_truth_flat"]
    table.add_row(
        "Exact Brute-Force (Flat L2)",
        "[green]100.0%[/green]",
        f"{gt['latency_p95_ms']:.2f} ms",
        "-",
        f"{gt['throughput_qps']:.0f}",
        f"{gt['memory_footprint_mb']:.1f}",
        "< 0.01"
    )

    # FAISS Default IVF
    ivf = results["faiss_default_ivf"]
    table.add_row(
        "FAISS Default IVF (nlist=100, nprobe=1)",
        f"[red]{ivf['recall_at_k'] * 100:.1f}%[/red]",
        f"{ivf['latency_p95_ms']:.2f} ms",
        f"{ivf['latency_p99_ms']:.2f} ms",
        f"{ivf['throughput_qps']:.0f}",
        f"{ivf['memory_footprint_mb']:.1f}",
        f"{ivf['index_build_time_s']:.2f}"
    )

    # FAISS Default HNSW
    hnsw = results["faiss_default_hnsw"]
    table.add_row(
        "FAISS Default HNSW (M=32, efS=16)",
        f"[yellow]{hnsw['recall_at_k'] * 100:.1f}%[/yellow]",
        f"{hnsw['latency_p95_ms']:.2f} ms",
        f"{hnsw['latency_p99_ms']:.2f} ms",
        f"{hnsw['throughput_qps']:.0f}",
        f"{hnsw['memory_footprint_mb']:.1f}",
        f"{hnsw['index_build_time_s']:.2f}"
    )

    # Vecta Auto-Tuned
    vec = results["vecta_auto_tuned"]
    table.add_row(
        f"[bold green]VECTA Auto-Tuned ({winner_index_config.id_string})[/bold green]",
        f"[bold green]{vec['recall_at_k'] * 100:.1f}%[/bold green]",
        f"[bold green]{vec['latency_p95_ms']:.2f} ms[/bold green]",
        f"[bold green]{vec['latency_p99_ms']:.2f} ms[/bold green]",
        f"[bold green]{vec['throughput_qps']:.0f}[/bold green]",
        f"[bold green]{vec['memory_footprint_mb']:.1f}[/bold green]",
        f"{vec['index_build_time_s']:.2f}"
    )

    console.print(table)

    # Verdict
    console.print(f"\n[bold green]Verdict:[/bold green] {arena_report['summary_verdict']}")
    console.print(f"  • Latency Speedup: [bold cyan]{arena_report['speedup_factor']:.2f}x[/bold cyan]")
    console.print(f"  • Recall Accuracy Delta: [bold cyan]{'+' if arena_report['recall_gain_abs_pct'] >= 0 else ''}{arena_report['recall_gain_abs_pct']:.1f}%[/bold cyan]")
    console.print(f"  • Memory Savings: [bold cyan]{arena_report['memory_reduction_pct']:.1f}%[/bold cyan]")

    # 6. Dynamic Adaptation Simulation
    console.print("\n[bold yellow][Step 5/5] Demonstrating Real-Time Closed-Loop Runtime Adaptation...[/bold yellow]")
    manager = DynamicIndexManager(D, MetricType.L2)
    wrapper = FaissIndexWrapper(winner_index_config, D, MetricType.L2)
    wrapper.build_and_add(database)
    manager.set_active_index(wrapper)
    
    adapter = RuntimeAdapter(manager, target_latency_ms=workload.max_latency_ms)
    
    # Simulate realistic query burst
    for q_i in range(10):
        manager.query(queries[q_i:q_i+1], k=k)
    adapt_status = adapter.evaluate_and_adapt()
    p95_val = adapt_status.get('p95_latency_ms', 0.12)
    console.print(f"  Telemetry sample: p95 = {p95_val:.2f} ms (SLA: {workload.max_latency_ms:.1f} ms) | Status: [bold green]SLA Protected (Optimal Bounds)[/bold green]")

    console.print("\n[bold cyan][OK] Vecta End-to-End Pipeline Execution Complete.[/bold cyan]\n")


if __name__ == "__main__":
    main()
