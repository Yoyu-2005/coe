/**
 * VECTA Vector Search Optimization Platform - Interactive Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    workload: {
      num_vectors: 15000,
      dimension: 128,
      metric: "L2",
      target_recall: 0.95,
      max_latency_ms: 1.5,
      memory_budget_mb: 128.0,
      expected_qps: 600.0,
      target_k: 10
    },
    paretoChart: null,
    arenaBarChart: null
  };

  // DOM Elements
  const el = {
    sliderVectors: document.getElementById('sliderVectors'),
    valVectors: document.getElementById('valVectors'),
    sliderDim: document.getElementById('sliderDim'),
    valDim: document.getElementById('valDim'),
    sliderRecall: document.getElementById('sliderRecall'),
    valRecall: document.getElementById('valRecall'),
    sliderLatency: document.getElementById('sliderLatency'),
    valLatency: document.getElementById('valLatency'),
    sliderMemory: document.getElementById('sliderMemory'),
    valMemory: document.getElementById('valMemory'),
    sliderK: document.getElementById('sliderK'),
    valK: document.getElementById('valK'),

    presetButtons: document.querySelectorAll('.preset-btn'),
    triggerAutoTuneBtn: document.getElementById('triggerAutoTuneBtn'),
    deployWinnerBtn: document.getElementById('deployWinnerBtn'),
    runArenaBtn: document.getElementById('runArenaBtn'),
    executeSearchBtn: document.getElementById('executeSearchBtn'),

    // Telemetry & Metrics
    telemetryCpu: document.getElementById('telemetryCpu'),
    telemetryRam: document.getElementById('telemetryRam'),
    mIntrinsicDim: document.getElementById('mIntrinsicDim'),
    mHopkins: document.getElementById('mHopkins'),
    mNormVar: document.getElementById('mNormVar'),
    mProfTime: document.getElementById('mProfTime'),

    // Recommendation Elements
    candidateCountBadge: document.getElementById('candidateCountBadge'),
    paretoCountBadge: document.getElementById('paretoCountBadge'),
    tuningDurationBadge: document.getElementById('tuningDurationBadge'),
    winnerConfigTitle: document.getElementById('winnerConfigTitle'),
    winnerFamilyTag: document.getElementById('winnerFamilyTag'),
    winnerRecall: document.getElementById('winnerRecall'),
    winnerRecallCi: document.getElementById('winnerRecallCi'),
    winnerLatency: document.getElementById('winnerLatency'),
    winnerLatencyCi: document.getElementById('winnerLatencyCi'),
    winnerMemory: document.getElementById('winnerMemory'),
    winnerBuildTime: document.getElementById('winnerBuildTime'),
    winnerRationaleText: document.getElementById('winnerRationaleText'),

    // Arena Table
    arenaVerdictBanner: document.getElementById('arenaVerdictBanner'),
    verdictText: document.getElementById('verdictText'),
    arenaTableBody: document.getElementById('arenaTableBody'),

    // Search Playground
    qLatencyVal: document.getElementById('qLatencyVal'),
    qGtMatchVal: document.getElementById('qGtMatchVal'),
    qTotalServed: document.getElementById('qTotalServed'),
    neighborsList: document.getElementById('neighborsList')
  };

  // Initialize System Telemetry & Event Bindings
  initTelemetry();
  bindEvents();

  // Run initial profile and tuning
  runWorkloadProfileAndTune();

  // Polling Telemetry every 5 seconds
  setInterval(initTelemetry, 5000);

  // -------------------------------------------------------------
  // Event Bindings
  // -------------------------------------------------------------
  function bindEvents() {
    // Slider Sync
    el.sliderVectors.addEventListener('input', (e) => {
      state.workload.num_vectors = parseInt(e.target.value, 10);
      el.valVectors.textContent = state.workload.num_vectors.toLocaleString();
    });
    el.sliderDim.addEventListener('input', (e) => {
      state.workload.dimension = parseInt(e.target.value, 10);
      el.valDim.textContent = state.workload.dimension;
    });
    el.sliderRecall.addEventListener('input', (e) => {
      state.workload.target_recall = parseFloat(e.target.value);
      el.valRecall.textContent = (state.workload.target_recall * 100).toFixed(1) + '%';
    });
    el.sliderLatency.addEventListener('input', (e) => {
      state.workload.max_latency_ms = parseFloat(e.target.value);
      el.valLatency.textContent = state.workload.max_latency_ms.toFixed(1) + ' ms';
    });
    el.sliderMemory.addEventListener('input', (e) => {
      state.workload.memory_budget_mb = parseFloat(e.target.value);
      el.valMemory.textContent = state.workload.memory_budget_mb.toFixed(0) + ' MB';
    });
    el.sliderK.addEventListener('input', (e) => {
      state.workload.target_k = parseInt(e.target.value, 10);
      el.valK.textContent = state.workload.target_k;
    });

    // Preset Buttons
    el.presetButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        el.presetButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        applyPreset(btn.dataset.preset);
      });
    });

    // Action Triggers
    el.triggerAutoTuneBtn.addEventListener('click', () => runWorkloadProfileAndTune());
    el.deployWinnerBtn.addEventListener('click', () => runArenaBenchmark());
    el.runArenaBtn.addEventListener('click', () => runArenaBenchmark());
    el.executeSearchBtn.addEventListener('click', () => executeLiveSearch());
  }

  // -------------------------------------------------------------
  // Preset Configuration
  // -------------------------------------------------------------
  function applyPreset(presetKey) {
    const presets = {
      rag_semantic_search: { vectors: 20000, dim: 128, recall: 0.98, latency: 2.0, memory: 180, k: 10 },
      recommendation_high_qps: { vectors: 25000, dim: 64, recall: 0.90, latency: 0.6, memory: 90, k: 20 },
      edge_memory_constrained: { vectors: 15000, dim: 128, recall: 0.88, latency: 2.5, memory: 35, k: 10 },
      sift_benchmark: { vectors: 15000, dim: 128, recall: 0.95, latency: 1.2, memory: 140, k: 10 }
    };
    const p = presets[presetKey] || presets.rag_semantic_search;
    
    state.workload.num_vectors = p.vectors;
    state.workload.dimension = p.dim;
    state.workload.target_recall = p.recall;
    state.workload.max_latency_ms = p.latency;
    state.workload.memory_budget_mb = p.memory;
    state.workload.target_k = p.k;

    el.sliderVectors.value = p.vectors;
    el.valVectors.textContent = p.vectors.toLocaleString();
    el.sliderDim.value = p.dim;
    el.valDim.textContent = p.dim;
    el.sliderRecall.value = p.recall;
    el.valRecall.textContent = (p.recall * 100).toFixed(1) + '%';
    el.sliderLatency.value = p.latency;
    el.valLatency.textContent = p.latency.toFixed(1) + ' ms';
    el.sliderMemory.value = p.memory;
    el.valMemory.textContent = p.memory + ' MB';
    el.sliderK.value = p.k;
    el.valK.textContent = p.k;

    runWorkloadProfileAndTune();
  }

  // -------------------------------------------------------------
  // Telemetry & Profiling
  // -------------------------------------------------------------
  async function initTelemetry() {
    try {
      const res = await fetch('/api/system');
      const data = await res.json();
      if (data.hardware) {
        el.telemetryCpu.textContent = `${data.hardware.logical_cpus} Threads`;
        el.telemetryRam.textContent = `${data.hardware.available_ram_gb} GB`;
      }
    } catch (err) {
      console.warn('Telemetry polling error:', err);
    }
  }

  async function runWorkloadProfileAndTune() {
    const t0 = performance.now();
    el.triggerAutoTuneBtn.innerHTML = 'Optimizing...';
    el.triggerAutoTuneBtn.disabled = true;

    try {
      // 1. Profile Workload
      const profRes = await fetch('/api/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.workload)
      });
      const profData = await profRes.json();
      const wp = profData.workload_profile;

      // Update Profile Metrics
      el.mIntrinsicDim.textContent = wp.intrinsic_dimension;
      el.mHopkins.textContent = wp.clustering_tendency_hopkins;
      el.mNormVar.textContent = wp.norm_variance;
      el.mProfTime.textContent = (performance.now() - t0).toFixed(1) + ' ms';

      // 2. Run Auto-Tuning
      const tuneRes = await fetch('/api/tune', { method: 'POST' });
      const tuneData = await tuneRes.json();

      updateTuningUI(tuneData);

      // 3. Run Benchmark automatically
      await runArenaBenchmark();

    } catch (err) {
      console.error('Tuning error:', err);
    } finally {
      el.triggerAutoTuneBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run Optimization';
      el.triggerAutoTuneBtn.disabled = false;
    }
  }

  function updateTuningUI(tuneData) {
    el.candidateCountBadge.textContent = `${tuneData.total_candidates_evaluated} Configurations`;
    el.paretoCountBadge.textContent = `${tuneData.pareto_optimal_count} Pareto Optimal`;
    el.tuningDurationBadge.textContent = `Tuned in ${tuneData.tuning_duration_ms} ms`;

    const winner = tuneData.recommended_config;
    const pred = tuneData.recommended_prediction;

    el.winnerConfigTitle.textContent = winner.id_string;
    el.winnerFamilyTag.textContent = winner.family === 'HNSW' 
      ? 'High-Exploration Graph-based Index' 
      : 'Voronoi Inverted File Cluster Index';

    el.winnerRecall.textContent = (pred.recall * 100).toFixed(1) + '%';
    el.winnerRecallCi.textContent = `± ${(pred.recall_ci_half_width * 100).toFixed(1)}% CI`;

    el.winnerLatency.textContent = pred.latency_p95_ms.toFixed(2) + ' ms';
    el.winnerLatencyCi.textContent = `± ${pred.latency_ci_half_width.toFixed(2)} ms CI`;

    el.winnerMemory.textContent = pred.memory_mb.toFixed(1) + ' MB';
    el.winnerBuildTime.textContent = pred.build_time_s.toFixed(2) + ' s';

    el.winnerRationaleText.textContent = tuneData.selection_rationale;

    // Render Pareto Frontier Scatter Plot
    renderParetoChart(tuneData.all_candidates, winner.id_string);
  }

  // -------------------------------------------------------------
  // Empirical Arena Benchmarking
  // -------------------------------------------------------------
  async function runArenaBenchmark() {
    el.runArenaBtn.disabled = true;
    el.runArenaBtn.textContent = 'Benchmarking...';
    el.deployWinnerBtn.disabled = true;
    el.deployWinnerBtn.textContent = 'Benchmarking Active Index...';

    try {
      const res = await fetch('/api/benchmark', { method: 'POST' });
      const data = await res.json();

      // Update Verdict
      el.verdictText.textContent = data.comparison_summary;

      // Update Table
      renderArenaTable(data.results);

      // Update Bar Chart
      renderArenaBarChart(data.results);

    } catch (err) {
      console.error('Benchmark error:', err);
    } finally {
      el.runArenaBtn.disabled = false;
      el.runArenaBtn.textContent = 'Re-run Benchmark';
      el.deployWinnerBtn.disabled = false;
      el.deployWinnerBtn.textContent = 'Deploy to Engine & Run Benchmark';
    }
  }

  function renderArenaTable(results) {
    const tbody = el.arenaTableBody;
    tbody.innerHTML = '';

    const items = [
      { key: 'ground_truth_flat', label: 'Exact Brute-Force (Flat L2)', data: results.ground_truth_flat, isWinner: false },
      { key: 'faiss_default_ivf', label: 'FAISS Default IVF (nlist=100, nprobe=1)', data: results.faiss_default_ivf, isWinner: false },
      { key: 'faiss_default_hnsw', label: 'FAISS Default HNSW (M=32, efS=16)', data: results.faiss_default_hnsw, isWinner: false },
      { key: 'vecta_auto_tuned', label: `VECTA Auto-Tuned (${results.vecta_auto_tuned.config.id_string})`, data: results.vecta_auto_tuned, isWinner: true }
    ];

    items.forEach(item => {
      const tr = document.createElement('tr');
      if (item.isWinner) tr.classList.add('winner-row');
      
      const rec = (item.data.recall_at_k * 100).toFixed(1) + '%';
      const p95 = (item.data.latency_p95_ms || 0.1).toFixed(2) + ' ms';
      const p99 = (item.data.latency_p99_ms || 0.15).toFixed(2) + ' ms';
      const qps = (item.data.throughput_qps || 1000).toLocaleString();
      const mem = item.data.memory_footprint_mb.toFixed(1) + ' MB';
      const bld = (item.data.index_build_time_s !== undefined) ? item.data.index_build_time_s.toFixed(2) + ' s' : '< 0.01 s';

      const recColor = item.data.recall_at_k > 0.9 ? '#10b981' : (item.data.recall_at_k > 0.5 ? '#f59e0b' : '#ef4444');

      tr.innerHTML = `
        <td>${item.label}</td>
        <td><span style="color: ${recColor}; font-weight: 600;">${rec}</span></td>
        <td>${p95}</td>
        <td>${p99}</td>
        <td>${qps}</td>
        <td>${mem}</td>
        <td>${bld}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------
  // Live Query Testing
  // -------------------------------------------------------------
  async function executeLiveSearch() {
    el.executeSearchBtn.disabled = true;
    el.executeSearchBtn.textContent = 'Executing Query...';

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k: state.workload.target_k, use_random_query: true })
      });
      const data = await res.json();

      el.qLatencyVal.textContent = data.query_latency_ms.toFixed(2) + ' ms';
      el.qGtMatchVal.textContent = `${data.exact_matches_in_top_k} / ${state.workload.target_k} (${data.accuracy_pct}%)`;
      el.qTotalServed.textContent = data.total_queries_served.toLocaleString();

      // Render Neighbors
      el.neighborsList.innerHTML = '';
      const dists = data.distances[0];
      const idxs = data.indices[0];

      idxs.slice(0, 8).forEach((id, i) => {
        const row = document.createElement('div');
        row.classList.add('neighbor-row');
        const dist = dists[i].toFixed(4);
        const sim = Math.max(0, 1.0 - dists[i] / 2.0).toFixed(3);
        row.innerHTML = `
          <span class="neighbor-id">Vector #${id}</span>
          <span class="neighbor-dist">Dist: ${dist}</span>
          <span class="neighbor-sim">Cosine Sim: ${sim}</span>
        `;
        el.neighborsList.appendChild(row);
      });

    } catch (err) {
      console.error('Search error:', err);
    } finally {
      el.executeSearchBtn.disabled = false;
      el.executeSearchBtn.textContent = 'Execute Test Query';
    }
  }

  // -------------------------------------------------------------
  // Chart.js: Pareto Frontier Scatter Plot
  // -------------------------------------------------------------
  function renderParetoChart(points, winnerId) {
    const ctx = document.getElementById('paretoChart').getContext('2d');

    const infeasiblePts = [];
    const paretoPts = [];
    let winnerPt = null;

    points.forEach(p => {
      const pt = { x: p.latency_p95, y: p.recall * 100, raw: p };
      if (p.is_winner || p.id === winnerId) {
        winnerPt = pt;
      } else if (p.is_pareto) {
        paretoPts.push(pt);
      } else {
        infeasiblePts.push(pt);
      }
    });

    const datasets = [
      {
        label: 'Sub-optimal / Exceeds SLA',
        data: infeasiblePts,
        backgroundColor: 'rgba(100, 116, 139, 0.4)',
        borderColor: 'rgba(100, 116, 139, 0.7)',
        pointRadius: 4,
        pointHoverRadius: 6
      },
      {
        label: 'Pareto Optimal Candidates',
        data: paretoPts,
        backgroundColor: 'rgba(59, 130, 246, 0.85)',
        borderColor: '#3b82f6',
        pointRadius: 5,
        pointHoverRadius: 8,
        showLine: false
      }
    ];

    if (winnerPt) {
      datasets.push({
        label: 'Selected Configuration',
        data: [winnerPt],
        backgroundColor: '#f59e0b',
        borderColor: '#ffffff',
        borderWidth: 2,
        pointRadius: 9,
        pointHoverRadius: 12,
        pointStyle: 'triangle'
      });
    }

    if (state.paretoChart) {
      state.paretoChart.destroy();
    }

    state.paretoChart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        scales: {
          x: {
            title: { 
              display: true, 
              text: 'Query Latency p95 (ms)', 
              color: '#94a3b8', 
              font: { family: 'Inter', size: 11, weight: 500 } 
            },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            title: { 
              display: true, 
              text: 'Recall (%)', 
              color: '#94a3b8', 
              font: { family: 'Inter', size: 11, weight: 500 } 
            },
            min: 10,
            max: 102,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            titleColor: '#38bdf8',
            bodyColor: '#f8fafc',
            borderColor: '#1e293b',
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: (ctx) => {
                const raw = ctx.raw.raw;
                return [
                  `Index: ${raw.id}`,
                  `Recall: ${(raw.recall * 100).toFixed(1)}%`,
                  `Latency p95: ${raw.latency_p95.toFixed(2)} ms`,
                  `Memory: ${raw.memory_mb.toFixed(1)} MB`
                ];
              }
            }
          }
        }
      }
    });
  }

  // -------------------------------------------------------------
  // Chart.js: Arena Comparative Bar Chart
  // -------------------------------------------------------------
  function renderArenaBarChart(results) {
    const ctx = document.getElementById('arenaBarChart').getContext('2d');

    const labels = ['Exact Flat', 'FAISS Default IVF', 'FAISS Default HNSW', 'VECTA Auto-Tuned'];
    const recallData = [
      100.0,
      results.faiss_default_ivf.recall_at_k * 100,
      results.faiss_default_hnsw.recall_at_k * 100,
      results.vecta_auto_tuned.recall_at_k * 100
    ];
    const latencyData = [
      results.ground_truth_flat.latency_p95_ms,
      results.faiss_default_ivf.latency_p95_ms,
      results.faiss_default_hnsw.latency_p95_ms,
      results.vecta_auto_tuned.latency_p95_ms
    ];

    if (state.arenaBarChart) {
      state.arenaBarChart.destroy();
    }

    state.arenaBarChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Recall@k (%)',
            data: recallData,
            backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#3b82f6'],
            borderRadius: 4,
            yAxisID: 'y'
          },
          {
            label: 'p95 Latency (ms)',
            data: latencyData,
            backgroundColor: 'rgba(148, 163, 184, 0.6)',
            borderRadius: 4,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          },
          y: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Recall (%)', color: '#94a3b8', font: { size: 10 } },
            min: 0,
            max: 105,
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          y1: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Latency (ms)', color: '#94a3b8', font: { size: 10 } },
            grid: { drawOnChartArea: false },
            ticks: { color: '#94a3b8' }
          }
        },
        plugins: {
          legend: {
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          }
        }
      }
    });
  }
});
