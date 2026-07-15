/* SYNTH Benchmark — interactive GitHub Pages dashboard */

const GENERATORS = [
  "CTGAN", "CopulaGAN", "TVAE", "GaussianCopula",
  "WGAN_GP", "CTABGAN", "TabDDPM", "ForestDiffusion",
];

const PLOTLY_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#e7ecf3", family: "Inter, system-ui, sans-serif", size: 12 },
  margin: { l: 50, r: 20, t: 40, b: 80 },
  xaxis: { gridcolor: "#2d3a4f", zerolinecolor: "#2d3a4f" },
  yaxis: { gridcolor: "#2d3a4f", zerolinecolor: "#2d3a4f" },
  legend: { bgcolor: "rgba(0,0,0,0)" },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: true, displaylogo: false };

let DATA = {};

async function loadJSON(name) {
  const url = `data/${name}`;
  try {
    const res = await fetch(url);
    if (!res.ok) return name === "meta.json" ? {} : (name === "statistics.json" ? {} : []);
    return res.json();
  } catch {
    return name.endsWith(".json") && name.includes("stat") ? {} : [];
  }
}

async function loadAllData() {
  const [meta, utilityAgg, utilityClf, utilityGaps, fidelity, privacy, tradeoff,
         weighted, borda, statistics, coverage] = await Promise.all([
    loadJSON("meta.json"),
    loadJSON("utility_agg.json"),
    loadJSON("utility_classifier.json"),
    loadJSON("utility_gaps.json"),
    loadJSON("fidelity.json"),
    loadJSON("privacy.json"),
    loadJSON("tradeoff.json"),
    loadJSON("rankings_weighted.json"),
    loadJSON("rankings_borda.json"),
    loadJSON("statistics.json"),
    loadJSON("coverage.json"),
  ]);

  DATA = {
    meta: meta || {},
    utilityAgg: utilityAgg || [],
    utilityClf: utilityClf || [],
    utilityGaps: utilityGaps || [],
    fidelity: fidelity || [],
    privacy: privacy || [],
    tradeoff: tradeoff || [],
    weighted: weighted || [],
    borda: borda || [],
    statistics: statistics || {},
    coverage: coverage || [],
  };
}

function unique(arr, key) {
  return [...new Set(arr.map(r => r[key]).filter(Boolean))].sort();
}

function plot(id, traces, layout = {}, config = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  Plotly.newPlot(id, traces, { ...PLOTLY_LAYOUT, ...layout }, { ...PLOTLY_CONFIG, ...config });
}

function fillSelect(id, options, defaultVal) {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = options.map(o => `<option value="${o}">${o}</option>`).join("");
  if (defaultVal && options.includes(defaultVal)) sel.value = defaultVal;
}

function renderMetrics() {
  const m = DATA.meta;
  document.getElementById("metric-datasets").textContent = m.n_datasets ?? "—";
  document.getElementById("metric-generators").textContent = (m.generators || GENERATORS).length;
  document.getElementById("metric-files").textContent = m.n_files ?? "—";
  document.getElementById("metric-utility").textContent = m.n_utility_rows?.toLocaleString() ?? "—";
}

function renderOverview() {
  const gaps = DATA.utilityGaps.filter(r => r.Metric === "Accuracy_Drop" || r.Metric === "R2_Drop");
  const datasets = unique(gaps, "Dataset");
  const generators = GENERATORS.filter(g => gaps.some(r => r.Generator === g));

  const z = generators.map(gen =>
    datasets.map(ds => {
      const row = gaps.find(r => r.Generator === gen && r.Dataset === ds);
      return row ? row.Mean : null;
    })
  );

  plot("overview-heatmap", [{
    type: "heatmap",
    x: datasets.map(d => d.replace(/^\d+\.\s*/, "")),
    y: generators,
    z,
    colorscale: "RdYlGn",
    reversescale: true,
    colorbar: { title: "Utility gap" },
  }], { title: "Utility gap heatmap (lower = better)", height: 460 });

  // Coverage
  const cov = DATA.coverage;
  if (cov.length) {
    const dsU = unique(cov, "Dataset");
    const zCov = GENERATORS.map(g =>
      dsU.map(ds => {
        const r = cov.find(x => x.Generator === g && x.Dataset === ds);
        return r ? r.Available : 0;
      })
    );
    plot("overview-coverage", [{
      type: "heatmap",
      x: dsU.map(d => d.replace(/^\d+\.\s*/, "")),
      y: GENERATORS,
      z: zCov,
      colorscale: [[0, "#2d3a4f"], [1, "#10b981"]],
      zmin: 0, zmax: 1,
      colorbar: { title: "Available" },
    }], { title: "Generator coverage", height: 400 });
  }
}

function renderUtility() {
  const task = document.getElementById("filter-task")?.value || "classification";
  const metric = document.getElementById("filter-metric")?.value || "Accuracy";
  const dataset = document.getElementById("filter-dataset")?.value;

  const rows = DATA.utilityAgg.filter(r =>
    r.TaskType === task && r.Metric === metric && (!dataset || r.Dataset === dataset)
  );

  const datasets = unique(DATA.utilityAgg.filter(r => r.TaskType === task), "Dataset");
  fillSelect("filter-dataset", ["All datasets", ...datasets], dataset || "All datasets");

  // TRTR vs TSTR grouped bar
  const gens = GENERATORS.filter(g => rows.some(r => r.Generator === g));
  const trtr = gens.map(g => {
    const r = rows.find(x => x.Generator === g && x.EvaluationType === "TRTR");
    return r ? r.Mean : null;
  });
  const tstr = gens.map(g => {
    const r = rows.find(x => x.Generator === g && x.EvaluationType === "TSTR");
    return r ? r.Mean : null;
  });

  plot("utility-trtr-tstr", [
    { x: gens, y: trtr, name: "TRTR", type: "bar", marker: { color: "#3b82f6" } },
    { x: gens, y: tstr, name: "TSTR", type: "bar", marker: { color: "#10b981" } },
  ], { barmode: "group", title: `${metric}: TRTR vs TSTR (${task})`, height: 440 });

  // Classifier detail
  const clfRows = DATA.utilityClf.filter(r =>
    r.Metric === metric && (!dataset || dataset === "All datasets" || r.Dataset === dataset)
  );
  const gen = document.getElementById("filter-generator")?.value || GENERATORS[0];
  fillSelect("filter-generator", GENERATORS.filter(g => clfRows.some(r => r.Generator === g)), gen);

  const ds = dataset && dataset !== "All datasets" ? dataset : (unique(clfRows, "Dataset")[0] || "");
  const detail = clfRows.filter(r => r.Generator === gen && r.Dataset === ds);
  const models = unique(detail, "Classifier");
  plot("utility-classifier", [
    { x: models, y: models.map(m => detail.find(r => r.Classifier === m && r.EvaluationType === "TRTR")?.Mean), name: "TRTR", type: "bar", marker: { color: "#3b82f6" } },
    { x: models, y: models.map(m => detail.find(r => r.Classifier === m && r.EvaluationType === "TSTR")?.Mean), name: "TSTR", type: "bar", marker: { color: "#10b981" } },
  ], { barmode: "group", title: `${ds} — ${gen}: ${metric} by classifier`, height: 440, xaxis: { tickangle: -45 } });
}

function renderFidelity() {
  const rows = DATA.fidelity;
  if (!rows.length) return;

  const metricKey = rows[0].Metric ? "Metric" : null;
  const scoreKey = rows[0].Mean !== undefined ? "Mean" : "Value";
  const byGen = {};
  rows.forEach(r => {
    const g = r.Generator;
    if (!g) return;
    byGen[g] = (byGen[g] || 0) + (r[scoreKey] || 0);
  });
  const gens = Object.keys(byGen).sort((a, b) => byGen[b] - byGen[a]);
  plot("fidelity-bar", [{
    x: gens, y: gens.map(g => byGen[g] / rows.filter(r => r.Generator === g).length),
    type: "bar", marker: { color: gens.map((_, i) => `hsl(${200 + i * 18}, 70%, 55%)`) },
  }], { title: "Fidelity / Quality score by generator", height: 420 });

  if (rows[0].Dataset) {
    const ds = unique(rows, "Dataset");
    const z = gens.map(g => ds.map(d => {
      const r = rows.find(x => x.Generator === g && x.Dataset === d);
      return r ? (r[scoreKey] || r.Mean) : null;
    }));
    plot("fidelity-heatmap", [{
      type: "heatmap", x: ds.map(d => d.replace(/^\d+\.\s*/, "")), y: gens, z,
      colorscale: "Viridis", colorbar: { title: "Score" },
    }], { title: "Fidelity by dataset × generator", height: 420 });
  }
}

function renderPrivacy() {
  const rows = DATA.privacy.filter(r => r.Metric === "Mean_Distance" || String(r.Metric).includes("Distance"));
  if (!rows.length) {
    plot("privacy-bar", [], { title: "No privacy data", height: 200 });
    return;
  }
  const scoreKey = rows[0].Mean !== undefined ? "Mean" : "Value";
  const byGen = {};
  rows.forEach(r => {
    if (!r.Generator) return;
    byGen[r.Generator] = (byGen[r.Generator] || []).concat(r[scoreKey]);
  });
  const gens = Object.keys(byGen);
  plot("privacy-bar", [{
    x: gens,
    y: gens.map(g => byGen[g].reduce((a, b) => a + b, 0) / byGen[g].length),
    type: "bar",
    marker: { color: "#f59e0b" },
  }], { title: "Mean Mahalanobis distance (lower = more private)", height: 420 });
}

function renderTradeoff() {
  const t = DATA.tradeoff;
  if (!t.length) return;

  plot("tradeoff-scatter", [{
    x: t.map(r => r.Utility),
    y: t.map(r => r.Privacy),
    text: t.map(r => r.Generator),
    mode: "markers+text",
    textposition: "top center",
    marker: {
      size: t.map(r => 12 + (r.Fidelity || 0) * 20),
      color: t.map(r => r.Fidelity || 0),
      colorscale: "Viridis",
      showscale: true,
      colorbar: { title: "Fidelity" },
    },
    type: "scatter",
  }], { title: "Privacy vs Utility (bubble size = Fidelity)", height: 480, xaxis: { title: "Utility" }, yaxis: { title: "Privacy" } });

  if (t[0].Fidelity !== undefined) {
    plot("tradeoff-3d", [{
      type: "scatter3d",
      x: t.map(r => r.Utility),
      y: t.map(r => r.Privacy),
      z: t.map(r => r.Fidelity),
      text: t.map(r => r.Generator),
      mode: "markers+text",
      marker: { size: 6, color: t.map(r => r.OverallScore || r.Utility), colorscale: "Portland" },
    }], { title: "3D trade-off", height: 500, scene: {
      xaxis: { title: "Utility" }, yaxis: { title: "Privacy" }, zaxis: { title: "Fidelity" },
    }});
  }
}

function renderRankings() {
  const w = DATA.weighted;
  const b = DATA.borda;
  if (w.length) {
    const scoreCol = w[0].OverallScore !== undefined ? "OverallScore" : "WeightedScore";
    plot("rankings-weighted", [{
      x: w.map(r => r.Generator),
      y: w.map(r => r[scoreCol]),
      type: "bar",
      marker: { color: "#8b5cf6" },
    }], { title: "Weighted overall ranking (40% utility, 30% privacy, 30% fidelity)", height: 400 });
  }
  if (b.length) {
    plot("rankings-borda", [{
      x: b.map(r => r.generator || r.Generator),
      y: b.map(r => r.BordaScore || r.wins),
      type: "bar",
      marker: { color: "#06b6d4" },
    }], { title: "Borda count ranking", height: 400 });
  }
}

function renderStatistics() {
  const w = DATA.statistics?.wilcoxon || [];
  if (!w.length) return;
  const metrics = [...new Set(w.map(r => r.Metric))].slice(0, 4);
  metrics.forEach((metric, i) => {
    const sub = w.filter(r => r.Metric === metric);
    const gens = [...new Set(sub.flatMap(r => [r.Generator_A, r.Generator_B]))];
    const mat = gens.map(g1 => gens.map(g2 => {
      if (g1 === g2) return 0;
      const r = sub.find(x =>
        (x.Generator_A === g1 && x.Generator_B === g2) ||
        (x.Generator_A === g2 && x.Generator_B === g1)
      );
      return r ? -Math.log10(r.p_bh || r.p_raw || 1) : null;
    }));
    plot(`stats-heat-${i}`, [{
      type: "heatmap", x: gens, y: gens, z: mat,
      colorscale: "Reds", colorbar: { title: "-log10(p)" },
    }], { title: `Significance: ${metric}`, height: 380 });
  });
}

function setupTabs() {
  document.querySelectorAll("nav.tabs button").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("nav.tabs button").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.panel).classList.add("active");
      window.dispatchEvent(new Event("resize"));
    });
  });
}

function setupFilters() {
  ["filter-task", "filter-metric", "filter-dataset", "filter-generator"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderUtility);
  });

  fillSelect("filter-task", ["classification", "regression"], "classification");
  fillSelect("filter-metric", ["Accuracy", "F1", "Precision", "Recall", "R2", "RMSE", "MAE"], "Accuracy");
}

async function init() {
  document.getElementById("app-loading").style.display = "block";
  await loadAllData();
  document.getElementById("app-loading").style.display = "none";
  document.getElementById("app-main").style.display = "block";

  renderMetrics();
  setupTabs();
  setupFilters();
  renderOverview();
  renderUtility();
  renderFidelity();
  renderPrivacy();
  renderTradeoff();
  renderRankings();
  renderStatistics();
}

init();
