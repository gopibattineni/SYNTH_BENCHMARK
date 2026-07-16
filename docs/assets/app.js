/* SYNTH Benchmark — interactive GitHub Pages dashboard */

const GENERATORS = [
  "CTGAN", "CopulaGAN", "TVAE", "GaussianCopula",
  "WGAN_GP", "CTABGAN", "TabDDPM", "ForestDiffusion",
];

const UNIT_INTERVAL_METRICS = new Set([
  "Accuracy", "F1", "Precision", "Recall",
  "OverallScore", "WeightedScore", "Utility", "Privacy", "Fidelity",
  "Quality_Score", "NormalizedScore",
]);

const LOWER_BETTER_METRICS = new Set(["RMSE", "MAE", "Mean_Distance"]);

const PRIVACY_LABELS = {
  NNDR: "NNDR (nearest-neighbour distance ratio)",
  Mahalanobis_Distance: "Mahalanobis distance",
  Hungarian_Cosine_Similarity: "Hungarian cosine similarity",
  Cosine_Similarity: "Cosine similarity",
  MIA_AUC: "MIA AUC (membership inference)",
  Mean_Distance: "Mean matching distance",
  Median_Distance: "Median matching distance",
  Min_Distance: "Min matching distance",
  Max_Distance: "Max matching distance",
  Std_Distance: "Std matching distance",
  Num_Matches: "Matching subsample size",
};

const PRIVACY_PREFERRED_ORDER = [
  "Mahalanobis_Distance",
  "NNDR",
  "Hungarian_Cosine_Similarity",
  "Cosine_Similarity",
  "MIA_AUC",
  "Mean_Distance",
  "Median_Distance",
  "Min_Distance",
  "Max_Distance",
  "Std_Distance",
  "Num_Matches",
];

const FIDELITY_LABELS = {
  Quality_Score: "Quality score (SDMetrics)",
  Quality: "Quality score",
  KS_Complement: "KS complement (1 − KS statistic)",
  JS_Divergence: "Jensen–Shannon divergence",
  Gower_Distance: "Gower distance",
  Cosine_Similarity: "Cosine similarity",
  MMD: "Maximum mean discrepancy (MMD)",
  MMD_Multivariate: "MMD (multivariate)",
  Wasserstein_Distance: "Wasserstein distance",
  PCA_Correlation_Diff: "PCA correlation difference",
  PCA_Mean_Error: "PCA mean error",
};

const FIDELITY_PREFERRED_ORDER = [
  "Quality_Score",
  "KS_Complement",
  "JS_Divergence",
  "Gower_Distance",
  "Cosine_Similarity",
  "MMD",
  "MMD_Multivariate",
  "Wasserstein_Distance",
  "PCA_Correlation_Diff",
  "PCA_Mean_Error",
];

const PLOTLY_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#e7ecf3", family: "Inter, system-ui, sans-serif", size: 12 },
  margin: { l: 56, r: 24, t: 44, b: 88 },
  xaxis: { gridcolor: "#2d3a4f", zerolinecolor: "#2d3a4f" },
  yaxis: { gridcolor: "#2d3a4f", zerolinecolor: "#2d3a4f", type: "linear" },
  legend: { bgcolor: "rgba(0,0,0,0)" },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: true, displaylogo: false };

let DATA = {};

function toNum(value) {
  if (value == null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function mean(values) {
  const nums = values.map(toNum).filter(v => v != null);
  if (!nums.length) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function isPercentMetric(metric, task) {
  // R2 / error metrics are not percentages (R2 can be negative → bars vanish on 0–100% axes)
  if (["RMSE", "MAE", "MSE", "R2", "Mean_Distance"].includes(metric)) return false;
  if (/^(R2|RMSE|MAE|MSE)_/.test(metric)) return false;
  if (metric.includes("Drop") || metric.includes("Gap") || metric.includes("Increase")) return true;
  if (task === "classification" && ["Accuracy", "F1", "Precision", "Recall"].includes(metric)) return true;
  return false;
}

function toPercentValues(values) {
  return values.map(v => {
    const n = toNum(v);
    return n == null ? null : n * 100;
  });
}

function percentYAxis(values) {
  const nums = toPercentValues(values).filter(v => v != null);
  const minV = nums.length ? Math.min(...nums) : 0;
  const maxV = nums.length ? Math.max(...nums) : 100;
  const pad = Math.max((maxV - minV) * 0.12, 6);
  const lo = Math.max(0, minV - pad);
  const hi = Math.min(100, maxV + pad + 10);
  return {
    type: "linear",
    autorange: false,
    range: [lo, Math.max(hi, lo + 18)],
    tickmode: "linear",
    dtick: hi - lo <= 30 ? 5 : 10,
    ticksuffix: "%",
    automargin: true,
  };
}

function valueLabel(v, { asPercent = false } = {}) {
  if (v == null) return "";
  if (asPercent) {
    // Keep enough precision so 99.9522% does not display as 100.0%
    const oneDec = Number(v.toFixed(1));
    if ((oneDec === 100 || oneDec === 0) && Math.abs(v - oneDec) > 1e-9) {
      return `${v.toFixed(2)}%`;
    }
    return `${v.toFixed(1)}%`;
  }
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(v) >= 10) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function numericYAxis(values, { asPercent = false, padRatio = 0.14 } = {}) {
  if (asPercent) return percentYAxis(values);
  const nums = values.map(toNum).filter(v => v != null);
  if (!nums.length) {
    return { type: "linear", range: [0, 1], autorange: false, automargin: true };
  }
  const minV = Math.min(...nums);
  const maxV = Math.max(...nums);
  const span = Math.max(maxV - minV, Math.abs(maxV - minV), Math.abs(maxV) * 0.05, Math.abs(minV) * 0.05, 1e-6);
  const pad = span * padRatio;
  // Cosine similarity can be negative — never floor at 0 or use nonnegative rangemode.
  const lo = minV >= 0 ? Math.max(0, minV - pad) : minV - pad;
  const hi = maxV <= 0 ? Math.min(0, maxV + pad) : maxV + pad + span * 0.08;
  const absMax = Math.max(Math.abs(minV), Math.abs(maxV));
  const tickformat = absMax >= 1000 ? ",.0f" : absMax >= 1 ? ".2f" : absMax >= 0.1 ? ".3f" : ".4f";
  return {
    type: "linear",
    autorange: false,
    rangemode: "normal",
    range: [lo, hi],
    automargin: true,
    tickformat,
    zeroline: true,
    zerolinecolor: "#64748b",
  };
}

function privacyMetricLabel(metricId) {
  return PRIVACY_LABELS[metricId] || String(metricId).replace(/_/g, " ");
}

function fidelityMetricLabel(metricId) {
  return FIDELITY_LABELS[metricId] || String(metricId).replace(/_/g, " ");
}

function fidelityUsesPercent(metricId, catalogEntry) {
  return Boolean(catalogEntry?.is_unit_interval)
    || ["Quality_Score", "Quality", "KS_Complement", "JS_Divergence", "Gower_Distance"].includes(metricId);
}

function fidelityScoreFor(row) {
  return toNum(row.NormalizedScore ?? row.Mean ?? row.MetricValue ?? row.Value);
}

function privacyUsesPercent(metricId, catalogEntry) {
  return Boolean(catalogEntry?.is_similarity)
    || metricId === "MIA_AUC"
    || metricId.includes("Cosine");
}

function valuesForAllGenerators(byGen) {
  return GENERATORS.map(g => (byGen[g]?.length ? mean(byGen[g]) : null));
}

function generatorXAxis() {
  return {
    type: "category",
    categoryorder: "array",
    categoryarray: GENERATORS,
    tickangle: -25,
  };
}

function generatorYAxis() {
  return {
    type: "category",
    categoryorder: "array",
    categoryarray: [...GENERATORS].reverse(),
  };
}

function missingBarLabelTrace(values, asPercent) {
  const nums = GENERATORS.map((_, i) => {
    const raw = Array.isArray(values) ? values[i] : null;
    if (raw == null) return null;
    return asPercent ? toNum(raw) * 100 : toNum(raw);
  });
  const missingGens = GENERATORS.filter((_, i) => nums[i] == null);
  if (!missingGens.length) return null;

  const present = nums.filter(v => v != null);
  let yPos = 0;
  if (present.length) {
    const minP = Math.min(...present);
    yPos = asPercent ? minP - 4 : minP - Math.abs(minP) * 0.08 - 0.002;
  }

  return {
    x: missingGens,
    y: missingGens.map(() => yPos),
    type: "scatter",
    mode: "text",
    text: missingGens.map(() => "N/A"),
    textfont: { color: "#94a3b8", size: 12 },
    hoverinfo: "skip",
    showlegend: false,
  };
}

function privacyBarTrace({ x, y, color, asPercent }) {
  const nums = GENERATORS.map((gen, i) => {
    const raw = Array.isArray(y) ? y[i] : null;
    if (raw == null) return null;
    return asPercent ? toNum(raw) * 100 : toNum(raw);
  });
  const barColor = nums.map(v => {
    if (v == null) return "rgba(100,116,139,0.35)";
    if (!asPercent && v < 0) return "#64748b";
    return color;
  });
  return {
    x: GENERATORS,
    y: nums,
    type: "bar",
    marker: { color: barColor },
    text: nums.map(v => (v == null ? "N/A" : valueLabel(v, { asPercent }))),
    textposition: "outside",
    cliponaxis: false,
    constraintext: "none",
    hovertemplate: asPercent
      ? "%{x}: %{y:.3f}%<extra></extra>"
      : "%{x}: %{y:.4f}<extra></extra>",
  };
}

function barChartTrace({ x, y, name, color, asPercent = true, decimals = 1 }) {
  const categories = Array.isArray(x) && x.length ? x : GENERATORS;
  const values = categories.map((_, i) => {
    const raw = Array.isArray(y) ? y[i] : null;
    if (raw == null) return null;
    return asPercent ? toNum(raw) * 100 : toNum(raw);
  });
  const barColor = values.map(v => {
    if (v == null) return "rgba(100,116,139,0.35)";
    if (!asPercent && v < 0) return "#f59e0b";
    return color;
  });
  return {
    x: categories,
    y: values,
    name,
    type: "bar",
    marker: { color: barColor },
    text: values.map(v => (v == null ? "N/A" : asPercent ? `${v.toFixed(decimals)}%` : v.toFixed(decimals))),
    textposition: "outside",
    cliponaxis: false,
    hovertemplate: asPercent
      ? `<b>${name}</b><br>%{x}: %{y:.2f}%<extra></extra>`
      : `<b>${name}</b><br>%{x}: %{y:.4f}<extra></extra>`,
  };
}

function classifierMean(rows, classifier, evaluationType) {
  const vals = rows
    .filter(r => r.Classifier === classifier && r.EvaluationType === evaluationType)
    .map(r => toNum(r.Mean));
  return mean(vals);
}

function isAllDatasets(dataset) {
  return !dataset || dataset === "All datasets";
}

function datasetProblemType(dataset) {
  if (!dataset) return "unknown";
  const match = String(dataset).match(/^(\d+)/);
  if (!match) return "unknown";
  return parseInt(match[1], 10) >= 10 ? "regression" : "classification";
}

function matchesProblemType(dataset, problemType) {
  if (!problemType || problemType === "all") return true;
  return datasetProblemType(dataset) === problemType;
}

function filterDatasetsByProblem(datasets, problemType) {
  return datasets.filter(ds => matchesProblemType(ds, problemType));
}

function problemTypeLabel(problemType) {
  if (problemType === "classification") return "classification";
  if (problemType === "regression") return "regression";
  return "all";
}

function setupProblemTypeSelect(id, onChange, defaultValue = "all") {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = [
    { value: "all", label: "All problem types" },
    { value: "classification", label: "Classification" },
    { value: "regression", label: "Regression" },
  ].map(o => `<option value="${o.value}">${o.label}</option>`).join("");
  sel.value = defaultValue;
  sel.addEventListener("change", onChange);
}

function axisSpec(metric, values, { clampUnit = false } = {}) {
  const nums = values.map(toNum).filter(v => v != null);
  const minV = nums.length ? Math.min(...nums) : 0;
  const maxV = nums.length ? Math.max(...nums) : 1;

  // R² can be strongly negative — use an auto range that includes negatives
  if (metric === "R2") {
    const span = Math.max(maxV - minV, 0.15);
    const lo = minV - span * 0.12;
    const hi = Math.max(maxV + span * 0.12, 0.05);
    return {
      type: "linear",
      range: [lo, hi],
      zeroline: true,
      zerolinecolor: "#94a3b8",
      tickformat: ".2f",
      title: { text: "R²" },
      automargin: true,
    };
  }

  if (
    (UNIT_INTERVAL_METRICS.has(metric) || clampUnit)
    && minV >= -0.02
    && maxV <= 1.05
  ) {
    return {
      type: "linear",
      range: [0, 1],
      tickmode: "linear",
      dtick: 0.1,
      tickformat: ".0%",
      automargin: true,
    };
  }

  if (metric.includes("Drop") || metric.includes("Gap")) {
    const hi = Math.min(1, Math.max(0.1, maxV * 1.12));
    return {
      type: "linear",
      range: [0, hi],
      tickmode: "linear",
      dtick: hi <= 0.2 ? 0.05 : 0.1,
      tickformat: ".2f",
      automargin: true,
    };
  }

  if (LOWER_BETTER_METRICS.has(metric)) {
    const span = Math.max(maxV - minV, maxV * 0.05, 1e-6);
    const lo = Math.max(0, minV - span * 0.08);
    const hi = maxV + span * 0.12;
    return {
      type: maxV / Math.max(lo, 1e-9) > 500 ? "log" : "linear",
      range: maxV / Math.max(lo, 1e-9) > 500 ? undefined : [lo, hi],
      tickformat: maxV >= 100 ? ",.0f" : ".3f",
      automargin: true,
    };
  }

  const span = Math.max(maxV - minV, 0.05);
  const lo = minV < 0 ? minV - span * 0.08 : Math.max(0, minV - span * 0.08);
  return {
    type: "linear",
    range: [lo, maxV + span * 0.12],
    tickformat: ".3f",
    automargin: true,
  };
}

const DATA_VERSION = "20260716e";

async function loadJSON(name) {
  const url = `data/${name}?v=${DATA_VERSION}`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return name === "meta.json" ? {} : (name === "statistics.json" ? {} : []);
    return res.json();
  } catch {
    return name.endsWith(".json") && name.includes("stat") ? {} : [];
  }
}

async function loadAllData() {
  const [meta, utilityAgg, utilityClf, utilityReg, utilityGaps, fidelity, fidelityMetrics, privacy, privacyMetrics, tradeoff,
         weighted, borda, statistics, coverage] = await Promise.all([
    loadJSON("meta.json"),
    loadJSON("utility_agg.json"),
    loadJSON("utility_classifier.json"),
    loadJSON("utility_regressor.json"),
    loadJSON("utility_gaps.json"),
    loadJSON("fidelity.json"),
    loadJSON("fidelity_metrics.json"),
    loadJSON("privacy.json"),
    loadJSON("privacy_metrics.json"),
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
    utilityReg: utilityReg || [],
    utilityGaps: utilityGaps || [],
    fidelity: fidelity || [],
    fidelityMetrics: fidelityMetrics || [],
    privacy: privacy || [],
    privacyMetrics: privacyMetrics || [],
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

/** Sort numbered datasets 1…15 by prefix, not lexicographically. */
function uniqueDatasets(rows) {
  return [...new Set(rows.map(r => r.Dataset).filter(Boolean))].sort((a, b) => {
    const na = parseInt(String(a).match(/^(\d+)/)?.[1] || "999", 10);
    const nb = parseInt(String(b).match(/^(\d+)/)?.[1] || "999", 10);
    if (na !== nb) return na - nb;
    return String(a).localeCompare(String(b));
  });
}

function shortDatasetLabel(dataset) {
  const raw = String(dataset).replace(/^\d+\.\s*/, "").trim();
  const aliases = {
    "Concrete Compressive Strength": "Concrete",
    "Real Estate Valuation": "Real Estate",
    "MAGIC Gamma Telescope": "MAGIC",
    "CDC diabetes dataset": "CDC Diabetes",
    "Forest cover dataset": "Forest Cover",
    "Mushroom dataset": "Mushroom",
    "Wine dataset": "Wine",
    "Bank Markting": "Bank Marketing",
    "Metro interstate": "Metro",
    "online shopping": "E-shop",
    "Air Quality": "Air Quality",
    Alzhimers: "Alzheimer",
    Cancer: "Cancer",
    Adult: "Adult",
    "Energy Efficiency": "Energy",
  };
  return aliases[raw] || raw;
}

function plot(id, traces, layout = {}, config = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  const isHeatmap = Array.isArray(traces) && traces.some(t => t && t.type === "heatmap");
  const yaxis = { ...PLOTLY_LAYOUT.yaxis, ...(layout.yaxis || {}) };
  // Heatmaps use categorical generator/dataset labels — never force a linear Y axis.
  if (isHeatmap && yaxis.type !== "category") {
    yaxis.type = "category";
    yaxis.autorange = true;
    delete yaxis.autotypenumbers;
  } else if (yaxis.type !== "category") {
    yaxis.type = "linear";
    yaxis.autotypenumbers = "strict";
  }
  const xaxis = { ...PLOTLY_LAYOUT.xaxis, ...(layout.xaxis || {}) };
  if (isHeatmap && xaxis.type !== "category" && xaxis.type !== "linear") {
    xaxis.type = "category";
  }
  Plotly.newPlot(
    id,
    traces,
    { ...PLOTLY_LAYOUT, ...layout, xaxis, yaxis },
    { ...PLOTLY_CONFIG, ...config },
  );
}

function fillSelect(id, options, defaultVal) {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = options.map(o => `<option value="${o}">${o}</option>`).join("");
  if (defaultVal && options.includes(defaultVal)) sel.value = defaultVal;
}

function aggregateGeneratorMeans(rows, evaluationType) {
  const buckets = {};
  rows
    .filter(r => r.EvaluationType === evaluationType)
    .forEach(r => {
      const value = toNum(r.Mean);
      if (value == null) return;
      if (!buckets[r.Generator]) buckets[r.Generator] = [];
      buckets[r.Generator].push(value);
    });
  return buckets;
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
  const datasets = uniqueDatasets(gaps);
  const generators = GENERATORS.filter(g => gaps.some(r => r.Generator === g));

  const z = generators.map(gen =>
    datasets.map(ds => {
      const row = gaps.find(r => r.Generator === gen && r.Dataset === ds);
      return row ? toNum(row.Mean) : null;
    }),
  );

  plot("overview-heatmap", [{
    type: "heatmap",
    x: datasets.map(shortDatasetLabel),
    y: generators,
    z,
    zmin: 0,
    zmax: 1,
    colorscale: "RdYlGn",
    reversescale: true,
    hovertemplate: "%{y} · %{x}<br>Gap: %{z:.1%}<extra></extra>",
    colorbar: { title: "Utility gap", tickformat: ".0%", thickness: 14, len: 0.75 },
    xgap: 1,
    ygap: 1,
  }], {
    title: { text: "" },
    height: 480,
    margin: { l: 130, r: 80, t: 20, b: 100 },
    xaxis: { type: "category", tickangle: -40, automargin: true },
    yaxis: { ...generatorYAxis(), automargin: true },
  });

  const cov = DATA.coverage;
  if (cov.length) {
    const dsU = uniqueDatasets(cov);
    const zCov = GENERATORS.map(g =>
      dsU.map(ds => {
        const r = cov.find(x => x.Generator === g && x.Dataset === ds);
        return r ? (toNum(r.Available) ? 1 : 0) : 0;
      }),
    );
    const textMat = zCov.map(row => row.map(v => (v ? "✓" : "✗")));
    const hoverMat = zCov.map((row, gi) =>
      row.map((v, di) => (v ? "Available" : "Missing")),
    );

    plot("overview-coverage", [{
      type: "heatmap",
      x: dsU.map(shortDatasetLabel),
      y: GENERATORS,
      z: zCov,
      zmin: 0,
      zmax: 1,
      colorscale: [
        [0, "#1e293b"],
        [0.5, "#1e293b"],
        [0.5, "#059669"],
        [1, "#34d399"],
      ],
      showscale: true,
      colorbar: {
        title: { text: "" },
        tickmode: "array",
        tickvals: [0, 1],
        ticktext: ["Missing", "Available"],
        thickness: 16,
        len: 0.45,
        outlinewidth: 0,
      },
      text: textMat,
      texttemplate: "%{text}",
      textfont: { size: 12, color: "#f8fafc" },
      customdata: hoverMat,
      hovertemplate: "<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
      xgap: 3,
      ygap: 3,
    }], {
      title: { text: "" },
      height: 480,
      margin: { l: 130, r: 100, t: 20, b: 100 },
      xaxis: {
        type: "category",
        tickangle: -40,
        automargin: true,
        side: "bottom",
      },
      yaxis: {
        ...generatorYAxis(),
        automargin: true,
      },
    });
  }
}

function metricsForTask(task) {
  return task === "regression"
    ? ["R2", "RMSE", "MAE"]
    : ["Accuracy", "F1", "Precision", "Recall"];
}

function syncUtilityMetricOptions(task, preferred) {
  const options = metricsForTask(task);
  const current = preferred || document.getElementById("filter-metric")?.value;
  const next = options.includes(current) ? current : options[0];
  fillSelect("filter-metric", options, next);
  return next;
}

function regressorMean(rows, regressor, evaluationType) {
  const vals = rows
    .filter(r => r.Regressor === regressor && r.EvaluationType === evaluationType)
    .map(r => toNum(r.Mean));
  return mean(vals);
}

function renderUtility() {
  const task = document.getElementById("filter-task")?.value || "classification";
  const metric = syncUtilityMetricOptions(task);
  const dataset = document.getElementById("filter-dataset")?.value;
  const allDs = isAllDatasets(dataset);

  const rows = DATA.utilityAgg.filter(r =>
    r.TaskType === task
    && r.Metric === metric
    && (allDs || r.Dataset === dataset),
  );

  const datasets = uniqueDatasets(DATA.utilityAgg.filter(r => r.TaskType === task));
  fillSelect("filter-dataset", ["All datasets", ...datasets], dataset || "All datasets");

  const trtrBuckets = aggregateGeneratorMeans(rows, "TRTR");
  const tstrBuckets = aggregateGeneratorMeans(rows, "TSTR");
  const gens = GENERATORS;
  const trtr = valuesForAllGenerators(trtrBuckets);
  const tstr = valuesForAllGenerators(tstrBuckets);
  const yValues = [...trtr, ...tstr].filter(v => v != null);
  const usePercent = isPercentMetric(metric, task);
  const hasBars = yValues.length > 0;

  plot("utility-trtr-tstr", hasBars ? [
    barChartTrace({ x: gens, y: trtr, name: "TRTR", color: "#3b82f6", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
    barChartTrace({ x: gens, y: tstr, name: "TSTR", color: "#10b981", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
  ] : [], {
    barmode: "group",
    bargap: 0.18,
    bargroupgap: 0.08,
    title: hasBars
      ? `${metric}: TRTR vs TSTR (${task}${allDs ? ", mean over datasets" : ""})`
      : `No ${metric} data for this ${task} selection`,
    height: 480,
    margin: { t: 44, b: 88, l: 64, r: 24 },
    yaxis: usePercent ? percentYAxis(yValues) : axisSpec(metric, yValues),
    xaxis: generatorXAxis(),
  });

  const modelTitleEl = document.querySelector("#panel-utility .chart-card:nth-child(2) h3");
  const isRegression = task === "regression";
  if (modelTitleEl) modelTitleEl.textContent = isRegression ? "By regressor" : "By classifier";

  const detailSource = isRegression ? (DATA.utilityReg || []) : (DATA.utilityClf || []);
  const modelKey = isRegression ? "Regressor" : "Classifier";
  const detailRows = detailSource.filter(r =>
    r.Metric === metric && (allDs || r.Dataset === dataset),
  );

  const gensWithDetail = GENERATORS.filter(g => detailRows.some(r => r.Generator === g));
  const genSel = document.getElementById("filter-generator")?.value;
  const gen = gensWithDetail.includes(genSel) ? genSel : (gensWithDetail[0] || GENERATORS[0]);
  fillSelect("filter-generator", gensWithDetail.length ? gensWithDetail : GENERATORS, gen);

  const ds = allDs ? (unique(detailRows, "Dataset")[0] || "") : dataset;
  const detail = detailRows.filter(r => r.Generator === gen && r.Dataset === ds);
  const models = unique(detail, modelKey);
  const trtrModel = models.map(m => (
    isRegression ? regressorMean(detail, m, "TRTR") : classifierMean(detail, m, "TRTR")
  ));
  const tstrModel = models.map(m => (
    isRegression ? regressorMean(detail, m, "TSTR") : classifierMean(detail, m, "TSTR")
  ));
  const modelValues = [...trtrModel, ...tstrModel].filter(v => v != null);
  const hasModelBars = modelValues.length > 0;

  plot("utility-classifier", hasModelBars ? [
    barChartTrace({ x: models, y: trtrModel, name: "TRTR", color: "#3b82f6", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
    barChartTrace({ x: models, y: tstrModel, name: "TSTR", color: "#10b981", asPercent: usePercent, decimals: usePercent ? 1 : 3 }),
  ] : [], {
    barmode: "group",
    bargap: 0.12,
    bargroupgap: 0.06,
    title: hasModelBars
      ? `${ds} — ${gen}: ${metric} by ${isRegression ? "regressor" : "classifier"}`
      : `No ${isRegression ? "regressor" : "classifier"} breakdown for ${metric}`,
    height: 480,
    margin: { t: 44, b: 110, l: 64, r: 24 },
    xaxis: { tickangle: -45, type: "category" },
    yaxis: usePercent ? percentYAxis(modelValues) : axisSpec(metric, modelValues),
  });
}

function renderFidelity() {
  const metric = document.getElementById("filter-fidelity-metric")?.value;
  const problemType = document.getElementById("filter-fidelity-problem")?.value || "all";
  const dataset = document.getElementById("filter-fidelity-dataset")?.value;
  const catalog = DATA.fidelityMetrics || [];
  const catalogEntry = catalog.find(m => m.id === metric) || {};
  const label = fidelityMetricLabel(metric);
  const asPercent = fidelityUsesPercent(metric, catalogEntry);
  const higherBetter = catalogEntry.higher_is_better !== false;

  const metricRows = DATA.fidelity.filter(r =>
    r.Metric === metric && matchesProblemType(r.Dataset, problemType),
  );
  const datasets = filterDatasetsByProblem(unique(metricRows, "Dataset"), problemType);
  const datasetDefault = dataset && (dataset === "All datasets" || datasets.includes(dataset))
    ? dataset
    : "All datasets";
  fillSelect("filter-fidelity-dataset", ["All datasets", ...datasets], datasetDefault);

  const selectedDataset = document.getElementById("filter-fidelity-dataset")?.value;
  const allDsSelected = isAllDatasets(selectedDataset);
  const rows = metricRows.filter(r => allDsSelected || r.Dataset === selectedDataset);

  if (!rows.length) {
    plot("fidelity-bar", [], { title: `No data for ${label}`, height: 220 });
    plot("fidelity-heatmap", [], { title: "No heatmap data", height: 220 });
    const titleEl = document.getElementById("fidelity-bar-title");
    if (titleEl) titleEl.textContent = label;
    return;
  }

  const byGen = {};
  rows.forEach(r => {
    const value = fidelityScoreFor(r);
    if (!r.Generator || value == null) return;
    if (!byGen[r.Generator]) byGen[r.Generator] = [];
    byGen[r.Generator].push(value);
  });

  const gens = GENERATORS;
  const barValues = valuesForAllGenerators(byGen);
  const direction = higherBetter ? "higher = better fidelity" : "lower = better fidelity";
  const missingCount = barValues.filter(v => v == null).length;
  const missingNote = missingCount ? ` · ${missingCount} generator(s) not evaluated` : "";
  const problemNote = problemType !== "all"
    ? ` · ${problemTypeLabel(problemType)} datasets`
    : "";

  const titleEl = document.getElementById("fidelity-bar-title");
  if (titleEl) titleEl.textContent = `${label} by generator`;

  const naTrace = missingBarLabelTrace(barValues, asPercent);
  plot("fidelity-bar", [
    privacyBarTrace({ x: gens, y: barValues, color: "#3b82f6", asPercent }),
    ...(naTrace ? [naTrace] : []),
  ], {
    title: `${label}${allDsSelected ? " (mean over datasets)" : ""}${problemNote} · ${direction}${missingNote}`,
    height: 460,
    margin: { t: 44, b: 88, l: 64, r: 24 },
    yaxis: numericYAxis(barValues, { asPercent }),
    xaxis: generatorXAxis(),
    showlegend: false,
  });

  const heatRows = metricRows;
  const heatDs = datasets;
  const heatGens = GENERATORS;
  const z = heatGens.map(g => heatDs.map(d => {
    const matches = heatRows.filter(r => r.Generator === g && r.Dataset === d);
    return mean(matches.map(fidelityScoreFor));
  }));

  const zFlat = z.flat().filter(v => v != null);
  const heatmapScale = asPercent
    ? { zmin: 0, zmax: 1 }
    : { zmin: zFlat.length ? Math.min(...zFlat) : undefined, zmax: zFlat.length ? Math.max(...zFlat) : undefined };

  plot("fidelity-heatmap", [{
    type: "heatmap",
    x: heatDs.map(d => d.replace(/^\d+\.\s*/, "")),
    y: heatGens,
    z,
    ...heatmapScale,
    colorscale: higherBetter ? "Viridis" : "YlOrRd",
    reversescale: !higherBetter,
    colorbar: {
      title: label,
      tickformat: asPercent ? ".0%" : ".2f",
    },
  }], {
    title: `${label} · dataset × generator${problemType !== "all" ? ` (${problemTypeLabel(problemType)})` : ""}`,
    height: 420,
    yaxis: generatorYAxis(),
  });
}

function setupFidelityFilters() {
  setupProblemTypeSelect("filter-fidelity-problem", renderFidelity);

  const catalog = DATA.fidelityMetrics || [];
  const available = new Set(catalog.map(m => m.id));
  const ordered = FIDELITY_PREFERRED_ORDER.filter(id => available.has(id));
  catalog.forEach(m => {
    if (!ordered.includes(m.id)) ordered.push(m.id);
  });

  const defaultMetric = ordered.includes("Quality_Score")
    ? "Quality_Score"
    : (ordered.includes("KS_Complement") ? "KS_Complement" : ordered[0]);

  const metricSel = document.getElementById("filter-fidelity-metric");
  if (metricSel) {
    metricSel.innerHTML = ordered
      .map(id => `<option value="${id}">${fidelityMetricLabel(id)}</option>`)
      .join("");
    metricSel.value = defaultMetric;
  }

  ["filter-fidelity-metric", "filter-fidelity-dataset"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderFidelity);
  });
}

function renderPrivacy() {
  const metric = document.getElementById("filter-privacy-metric")?.value;
  const problemType = document.getElementById("filter-privacy-problem")?.value || "all";
  const dataset = document.getElementById("filter-privacy-dataset")?.value;
  const catalog = DATA.privacyMetrics || [];
  const catalogEntry = catalog.find(m => m.id === metric) || {};
  const label = privacyMetricLabel(metric);
  const asPercent = privacyUsesPercent(metric, catalogEntry);
  const lowerBetter = catalogEntry.lower_is_better !== false;

  const metricRows = DATA.privacy.filter(r =>
    r.Metric === metric && matchesProblemType(r.Dataset, problemType),
  );
  const datasets = filterDatasetsByProblem(unique(metricRows, "Dataset"), problemType);
  const datasetDefault = dataset && (dataset === "All datasets" || datasets.includes(dataset))
    ? dataset
    : "All datasets";
  fillSelect("filter-privacy-dataset", ["All datasets", ...datasets], datasetDefault);

  const selectedDataset = document.getElementById("filter-privacy-dataset")?.value;
  const allDsSelected = isAllDatasets(selectedDataset);
  const rows = metricRows.filter(r => allDsSelected || r.Dataset === selectedDataset);

  if (!rows.length) {
    plot("privacy-bar", [], { title: `No data for ${label}`, height: 220 });
    plot("privacy-heatmap", [], { title: "No heatmap data", height: 220 });
    const titleEl = document.getElementById("privacy-bar-title");
    if (titleEl) titleEl.textContent = label;
    return;
  }

  const byGen = {};
  rows.forEach(r => {
    const value = toNum(r.Mean ?? r.MetricValue ?? r.Value);
    if (!r.Generator || value == null) return;
    if (!byGen[r.Generator]) byGen[r.Generator] = [];
    byGen[r.Generator].push(value);
  });

  const gens = GENERATORS;
  const barValues = valuesForAllGenerators(byGen);
  const direction = catalogEntry.is_sample_size
    ? "Hungarian matching subsample size (not a privacy risk score)"
    : lowerBetter
      ? "lower = more private"
      : "lower similarity = more private";
  const missingCount = barValues.filter(v => v == null).length;
  const missingNote = missingCount ? ` · ${missingCount} generator(s) not evaluated` : "";
  const problemNote = problemType !== "all"
    ? ` · ${problemTypeLabel(problemType)} datasets`
    : "";

  const titleEl = document.getElementById("privacy-bar-title");
  if (titleEl) titleEl.textContent = `${label} by generator`;

  const naTrace = missingBarLabelTrace(barValues, asPercent);
  plot("privacy-bar", [
    privacyBarTrace({ x: gens, y: barValues, color: "#f59e0b", asPercent }),
    ...(naTrace ? [naTrace] : []),
  ], {
    title: `${label}${allDsSelected ? " (mean over datasets)" : ""}${problemNote} · ${direction}${missingNote}`,
    height: 460,
    margin: { t: 44, b: 88, l: 64, r: 24 },
    yaxis: numericYAxis(barValues, { asPercent }),
    xaxis: generatorXAxis(),
    showlegend: false,
  });

  const heatRows = metricRows;
  const heatDs = datasets;
  const heatGens = GENERATORS;
  const z = heatGens.map(g => heatDs.map(d => {
    const matches = heatRows.filter(r => r.Generator === g && r.Dataset === d);
    return mean(matches.map(r => toNum(r.Mean ?? r.MetricValue ?? r.Value)));
  }));

  plot("privacy-heatmap", [{
    type: "heatmap",
    x: heatDs.map(d => d.replace(/^\d+\.\s*/, "")),
    y: heatGens,
    z,
    colorscale: lowerBetter ? "YlOrRd" : "RdYlGn",
    reversescale: !lowerBetter,
    colorbar: {
      title: label,
      tickformat: asPercent ? ".0%" : ".2f",
    },
  }], {
    title: `${label} · dataset × generator${problemType !== "all" ? ` (${problemTypeLabel(problemType)})` : ""}`,
    height: 420,
    yaxis: generatorYAxis(),
  });
}

function setupPrivacyFilters() {
  setupProblemTypeSelect("filter-privacy-problem", renderPrivacy);

  const catalog = DATA.privacyMetrics || [];
  const available = new Set(catalog.map(m => m.id));
  const ordered = PRIVACY_PREFERRED_ORDER.filter(id => available.has(id));
  catalog.forEach(m => {
    if (!ordered.includes(m.id)) ordered.push(m.id);
  });

  const defaultMetric = ordered.includes("Mahalanobis_Distance")
    ? "Mahalanobis_Distance"
    : (ordered.includes("NNDR") ? "NNDR" : ordered[0]);

  const metricSel = document.getElementById("filter-privacy-metric");
  if (metricSel) {
    metricSel.innerHTML = ordered
      .map(id => `<option value="${id}">${privacyMetricLabel(id)}</option>`)
      .join("");
    metricSel.value = defaultMetric;
  }

  ["filter-privacy-metric", "filter-privacy-dataset"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderPrivacy);
  });
}

function renderTradeoff() {
  const t = DATA.tradeoff;
  if (!t.length) return;

  const utility = t.map(r => toNum(r.Utility));
  const privacy = t.map(r => toNum(r.Privacy));
  const fidelity = t.map(r => toNum(r.Fidelity));

  plot("tradeoff-scatter", [{
    x: utility,
    y: privacy,
    text: t.map(r => r.Generator),
    mode: "markers+text",
    textposition: "top center",
    marker: {
      size: fidelity.map(v => 12 + (v || 0) * 20),
      color: fidelity,
      colorscale: "Viridis",
      cmin: 0,
      cmax: 1,
      showscale: true,
      colorbar: { title: "Fidelity", tickformat: ".0%" },
    },
    type: "scatter",
  }], {
    title: "Privacy vs Utility (bubble size = Fidelity)",
    height: 480,
    xaxis: { title: "Utility", ...axisSpec("Utility", utility, { clampUnit: true }) },
    yaxis: { title: "Privacy", ...axisSpec("Privacy", privacy, { clampUnit: true }) },
  });

  if (t[0].Fidelity !== undefined) {
    plot("tradeoff-3d", [{
      type: "scatter3d",
      x: utility,
      y: privacy,
      z: fidelity,
      text: t.map(r => r.Generator),
      mode: "markers+text",
      marker: {
        size: 6,
        color: t.map(r => toNum(r.OverallScore ?? r.Utility)),
        colorscale: "Portland",
        cmin: 0,
        cmax: 1,
      },
    }], {
      title: "3D trade-off",
      height: 500,
      scene: {
        xaxis: { title: "Utility", range: [0, 1], tickformat: ".0%" },
        yaxis: { title: "Privacy", range: [0, 1], tickformat: ".0%" },
        zaxis: { title: "Fidelity", range: [0, 1], tickformat: ".0%" },
      },
    });
  }
}

function renderRankings() {
  const w = DATA.weighted;
  const b = DATA.borda;

  if (w.length) {
    const scoreCol = w[0].OverallScore !== undefined ? "OverallScore" : "WeightedScore";
    const values = w.map(r => toNum(r[scoreCol]));
    plot("rankings-weighted", [
      barChartTrace({ x: w.map(r => r.Generator), y: values, name: "Score", color: "#8b5cf6", asPercent: true }),
    ], {
      title: "Weighted overall ranking (40% utility, 30% privacy, 30% fidelity)",
      height: 420,
      margin: { t: 44, b: 88, l: 56, r: 24 },
      yaxis: percentYAxis(values),
      showlegend: false,
    });
  }

  if (b.length) {
    const values = b.map(r => toNum(r.BordaScore ?? r.wins));
    const maxScore = Math.max(...values.filter(v => v != null), 1);
    plot("rankings-borda", [
      barChartTrace({ x: b.map(r => r.generator || r.Generator), y: values, name: "Borda", color: "#06b6d4", asPercent: false, decimals: 0 }),
    ], {
      title: "Borda count ranking",
      height: 420,
      yaxis: {
        type: "linear",
        range: [0, maxScore * 1.15],
        tickformat: ".0f",
        automargin: true,
      },
      showlegend: false,
    });
  }
}

const STATS_METRIC_OPTIONS = [
  { value: "Mean_Error_Pct", label: "Mean Error %" },
  { value: "Median_Error_Pct", label: "Median Error %" },
  { value: "Std_Error_Pct", label: "Std Error %" },
];

function pcaErrorRows() {
  return DATA.statistics?.pca_errors || [];
}

function statsMetricField() {
  return document.getElementById("filter-stats-metric")?.value || "Mean_Error_Pct";
}

function statsMetricLabel(field) {
  return STATS_METRIC_OPTIONS.find(o => o.value === field)?.label || field;
}

function renderPcaErrorTable(rows) {
  const tbody = document.querySelector("#stats-error-table tbody");
  if (!tbody) return;
  const ordered = GENERATORS
    .map(g => rows.find(r => r.Generator === g))
    .filter(Boolean);
  if (!ordered.length) {
    tbody.innerHTML = `<tr><td colspan="5">No average-error statistics for this selection.</td></tr>`;
    return;
  }
  tbody.innerHTML = ordered.map(r => `
    <tr>
      <td>${r.Generator}</td>
      <td>${toNum(r.Mean_Error_Pct)?.toFixed(6) ?? "—"}</td>
      <td>${toNum(r.Median_Error_Pct)?.toFixed(6) ?? "—"}</td>
      <td>${toNum(r.Std_Error_Pct)?.toFixed(6) ?? "—"}</td>
      <td>${r.Source_Group || "—"}</td>
    </tr>
  `).join("");
}

function renderPcaErrorCharts() {
  const rows = pcaErrorRows();
  const field = statsMetricField();
  const datasetSel = document.getElementById("filter-stats-dataset")?.value || "All datasets";

  const datasets = uniqueDatasets(rows);
  fillSelect("filter-stats-dataset", ["All datasets", ...datasets], datasetSel);
  fillSelect(
    "filter-stats-metric",
    STATS_METRIC_OPTIONS.map(o => o.value),
    field,
  );
  const metricEl = document.getElementById("filter-stats-metric");
  if (metricEl) {
    [...metricEl.options].forEach(opt => {
      opt.textContent = statsMetricLabel(opt.value);
    });
  }

  const selectedDataset = document.getElementById("filter-stats-dataset")?.value || "All datasets";
  const metricField = statsMetricField();
  const metricLabel = statsMetricLabel(metricField);

  const barRows = selectedDataset === "All datasets"
    ? rows
    : rows.filter(r => r.Dataset === selectedDataset);

  const barTitle = document.getElementById("stats-bar-title");
  if (barTitle) {
    barTitle.textContent = selectedDataset === "All datasets"
      ? `Average ${metricLabel} across datasets`
      : `Average error by model — ${shortDatasetLabel(selectedDataset)}`;
  }

  const barY = GENERATORS.map(g => {
    const subset = barRows.filter(r => r.Generator === g);
    if (!subset.length) return null;
    return mean(subset.map(r => toNum(r[metricField])).filter(v => v != null));
  });

  plot("stats-error-bar", [
    barChartTrace({
      x: GENERATORS,
      y: barY.map(v => (v == null ? null : v / 100)),
      name: metricLabel,
      color: "#3b82f6",
      asPercent: true,
      decimals: 2,
    }),
  ], {
    height: 420,
    showlegend: false,
    yaxis: { title: metricLabel, ticksuffix: "%", automargin: true },
    xaxis: { tickangle: -25, automargin: true },
  });

  const heatDatasets = uniqueDatasets(rows);
  const z = heatDatasets.map(ds =>
    GENERATORS.map(g => {
      const hit = rows.find(r => r.Dataset === ds && r.Generator === g);
      return hit ? toNum(hit[metricField]) : null;
    }),
  );
  const flat = z.flat().filter(v => v != null);
  const zmax = flat.length ? Math.max(...flat) : 1;

  const heatTitle = document.getElementById("stats-heat-title");
  if (heatTitle) heatTitle.textContent = `${metricLabel} heatmap`;

  plot("stats-error-heatmap", [{
    type: "heatmap",
    x: GENERATORS,
    y: heatDatasets.map(shortDatasetLabel),
    z,
    zmin: 0,
    zmax,
    colorscale: "YlOrRd",
    colorbar: { title: "%" },
    hovertemplate: "%{y} · %{x}<br>%{z:.2f}%<extra></extra>",
  }], {
    height: 480,
    xaxis: { tickangle: -25, automargin: true, type: "category", categoryarray: GENERATORS },
    yaxis: {
      type: "category",
      categoryarray: [...heatDatasets.map(shortDatasetLabel)].reverse(),
      automargin: true,
    },
    margin: { l: 110, r: 40, t: 20, b: 80 },
  });

  const tableRows = selectedDataset === "All datasets"
    ? GENERATORS.map(g => {
        const subset = rows.filter(r => r.Generator === g);
        if (!subset.length) return null;
        return {
          Generator: g,
          Mean_Error_Pct: mean(subset.map(r => toNum(r.Mean_Error_Pct)).filter(v => v != null)),
          Median_Error_Pct: mean(subset.map(r => toNum(r.Median_Error_Pct)).filter(v => v != null)),
          Std_Error_Pct: mean(subset.map(r => toNum(r.Std_Error_Pct)).filter(v => v != null)),
          Source_Group: "mean across datasets",
        };
      }).filter(Boolean)
    : rows.filter(r => r.Dataset === selectedDataset);

  const tableTitle = document.querySelector("#panel-statistics .table-wrap .section-title");
  if (tableTitle) {
    tableTitle.textContent = selectedDataset === "All datasets"
      ? "Average error by model (mean across 15 datasets)"
      : `Average error by model — ${shortDatasetLabel(selectedDataset)}`;
  }
  renderPcaErrorTable(tableRows);
}

function renderWilcoxonHeatmaps() {
  const w = DATA.statistics?.wilcoxon || [];
  if (!w.length) return;

  const metrics = [...new Set(w.map(r => r.Metric))].slice(0, 4);
  metrics.forEach((metric, i) => {
    const sub = w.filter(r => r.Metric === metric);
    const gens = [...new Set(sub.flatMap(r => [r.Generator_A, r.Generator_B]))];
    const mat = gens.map(g1 => gens.map(g2 => {
      if (g1 === g2) return 0;
      const r = sub.find(x =>
        (x.Generator_A === g1 && x.Generator_B === g2)
        || (x.Generator_A === g2 && x.Generator_B === g1),
      );
      return r ? -Math.log10(toNum(r.p_bh ?? r.p_raw) || 1) : null;
    }));
    const flat = mat.flat().filter(v => v != null && v > 0);
    const zmax = flat.length ? Math.max(...flat) : 5;
    plot(`stats-heat-${i}`, [{
      type: "heatmap",
      x: gens,
      y: gens,
      z: mat,
      zmin: 0,
      zmax,
      colorscale: "Reds",
      colorbar: { title: "-log10(p)" },
    }], { title: `Significance: ${metric}`, height: 380 });
  });
}

function renderStatistics() {
  renderPcaErrorCharts();
  renderWilcoxonHeatmaps();
}

function setupStatisticsFilters() {
  ["filter-stats-dataset", "filter-stats-metric"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderPcaErrorCharts);
  });
  const datasets = uniqueDatasets(pcaErrorRows());
  fillSelect("filter-stats-dataset", ["All datasets", ...datasets], datasets[0] || "All datasets");
  fillSelect(
    "filter-stats-metric",
    STATS_METRIC_OPTIONS.map(o => o.value),
    "Mean_Error_Pct",
  );
  const metricEl = document.getElementById("filter-stats-metric");
  if (metricEl) {
    [...metricEl.options].forEach(opt => {
      opt.textContent = statsMetricLabel(opt.value);
    });
  }
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
  const taskEl = document.getElementById("filter-task");
  if (taskEl) {
    taskEl.addEventListener("change", () => {
      syncUtilityMetricOptions(taskEl.value);
      renderUtility();
    });
  }
  ["filter-metric", "filter-dataset", "filter-generator"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", renderUtility);
  });

  fillSelect("filter-task", ["classification", "regression"], "classification");
  syncUtilityMetricOptions("classification", "Accuracy");
}

async function init() {
  const loading = document.getElementById("app-loading");
  const main = document.getElementById("app-main");
  try {
    loading.style.display = "block";
    loading.textContent = "Loading benchmark data…";
    await loadAllData();
    loading.style.display = "none";
    main.style.display = "block";

    renderMetrics();
    setupTabs();
    setupFilters();
    setupFidelityFilters();
    setupPrivacyFilters();
    setupStatisticsFilters();
    renderOverview();
    renderUtility();
    renderFidelity();
    renderPrivacy();
    renderTradeoff();
    renderRankings();
    renderStatistics();
  } catch (err) {
    console.error(err);
    loading.textContent = `Failed to load dashboard data: ${err.message}. Hard refresh (Ctrl+Shift+R) or rebuild with python dashboard/build_pages.py`;
    main.style.display = "none";
  }
}

init();
