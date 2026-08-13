const state = {
  datasets: [],
  generators: [],
  selectedDataset: null,
  selectedGenerators: new Set(),
  taskFilter: "all",
  activeJobId: null,
  activeGenerator: null,
  pollTimer: null,
};

const els = {
  datasetGrid: document.getElementById("dataset-grid"),
  generatorGroups: document.getElementById("generator-groups"),
  generateBtn: document.getElementById("generate-btn"),
  selectAllGenerators: document.getElementById("select-all-generators"),
  clearGenerators: document.getElementById("clear-generators"),
  nSamples: document.getElementById("n-samples"),
  seed: document.getElementById("seed"),
  selectionSummary: document.getElementById("selection-summary"),
  resultsPanel: document.getElementById("results-panel"),
  jobStatusChip: document.getElementById("job-status-chip"),
  progressBar: document.getElementById("progress-bar"),
  statusMessage: document.getElementById("status-message"),
  resultCards: document.getElementById("result-cards"),
  previewSection: document.getElementById("preview-section"),
  previewTitle: document.getElementById("preview-title"),
  previewTable: document.getElementById("preview-table"),
  downloadActive: document.getElementById("download-active"),
  errorBox: document.getElementById("error-box"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function updateSelectionSummary() {
  const ds = state.datasets.find((d) => d.id === state.selectedDataset);
  const count = state.selectedGenerators.size;
  if (!ds) {
    els.selectionSummary.textContent = "Select a dataset and at least one generator.";
  } else if (count === 0) {
    els.selectionSummary.textContent = `${ds.name} selected — choose one or more generators.`;
  } else {
    els.selectionSummary.textContent = `${ds.name} · ${count} generator${count > 1 ? "s" : ""} · ${els.nSamples.value} rows · seed ${els.seed.value}`;
  }
  els.generateBtn.disabled = !ds || count === 0;
}

function renderDatasets() {
  const filtered = state.datasets.filter((ds) =>
    state.taskFilter === "all" ? true : ds.task === state.taskFilter
  );

  els.datasetGrid.innerHTML = filtered
    .map(
      (ds) => `
      <button
        type="button"
        class="dataset-card ${state.selectedDataset === ds.id ? "selected" : ""}"
        data-id="${ds.id}"
        aria-pressed="${state.selectedDataset === ds.id}"
      >
        <span class="badge badge-${ds.task}">${ds.task}</span>
        <h3>${ds.name}</h3>
        <p>${ds.description || `Target: ${ds.target_col}`}</p>
        <div class="meta-line">${ds.row_count} rows · ${ds.column_count} cols</div>
      </button>
    `
    )
    .join("");

  els.datasetGrid.querySelectorAll(".dataset-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedDataset = card.dataset.id;
      renderDatasets();
      updateSelectionSummary();
    });
  });
}

function renderGenerators() {
  const families = ["GAN", "Diffusion", "SDV"];
  els.generatorGroups.innerHTML = families
    .map((family) => {
      const gens = state.generators.filter((g) => g.family === family);
      return `
        <div class="generator-family">
          <h3>${family}</h3>
          <div class="cards">
            ${gens
              .map((g) => {
                const selected = state.selectedGenerators.has(g.id);
                const unavailable = !g.available;
                return `
                  <button
                    type="button"
                    class="generator-card ${selected ? "selected" : ""} ${unavailable ? "unavailable" : ""}"
                    data-id="${g.id}"
                    ${unavailable ? "disabled title=\"" + (g.reason || "Unavailable") + "\"" : ""}
                    aria-pressed="${selected}"
                  >
                    <h3>${g.name}</h3>
                    <p>${g.description}</p>
                    ${g.auto_setup ? '<div class="meta-line">Auto-setup on first use</div>' : ""}
                  </button>
                `;
              })
              .join("")}
          </div>
        </div>
      `;
    })
    .join("");

  els.generatorGroups.querySelectorAll(".generator-card:not(:disabled)").forEach((card) => {
    card.addEventListener("click", () => {
      const id = card.dataset.id;
      if (state.selectedGenerators.has(id)) {
        state.selectedGenerators.delete(id);
      } else {
        state.selectedGenerators.add(id);
      }
      renderGenerators();
      updateSelectionSummary();
    });
  });
}

function bindFilters() {
  document.querySelectorAll(".filter-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".filter-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.taskFilter = tab.dataset.filter;
      renderDatasets();
    });
  });

  els.selectAllGenerators.addEventListener("click", () => {
    state.generators.filter((g) => g.available).forEach((g) => state.selectedGenerators.add(g.id));
    renderGenerators();
    updateSelectionSummary();
  });

  els.clearGenerators.addEventListener("click", () => {
    state.selectedGenerators.clear();
    renderGenerators();
    updateSelectionSummary();
  });

  els.nSamples.addEventListener("input", updateSelectionSummary);
  els.seed.addEventListener("input", updateSelectionSummary);
}

async function startGeneration() {
  els.generateBtn.disabled = true;
  els.resultsPanel.classList.remove("hidden");
  els.previewSection.classList.add("hidden");
  els.errorBox.classList.add("hidden");
  els.resultCards.innerHTML = "";
  setJobUi({ status: "queued", progress: 0, message: "Starting job…" });

  try {
    const job = await api("/api/generate", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: state.selectedDataset,
        generator_ids: [...state.selectedGenerators],
        n_samples: Number(els.nSamples.value),
        seed: Number(els.seed.value),
      }),
    });
    state.activeJobId = job.job_id;
    pollJob();
  } catch (err) {
    setJobUi({ status: "failed", progress: 1, message: "Failed to start", error: err.message });
    els.generateBtn.disabled = false;
  }
}

function setJobUi({ status, progress, message, error }) {
  els.jobStatusChip.textContent = status;
  els.jobStatusChip.className = `status-chip ${status}`;
  els.progressBar.style.width = `${Math.round((progress || 0) * 100)}%`;
  els.statusMessage.textContent = message || "";
  if (error) {
    els.errorBox.textContent = error;
    els.errorBox.classList.remove("hidden");
  }
}

async function pollJob() {
  if (!state.activeJobId) return;
  try {
    const job = await api(`/api/jobs/${state.activeJobId}`);
    setJobUi(job);
    if (job.status === "running" || job.status === "queued") {
      state.pollTimer = setTimeout(pollJob, 1200);
      return;
    }
    if (job.status === "completed") {
      renderResultCards(job);
      els.generateBtn.disabled = false;
      return;
    }
    if (job.status === "failed") {
      setJobUi({ ...job, error: job.error || "Generation failed" });
      els.generateBtn.disabled = false;
    }
  } catch (err) {
    setJobUi({ status: "failed", progress: 1, message: "Polling failed", error: err.message });
    els.generateBtn.disabled = false;
  }
}

function renderResultCards(job) {
  const entries = Object.entries(job.results || {});
  els.resultCards.innerHTML = entries
    .map(
      ([id, meta]) => `
      <div class="result-card" data-id="${id}">
        <h4>${meta.generator_name}</h4>
        <div class="score">${meta.quality_score != null ? `Quality: ${meta.quality_score.toFixed(3)}` : "Quality: n/a"}</div>
        <div class="meta-line">${meta.rows} rows</div>
      </div>
    `
    )
    .join("");

  els.resultCards.querySelectorAll(".result-card").forEach((card) => {
    card.addEventListener("click", () => showPreview(card.dataset.id));
  });

  if (entries.length) {
    showPreview(entries[0][0]);
  }
}

async function showPreview(generatorId) {
  state.activeGenerator = generatorId;
  els.resultCards.querySelectorAll(".result-card").forEach((c) => {
    c.classList.toggle("active", c.dataset.id === generatorId);
  });

  const gen = state.generators.find((g) => g.id === generatorId);
  els.previewTitle.textContent = `${gen?.name || generatorId} — preview`;
  els.previewSection.classList.remove("hidden");

  const preview = await api(`/api/jobs/${state.activeJobId}/preview/${generatorId}?limit=15`);
  const thead = els.previewTable.querySelector("thead");
  const tbody = els.previewTable.querySelector("tbody");
  thead.innerHTML = `<tr>${preview.columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  tbody.innerHTML = preview.rows
    .map(
      (row) =>
        `<tr>${preview.columns.map((c) => `<td>${row[c] ?? ""}</td>`).join("")}</tr>`
    )
    .join("");
}

els.downloadActive.addEventListener("click", () => {
  if (state.activeJobId && state.activeGenerator) {
    window.location.href = `/api/jobs/${state.activeJobId}/download/${state.activeGenerator}`;
  }
});

els.generateBtn.addEventListener("click", startGeneration);

async function init() {
  bindFilters();
  [state.datasets, state.generators] = await Promise.all([
    api("/api/datasets"),
    api("/api/generators"),
  ]);

  state.generators
    .filter((g) => g.available)
    .slice(0, 2)
    .forEach((g) => state.selectedGenerators.add(g.id));

  renderDatasets();
  renderGenerators();
  updateSelectionSummary();
}

init().catch((err) => {
  els.selectionSummary.textContent = `Failed to load app data: ${err.message}`;
});
