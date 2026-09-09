/**
 * CreditPulse MLOps Simulator - Core Frontend Controller
 * Integrates live FastAPI inference with fallback client-side prediction engine.
 */

(() => {
  // DOM Elements - Config & Status
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const btnSettings = document.getElementById("btnSettings");
  const endpointLabel = document.getElementById("endpointLabel");
  const settingsModal = document.getElementById("settingsModal");
  const btnCloseModal = document.getElementById("btnCloseModal");
  const btnCancelModal = document.getElementById("btnCancelModal");
  const btnSaveEndpoint = document.getElementById("btnSaveEndpoint");
  const apiEndpointInput = document.getElementById("apiEndpointInput");
  const btnSetLocal = document.getElementById("btnSetLocal");
  const btnSetDemo = document.getElementById("btnSetDemo");

  // DOM Elements - Form Fields
  const form = document.getElementById("creditForm");
  const applicantId = document.getElementById("applicantId");
  const btnRandomId = document.getElementById("btnRandomId");
  const autoEvalToggle = document.getElementById("autoEvalToggle");

  const ageInput = document.getElementById("ageInput");
  const ageSlider = document.getElementById("ageSlider");
  const ageValDisplay = document.getElementById("ageValDisplay");

  const incomeInput = document.getElementById("incomeInput");
  const incomeSlider = document.getElementById("incomeSlider");
  const incomeValDisplay = document.getElementById("incomeValDisplay");

  const loanInput = document.getElementById("loanInput");
  const loanSlider = document.getElementById("loanSlider");
  const loanValDisplay = document.getElementById("loanValDisplay");

  const creditScoreInput = document.getElementById("creditScoreInput");
  const creditScoreSlider = document.getElementById("creditScoreSlider");
  const scoreTierPill = document.getElementById("scoreTierPill");

  const dtiInput = document.getElementById("dtiInput");
  const dtiSlider = document.getElementById("dtiSlider");
  const dtiValDisplay = document.getElementById("dtiValDisplay");

  const loanIntentSelect = document.getElementById("loanIntentSelect");
  const employmentSelect = document.getElementById("employmentSelect");
  const defaultToggle = document.getElementById("defaultToggle");
  const btnEvaluate = document.getElementById("btnEvaluate");

  // DOM Elements - Results & Visuals
  const decisionCard = document.getElementById("decisionCard");
  const riskTierBadge = document.getElementById("riskTierBadge");
  const gaugeFill = document.getElementById("gaugeFill");
  const probPercent = document.getElementById("probPercent");
  const decisionBanner = document.getElementById("decisionBanner");
  const decisionIcon = document.getElementById("decisionIcon");
  const decisionText = document.getElementById("decisionText");
  const decisionSummary = document.getElementById("decisionSummary");

  // Telemetry Elements
  const latencyVal = document.getElementById("latencyVal");
  const modelVersionVal = document.getElementById("modelVersionVal");
  const engineVal = document.getElementById("engineVal");

  // Inspector Elements
  const btnToggleInspector = document.getElementById("btnToggleInspector");
  const inspectorContent = document.getElementById("inspectorContent");
  const inspectorArrow = document.getElementById("inspectorArrow");
  const jsonCodeViewer = document.getElementById("jsonCodeViewer");
  const tabBtns = document.querySelectorAll(".tab-btn");

  // What-If Elements
  const btnBoostIncome = document.getElementById("btnBoostIncome");
  const btnBoostScore = document.getElementById("btnBoostScore");
  const btnToggleDefault = document.getElementById("btnToggleDefault");

  // Presets
  const presetPrime = document.getElementById("presetPrime");
  const presetStandard = document.getElementById("presetStandard");
  const presetSubprime = document.getElementById("presetSubprime");

  // App State
  let apiBaseUrl = localStorage.getItem("creditpulse_api") || "http://localhost:8000";
  let isBackendOnline = false;
  let isDemoMode = false;
  let activeTab = "response";
  let lastRequestJson = {};
  let lastResponseJson = {};
  let debounceTimer = null;

  // Preset Definitions
  const PRESETS = {
    prime: {
      age: 36,
      annual_income: 95000,
      loan_amount: 14000,
      credit_score: 790,
      debt_to_income_ratio: 0.15,
      loan_intent: "home_improvement",
      employment_status: "employed",
      historical_default: false,
    },
    standard: {
      age: 31,
      annual_income: 55000,
      loan_amount: 15000,
      credit_score: 680,
      debt_to_income_ratio: 0.27,
      loan_intent: "debt_consolidation",
      employment_status: "employed",
      historical_default: false,
    },
    subprime: {
      age: 22,
      annual_income: 22000,
      loan_amount: 25000,
      credit_score: 490,
      debt_to_income_ratio: 1.14,
      loan_intent: "personal",
      employment_status: "unemployed",
      historical_default: true,
    },
  };

  // Utilities
  const formatCurrency = (val) => "$" + Number(val).toLocaleString("en-US");

  const generateRandomId = () => "app_" + Math.floor(10000 + Math.random() * 90000);

  // Synchronize Controls
  function bindDualControl(slider, input, displayEl, formatFn) {
    slider.addEventListener("input", () => {
      input.value = slider.value;
      if (displayEl) displayEl.textContent = formatFn ? formatFn(slider.value) : slider.value;
      triggerAutoEvaluation();
    });

    input.addEventListener("input", () => {
      slider.value = input.value;
      if (displayEl) displayEl.textContent = formatFn ? formatFn(input.value) : input.value;
      triggerAutoEvaluation();
    });
  }

  // Update Score Tier Pill
  function updateScoreTier(score) {
    const num = Number(score);
    if (num >= 740) {
      scoreTierPill.textContent = `Excellent (${num})`;
      scoreTierPill.style.background = "rgba(16, 185, 129, 0.15)";
      scoreTierPill.style.color = "#10b981";
      scoreTierPill.style.borderColor = "rgba(16, 185, 129, 0.3)";
    } else if (num >= 670) {
      scoreTierPill.textContent = `Good (${num})`;
      scoreTierPill.style.background = "rgba(14, 165, 233, 0.15)";
      scoreTierPill.style.color = "#38bdf8";
      scoreTierPill.style.borderColor = "rgba(14, 165, 233, 0.3)";
    } else if (num >= 580) {
      scoreTierPill.textContent = `Fair (${num})`;
      scoreTierPill.style.background = "rgba(245, 158, 11, 0.15)";
      scoreTierPill.style.color = "#f59e0b";
      scoreTierPill.style.borderColor = "rgba(245, 158, 11, 0.3)";
    } else {
      scoreTierPill.textContent = `Poor (${num})`;
      scoreTierPill.style.background = "rgba(244, 63, 94, 0.15)";
      scoreTierPill.style.color = "#f43f5e";
      scoreTierPill.style.borderColor = "rgba(244, 63, 94, 0.3)";
    }
  }

  // Apply Persona Preset
  function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;

    applicantId.value = generateRandomId();
    ageInput.value = p.age;
    ageSlider.value = p.age;
    ageValDisplay.textContent = `${p.age} yrs`;

    incomeInput.value = p.annual_income;
    incomeSlider.value = p.annual_income;
    incomeValDisplay.textContent = formatCurrency(p.annual_income);

    loanInput.value = p.loan_amount;
    loanSlider.value = p.loan_amount;
    loanValDisplay.textContent = formatCurrency(p.loan_amount);

    creditScoreInput.value = p.credit_score;
    creditScoreSlider.value = p.credit_score;
    updateScoreTier(p.credit_score);

    dtiInput.value = p.debt_to_income_ratio;
    dtiSlider.value = p.debt_to_income_ratio;
    dtiValDisplay.textContent = p.debt_to_income_ratio.toFixed(2);

    loanIntentSelect.value = p.loan_intent;
    employmentSelect.value = p.employment_status;
    defaultToggle.checked = p.historical_default;

    scoreApplication();
  }

  // Healthcheck Ping
  async function checkBackendHealth() {
    if (isDemoMode) {
      statusDot.className = "status-dot offline";
      statusText.textContent = "Offline Demo Engine";
      endpointLabel.textContent = "Engine: Client-Side";
      return;
    }

    try {
      const resp = await fetch(`${apiBaseUrl}/health`, { method: "GET" });
      if (resp.ok) {
        const data = await resp.json();
        isBackendOnline = data.model_loaded === true;
        statusDot.className = "status-dot online";
        statusText.textContent = `Live API (${data.status.toUpperCase()})`;
        endpointLabel.textContent = `API: ${apiBaseUrl.replace(/^https?:\/\//, "")}`;
        if (data.model_version) modelVersionVal.textContent = `v${data.model_version} (DVC)`;
      } else {
        throw new Error("HTTP " + resp.status);
      }
    } catch {
      isBackendOnline = false;
      statusDot.className = "status-dot error";
      statusText.textContent = "Backend Offline (Demo Fallback)";
      endpointLabel.textContent = "API: Offline";
    }
  }

  // Build Payload
  function getPayload() {
    return {
      applicant_id: applicantId.value.trim() || generateRandomId(),
      age: parseInt(ageInput.value, 10),
      annual_income: parseFloat(incomeInput.value),
      loan_amount: parseFloat(loanInput.value),
      credit_score: parseInt(creditScoreInput.value, 10),
      debt_to_income_ratio: parseFloat(dtiInput.value),
      loan_intent: loanIntentSelect.value,
      employment_status: employmentSelect.value,
      historical_default: defaultToggle.checked,
    };
  }

  // Client-Side Approximation Fallback Engine
  function computeClientSideFallback(payload) {
    const t0 = performance.now();
    const age = payload.age;
    const income = payload.annual_income;
    const loan = payload.loan_amount;
    const score = payload.credit_score;
    const dti = payload.debt_to_income_ratio;
    const isUnemployed = payload.employment_status === "unemployed";
    const histDefault = payload.historical_default ? 1 : 0;

    // Direct mathematical mirror of training log-odds function
    const logOdds =
      -1.5 -
      (score - 650) / 75.0 +
      (dti - 0.25) * 3.5 -
      (income - 50000) / 45000.0 +
      (loan - 15000) / 20000.0 +
      histDefault * 1.8 +
      (isUnemployed ? 1.2 : 0);

    const prob = 1.0 / (1.0 + Math.exp(-logOdds));
    const defaultProb = Math.min(Math.max(parseFloat(prob.toFixed(4)), 0.0001), 0.9999);

    let riskTier = "LOW";
    if (defaultProb >= 0.45) riskTier = "HIGH";
    else if (defaultProb >= 0.20) riskTier = "MEDIUM";

    const approved = defaultProb < 0.35;
    const latency = parseFloat((performance.now() - t0).toFixed(2));

    return {
      applicant_id: payload.applicant_id,
      default_probability: defaultProb,
      risk_tier: riskTier,
      approved: approved,
      model_version: "1.0.0 (Client Simulator)",
      latency_ms: latency,
    };
  }

  // Render Prediction Result in UI
  function renderResult(result, isLiveApi) {
    lastResponseJson = result;

    const prob = result.default_probability;
    const percentStr = (prob * 100).toFixed(1) + "%";
    probPercent.textContent = percentStr;

    // Radial Gauge SVG Arc: Total circumference of half circle (r=80) is 251.2
    const totalArc = 251.2;
    const offset = Math.max(totalArc - prob * totalArc, 0);
    gaugeFill.style.strokeDashoffset = offset;

    // Visual Themes by Tier
    decisionCard.classList.remove("high-risk", "medium-risk");
    riskTierBadge.classList.remove("medium", "high");
    gaugeFill.classList.remove("medium", "high");

    if (result.risk_tier === "HIGH") {
      decisionCard.classList.add("high-risk");
      riskTierBadge.classList.add("high");
      gaugeFill.classList.add("high");
      riskTierBadge.textContent = "HIGH RISK";
    } else if (result.risk_tier === "MEDIUM") {
      decisionCard.classList.add("medium-risk");
      riskTierBadge.classList.add("medium");
      gaugeFill.classList.add("medium");
      riskTierBadge.textContent = "MEDIUM RISK";
    } else {
      riskTierBadge.textContent = "LOW RISK";
    }

    // Decision Banner
    if (result.approved) {
      decisionBanner.className = "decision-banner approved";
      decisionIcon.textContent = "✓";
      decisionText.textContent = "APPLICATION APPROVED";
      decisionSummary.textContent = `Credit application passed underwriting standards. Probability of default (${percentStr}) is within safe bounds (< 35.0%).`;
    } else {
      decisionBanner.className = "decision-banner rejected";
      decisionIcon.textContent = "✕";
      decisionText.textContent = "APPLICATION REJECTED";
      decisionSummary.textContent = `Application flagged due to elevated default risk (${percentStr}). Exceeds the automated credit risk tolerance limit of 35.0%.`;
    }

    // Telemetry Update
    latencyVal.textContent = `${result.latency_ms} ms`;
    modelVersionVal.textContent = result.model_version;
    engineVal.textContent = isLiveApi ? "FastAPI Live REST" : "Client Engine Fallback";

    // Inspector Viewer
    updateJsonViewer();
  }

  // Execute Scoring Request
  async function scoreApplication() {
    const payload = getPayload();
    lastRequestJson = payload;

    btnEvaluate.classList.add("loading");

    if (isBackendOnline && !isDemoMode) {
      try {
        const t0 = performance.now();
        const response = await fetch(`${apiBaseUrl}/v1/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (response.ok) {
          const data = await response.json();
          // Read server process header if available
          const headerLatency = response.headers.get("X-Process-Time-Ms");
          if (headerLatency) data.latency_ms = parseFloat(headerLatency);
          renderResult(data, true);
        } else {
          // If 422 or 500, display error in inspector
          const errData = await response.json();
          lastResponseJson = { http_status: response.status, ...errData };
          updateJsonViewer();
        }
      } catch (err) {
        console.warn("Backend error, falling back to local simulator:", err);
        isBackendOnline = false;
        statusDot.className = "status-dot error";
        statusText.textContent = "Backend Error (Demo Fallback)";
        const fallback = computeClientSideFallback(payload);
        renderResult(fallback, false);
      }
    } else {
      const fallback = computeClientSideFallback(payload);
      renderResult(fallback, false);
    }

    btnEvaluate.classList.remove("loading");
  }

  // Debounced auto evaluation
  function triggerAutoEvaluation() {
    if (!autoEvalToggle.checked) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(scoreApplication, 120);
  }

  // Update JSON Code Viewer
  function updateJsonViewer() {
    const obj = activeTab === "response" ? lastResponseJson : lastRequestJson;
    jsonCodeViewer.textContent = JSON.stringify(obj, null, 2);
  }

  // Setup Event Listeners
  function initListeners() {
    // Sliders Dual Controls
    bindDualControl(ageSlider, ageInput, ageValDisplay, (v) => `${v} yrs`);
    bindDualControl(incomeSlider, incomeInput, incomeValDisplay, formatCurrency);
    bindDualControl(loanSlider, loanInput, loanValDisplay, formatCurrency);
    bindDualControl(creditScoreSlider, creditScoreInput, null, (v) => {
      updateScoreTier(v);
      return v;
    });
    creditScoreSlider.addEventListener("input", (e) => updateScoreTier(e.target.value));
    creditScoreInput.addEventListener("input", (e) => updateScoreTier(e.target.value));

    bindDualControl(dtiSlider, dtiInput, dtiValDisplay, (v) => parseFloat(v).toFixed(2));

    loanIntentSelect.addEventListener("change", triggerAutoEvaluation);
    employmentSelect.addEventListener("change", triggerAutoEvaluation);
    defaultToggle.addEventListener("change", triggerAutoEvaluation);

    // Form submission
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      scoreApplication();
    });

    btnRandomId.addEventListener("click", () => {
      applicantId.value = generateRandomId();
      triggerAutoEvaluation();
    });

    // Presets
    presetPrime.addEventListener("click", () => applyPreset("prime"));
    presetStandard.addEventListener("click", () => applyPreset("standard"));
    presetSubprime.addEventListener("click", () => applyPreset("subprime"));

    // What-If Sensitivity Lab
    btnBoostIncome.addEventListener("click", () => {
      const current = parseFloat(incomeInput.value) || 50000;
      const next = Math.min(current + 25000, 300000);
      incomeInput.value = next;
      incomeSlider.value = next;
      incomeValDisplay.textContent = formatCurrency(next);

      // Recalculate DTI
      const loan = parseFloat(loanInput.value) || 15000;
      const newDti = parseFloat((loan / next).toFixed(2));
      dtiInput.value = newDti;
      dtiSlider.value = newDti;
      dtiValDisplay.textContent = newDti.toFixed(2);

      scoreApplication();
    });

    btnBoostScore.addEventListener("click", () => {
      const current = parseInt(creditScoreInput.value, 10) || 650;
      const next = Math.min(current + 50, 850);
      creditScoreInput.value = next;
      creditScoreSlider.value = next;
      updateScoreTier(next);
      scoreApplication();
    });

    btnToggleDefault.addEventListener("click", () => {
      defaultToggle.checked = !defaultToggle.checked;
      scoreApplication();
    });

    // Inspector Drawer
    btnToggleInspector.addEventListener("click", () => {
      const isHidden = inspectorContent.style.display === "none";
      inspectorContent.style.display = isHidden ? "block" : "none";
      inspectorArrow.classList.toggle("open", isHidden);
    });

    tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        tabBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeTab = btn.dataset.tab;
        updateJsonViewer();
      });
    });

    // Settings Modal
    btnSettings.addEventListener("click", () => {
      apiEndpointInput.value = apiBaseUrl;
      settingsModal.classList.add("open");
    });

    const closeModal = () => settingsModal.classList.remove("open");
    btnCloseModal.addEventListener("click", closeModal);
    btnCancelModal.addEventListener("click", closeModal);

    btnSetLocal.addEventListener("click", () => {
      apiEndpointInput.value = "http://localhost:8000";
    });

    btnSetDemo.addEventListener("click", () => {
      isDemoMode = true;
      closeModal();
      checkBackendHealth();
      scoreApplication();
    });

    btnSaveEndpoint.addEventListener("click", async () => {
      const val = apiEndpointInput.value.trim().replace(/\/+$/, "");
      if (val) {
        apiBaseUrl = val;
        localStorage.setItem("creditpulse_api", apiBaseUrl);
        isDemoMode = false;
        closeModal();
        await checkBackendHealth();
        scoreApplication();
      }
    });
  }

  // Initialize Application
  async function init() {
    initListeners();
    updateScoreTier(creditScoreInput.value);
    await checkBackendHealth();
    scoreApplication();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
