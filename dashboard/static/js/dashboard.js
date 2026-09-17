/**
 * AI-Based Real-Time Stress & Affect Detection Dashboard Frontend Logic
 * Real-Time WESAD Simulation, Dynamic Sensors, Chart.js Visualizations & XAI Integration
 */

document.addEventListener("DOMContentLoaded", function () {
    // Simulation state variables
    let currentWindowIndex = 0;
    let totalWindows = 0; // Initialized dynamically after fetching /api/simulation/windows
    let isReplaying = false;
    let replayTimer = null;
    const REPLAY_INTERVAL_MS = 2500; // 2.5 seconds per window

    // Chart instances
    let confidenceChart = null;
    let shapChart = null;

    // DOM Elements
    const btnToggleSim = document.getElementById("btn-toggle-sim");
    const btnNextStep = document.getElementById("btn-next-step");
    const simStatusLabel = document.getElementById("sim-status-label");

    const simSubjectId = document.getElementById("sim-subject-id");
    const simWindowId = document.getElementById("sim-window-id");
    const simTrueLabel = document.getElementById("sim-true-label");
    const simPredLabel = document.getElementById("sim-pred-label");
    const simConfidenceVal = document.getElementById("sim-confidence-val");
    const heroBadge = document.getElementById("hero-badge");
    const simTimestamp = document.getElementById("sim-timestamp");

    // Sensor DOM Elements
    const snsWTemp = document.getElementById("sns-w-temp");
    const snsWTempRange = document.getElementById("sns-w-temp-range");
    const snsWHr = document.getElementById("sns-w-hr");
    const snsWHrRange = document.getElementById("sns-w-hr-range");
    const snsWEda = document.getElementById("sns-w-eda");
    const snsCEcg = document.getElementById("sns-c-ecg");
    const snsCEcgStd = document.getElementById("sns-c-ecg-std");
    const snsCEda = document.getElementById("sns-c-eda");
    const snsCResp = document.getElementById("sns-c-resp");
    const snsAccMag = document.getElementById("sns-acc-mag");

    // XAI DOM Elements
    const xaiSampleTrue = document.getElementById("xai-sample-true");
    const xaiSamplePred = document.getElementById("xai-sample-pred");
    const xaiSampleConf = document.getElementById("xai-sample-conf");
    const xaiPosList = document.getElementById("xai-pos-list");
    const xaiNegList = document.getElementById("xai-neg-list");

    // Performance Summary Metric DOM Elements
    const perfAccuracy = document.getElementById("perf-accuracy");
    const perfPrecision = document.getElementById("perf-precision");
    const perfRecall = document.getElementById("perf-recall");
    const perfF1 = document.getElementById("perf-f1");

    // Initialize Charts
    initConfidenceChart();
    initShapChart();

    // Fetch initial dataset, performance metrics, and XAI information
    fetchPerformanceMetrics();
    fetchSimulationWindowsCount().then(() => {
        // Run first initial prediction (Window 0) after totalWindows is dynamically initialized
        fetchAndDisplayPrediction(0);
    });
    fetchXAIImportance();
    fetchXAISampleExplanation();

    // Event Listeners for Simulation Controls
    if (btnToggleSim) {
        btnToggleSim.addEventListener("click", toggleSimulationReplay);
    }
    if (btnNextStep) {
        btnNextStep.addEventListener("click", stepNextWindow);
    }

    // Controlled Sidebar Navigation Handler
    const navItems = document.querySelectorAll(".nav-item");
    const navbarHeight = 85; // 70px sticky navbar + 15px margin offset

    navItems.forEach(item => {
        item.addEventListener("click", function (e) {
            e.preventDefault();

            const targetId = this.getAttribute("data-target");
            if (!targetId) return;

            const targetSection = document.getElementById(targetId);
            if (!targetSection) return;

            // Highlight active sidebar button
            navItems.forEach(i => i.classList.remove("active"));
            this.classList.add("active");

            // Calculate exact position relative to window top, accounting for sticky navbar offset
            const targetPosition = targetSection.getBoundingClientRect().top + window.pageYOffset - navbarHeight;

            // Execute INSTANT single scroll pass
            window.scrollTo({
                top: targetPosition,
                behavior: "auto"
            });

            // Update URL hash state silently without triggering browser scroll events
            if (window.history && window.history.replaceState) {
                window.history.replaceState(null, "", "#" + targetId);
            }
        });
    });

    // --- FUNCTIONS ---

    function initConfidenceChart() {
        const ctx = document.getElementById("confidenceChart").getContext("2d");
        confidenceChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Neutral", "Stress", "Amusement"],
                datasets: [{
                    label: "Probability (%)",
                    data: [0, 0, 0],
                    backgroundColor: [
                        "rgba(14, 165, 233, 0.75)", // Neutral (blue)
                        "rgba(239, 68, 68, 0.75)",  // Stress (red)
                        "rgba(16, 185, 129, 0.75)"  // Amusement (green)
                    ],
                    borderColor: [
                        "#0ea5e9",
                        "#ef4444",
                        "#10b981"
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(255, 255, 255, 0.08)" },
                        ticks: { color: "#94a3b8" }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#f8fafc", font: { weight: "700" } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return context.parsed.y.toFixed(2) + "%";
                            }
                        }
                    }
                }
            }
        });
    }

    function initShapChart() {
        const ctx = document.getElementById("shapImportanceChart").getContext("2d");
        shapChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "Mean |SHAP Value|",
                    data: [],
                    backgroundColor: "rgba(99, 102, 241, 0.75)",
                    borderColor: "#6366f1",
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: { color: "rgba(255, 255, 255, 0.08)" },
                        ticks: { color: "#94a3b8" }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: "#f8fafc", font: { size: 11 } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function fetchPerformanceMetrics() {
        fetch("/api/performance/comparison")
            .then(res => res.json())
            .then(data => {
                if (data.federated) {
                    const fl = data.federated;
                    if (perfAccuracy) perfAccuracy.textContent = fl.accuracy;
                    if (perfPrecision) perfPrecision.textContent = fl.precision_macro;
                    if (perfRecall) perfRecall.textContent = fl.recall_macro;
                    if (perfF1) perfF1.textContent = fl.f1_macro;
                }
            })
            .catch(err => console.error("Error fetching performance metrics:", err));
    }

    function fetchSimulationWindowsCount() {
        return fetch("/api/simulation/windows")
            .then(res => res.json())
            .then(data => {
                if (data.total_test_windows) {
                    totalWindows = data.total_test_windows;
                }
            })
            .catch(err => console.error("Error fetching simulation windows:", err));
    }

    function toggleSimulationReplay() {
        if (isReplaying) {
            pauseSimulation();
        } else {
            startSimulation();
        }
    }

    function startSimulation() {
        isReplaying = true;
        btnToggleSim.textContent = "⏸ Pause Replay";
        btnToggleSim.className = "btn btn-secondary";
        simStatusLabel.textContent = "Replaying (2.5s per window)";
        simStatusLabel.style.color = "#34d399";

        replayTimer = setInterval(() => {
            stepNextWindow();
        }, REPLAY_INTERVAL_MS);
    }

    function pauseSimulation() {
        isReplaying = false;
        clearInterval(replayTimer);
        btnToggleSim.textContent = "▶ Start Simulation Replay";
        btnToggleSim.className = "btn btn-primary";
        simStatusLabel.textContent = "Paused";
        simStatusLabel.style.color = "#94a3b8";
    }

    function stepNextWindow() {
        if (!totalWindows || totalWindows <= 0) return;
        currentWindowIndex = (currentWindowIndex + 1) % totalWindows;
        fetchAndDisplayPrediction(currentWindowIndex);
    }

    function fetchAndDisplayPrediction(idx) {
        fetch(`/api/simulation/predict/${idx}`)
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    console.error("Prediction API error:", data.error);
                    return;
                }

                // Update Overview/Simulation DOM
                simSubjectId.textContent = data.subject_id;
                simWindowId.textContent = data.window_id;
                simTrueLabel.textContent = data.true_label;

                simPredLabel.textContent = data.predicted_label;
                setLabelStyle(simPredLabel, data.predicted_label);
                setHeroBadgeStyle(heroBadge, data.predicted_label);

                simConfidenceVal.textContent = data.prediction_confidence.toFixed(1) + "%";

                // Update Timestamp
                const now = new Date();
                if (simTimestamp) {
                    simTimestamp.textContent = now.toTimeString().split(' ')[0];
                }

                // Update Confidence Chart
                if (confidenceChart && data.class_probabilities) {
                    confidenceChart.data.datasets[0].data = [
                        data.class_probabilities.Neutral || 0,
                        data.class_probabilities.Stress || 0,
                        data.class_probabilities.Amusement || 0
                    ];
                    confidenceChart.update();
                }

                // Update Sensor Panel
                if (data.sensor_values) {
                    const s = data.sensor_values;
                    snsWTemp.textContent = `${s.wrist_temp_mean} °C`;
                    snsWTempRange.textContent = `Min: ${s.wrist_temp_min}°C | Max: ${s.wrist_temp_max}°C`;

                    snsWHr.textContent = `${s.wrist_hr_mean} BPM`;
                    snsWHrRange.textContent = `Min: ${s.wrist_hr_min} | Max: ${s.wrist_hr_max}`;

                    snsWEda.textContent = `${s.wrist_eda_mean} μS`;

                    snsCEcg.textContent = `${s.chest_ecg_mean} mV`;
                    snsCEcgStd.textContent = `Std Dev: ${s.chest_ecg_std}`;

                    snsCEda.textContent = `${s.chest_eda_mean} μS`;
                    snsCResp.textContent = `${s.chest_resp_mean}`;

                    const wristG = (s.wrist_acc_mag_mean / 64.0).toFixed(2);
                    snsAccMag.textContent = `Chest: ${s.chest_acc_mag_mean}g | Wrist: ${wristG}g (${s.wrist_acc_mag_mean} LSB)`;
                }
            })
            .catch(err => console.error("Error fetching window prediction:", err));
    }

    function setLabelStyle(element, labelName) {
        if (!element) return;
        if (labelName === "Neutral") {
            element.style.color = "#38bdf8";
        } else if (labelName === "Stress") {
            element.style.color = "#f87171";
        } else if (labelName === "Amusement") {
            element.style.color = "#34d399";
        }
    }

    function setHeroBadgeStyle(element, labelName) {
        if (!element) return;
        element.textContent = labelName.toUpperCase();
        element.className = "badge";
        if (labelName === "Neutral") {
            element.classList.add("badge-neutral");
        } else if (labelName === "Stress") {
            element.classList.add("badge-stress");
        } else if (labelName === "Amusement") {
            element.classList.add("badge-success");
        } else {
            element.classList.add("badge-secondary");
        }
    }

    function fetchXAIImportance() {
        fetch("/api/xai/importance")
            .then(res => res.json())
            .then(data => {
                if (Array.isArray(data) && shapChart) {
                    // Top 10 features
                    const topFeatures = data.slice(0, 10);
                    shapChart.data.labels = topFeatures.map(item => item.feature);
                    shapChart.data.datasets[0].data = topFeatures.map(item => item.mean_abs_shap);
                    shapChart.update();
                }
            })
            .catch(err => console.error("Error fetching SHAP importance:", err));
    }

    function fetchXAISampleExplanation() {
        fetch("/api/xai/sample_explanation")
            .then(res => res.json())
            .then(data => {
                if (data.error) return;

                xaiSampleTrue.textContent = data.true_label;
                xaiSamplePred.textContent = data.predicted_label;
                xaiSampleConf.textContent = `${(data.prediction_confidence * 100).toFixed(1)}%`;

                // Render Positive Attributions
                xaiPosList.innerHTML = "";
                if (data.top_positive_attributions) {
                    data.top_positive_attributions.forEach(item => {
                        const li = document.createElement("li");
                        li.innerHTML = `<span><strong>${item.feature}</strong> = ${item.feature_value.toFixed(3)}</span> <span class="pos-heading">+${item.shap_value.toFixed(4)}</span>`;
                        xaiPosList.appendChild(li);
                    });
                }

                // Render Negative Attributions
                xaiNegList.innerHTML = "";
                if (data.top_negative_attributions) {
                    data.top_negative_attributions.forEach(item => {
                        const li = document.createElement("li");
                        li.innerHTML = `<span><strong>${item.feature}</strong> = ${item.feature_value.toFixed(3)}</span> <span class="neg-heading">${item.shap_value.toFixed(4)}</span>`;
                        xaiNegList.appendChild(li);
                    });
                }
            })
            .catch(err => console.error("Error fetching XAI sample explanation:", err));
    }
});
