let goldChart = null;

const fmt = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
};

async function getJson(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "API error");
  return data;
}

function renderTechnical(data) {
  const items = [
    ["MA 7", fmt(data.ma7)], ["MA 20", fmt(data.ma20)],
    ["MA 50", fmt(data.ma50)], ["MA 200", fmt(data.ma200)],
    ["RSI 14", fmt(data.rsi14)], ["MACD", fmt(data.macd, 4)],
    ["MACD Signal", fmt(data.macd_signal, 4)], ["Volatility 20", `${fmt(data.volatility20 * 100, 2)}%`],
  ];
  document.getElementById("technicalMetrics").innerHTML = items.map(([label, value]) =>
    `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");
}

function renderModel(data) {
  const m = data.metrics || {};
  const items = [
    ["Model", data.model || "--"],
    ["MAE", fmt(m.MAE)],
    ["RMSE", fmt(m.RMSE)],
    ["MAPE", `${fmt(m.MAPE_percent)}%`],
    ["Train rows", Number(data.train_rows || 0).toLocaleString()],
    ["Test rows", Number(data.test_rows || 0).toLocaleString()],
  ];
  document.getElementById("modelMetrics").innerHTML = items.map(([label, value]) =>
    `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");
}

function renderChart(data) {
  const labels = data.data.map(x => x.date);
  const close = data.data.map(x => x.close);
  const ma20 = data.data.map(x => x.ma20);
  const ma50 = data.data.map(x => x.ma50);

  const ctx = document.getElementById("goldChart");
  if (goldChart) goldChart.destroy();
  goldChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Close", data: close, tension: 0.2, pointRadius: 0 },
        { label: "MA20", data: ma20, tension: 0.2, pointRadius: 0 },
        { label: "MA50", data: ma50, tension: 0.2, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: { x: { ticks: { maxTicksLimit: 10 } } },
    },
  });

  const dates = labels;
  document.getElementById("dateRange").textContent = dates.length
    ? `${dates[0]} → ${dates[dates.length - 1]}`
    : "Không có dữ liệu";
}

async function loadDashboard() {
  const days = document.getElementById("daysSelect").value;
  const [history, technical, forecast, model] = await Promise.all([
    getJson(`/api/history?limit=${days}`),
    getJson("/api/technical"),
    getJson("/api/forecast"),
    getJson("/api/model"),
  ]);

  document.getElementById("currentPrice").textContent = fmt(forecast.current_close);
  document.getElementById("predictedPrice").textContent = fmt(forecast.predicted_next_close);
  document.getElementById("changePct").textContent = `${forecast.expected_change_percent >= 0 ? "+" : ""}${fmt(forecast.expected_change_percent)}%`;
  document.getElementById("rsi").textContent = fmt(technical.rsi14);

  renderChart(history);
  renderTechnical(technical);
  renderModel(model);
}

document.getElementById("refreshBtn").addEventListener("click", () => loadDashboard().catch(showError));
document.getElementById("daysSelect").addEventListener("change", () => loadDashboard().catch(showError));

function showError(error) {
  console.error(error);
  alert(`Không tải được dashboard: ${error.message}`);
}

loadDashboard().catch(showError);
