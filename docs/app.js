// Ideal points of Polish Sejm MPs — interactive 1D visualization (D3 v7).
// Loads ideal_points.json (precomputed MCMC results) and renders a beeswarm.

const CLUB_COLORS = {
  "PiS": "#003087", "KO": "#F5821F", "PSL-TD": "#00A550",
  "Polska2050": "#FACC15", "Polska2050-TD": "#FACC15", "Lewica": "#E31E24",
  "Razem": "#951B81", "Konfederacja": "#1A1A1A", "Konfederacja_KP": "#4A4A4A",
  "Centrum": "#1DACD6", "Demokracja": "#00BFA5", "niez.": "#AAAAAA",
};
const color = (c) => CLUB_COLORS[c] || "#888";
const R = 4;                                  // dot radius
const norm = (s) => s.toLowerCase().normalize("NFD").replace(/\p{Diacritic}/gu, "");

let DATA = null;
let activeClub = null;                         // club isolation filter
let showCI = false;

const tooltip = document.getElementById("tooltip");

fetch("ideal_points.json")
  .then((r) => r.json())
  .then((data) => {
    DATA = data;
    document.getElementById("subtitle").textContent =
      `${data.meta.term} · ${data.meta.n_mps} posłów · ${data.meta.n_votes} głosowań`;
    document.getElementById("meta-line").textContent =
      `Wygenerowano: ${data.meta.generated}. Źródło: ${data.meta.source}.`;
    buildLegend(data);
    render();
    renderClubs(data);
  })
  .catch((e) => {
    document.getElementById("subtitle").textContent = "Błąd ładowania danych.";
    console.error(e);
  });

// ---------- legend ----------
function buildLegend(data) {
  const legend = document.getElementById("legend");
  const clubs = data.clubs.map((c) => c.club);
  legend.innerHTML = "";
  clubs.forEach((club) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.dataset.club = club;
    chip.innerHTML = `<span class="dot" style="background:${color(club)}"></span>${club}`;
    chip.onclick = () => {
      activeClub = activeClub === club ? null : club;
      updateChips();
      applyFilter();
    };
    legend.appendChild(chip);
  });
}
function updateChips() {
  document.querySelectorAll(".chip").forEach((c) => {
    c.classList.toggle("dim", activeClub !== null && c.dataset.club !== activeClub);
  });
}

// ---------- main beeswarm ----------
function render() {
  const svg = d3.select("#chart");
  svg.selectAll("*").remove();
  const width = document.querySelector(".chart-wrap").clientWidth - 28;
  const height = 440;
  const margin = { top: 28, right: 24, bottom: 40, left: 24 };
  svg.attr("viewBox", `0 0 ${width} ${height}`).attr("height", height);

  const ext = d3.extent(DATA.mps, (d) => d.x);
  const pad = (ext[1] - ext[0]) * 0.05;
  const x = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad])
    .range([margin.left, width - margin.right]);

  // axis + zero line + L/R labels
  svg.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(7));
  svg.append("line").attr("class", "zero-line")
    .attr("x1", x(0)).attr("x2", x(0))
    .attr("y1", margin.top).attr("y2", height - margin.bottom);
  svg.append("text").attr("class", "axis-label").attr("x", margin.left)
    .attr("y", height - 8).attr("text-anchor", "start").text("← lewica");
  svg.append("text").attr("class", "axis-label").attr("x", width - margin.right)
    .attr("y", height - 8).attr("text-anchor", "end").text("prawica →");

  // beeswarm layout
  const nodes = DATA.mps.map((d) => ({ ...d, ideal: d.x }));
  const sim = d3.forceSimulation(nodes)
    .force("x", d3.forceX((d) => x(d.ideal)).strength(1))
    .force("y", d3.forceY((height - margin.top - margin.bottom) / 2 + margin.top).strength(0.05))
    .force("collide", d3.forceCollide(R + 0.6))
    .stop();
  for (let i = 0; i < 220; i++) sim.tick();
  const yMin = margin.top + R, yMax = height - margin.bottom - R;
  nodes.forEach((d) => { d.y = Math.max(yMin, Math.min(yMax, d.y)); });

  // CI lines (optional)
  svg.append("g").attr("id", "ci-group").attr("display", showCI ? null : "none")
    .selectAll("line").data(nodes).join("line")
    .attr("class", "ci-line").attr("stroke", (d) => color(d.club))
    .attr("x1", (d) => x(d.lo)).attr("x2", (d) => x(d.hi))
    .attr("y1", (d) => d.y).attr("y2", (d) => d.y);

  // dots
  svg.append("g").selectAll("circle").data(nodes).join("circle")
    .attr("class", "dot-mp").attr("r", R)
    .attr("cx", (d) => d.x).attr("cy", (d) => d.y)
    .attr("fill", (d) => color(d.club))
    .on("mousemove", showTip).on("mouseleave", hideTip);

  applyFilter();
}

function showTip(event, d) {
  tooltip.hidden = false;
  tooltip.innerHTML =
    `<b>${d.name}</b><br><span class="club">${d.club}</span>` +
    `<div class="pos">pozycja: ${d.x.toFixed(2)} &nbsp;(90% CI: ${d.lo.toFixed(2)}…${d.hi.toFixed(2)})</div>`;
  const wrap = document.querySelector(".chart-wrap").getBoundingClientRect();
  let left = event.clientX - wrap.left + 12;
  if (left > wrap.width - 250) left = event.clientX - wrap.left - 250;
  tooltip.style.left = left + "px";
  tooltip.style.top = (event.clientY - wrap.top + 12) + "px";
}
function hideTip() { tooltip.hidden = true; }

// ---------- filtering / search ----------
function applyFilter() {
  const q = norm(document.getElementById("search").value.trim());
  d3.selectAll(".dot-mp")
    .classed("dimmed", (d) => {
      if (activeClub && d.club !== activeClub) return true;
      if (q && !norm(d.name).includes(q)) return true;
      return false;
    })
    .classed("hit", (d) => q && norm(d.name).includes(q));
}

document.getElementById("search").addEventListener("input", applyFilter);
document.getElementById("ci-toggle").addEventListener("change", (e) => {
  showCI = e.target.checked;
  const g = document.getElementById("ci-group");
  if (g) g.setAttribute("display", showCI ? "inline" : "none");
});
document.getElementById("reset").addEventListener("click", () => {
  activeClub = null; document.getElementById("search").value = "";
  updateChips(); applyFilter();
});

// ---------- club averages panel ----------
function renderClubs(data) {
  const svg = d3.select("#clubs");
  svg.selectAll("*").remove();
  const width = document.querySelector(".chart-wrap").clientWidth - 28;
  const rowH = 26, margin = { top: 10, right: 24, bottom: 30, left: 110 };
  const height = margin.top + margin.bottom + data.clubs.length * rowH;
  svg.attr("viewBox", `0 0 ${width} ${height}`).attr("height", height);

  const ext = d3.extent(data.clubs, (d) => d.mean);
  const x = d3.scaleLinear().domain([ext[0] - 0.2, ext[1] + 0.2])
    .range([margin.left, width - margin.right]);

  svg.append("line").attr("class", "zero-line")
    .attr("x1", x(0)).attr("x2", x(0)).attr("y1", margin.top).attr("y2", height - margin.bottom);
  svg.append("g").attr("class", "axis")
    .attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(7));

  const rows = svg.selectAll("g.row").data(data.clubs).join("g")
    .attr("transform", (d, i) => `translate(0,${margin.top + i * rowH + rowH / 2})`);
  rows.append("text").attr("x", margin.left - 10).attr("dy", "0.32em")
    .attr("text-anchor", "end").style("font-size", "12px").style("fill", "#333")
    .text((d) => `${d.club} (${d.n})`);
  rows.append("line").attr("x1", x(0)).attr("x2", (d) => x(d.mean))
    .attr("stroke", (d) => color(d.club)).attr("stroke-width", 2).attr("stroke-opacity", .4);
  rows.append("circle").attr("cx", (d) => x(d.mean)).attr("r", 6)
    .attr("fill", (d) => color(d.club));
}

// ---------- responsive ----------
let rt;
window.addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(() => { if (DATA) { render(); renderClubs(DATA); } }, 200);
});
