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
let selected = null;                           // MP shown in the profile panel
let XEXT = null;                               // global ideal-point extent
let CLUB_MEAN = {};                            // club -> mean position

const tooltip = document.getElementById("tooltip");

fetch("ideal_points.json")
  .then((r) => r.json())
  .then((data) => {
    DATA = data;
    XEXT = d3.extent(data.mps, (d) => d.x);
    data.clubs.forEach((c) => { CLUB_MEAN[c.club] = c.mean; });
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
    .classed("selected", (d) => selected && d.id === selected.id)
    .on("mousemove", showTip).on("mouseleave", hideTip)
    .on("click", (e, d) => openProfile(d));

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

// ---------- MP profile panel ----------
const fmtPct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");

function openProfile(mp) {
  selected = mp;
  d3.selectAll(".dot-mp").classed("selected", (d) => d.id === mp.id);

  const neighbors = DATA.mps
    .filter((d) => d.id !== mp.id)
    .map((d) => ({ ...d, dist: Math.abs(d.x - mp.x) }))
    .sort((a, b) => a.dist - b.dist).slice(0, 3);

  const rhatOk = mp.rhat < 1.05;
  const el = document.getElementById("profile");
  el.innerHTML = `
    <button class="close" aria-label="Zamknij">×</button>
    <h2>${mp.name}</h2>
    <div class="club-chip"><span class="dot" style="background:${color(mp.club)}"></span>${mp.club}</div>
    <div class="big-pos">${mp.x.toFixed(2)}
      <span class="ci">90% CI: ${mp.lo.toFixed(2)} … ${mp.hi.toFixed(2)}</span></div>
    <svg id="mini-axis" role="img" aria-label="Pozycja na osi"></svg>
    <dl class="stats">
      <dt>Ranga (lewica→prawica)</dt><dd>#${mp.rank} / ${DATA.meta.n_mps}</dd>
      <dt>W klubie</dt><dd>#${mp.club_rank} / ${mp.club_size}</dd>
      <dt>Frekwencja</dt><dd>${fmtPct(mp.turnout)} <span class="muted">(${mp.votes}/${DATA.meta.n_votes})</span></dd>
      <dt>Lojalność klubowa</dt><dd>${fmtPct(mp.loyalty)}</dd>
      <dt>Zbieżność (R̂)</dt><dd>${mp.rhat.toFixed(3)} <span class="${rhatOk ? "ok" : "warn"}">${rhatOk ? "✓" : "⚠"}</span></dd>
    </dl>
    <h3>Najbliżsi ideologicznie</h3>
    <ul class="neighbors">
      ${neighbors.map((d) => `<li data-id="${d.id}"><span class="dot" style="background:${color(d.club)}"></span><span class="nm">${d.name}</span><span class="nx">${d.x.toFixed(2)}</span></li>`).join("")}
    </ul>`;

  el.querySelector(".close").onclick = closeProfile;
  el.querySelectorAll(".neighbors li").forEach((li) => {
    li.onclick = () => {
      const m = DATA.mps.find((d) => d.id === +li.dataset.id);
      if (m) openProfile(m);
    };
  });

  el.hidden = false; el.classList.add("open");
  document.getElementById("backdrop").hidden = false;
  renderMiniAxis(mp);
}

function closeProfile() {
  selected = null;
  d3.selectAll(".dot-mp").classed("selected", false);
  const el = document.getElementById("profile");
  el.classList.remove("open"); el.hidden = true;
  document.getElementById("backdrop").hidden = true;
}

function renderMiniAxis(mp) {
  const svg = d3.select("#mini-axis");
  svg.selectAll("*").remove();
  const width = document.getElementById("profile").clientWidth - 40;
  const height = 56, yc = 28;
  svg.attr("viewBox", `0 0 ${width} ${height}`).attr("height", height);
  const x = d3.scaleLinear().domain([XEXT[0] - 0.1, XEXT[1] + 0.1]).range([10, width - 10]);

  svg.append("line").attr("x1", 10).attr("x2", width - 10).attr("y1", yc).attr("y2", yc).attr("stroke", "#ccc");
  svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 6).attr("y2", yc + 6)
    .attr("stroke", "#bbb").attr("stroke-dasharray", "3 2");
  svg.append("g").selectAll("circle").data(DATA.mps).join("circle")
    .attr("cx", (d) => x(d.x)).attr("cy", yc).attr("r", 1.5).attr("fill", "#d0d0d0");
  const cm = CLUB_MEAN[mp.club];
  svg.append("line").attr("x1", x(cm)).attr("x2", x(cm)).attr("y1", yc - 9).attr("y2", yc + 9)
    .attr("stroke", color(mp.club)).attr("stroke-width", 2).attr("stroke-opacity", .6);
  svg.append("line").attr("x1", x(mp.lo)).attr("x2", x(mp.hi)).attr("y1", yc).attr("y2", yc)
    .attr("stroke", color(mp.club)).attr("stroke-width", 2).attr("stroke-opacity", .5);
  svg.append("circle").attr("cx", x(mp.x)).attr("cy", yc).attr("r", 5)
    .attr("fill", color(mp.club)).attr("stroke", "#000").attr("stroke-width", 1);
  svg.append("text").attr("x", 10).attr("y", height - 3).attr("class", "axis-label").text("lewica");
  svg.append("text").attr("x", width - 10).attr("y", height - 3).attr("text-anchor", "end")
    .attr("class", "axis-label").text("prawica");
}

document.getElementById("backdrop").addEventListener("click", closeProfile);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeProfile(); });

// ---------- responsive ----------
let rt;
window.addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(() => {
    if (DATA) { render(); renderClubs(DATA); if (selected) renderMiniAxis(selected); }
  }, 200);
});
