"use strict";
const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const labels = {news:"World & news",sports:"Sport",finance:"Business",lifestyle:"Life & style",health:"Wellbeing",foodanddrink:"Food & drink",travel:"Travel",entertainment:"Culture",tv:"Television",movies:"Film",music:"Music",autos:"Motoring",video:"Video",weather:"Weather",kids:"Family",northamerica:"North America"};
const label = (category) => labels[category] || category || "Discover";
let articles = [], readerState = null, selected = null, topic = "all", visible = 6, busy = false, toastTimer;
const allArticles = new Map();

async function api(path, body) {
  const response = await fetch(path, {credentials:"same-origin", ...(body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)})});
  if (!response.ok) throw new Error(`Request failed (${response.status}). Please try again.`);
  return response.json();
}

function setBusy(value) {
  busy = value;
  $("recommendations").setAttribute("aria-busy", String(value));
  ["mark-read","new-session","reset-history","retry"].forEach((id) => { $(id).disabled = value; });
  if (!value) $("reset-history").disabled = !readerState?.read_count;
}

function toast(message) {
  clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4000);
}

function renderState(data) {
  readerState = data;
  [...data.recommendations, ...data.history].forEach((a) => allArticles.set(a.news_id, a));
  $("session-label").textContent = `Session ${data.session}`;
  $("session-type").textContent = data.read_count ? "Returning reader" : "New reader";
  $("read-count").textContent = data.read_count;
  $("recommendation-kicker").textContent = data.read_count ? "FOLLOWING YOUR CURIOSITY" : "A PLACE TO BEGIN";
  $("recommendation-mode").textContent = data.read_count ? "Based on your reading" : "First discoveries";
  $("recommendation-description").textContent = data.read_count ? "A fresh selection, shaped by the stories you’ve read." : "Your first discoveries. Read something to make this space your own.";
  $("error-banner").hidden = !data.error;
  $("error-message").textContent = data.error || "";
  $("connection-dot").classList.toggle("online", data.api_connected);
  $("connection-label").textContent = data.api_connected ? "Connected to recommendation API" : "Recommendation API unavailable";
  $("model-timing").textContent = data.latency_ms === null ? "" : `Last inference · ${data.latency_ms} ms`;
  $("recommendations").innerHTML = data.recommendations.length ? data.recommendations.map((a, i) => `
    <button class="recommendation-card" data-article="${escapeHtml(a.news_id)}">
      <span class="card-top"><span class="category-label">${escapeHtml(label(a.category))}</span><span class="card-number">0${i + 1}</span></span>
      <h3>${escapeHtml(a.title)}</h3><span class="card-bottom"><span>ARTICLE PREVIEW</span><span class="arrow" aria-hidden="true">↗</span></span>
    </button>`).join("") : `<div class="empty-recommendations">${data.api_connected ? "No new recommendations just yet. Explore the starter shelf below to discover something different." : "Your reading room is ready. Connect the recommendation API to see your next four, or explore the starter shelf below."}</div>`;
  $("history").innerHTML = data.history.length ? data.history.map((a) => `<div class="history-item"><span aria-hidden="true">✓</span><button data-article="${escapeHtml(a.news_id)}">${escapeHtml(a.title)}</button></div>`).join("") : `<div class="empty-trail"><span aria-hidden="true">⌁</span><p>Your curiosity starts here.</p><small>Open an article and mark it as read.<br>Your trail will appear right here.</small></div>`;
  renderCatalog();
}

function renderCatalog() {
  const filtered = articles.filter((a) => topic === "all" || a.category === topic);
  const readIds = new Set((readerState?.history || []).map((a) => a.news_id));
  $("catalog").innerHTML = filtered.slice(0, visible).map((a, i) => `
    <button class="catalog-row" data-article="${escapeHtml(a.news_id)}"><span class="catalog-number">${String(i + 1).padStart(2,"0")}</span><span class="catalog-content"><h3>${escapeHtml(a.title)}</h3><span class="catalog-meta">${escapeHtml(label(a.category))} · ARTICLE PREVIEW${readIds.has(a.news_id) ? '<span class="read-label">READ</span>' : ""}</span></span><span class="catalog-arrow" aria-hidden="true">↗</span></button>`).join("") || '<p class="section-description">No articles in this topic yet.</p>';
  $("show-more").hidden = filtered.length <= visible;
}

function renderTopics(categories) {
  const preferred = ["news","lifestyle","sports","health","finance","travel","foodanddrink","entertainment"];
  const chosen = preferred.filter((c) => categories.includes(c));
  $("topics").innerHTML = [{id:"all",name:"All stories"}, ...chosen.map((c) => ({id:c,name:label(c)}))].map((c) => `<button class="topic${c.id === topic ? " active" : ""}" data-topic="${escapeHtml(c.id)}" aria-pressed="${c.id === topic}">${escapeHtml(c.name)}</button>`).join("");
}

function openArticle(id) {
  if (busy) return;
  const article = allArticles.get(id);
  if (!article) return;
  selected = article;
  $("reader-category").textContent = label(article.category);
  $("reader-title").textContent = article.title;
  $("reader-abstract").textContent = article.abstract || "An abstract isn’t available for this article. You can try the original archive link below.";
  const publisher = $("publisher-link");
  publisher.hidden = !article.url;
  if (article.url) publisher.href = article.url; else publisher.removeAttribute("href");
  const isRead = readerState?.history.some((a) => a.news_id === id);
  $("mark-read").textContent = isRead ? "Read again →" : "Mark as read →";
  $("reader").showModal();
}

async function update(path, body, message) {
  if (busy) return;
  setBusy(true);
  try {
    const data = await api(path, body);
    renderState(data);
    if (message) toast(data.api_connected ? message : "Reading history saved. Recommendations will return when the API reconnects.");
  } catch (error) {
    $("error-message").textContent = error.message;
    $("error-banner").hidden = false;
  } finally { setBusy(false); }
}

document.addEventListener("click", (event) => {
  const article = event.target.closest("[data-article]");
  if (article) openArticle(article.dataset.article);
  const chip = event.target.closest("[data-topic]");
  if (chip) {
    topic = chip.dataset.topic; visible = 6;
    document.querySelectorAll("[data-topic]").forEach((el) => {el.classList.toggle("active", el === chip); el.setAttribute("aria-pressed", String(el === chip));});
    renderCatalog();
  }
});
$("close-reader").addEventListener("click", () => $("reader").close());
$("reader").addEventListener("click", (event) => {if (event.target === $("reader")) {const r = $("reader").getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $("reader").close();}});
$("mark-read").addEventListener("click", async () => {
  if (!selected || busy) return;
  const id = selected.news_id;
  $("reader").close();
  await update("/api/read", {news_id:id}, "Added to your trail. Your next reads are ready.");
});
$("new-session").addEventListener("click", () => update("/api/session/new", {}, "A fresh reading room. Make yourself at home."));
$("reset-history").addEventListener("click", () => {if (confirm("Clear the reading history for this session?")) update("/api/session/reset", {}, "Your reading trail has been reset.");});
$("retry").addEventListener("click", () => update("/api/session"));
$("show-more").addEventListener("click", () => {visible += 6; renderCatalog();});

async function init() {
  setBusy(true);
  try {
    const catalog = await api("/api/catalog");
    // Interleave categories so the initial shelf includes different interests.
    const groups = catalog.categories.map((c) => catalog.articles.filter((a) => a.category === c));
    articles = Array.from({length:4}, (_, i) => groups.map((g) => g[i]).filter(Boolean)).flat();
    articles.forEach((a) => allArticles.set(a.news_id, a));
    renderTopics(catalog.categories); renderCatalog();
    renderState(await api("/api/session"));
  } catch (error) {
    $("error-message").textContent = error.message; $("error-banner").hidden = false;
    $("recommendations").innerHTML = '<div class="empty-recommendations">We couldn’t open your session. Please try again.</div>';
    $("connection-label").textContent = "Demo connection unavailable";
  } finally { setBusy(false); }
}
init();
