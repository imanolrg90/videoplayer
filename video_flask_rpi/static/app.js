const searchInput = document.getElementById("searchInput");
const refreshBtn = document.getElementById("refreshBtn");
const recursiveToggle = document.getElementById("recursiveToggle");
const logBtn = document.getElementById("logBtn");
const logoutBtn = document.getElementById("logoutBtn");

const filterAll = document.getElementById("filterAll");
const filterFav = document.getElementById("filterFav");
const filterUnseen = document.getElementById("filterUnseen");
const filterSeen = document.getElementById("filterSeen");
const filterMostViews = document.getElementById("filterMostViews");
const filterMostTime = document.getElementById("filterMostTime");
const filterHeavy = document.getElementById("filterHeavy");

const folderTree = document.getElementById("folderTree");
const videoTable = document.getElementById("videoTable");
const countBadge = document.getElementById("countBadge");

const videoPlayer = document.getElementById("videoPlayer");
const selectedTitle = document.getElementById("selectedTitle");
const selectedMeta = document.getElementById("selectedMeta");
const statusLine = document.getElementById("statusLine");

const playBtn = document.getElementById("playBtn");
const favBtn = document.getElementById("favBtn");
const watchedBtn = document.getElementById("watchedBtn");
const deleteBtn = document.getElementById("deleteBtn");
const logModal = document.getElementById("logModal");
const logContent = document.getElementById("logContent");
const closeLogBtn = document.getElementById("closeLogBtn");

let folders = [];
let allVideos = [];
let selectedFolder = "";
let selectedVideo = null;
let currentFilter = "all";

function setStatus(text) {
  statusLine.textContent = text || "";
}

async function apiGet(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

async function apiPost(url) {
  const response = await fetch(url, { method: "POST" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

async function loadFolders() {
  const data = await apiGet("/api/folders");
  folders = data.items || [];
  renderFolders();
}

async function loadVideos() {
  const params = new URLSearchParams({
    search: searchInput.value.trim(),
    folder: selectedFolder,
    recursive: String(recursiveToggle.checked),
  });

  const data = await apiGet(`/api/videos?${params.toString()}`);
  allVideos = data.items || [];
  renderVideos();
}

function applyFilter(videos) {
  let output = [...videos];
  if (currentFilter === "fav") output = output.filter((v) => v.favorite);
  if (currentFilter === "unseen") output = output.filter((v) => !v.watched);
  if (currentFilter === "seen") output = output.filter((v) => v.watched);
  if (currentFilter === "most_views") output.sort((a, b) => (b.views || 0) - (a.views || 0));
  if (currentFilter === "most_time") output.sort((a, b) => (b.watched_seconds || 0) - (a.watched_seconds || 0));
  if (currentFilter === "heavy") output.sort((a, b) => (b.size_bytes || 0) - (a.size_bytes || 0));
  return output;
}

function fmtMinutes(totalSeconds) {
  return `${Math.floor(Number(totalSeconds || 0) / 60)}m`;
}

function renderFolders() {
  folderTree.innerHTML = "";
  if (!folders.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Sin carpetas";
    folderTree.appendChild(empty);
    return;
  }

  const frag = document.createDocumentFragment();
  folders.forEach((folder) => {
    const item = document.createElement("div");
    item.className = "folder-item";
    if (folder.relative_path === selectedFolder) item.classList.add("active");
    item.style.paddingLeft = `${10 + folder.depth * 14}px`;

    const left = document.createElement("div");
    left.className = "folder-left";

    const thumb = document.createElement("img");
    thumb.className = "folder-thumb";
    thumb.loading = "lazy";
    thumb.src = folder.thumbnail_url || `/api/thumb/folder?path=${encodeURIComponent(folder.relative_path || "")}`;
    thumb.alt = "thumb";
    thumb.addEventListener("error", () => { thumb.style.visibility = "hidden"; });

    const name = document.createElement("span");
    name.className = "folder-name";
    name.textContent = folder.depth === 0 ? `📁 ${folder.name}` : folder.name;

    const count = document.createElement("span");
    count.className = "folder-count";
    count.textContent = String(folder.video_count || 0);

    left.appendChild(thumb);
    left.appendChild(name);
    item.appendChild(left);
    item.appendChild(count);

    item.addEventListener("click", () => {
      selectedFolder = folder.relative_path;
      selectedVideo = null;
      renderFolders();
      loadVideos().catch(showError);
      updateSelectionUi();
    });

    frag.appendChild(item);
  });

  folderTree.appendChild(frag);
}

function renderVideos() {
  videoTable.innerHTML = "";
  const filtered = applyFilter(allVideos);
  countBadge.textContent = String(filtered.length);

  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No hay videos en esta vista";
    videoTable.appendChild(empty);
    return;
  }

  const frag = document.createDocumentFragment();
  filtered.forEach((video) => {
    const row = document.createElement("div");
    row.className = "video-row";
    if (selectedVideo && selectedVideo.relative_path === video.relative_path) row.classList.add("active");

    const main = document.createElement("div");
    main.className = "video-main";

    const thumb = document.createElement("img");
    thumb.className = "video-thumb";
    thumb.loading = "lazy";
    thumb.src = video.thumbnail_url || `/api/thumb/video?path=${encodeURIComponent(video.relative_path)}`;
    thumb.alt = "thumb";
    thumb.addEventListener("error", () => { thumb.style.visibility = "hidden"; });

    const textWrap = document.createElement("div");
    textWrap.className = "video-main-text";

    const name = document.createElement("div");
    name.className = "video-name";
    name.textContent = video.name;

    const path = document.createElement("div");
    path.className = "video-path";
    path.textContent = video.relative_path;

    textWrap.appendChild(name);
    textWrap.appendChild(path);
    main.appendChild(thumb);
    main.appendChild(textWrap);

    const views = chip(String(video.views || 0));
    const time = chip(fmtMinutes(video.watched_seconds || 0));
    const size = chip(`${video.size_mb} MB`);
    const watched = chip(video.watched ? "Si" : "No", video.watched ? "on" : "");
    const favorite = chip(video.favorite ? "Si" : "No", video.favorite ? "on" : "");
    const hashChip = chip(video.has_hash ? "OK" : "-", video.has_hash ? "hash-ok" : "hash-miss");

    row.appendChild(main);
    row.appendChild(views);
    row.appendChild(time);
    row.appendChild(size);
    row.appendChild(watched);
    row.appendChild(favorite);
    row.appendChild(hashChip);

    row.addEventListener("click", () => {
      selectedVideo = video;
      renderVideos();
      updateSelectionUi();
    });

    row.addEventListener("dblclick", () => {
      selectedVideo = video;
      playSelected();
    });

    frag.appendChild(row);
  });

  videoTable.appendChild(frag);
}

function chip(text, extraClass = "") {
  const el = document.createElement("span");
  el.className = `chip ${extraClass}`.trim();
  el.textContent = text;
  return el;
}

function updateSelectionUi() {
  const has = Boolean(selectedVideo);
  playBtn.disabled = !has;
  favBtn.disabled = !has;
  watchedBtn.disabled = !has;
  deleteBtn.disabled = !has;

  if (!has) {
    selectedTitle.textContent = "Selecciona un video";
    selectedMeta.textContent = "";
    favBtn.textContent = "Favorito";
    watchedBtn.textContent = "Marcar visto";
    favBtn.classList.remove("active");
    watchedBtn.classList.remove("active");
    return;
  }

  selectedTitle.textContent = selectedVideo.name;
  selectedMeta.textContent = `${selectedVideo.relative_path} | ${selectedVideo.size_mb} MB | ${selectedVideo.last_viewed || "sin reproduccion"}`;
  favBtn.textContent = selectedVideo.favorite ? "Quitar favorito" : "Favorito";
  watchedBtn.textContent = selectedVideo.watched ? "Quitar visto" : "Marcar visto";
  favBtn.classList.toggle("active", selectedVideo.favorite);
  watchedBtn.classList.toggle("active", selectedVideo.watched);
}

function playSelected() {
  if (!selectedVideo) return;
  videoPlayer.src = `/api/stream?path=${encodeURIComponent(selectedVideo.relative_path)}`;
  videoPlayer.load();
  videoPlayer.play().catch(() => {});
  setStatus(`Reproduciendo: ${selectedVideo.name}`);
}

async function updateSelectedState(payload) {
  if (!selectedVideo) return;
  const query = new URLSearchParams({ path: selectedVideo.relative_path });
  if (Object.hasOwn(payload, "favorite")) query.set("favorite", String(Boolean(payload.favorite)));
  if (Object.hasOwn(payload, "watched")) query.set("watched", String(Boolean(payload.watched)));

  const data = await apiPost(`/api/video/state?${query.toString()}`);
  Object.assign(selectedVideo, data.state);

  const match = allVideos.find((v) => v.relative_path === selectedVideo.relative_path);
  if (match) Object.assign(match, selectedVideo);

  renderVideos();
  updateSelectionUi();

  if (Object.hasOwn(payload, "favorite")) {
    setStatus("Top diferido guardado. Se aplicara al iniciar la app de escritorio.");
  }
  if (Object.hasOwn(payload, "watched") && payload.watched) {
    setStatus("Marcado para RWD diferido.");
  }
}

async function markViewedOnEnd() {
  if (!selectedVideo || selectedVideo.watched) return;
  const q = new URLSearchParams({ path: selectedVideo.relative_path });
  const data = await apiPost(`/api/video/viewed?${q.toString()}`);
  Object.assign(selectedVideo, data.state);
  const match = allVideos.find((v) => v.relative_path === selectedVideo.relative_path);
  if (match) Object.assign(match, selectedVideo);
  renderVideos();
  updateSelectionUi();
  setStatus("Marcado como visto y en cola para RWD.");
}

async function queueDeleteSelected() {
  if (!selectedVideo) return;
  const confirmed = window.confirm("Se enviara a borrado diferido para la app de escritorio. Continuar?");
  if (!confirmed) return;
  const q = new URLSearchParams({ path: selectedVideo.relative_path });
  await apiPost(`/api/video/delete?${q.toString()}`);
  setStatus("Borrado diferido guardado.");
}

function setFilter(filter) {
  currentFilter = filter;
  filterAll.classList.toggle("active", filter === "all");
  filterFav.classList.toggle("active", filter === "fav");
  filterUnseen.classList.toggle("active", filter === "unseen");
  filterSeen.classList.toggle("active", filter === "seen");
  filterMostViews.classList.toggle("active", filter === "most_views");
  filterMostTime.classList.toggle("active", filter === "most_time");
  filterHeavy.classList.toggle("active", filter === "heavy");
  renderVideos();
}

async function openLogModal() {
  logModal.classList.remove("hidden");
  logContent.textContent = "Cargando log...";
  try {
    const response = await fetch("/api/log?lines=350");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    logContent.textContent = text || "Log vacio";
  } catch (err) {
    logContent.textContent = `Error cargando log: ${err.message || err}`;
  }
}

function closeLogModal() {
  logModal.classList.add("hidden");
}

function showError(err) {
  setStatus(`Error: ${err.message || err}`);
}

let debounceTimer = null;
searchInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => loadVideos().catch(showError), 220);
});

refreshBtn.addEventListener("click", () => {
  Promise.all([loadFolders(), loadVideos()]).catch(showError);
});

recursiveToggle.addEventListener("change", () => {
  loadVideos().catch(showError);
});

filterAll.addEventListener("click", () => setFilter("all"));
filterFav.addEventListener("click", () => setFilter("fav"));
filterUnseen.addEventListener("click", () => setFilter("unseen"));
filterSeen.addEventListener("click", () => setFilter("seen"));
filterMostViews.addEventListener("click", () => setFilter("most_views"));
filterMostTime.addEventListener("click", () => setFilter("most_time"));
filterHeavy.addEventListener("click", () => setFilter("heavy"));

playBtn.addEventListener("click", () => playSelected());
favBtn.addEventListener("click", () => updateSelectedState({ favorite: !selectedVideo.favorite }).catch(showError));
watchedBtn.addEventListener("click", () => updateSelectedState({ watched: !selectedVideo.watched }).catch(showError));
deleteBtn.addEventListener("click", () => queueDeleteSelected().catch(showError));

videoPlayer.addEventListener("ended", () => markViewedOnEnd().catch(() => {}));

logBtn.addEventListener("click", () => openLogModal().catch(showError));
closeLogBtn.addEventListener("click", () => closeLogModal());
logModal.addEventListener("click", (event) => {
  if (event.target === logModal) closeLogModal();
});

logoutBtn.addEventListener("click", async () => {
  await apiPost("/api/logout").catch(() => {});
  window.location.href = "/login";
});

updateSelectionUi();
Promise.all([loadFolders(), loadVideos()])
  .then(() => setStatus("Conectado"))
  .catch((err) => {
    if (String(err.message || "").includes("401")) {
      window.location.href = "/login";
      return;
    }
    showError(err);
  });
