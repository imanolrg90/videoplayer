const tokenInput = document.getElementById("tokenInput");
const searchInput = document.getElementById("searchInput");
const refreshBtn = document.getElementById("refreshBtn");
const recursiveToggle = document.getElementById("recursiveToggle");
const logBtn = document.getElementById("logBtn");

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
const logModal = document.getElementById("logModal");
const logContent = document.getElementById("logContent");
const closeLogBtn = document.getElementById("closeLogBtn");

const savedToken = localStorage.getItem("video_web_token") || "";
if (savedToken) {
  tokenInput.value = savedToken;
}

let folders = [];
let allVideos = [];
let selectedFolder = "";
let selectedVideo = null;
let currentFilter = "all";

function getToken() {
  return tokenInput.value.trim();
}

function withTokenHeaders(base = {}) {
  const token = getToken();
  if (!token) {
    return base;
  }
  return {
    ...base,
    "X-Access-Token": token,
  };
}

function tokenQuery() {
  const token = getToken();
  return token ? `token=${encodeURIComponent(token)}` : "";
}

function withTokenUrl(url) {
  const tokenPart = tokenQuery();
  if (!tokenPart) {
    return url;
  }
  return url.includes("?") ? `${url}&${tokenPart}` : `${url}?${tokenPart}`;
}

function setStatus(text) {
  statusLine.textContent = text || "";
}

async function apiGet(url) {
  const response = await fetch(url, { headers: withTokenHeaders() });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

async function apiPost(url) {
  const response = await fetch(url, { method: "POST", headers: withTokenHeaders() });
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

  if (currentFilter === "fav") {
    output = output.filter((v) => v.favorite);
  }
  if (currentFilter === "unseen") {
    output = output.filter((v) => !v.watched);
  }
  if (currentFilter === "seen") {
    output = output.filter((v) => v.watched);
  }
  if (currentFilter === "most_views") {
    output.sort((a, b) => (b.views || 0) - (a.views || 0));
  }
  if (currentFilter === "most_time") {
    output.sort((a, b) => (b.watched_seconds || 0) - (a.watched_seconds || 0));
  }
  if (currentFilter === "heavy") {
    output.sort((a, b) => (b.size_bytes || 0) - (a.size_bytes || 0));
  }
  return output;
}

function fmtMinutes(totalSeconds) {
  const sec = Number(totalSeconds || 0);
  return `${Math.floor(sec / 60)}m`;
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

  const fragment = document.createDocumentFragment();
  folders.forEach((folder) => {
    const item = document.createElement("div");
    item.className = "folder-item";
    if (folder.relative_path === selectedFolder) {
      item.classList.add("active");
    }

    item.style.paddingLeft = `${10 + folder.depth * 14}px`;

    const name = document.createElement("span");
    name.className = "folder-name";
    name.textContent = folder.depth === 0 ? `📁 ${folder.name}` : folder.name;

    const left = document.createElement("div");
    left.className = "folder-left";

    const thumb = document.createElement("img");
    thumb.className = "folder-thumb";
    thumb.loading = "lazy";
    thumb.src = withTokenUrl(folder.thumbnail_url || `/api/thumb/folder?path=${encodeURIComponent(folder.relative_path || "")}`);
    thumb.alt = "thumb";
    thumb.addEventListener("error", () => {
      thumb.style.visibility = "hidden";
    });

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

    fragment.appendChild(item);
  });

  folderTree.appendChild(fragment);
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

  const fragment = document.createDocumentFragment();
  filtered.forEach((video) => {
    const row = document.createElement("div");
    row.className = "video-row";
    if (selectedVideo && selectedVideo.relative_path === video.relative_path) {
      row.classList.add("active");
    }

    const main = document.createElement("div");
    main.className = "video-main";

    const thumb = document.createElement("img");
    thumb.className = "video-thumb";
    thumb.loading = "lazy";
    thumb.src = withTokenUrl(video.thumbnail_url || `/api/thumb/video?path=${encodeURIComponent(video.relative_path)}`);
    thumb.alt = "thumb";
    thumb.addEventListener("error", () => {
      thumb.style.visibility = "hidden";
    });

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

    const size = document.createElement("span");
    size.className = "chip";
    size.textContent = `${video.size_mb} MB`;

    const views = document.createElement("span");
    views.className = "chip";
    views.textContent = String(video.views || 0);

    const time = document.createElement("span");
    time.className = "chip";
    time.textContent = fmtMinutes(video.watched_seconds || 0);

    const watched = document.createElement("span");
    watched.className = `chip ${video.watched ? "on" : ""}`;
    watched.textContent = video.watched ? "Si" : "No";

    const favorite = document.createElement("span");
    favorite.className = `chip ${video.favorite ? "on" : ""}`;
    favorite.textContent = video.favorite ? "Si" : "No";

    const hashChip = document.createElement("span");
    hashChip.className = `chip ${video.has_hash ? "hash-ok" : "hash-miss"}`;
    hashChip.textContent = video.has_hash ? "OK" : "-";

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

    fragment.appendChild(row);
  });

  videoTable.appendChild(fragment);
}

function updateSelectionUi() {
  const hasSelection = Boolean(selectedVideo);
  playBtn.disabled = !hasSelection;
  favBtn.disabled = !hasSelection;
  watchedBtn.disabled = !hasSelection;

  if (!hasSelection) {
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
  if (!selectedVideo) {
    return;
  }

  const src = withTokenUrl(`/api/stream?path=${encodeURIComponent(selectedVideo.relative_path)}`);
  videoPlayer.src = src;
  videoPlayer.load();
  videoPlayer.play().catch(() => {});
  setStatus(`Reproduciendo: ${selectedVideo.name}`);
}

async function openLogModal() {
  logModal.classList.remove("hidden");
  logContent.textContent = "Cargando log...";
  try {
    const response = await fetch(withTokenUrl("/api/log?lines=350"), {
      headers: withTokenHeaders(),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const text = await response.text();
    logContent.textContent = text || "Log vacio";
  } catch (err) {
    logContent.textContent = `Error cargando log: ${err.message || err}`;
  }
}

function closeLogModal() {
  logModal.classList.add("hidden");
}

async function updateSelectedState(payload) {
  if (!selectedVideo) {
    return;
  }
  const query = new URLSearchParams({ path: selectedVideo.relative_path });
  if (Object.hasOwn(payload, "favorite")) {
    query.set("favorite", String(Boolean(payload.favorite)));
  }
  if (Object.hasOwn(payload, "watched")) {
    query.set("watched", String(Boolean(payload.watched)));
  }

  const data = await apiPost(`/api/video/state?${query.toString()}`);
  selectedVideo.favorite = data.state.favorite;
  selectedVideo.watched = data.state.watched;
  selectedVideo.last_viewed = data.state.last_viewed;

  const match = allVideos.find((v) => v.relative_path === selectedVideo.relative_path);
  if (match) {
    Object.assign(match, selectedVideo);
  }

  renderVideos();
  updateSelectionUi();
}

async function markViewedOnEnd() {
  if (!selectedVideo || selectedVideo.watched) {
    return;
  }
  const q = new URLSearchParams({ path: selectedVideo.relative_path });
  const data = await apiPost(`/api/video/viewed?${q.toString()}`);

  selectedVideo.watched = data.state.watched;
  selectedVideo.last_viewed = data.state.last_viewed;
  const match = allVideos.find((v) => v.relative_path === selectedVideo.relative_path);
  if (match) {
    Object.assign(match, selectedVideo);
  }

  renderVideos();
  updateSelectionUi();
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

function showError(err) {
  setStatus(`Error: ${err.message || err}`);
}

let debounceTimer = null;
searchInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    loadVideos().catch(showError);
  }, 220);
});

tokenInput.addEventListener("change", () => {
  localStorage.setItem("video_web_token", getToken());
  Promise.all([loadFolders(), loadVideos()]).catch(showError);
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

playBtn.addEventListener("click", () => {
  playSelected();
});

favBtn.addEventListener("click", () => {
  updateSelectedState({ favorite: !selectedVideo.favorite }).catch(showError);
});

watchedBtn.addEventListener("click", () => {
  updateSelectedState({ watched: !selectedVideo.watched }).catch(showError);
});

logBtn.addEventListener("click", () => {
  openLogModal().catch(showError);
});

closeLogBtn.addEventListener("click", () => {
  closeLogModal();
});

logModal.addEventListener("click", (event) => {
  if (event.target === logModal) {
    closeLogModal();
  }
});

videoPlayer.addEventListener("ended", () => {
  markViewedOnEnd().catch(() => {});
});

updateSelectionUi();
Promise.all([loadFolders(), loadVideos()])
  .then(() => setStatus("Conectado"))
  .catch(showError);
