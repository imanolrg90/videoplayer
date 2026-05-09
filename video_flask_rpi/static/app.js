const searchInput = document.getElementById("searchInput");
const refreshBtn = document.getElementById("refreshBtn");
const recursiveToggle = document.getElementById("recursiveToggle");
const logBtn = document.getElementById("logBtn");
const logoutBtn = document.getElementById("logoutBtn");
const pendingBadge = document.getElementById("pendingBadge");
const privacyBtn = document.getElementById("privacyBtn");

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
const playerBox = document.querySelector(".player-box");
const selectedTitle = document.getElementById("selectedTitle");
const selectedMeta = document.getElementById("selectedMeta");
const statusLine = document.getElementById("statusLine");
const fsOverlay = document.getElementById("fsOverlay");
const fsFavBtn = document.getElementById("fsFavBtn");
const fsDeleteBtn = document.getElementById("fsDeleteBtn");
const fsNextBtn = document.getElementById("fsNextBtn");
const fsTimeLabel = document.getElementById("fsTimeLabel");
const fsSeekSlider = document.getElementById("fsSeekSlider");
const fsVolumeSlider = document.getElementById("fsVolumeSlider");

const playBtn = document.getElementById("playBtn");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const randomBtn = document.getElementById("randomBtn");
const favBtn = document.getElementById("favBtn");
const watchedBtn = document.getElementById("watchedBtn");
const deleteBtn = document.getElementById("deleteBtn");
const repeatBtn = document.getElementById("repeatBtn");
const muteBtn = document.getElementById("muteBtn");
const volumeSlider = document.getElementById("volumeSlider");
const logModal = document.getElementById("logModal");
const logContent = document.getElementById("logContent");
const closeLogBtn = document.getElementById("closeLogBtn");
const saveLogBtn = document.getElementById("saveLogBtn");
const logFileName = document.getElementById("logFileName");
const privacyOverlay = document.getElementById("privacyOverlay");
const privacyPassword = document.getElementById("privacyPassword");
const privacyUnlockBtn = document.getElementById("privacyUnlockBtn");
const privacyError = document.getElementById("privacyError");

let folders = [];
let allVideos = [];
let selectedFolder = "";
let selectedVideo = null;
let currentFilter = "all";
let filteredVideos = [];
let repeatEnabled = false;
let privacyLocked = false;
let fsHideTimer = null;

function formatTime(value) {
  const sec = Math.max(0, Math.floor(Number(value || 0)));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function syncFullscreenOverlay() {
  const current = Number(videoPlayer.currentTime || 0);
  const duration = Number(videoPlayer.duration || 0);
  fsSeekSlider.max = String(duration > 0 ? duration : 0);
  fsSeekSlider.value = String(Math.min(current, duration || current));
  fsTimeLabel.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
  fsVolumeSlider.value = String(Math.round((videoPlayer.volume || 0) * 100));
}

function isInFullscreen() {
  return document.fullscreenElement === playerBox;
}

function showFsOverlayTemporarily() {
  if (!isInFullscreen()) {
    fsOverlay.classList.add("hidden");
    return;
  }
  fsOverlay.classList.remove("hidden");
  if (fsHideTimer) clearTimeout(fsHideTimer);
  fsHideTimer = setTimeout(() => {
    fsOverlay.classList.add("hidden");
  }, 1700);
}

async function toggleFullscreen() {
  if (!playerBox) return;
  if (isInFullscreen()) {
    await document.exitFullscreen().catch(() => {});
  } else {
    await playerBox.requestFullscreen().catch(() => {});
  }
}

function lockPrivacy() {
  privacyLocked = true;
  document.body.classList.add("privacy-locked");
  privacyOverlay.classList.remove("hidden");
  privacyError.textContent = "";
  privacyPassword.value = "";
  videoPlayer.pause();
  setTimeout(() => privacyPassword.focus(), 0);
}

async function unlockPrivacy() {
  try {
    const password = privacyPassword.value || "";
    await apiPostJson("/api/privacy/unlock", { password });
    privacyLocked = false;
    document.body.classList.remove("privacy-locked");
    privacyOverlay.classList.add("hidden");
    privacyError.textContent = "";
    privacyPassword.value = "";
    setStatus("Privacidad desactivada");
  } catch (err) {
    privacyError.textContent = err.message || "Contraseña incorrecta";
    privacyPassword.select();
    privacyPassword.focus();
  }
}

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

async function apiPostJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${response.status}`);
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
  await refreshPendingSummary();
  renderVideos();
}

async function refreshPendingSummary() {
  try {
    const data = await apiGet("/api/pending/summary");
    const c = data.counts || {};
    pendingBadge.textContent = `Pendientes: ${c.total || 0}`;
    pendingBadge.title = `rwd: ${c.rwd || 0} | borrar: ${c.borrar || 0} | top: ${c.top || 0} | meta: ${c.meta || 0}`;
  } catch (_err) {
    pendingBadge.textContent = "Pendientes: ?";
  }
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
  filteredVideos = filtered;
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
  prevBtn.disabled = !has;
  nextBtn.disabled = !has;
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
    repeatBtn.classList.toggle("active", repeatEnabled);
    repeatBtn.textContent = `Repeat: ${repeatEnabled ? "ON" : "OFF"}`;
    muteBtn.classList.toggle("active", videoPlayer.muted);
    muteBtn.textContent = `Mute: ${videoPlayer.muted ? "ON" : "OFF"}`;
    fullscreenBtn.textContent = isInFullscreen() ? "Salir pantalla completa" : "Pantalla completa";
    syncFullscreenOverlay();
    return;
  }

  selectedTitle.textContent = selectedVideo.name;
  selectedMeta.textContent = `${selectedVideo.relative_path} | ${selectedVideo.size_mb} MB | ${selectedVideo.last_viewed || "sin reproduccion"}`;
  favBtn.textContent = selectedVideo.favorite ? "Quitar favorito" : "Favorito";
  watchedBtn.textContent = selectedVideo.watched ? "Quitar visto" : "Marcar visto";
  favBtn.classList.toggle("active", selectedVideo.favorite);
  watchedBtn.classList.toggle("active", selectedVideo.watched);
  repeatBtn.classList.toggle("active", repeatEnabled);
  repeatBtn.textContent = `Repeat: ${repeatEnabled ? "ON" : "OFF"}`;
  muteBtn.classList.toggle("active", videoPlayer.muted);
  muteBtn.textContent = `Mute: ${videoPlayer.muted ? "ON" : "OFF"}`;
  fullscreenBtn.textContent = isInFullscreen() ? "Salir pantalla completa" : "Pantalla completa";
  fsFavBtn.textContent = selectedVideo.favorite ? "Quitar favorito" : "Favorito";
  syncFullscreenOverlay();
}

function playSelected() {
  if (!selectedVideo) return;
  videoPlayer.src = `/api/stream?path=${encodeURIComponent(selectedVideo.relative_path)}`;
  videoPlayer.load();
  videoPlayer.play().catch(() => {});
  setStatus(`Reproduciendo: ${selectedVideo.name}`);
}

function selectedIndexInFiltered() {
  if (!selectedVideo) return -1;
  return filteredVideos.findIndex((v) => v.relative_path === selectedVideo.relative_path);
}

function pickAndRender(video) {
  if (!video) return;
  selectedVideo = video;
  renderVideos();
  updateSelectionUi();
}

function playNextInList() {
  if (!filteredVideos.length) return;
  const idx = selectedIndexInFiltered();
  const next = idx < 0 ? filteredVideos[0] : filteredVideos[(idx + 1) % filteredVideos.length];
  pickAndRender(next);
  playSelected();
}

function playPrevInList() {
  if (!filteredVideos.length) return;
  const idx = selectedIndexInFiltered();
  const prev = idx < 0 ? filteredVideos[0] : filteredVideos[(idx - 1 + filteredVideos.length) % filteredVideos.length];
  pickAndRender(prev);
  playSelected();
}

function playRandomFromList() {
  if (!filteredVideos.length) return;
  const i = Math.floor(Math.random() * filteredVideos.length);
  pickAndRender(filteredVideos[i]);
  playSelected();
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
  await refreshPendingSummary();
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
  await refreshPendingSummary();
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
  logContent.value = "Cargando log diario...";
  logFileName.textContent = "-";
  try {
    const data = await apiGet("/api/log/folder-views/today");
    logFileName.textContent = data.file_name || "folder_views";
    logContent.value = data.content || "";
  } catch (err) {
    logContent.value = `Error cargando log: ${err.message || err}`;
  }
}

async function saveLogModal() {
  const content = logContent.value || "";
  const data = await apiPostJson("/api/log/folder-views/today", { content });
  logFileName.textContent = data.file_name || logFileName.textContent;
  setStatus("Log diario guardado.");
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
fullscreenBtn.addEventListener("click", () => toggleFullscreen().catch(showError));
prevBtn.addEventListener("click", () => playPrevInList());
nextBtn.addEventListener("click", () => playNextInList());
randomBtn.addEventListener("click", () => playRandomFromList());
favBtn.addEventListener("click", () => updateSelectedState({ favorite: !selectedVideo.favorite }).catch(showError));
watchedBtn.addEventListener("click", () => updateSelectedState({ watched: !selectedVideo.watched }).catch(showError));
deleteBtn.addEventListener("click", () => queueDeleteSelected().catch(showError));
repeatBtn.addEventListener("click", () => {
  repeatEnabled = !repeatEnabled;
  updateSelectionUi();
});
muteBtn.addEventListener("click", () => {
  videoPlayer.muted = !videoPlayer.muted;
  updateSelectionUi();
});
volumeSlider.addEventListener("input", () => {
  const v = Number(volumeSlider.value || 100);
  videoPlayer.volume = Math.max(0, Math.min(1, v / 100));
  fsVolumeSlider.value = String(v);
});

fsFavBtn.addEventListener("click", () => {
  if (!selectedVideo) return;
  updateSelectedState({ favorite: !selectedVideo.favorite }).catch(showError);
});
fsDeleteBtn.addEventListener("click", () => {
  if (!selectedVideo) return;
  queueDeleteSelected().catch(showError);
});
fsNextBtn.addEventListener("click", () => playNextInList());
fsSeekSlider.addEventListener("input", () => {
  videoPlayer.currentTime = Number(fsSeekSlider.value || 0);
  syncFullscreenOverlay();
});
fsVolumeSlider.addEventListener("input", () => {
  const v = Number(fsVolumeSlider.value || 100);
  videoPlayer.volume = Math.max(0, Math.min(1, v / 100));
  volumeSlider.value = String(v);
});

playerBox.addEventListener("mousemove", () => showFsOverlayTemporarily());
playerBox.addEventListener("mousedown", () => showFsOverlayTemporarily());
document.addEventListener("fullscreenchange", () => {
  fullscreenBtn.textContent = isInFullscreen() ? "Salir pantalla completa" : "Pantalla completa";
  showFsOverlayTemporarily();
  syncFullscreenOverlay();
});

videoPlayer.addEventListener("ended", () => {
  markViewedOnEnd().catch(() => {});
  if (repeatEnabled) {
    playSelected();
    return;
  }
  playNextInList();
});
videoPlayer.addEventListener("timeupdate", () => syncFullscreenOverlay());
videoPlayer.addEventListener("loadedmetadata", () => syncFullscreenOverlay());

privacyBtn.addEventListener("click", () => {
  if (privacyLocked) {
    privacyPassword.focus();
    privacyPassword.select();
    return;
  }
  lockPrivacy();
});
privacyUnlockBtn.addEventListener("click", () => unlockPrivacy().catch(showError));
privacyPassword.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    unlockPrivacy().catch(showError);
  }
});

logBtn.addEventListener("click", () => openLogModal().catch(showError));
saveLogBtn.addEventListener("click", () => saveLogModal().catch(showError));
closeLogBtn.addEventListener("click", () => closeLogModal());
logModal.addEventListener("click", (event) => {
  if (event.target === logModal) closeLogModal();
});

logoutBtn.addEventListener("click", async () => {
  await apiPost("/api/logout").catch(() => {});
  window.location.href = "/login";
});

document.addEventListener("keydown", (event) => {
  if (privacyLocked) {
    if (event.key === "Enter") {
      event.preventDefault();
      unlockPrivacy().catch(showError);
    }
    return;
  }

  const tag = (event.target && event.target.tagName) ? event.target.tagName.toLowerCase() : "";
  const inText = tag === "input" || tag === "textarea";

  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f") {
    event.preventDefault();
    searchInput.focus();
    searchInput.select();
    return;
  }

  if (inText) return;

  if (event.key === "Enter") {
    event.preventDefault();
    playSelected();
  } else if (event.key === "Backspace") {
    event.preventDefault();
    playPrevInList();
  } else if (event.key === " ") {
    event.preventDefault();
    playNextInList();
  } else if (event.key === "Delete") {
    event.preventDefault();
    if (!deleteBtn.disabled) queueDeleteSelected().catch(showError);
  } else if (event.key.toLowerCase() === "r") {
    event.preventDefault();
    repeatEnabled = !repeatEnabled;
    updateSelectionUi();
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    videoPlayer.currentTime = Math.max(0, (videoPlayer.currentTime || 0) - 5);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    videoPlayer.currentTime = (videoPlayer.currentTime || 0) + 5;
  } else if (event.key.toLowerCase() === "m") {
    event.preventDefault();
    videoPlayer.muted = !videoPlayer.muted;
    updateSelectionUi();
  } else if (event.ctrlKey && event.key === "ArrowUp") {
    event.preventDefault();
    const v = Math.min(100, Number(volumeSlider.value || 100) + 10);
    volumeSlider.value = String(v);
    videoPlayer.volume = v / 100;
  } else if (event.ctrlKey && event.key === "ArrowDown") {
    event.preventDefault();
    const v = Math.max(0, Number(volumeSlider.value || 100) - 10);
    volumeSlider.value = String(v);
    videoPlayer.volume = v / 100;
  } else if (event.key === "F11") {
    event.preventDefault();
    toggleFullscreen().catch(showError);
  } else if (event.key === "Escape" && isInFullscreen()) {
    event.preventDefault();
    document.exitFullscreen().catch(() => {});
  }
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
