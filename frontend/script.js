
"use strict";

/* =========================================================
   VIDEOMOMENT - DASHBOARD CONTROLLER
   PART 1 OF 4
   ========================================================= */

const API_BASE = (
  window.VIDEOMOMENT_API_BASE ||
  "http://127.0.0.1:8001"
).replace(/\/+$/, "");

const $ = (id) => document.getElementById(id);

const UI = {
  sampleList: $("sampleVideoList"),
  libraryStatus: $("libraryStatus"),

  uploadButton: $("uploadButton"),
  fileInput: $("videoFileInput"),
  youtubeButton: $("youtubeButton"),
  youtubeWrapper: $("youtubeInputWrapper"),
  youtubeInput: $("youtubeUrlInput"),
  loadYoutubeButton: $("loadYoutubeButton"),
  youtubeNotice: $("youtubeNotice"),
  sourceStatus: $("sourceStatus"),

  searchForm: $("searchForm"),
  searchInput: $("searchQuery"),
  searchSubmit: $("searchSubmit"),
  searchStatus: $("searchStatus"),
  examplesButton: $("exampleQueriesButton"),
  examples: $("exampleQueries"),

  answerPanel: $("askVideoAnswer"),
  answerText: $("askVideoAnswerText"),

  player: $("videoPlayer"),
  playerEmpty: $("playerEmptyState"),
  playerTitle: $("currentVideoTitle"),
  playerDuration: $("currentVideoDuration"),
  playerState: $("currentVideoState"),

  resultsList: $("resultsList"),
  resultsCount: $("resultsCount"),
  resultsFooter: $("resultsFooter")
};

const state = {
  currentVideoId: null,
  currentVideoName: null,
  currentVideoUrl: null,

  samples: [],
  results: [],
  activeResult: -1,

  isSearching: false,
  isUploading: false,
  selectionVersion: 0,

  visitorToken: null,
  visitorSessionPromise: null,
  sessionMode: "unknown"
};

/* =========================================================
   01. HELPERS
   ========================================================= */

function formatTime(seconds) {
  const value = Math.max(
    0,
    Math.floor(Number(seconds) || 0)
  );

  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = value % 60;

  const pad = (n) => String(n).padStart(2, "0");

  return hours > 0
    ? `${pad(hours)}:${pad(minutes)}:${pad(secs)}`
    : `${pad(minutes)}:${pad(secs)}`;
}

function getValue(object, keys, fallback = null) {
  if (!object || typeof object !== "object") {
    return fallback;
  }

  for (const key of keys) {
    const value = object[key];

    if (
      value !== undefined &&
      value !== null &&
      value !== ""
    ) {
      return value;
    }
  }

  return fallback;
}

function setMessage(element, message, isError = false) {
  if (!element) return;

  element.textContent = message || "";
  element.style.color = isError ? "#b93417" : "";
}

function setLoading(button, loading, label = null) {
  if (!button) return;

  button.disabled = loading;
  button.classList.toggle("is-loading", loading);

  if (label !== null) {
    const span = button.querySelector("span");

    if (span) {
      span.textContent = label;
    } else {
      button.textContent = label;
    }
  }
}

function createElement(tag, className = "", text = null) {
  const element = document.createElement(tag);

  if (className) {
    element.className = className;
  }

  if (text !== null) {
    element.textContent = String(text);
  }

  return element;
}

function resolveMediaUrl(value) {
  if (!value || typeof value !== "string") {
    return null;
  }

  if (value.startsWith("blob:")) {
    return value;
  }

  if (/^https?:\/\//i.test(value)) {
    return value;
  }

  if (value.startsWith("/")) {
    return `${API_BASE}${value}`;
  }

  return `${API_BASE}/${value}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* =========================================================
   02. VISITOR AUTHENTICATION
   ========================================================= */

async function initializeVisitorSession() {
  if (state.sessionMode === "local") {
    return null;
  }

  if (state.visitorToken) {
    return state.visitorToken;
  }

  if (state.visitorSessionPromise) {
    return state.visitorSessionPromise;
  }

  state.visitorSessionPromise = (async () => {
    const response = await fetch(
      `${API_BASE}/visitors/session`,
      {
        method: "POST",
        cache: "no-store"
      }
    );

    if (response.status === 404) {
      state.sessionMode = "local";
      return null;
    }

    if (!response.ok) {
      throw new Error(
        `Visitor session failed (${response.status}).`
      );
    }

    const data = await response.json();

    const token = getValue(
  data,
  ["visitor_token", "access_token", "token"]
);

    if (!token) {
      throw new Error(
        "Visitor session response did not include a token."
      );
    }

    state.visitorToken = token;
    state.sessionMode = "public";

    return token;
  })();

  try {
    return await state.visitorSessionPromise;
  } finally {
    state.visitorSessionPromise = null;
  }
}

async function apiRequest(path, options = {}) {
  if (path !== "/visitors/session") {
    await initializeVisitorSession();
  }

  const headers = new Headers(options.headers || {});

  if (state.visitorToken) {
    headers.set(
      "Authorization",
      `Bearer ${state.visitorToken}`
    );
  }

  const response = await fetch(
    `${API_BASE}${path}`,
    {
      ...options,
      headers,
      cache: "no-store"
    }
  );

  if (!response.ok) {
    let message = `Request failed (${response.status})`;

    try {
      const data = await response.json();

      const detail =
        data.detail ||
        data.message ||
        data.error;

      if (detail) {
        message =
          typeof detail === "string"
            ? detail
            : JSON.stringify(detail);
      }
    } catch {
      // Preserve the HTTP error.
    }

    throw new Error(message);
  }

  if (response.status === 204) {
    return null;
  }

  const contentType =
    response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}

/* =========================================================
   03. PLAYBACK TICKETS
   ========================================================= */

async function getPlaybackUrl(videoId) {
  const id = encodeURIComponent(videoId);

  if (state.sessionMode !== "public") {
    return `${API_BASE}/videos/${id}/stream`;
  }

  const data = await apiRequest(
    `/videos/${id}/playback-ticket`,
    {
      method: "POST"
    }
  );

  const streamUrl = getValue(
    data,
    ["stream_url", "playback_url", "url"]
  );

  if (!streamUrl) {
    throw new Error(
      "Backend did not return a playback URL."
    );
  }

  const url = new URL(
    resolveMediaUrl(streamUrl)
  );

  // The backend already embeds the ticket in stream_url.
  if (!url.searchParams.has("ticket")) {
    throw new Error(
      "Playback URL is missing its security ticket."
    );
  }

  return url.toString();
}

/* =========================================================
   04. CLEAR RESULTS AND ANSWERS
   ========================================================= */

function clearAnswer() {
  if (UI.answerPanel) {
    UI.answerPanel.hidden = true;
  }

  if (UI.answerText) {
    UI.answerText.textContent = "";
  }
}

function clearResults(
  message = "Your matching moments will appear here after a search."
) {
  state.results = [];
  state.activeResult = -1;

  if (UI.resultsCount) {
    UI.resultsCount.textContent = "0 results";
  }

  if (UI.resultsFooter) {
    UI.resultsFooter.hidden = true;
  }

  if (!UI.resultsList) return;

  UI.resultsList.replaceChildren();

  const empty = createElement(
    "div",
    "results-empty-state"
  );

  empty.append(
    createElement("span", "results-empty-number", "00"),
    createElement("p", "", message)
  );

  UI.resultsList.appendChild(empty);
}

/* =========================================================
   05. VIDEO SELECTION
   ========================================================= */

async function selectVideo(video) {
  const id = getValue(video, ["video_id", "id"]);

  if (!id) {
    setMessage(
      UI.sourceStatus,
      "This video does not have a valid ID.",
      true
    );
    return;
  }

  const version = ++state.selectionVersion;

  state.currentVideoId = String(id);

  state.currentVideoName = getValue(
    video,
    ["title", "filename", "name", "original_filename"],
    "Untitled video"
  );

  state.currentVideoUrl = null;

  if (UI.player) {
    UI.player.pause();
    UI.player.removeAttribute("src");
    UI.player.load();
  }

  if (UI.playerEmpty) {
    UI.playerEmpty.hidden = true;
  }

  if (UI.playerTitle) {
    UI.playerTitle.textContent = state.currentVideoName;
  }

  if (UI.playerDuration) {
    UI.playerDuration.textContent = "--:--";
  }

  if (UI.playerState) {
    UI.playerState.textContent = "LOADING";
  }

  clearResults();
  clearAnswer();

  setMessage(UI.searchStatus, "");
  setMessage(UI.sourceStatus, "");

  updateSelectedSample();
  updateExampleQueries(video);

  try {
    const playbackUrl = await getPlaybackUrl(
      state.currentVideoId
    );

    if (version !== state.selectionVersion) {
      return;
    }

    state.currentVideoUrl = playbackUrl;

    if (UI.player) {
      UI.player.src = playbackUrl;
      UI.player.load();
    }
  } catch (error) {
    if (version !== state.selectionVersion) {
      return;
    }

    if (UI.playerState) {
      UI.playerState.textContent = "UNAVAILABLE";
    }

    setMessage(
      UI.sourceStatus,
      `Could not load video: ${error.message}`,
      true
    );

    console.error("Playback error:", error);
  }
}

function updateSelectedSample() {
  if (!UI.sampleList) return;

  UI.sampleList
    .querySelectorAll(".sample-video-card")
    .forEach((card) => {
      const active =
        card.dataset.videoId === state.currentVideoId;

      card.classList.toggle("active", active);
      card.setAttribute("aria-pressed", String(active));
    });
}

function initializePlayerEvents() {
  if (!UI.player) return;

  UI.player.addEventListener("loadedmetadata", () => {
    if (UI.playerDuration) {
      UI.playerDuration.textContent =
        formatTime(UI.player.duration);
    }

    if (UI.playerState) {
      UI.playerState.textContent = "READY";
    }
  });

  UI.player.addEventListener("playing", () => {
    if (UI.playerState) {
      UI.playerState.textContent = "PLAYING";
    }
  });

  UI.player.addEventListener("pause", () => {
    if (
      state.currentVideoId &&
      !UI.player.ended &&
      UI.playerState
    ) {
      UI.playerState.textContent = "PAUSED";
    }
  });

  UI.player.addEventListener("ended", () => {
    if (UI.playerState) {
      UI.playerState.textContent = "ENDED";
    }
  });

  UI.player.addEventListener("error", () => {
    if (!state.currentVideoId) return;

    if (UI.playerState) {
      UI.playerState.textContent = "UNAVAILABLE";
    }

    setMessage(
      UI.sourceStatus,
      "The video could not be played. Check the backend stream endpoint.",
      true
    );
  });
}


/* =========================================================
   06. SAMPLE VIDEO LIBRARY
   ========================================================= */

function normalizeSamples(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }

  for (const key of [
    "samples",
    "videos",
    "items",
    "results"
  ]) {
    if (Array.isArray(payload?.[key])) {
      return payload[key];
    }
  }

  return [];
}

function renderSamples(samples) {
  if (!UI.sampleList) return;

  UI.sampleList.replaceChildren();

  if (!samples.length) {
    UI.sampleList.appendChild(
      createElement(
        "p",
        "library-message",
        "No sample videos are currently available."
      )
    );
    return;
  }

  samples.forEach((video) => {
    const id = getValue(video, ["video_id", "id"]);

    if (!id) return;

    const title = getValue(
      video,
      ["title", "filename", "name", "original_filename"],
      "Untitled video"
    );

    const duration = getValue(
      video,
      ["duration_seconds", "duration"],
      null
    );

    const thumbnail = getValue(
      video,
      ["thumbnail_url", "thumbnail", "poster_url"],
      null
    );

    const card = createElement(
      "button",
      "sample-video-card"
    );

    card.type = "button";
    card.dataset.videoId = String(id);

    card.setAttribute("aria-pressed", "false");
    card.setAttribute("aria-label", `Select ${title}`);

    const imageBox = createElement(
      "span",
      "sample-thumbnail"
    );

    if (thumbnail) {
      const img = document.createElement("img");

      img.src = resolveMediaUrl(thumbnail);
      img.alt = "";
      img.loading = "lazy";

      imageBox.appendChild(img);
    } else {
      // Stream URLs require playback tickets in public mode.
      // Obtain a ticket before attempting thumbnail generation.
      getPlaybackUrl(String(id))
        .then((url) => {
          if (imageBox.isConnected) {
            generateMomentThumbnail(
              url,
              6,
              imageBox
            );
          }
        })
        .catch((error) => {
          console.warn(
            "Sample thumbnail unavailable:",
            error
          );
        });
    }

    const details = createElement(
      "span",
      "sample-details"
    );

    const displayTitle =
      String(title).toLowerCase().includes("artemis ii")
        ? "NASA Artemis II"
        : title;

    details.append(
      createElement(
        "span",
        "sample-title",
        displayTitle
      ),
      createElement(
        "span",
        "sample-duration",
        duration !== null
          ? formatTime(duration)
          : "VIDEO"
      )
    );

    card.append(imageBox, details);

    card.addEventListener("click", () => {
      void selectVideo(video);
    });

    UI.sampleList.appendChild(card);
  });

  updateSelectedSample();
}

async function loadSampleVideos() {
  setMessage(
    UI.libraryStatus,
    "Connecting to library"
  );

  try {
    const payload = await apiRequest("/videos/samples");

    state.samples = normalizeSamples(payload);

    renderSamples(state.samples);

    const count = state.samples.length;

    setMessage(
      UI.libraryStatus,
      `${count} sample video${count === 1 ? "" : "s"} available`
    );

  } catch (error) {
    if (UI.sampleList) {
      UI.sampleList.replaceChildren(
        createElement(
          "p",
          "library-message",
          "Could not load sample videos."
        )
      );
    }

    setMessage(
      UI.libraryStatus,
      "Library unavailable"
    );

    setMessage(
      UI.sourceStatus,
      "Cannot connect to the sample library. Ensure FastAPI is running.",
      true
    );

    console.error("Sample library error:", error);
  }
}

/* =========================================================
   07. UPLOAD AND INDEXING
   ========================================================= */

function initializeUploadEvents() {
  if (!UI.uploadButton || !UI.fileInput) return;

  UI.uploadButton.addEventListener("click", () => {
    if (!state.isUploading) {
      UI.fileInput.click();
    }
  });

  UI.fileInput.addEventListener("change", async () => {
    const file = UI.fileInput.files?.[0];

    if (!file) return;

    const validType =
      file.type === "video/mp4" ||
      file.name.toLowerCase().endsWith(".mp4");

    if (!validType) {
      setMessage(
        UI.sourceStatus,
        "Please select an MP4 video.",
        true
      );

      UI.fileInput.value = "";
      return;
    }

    await uploadVideo(file);

    UI.fileInput.value = "";
  });
}

async function uploadVideo(file) {
  if (state.isUploading) return;

  state.isUploading = true;

  setLoading(
    UI.uploadButton,
    true,
    "Uploading..."
  );

  setMessage(
    UI.sourceStatus,
    `Uploading ${file.name}...`
  );

  try {
    const form = new FormData();

    form.append("file", file);

    const uploaded = await apiRequest(
      "/videos/upload",
      {
        method: "POST",
        body: form
      }
    );

    const videoId = getValue(
      uploaded,
      ["video_id", "id"]
    );

    if (!videoId) {
      throw new Error(
        "The upload response did not include a video ID."
      );
    }

    await selectVideo({
      video_id: videoId,
      title: file.name
    });

    setMessage(
      UI.sourceStatus,
      "Upload complete. Starting video indexing..."
    );

    await startIndexing(videoId);

  } catch (error) {
    setMessage(
      UI.sourceStatus,
      error.message,
      true
    );

    console.error("Upload error:", error);

  } finally {
    state.isUploading = false;

    setLoading(
      UI.uploadButton,
      false,
      "Upload MP4"
    );
  }
}

async function startIndexing(videoId) {
  await apiRequest(
    `/videos/${encodeURIComponent(videoId)}/index`,
    {
      method: "POST"
    }
  );

  setMessage(
    UI.sourceStatus,
    "Indexing started. This may take several minutes..."
  );

  await pollIndexStatus(videoId);
}

async function pollIndexStatus(videoId) {
  const maxAttempts = 600;
  const delayMs = 3000;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (state.currentVideoId !== String(videoId)) {
      return;
    }

    let payload;

    try {
      payload = await apiRequest(
        `/videos/${encodeURIComponent(videoId)}/status`
      );
    } catch (error) {
      setMessage(
        UI.sourceStatus,
        `Unable to check indexing status: ${error.message}`,
        true
      );
      return;
    }

    const status = String(
      getValue(
        payload,
        ["status", "index_status", "state"],
        ""
      )
    ).toLowerCase();

    const indexedFlag = getValue(
      payload,
      ["is_indexed", "indexed"],
      false
    );

    if (
      indexedFlag === true ||
      [
        "completed",
        "complete",
        "indexed",
        "ready",
        "success"
      ].includes(status)
    ) {
      setMessage(
        UI.sourceStatus,
        "Video indexed successfully. You can now search or ask questions."
      );

      if (UI.playerState) {
        UI.playerState.textContent = "INDEXED";
      }

      return;
    }

    if (["failed", "error"].includes(status)) {
      setMessage(
        UI.sourceStatus,
        getValue(
          payload,
          ["error", "message"],
          "Video indexing failed."
        ),
        true
      );
      return;
    }

    const progress = Number(payload.progress);

    const progressText = Number.isFinite(progress)
      ? ` ${Math.round(progress)}%`
      : "";

    const stage = getValue(
      payload,
      ["stage", "message"],
      status || "indexing"
    );

    setMessage(
      UI.sourceStatus,
      `Processing video... ${stage}${progressText}`
    );

    await sleep(delayMs);
  }

  setMessage(
    UI.sourceStatus,
    "Indexing is taking longer than expected. Check backend status."
  );
}

/* =========================================================
   08. YOUTUBE URL IMPORT
   ========================================================= */

function initializeYoutubeEvents() {
  if (UI.youtubeNotice) {
    UI.youtubeNotice.textContent =
      "Paste a public YouTube video URL to import it.";
  }

  if (UI.youtubeButton && UI.youtubeWrapper) {
    UI.youtubeButton.addEventListener("click", () => {
      UI.youtubeWrapper.hidden =
        !UI.youtubeWrapper.hidden;

      if (!UI.youtubeWrapper.hidden) {
        UI.youtubeInput?.focus();
      }
    });
  }

  async function importYoutubeVideo() {
    if (state.isUploading) return;

    const url = UI.youtubeInput?.value.trim();

    if (!url) {
      setMessage(
        UI.sourceStatus,
        "Please enter a YouTube URL.",
        true
      );
      return;
    }

    try {
      const parsed = new URL(url);

      const allowedHosts = [
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com"
      ];

      if (
        parsed.protocol !== "https:" ||
        !allowedHosts.includes(parsed.hostname)
      ) {
        throw new Error("Invalid YouTube URL.");
      }
    } catch {
      setMessage(
        UI.sourceStatus,
        "Please enter a valid HTTPS YouTube URL.",
        true
      );
      return;
    }

    state.isUploading = true;

    if (UI.loadYoutubeButton) {
      UI.loadYoutubeButton.disabled = true;
      UI.loadYoutubeButton.textContent = "IMPORTING...";
    }

    setMessage(
      UI.sourceStatus,
      "Importing YouTube video. This may take a few minutes..."
    );

    try {
      const imported = await apiRequest(
        "/videos/import-url",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ url })
        }
      );

      const videoId = getValue(
        imported,
        ["video_id", "id"]
      );

      if (!videoId) {
        throw new Error(
          "Import succeeded but no video ID was returned."
        );
      }

      await selectVideo({
        video_id: videoId,
        title:
          imported.title ||
          imported.filename ||
          "Imported video"
      });

      setMessage(
        UI.sourceStatus,
        "Video imported successfully. Starting indexing..."
      );

      await startIndexing(videoId);

    } catch (error) {
      setMessage(
        UI.sourceStatus,
        error.message,
        true
      );

      console.error("YouTube import error:", error);

    } finally {
      state.isUploading = false;

      if (UI.loadYoutubeButton) {
        UI.loadYoutubeButton.disabled = false;
        UI.loadYoutubeButton.textContent = "LOAD";
      }
    }
  }

  UI.loadYoutubeButton?.addEventListener(
    "click",
    importYoutubeVideo
  );

  UI.youtubeInput?.addEventListener(
    "keydown",
    (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void importYoutubeVideo();
      }
    }
  );
}


/* =========================================================
   09. VIDEO-SPECIFIC EXAMPLE QUERIES
   ========================================================= */

function updateExampleQueries(video) {
  if (!UI.examples) return;

  const queries = Array.isArray(video?.example_queries)
    ? video.example_queries.filter(
        (query) =>
          typeof query === "string" &&
          query.trim().length > 0
      )
    : [];

  UI.examples.replaceChildren();

  queries.forEach((query) => {
    const button = createElement(
      "button",
      "example-query",
      query
    );

    button.type = "button";
    button.dataset.query = query;

    button.addEventListener("click", () => {
      if (!UI.searchInput) return;

      UI.searchInput.value = query;
      UI.examples.hidden = true;

      UI.examplesButton?.setAttribute(
        "aria-expanded",
        "false"
      );

      UI.searchInput.focus();
    });

    UI.examples.appendChild(button);
  });

  UI.examples.hidden = true;

  UI.examplesButton?.setAttribute(
    "aria-expanded",
    "false"
  );

  if (UI.examplesButton) {
    UI.examplesButton.disabled =
      queries.length === 0;
  }
}

function initializeExampleQueries() {
  if (UI.examplesButton && UI.examples) {
    UI.examplesButton.addEventListener(
      "click",
      () => {
        if (UI.examplesButton.disabled) return;

        UI.examples.hidden = !UI.examples.hidden;

        UI.examplesButton.setAttribute(
          "aria-expanded",
          String(!UI.examples.hidden)
        );
      }
    );
  }

  updateExampleQueries(null);
}

/* =========================================================
   10. SEARCH AND VIDEO QUESTIONS
   ========================================================= */

function normalizeResults(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }

  for (const key of [
    "moments",
    "results",
    "matches",
    "segments"
  ]) {
    if (Array.isArray(payload?.[key])) {
      return payload[key];
    }
  }

  return [];
}

function getMomentStart(moment) {
  return Number(
    getValue(
      moment,
      ["start", "start_time", "start_seconds"],
      0
    )
  );
}

function getMomentEnd(moment) {
  return Number(
    getValue(
      moment,
      ["end", "end_time", "end_seconds"],
      getMomentStart(moment)
    )
  );
}

function getMomentDescription(moment) {
  const evidence = moment?.evidence;

  if (Array.isArray(evidence)) {
    const preferredTypes = [
      "audio",
      "transcript",
      "speech",
      "ocr"
    ];

    for (const type of preferredTypes) {
      for (const item of evidence) {
        const modality = String(
          getValue(
            item,
            ["modality", "type", "source"],
            ""
          )
        ).toLowerCase();

        if (!modality.includes(type)) {
          continue;
        }

        const text = getValue(
          item,
          ["text", "transcript", "description"],
          null
        );

        if (
          typeof text === "string" &&
          text.trim()
        ) {
          return text.trim();
        }
      }
    }
  }

  const direct = getValue(
    moment,
    [
      "description",
      "caption",
      "text",
      "summary",
      "explanation"
    ],
    null
  );

  if (
    typeof direct === "string" &&
    direct.trim()
  ) {
    return direct.trim();
  }

  if (Array.isArray(evidence)) {
    for (const item of evidence) {
      const text = getValue(
        item,
        ["text", "transcript", "description"],
        null
      );

      if (
        typeof text === "string" &&
        text.trim()
      ) {
        return text.trim();
      }
    }
  }

  return "Matching video moment";
}

function initializeSearchEvents() {
  if (!UI.searchForm) return;

  UI.searchForm.addEventListener(
    "submit",
    async (event) => {
      event.preventDefault();

      const query =
        UI.searchInput?.value.trim() || "";

      if (!query) {
        setMessage(
          UI.searchStatus,
          "Enter a question or search query.",
          true
        );
        return;
      }

      if (!state.currentVideoId) {
        setMessage(
          UI.searchStatus,
          "Select or upload a video first.",
          true
        );
        return;
      }

      if (state.isSearching) return;

      await searchMoments(query);
    }
  );
}

async function searchMoments(query) {
  state.isSearching = true;

  const videoId = state.currentVideoId;
  const selectionVersion = state.selectionVersion;

  setLoading(
    UI.searchSubmit,
    true,
    "Searching..."
  );

  setMessage(
    UI.searchStatus,
    "Analyzing your video query..."
  );

  clearResults("Analyzing video...");
  clearAnswer();

  try {
    const payload = await apiRequest(
      `/videos/${encodeURIComponent(videoId)}/query`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query,
          top_k: 5,
          fusion_method: "reliability_aware"
        })
      }
    );

    if (selectionVersion !== state.selectionVersion) {
      return;
    }

    console.info("Video query response:", payload);

    const results = normalizeResults(payload);

    state.results = results;
    renderResults(results);

    const answer =
      typeof payload?.answer === "string"
        ? payload.answer.trim()
        : "";

    if (answer) {
      if (UI.answerText) {
        UI.answerText.textContent = answer;
      }

      if (UI.answerPanel) {
        UI.answerPanel.hidden = false;
      }
    }

    const intent = payload?.intent || "moment_search";

    if (answer) {
      setMessage(
        UI.searchStatus,
        results.length
          ? `Answer generated with ${results.length} supporting video moment${results.length === 1 ? "" : "s"}.`
          : "Video answer generated."
      );
    } else if (intent === "moment_search") {
      setMessage(
        UI.searchStatus,
        results.length
          ? `Found ${results.length} matching moment${results.length === 1 ? "" : "s"}.`
          : "No matching moments found."
      );
    } else {
      setMessage(
        UI.searchStatus,
        "The query completed, but the backend did not return an answer.",
        true
      );
    }

  } catch (error) {
    if (selectionVersion !== state.selectionVersion) {
      return;
    }

    clearResults("Search could not be completed.");
    clearAnswer();

    setMessage(
      UI.searchStatus,
      error.message,
      true
    );

    console.error("Video query error:", error);

  } finally {
    state.isSearching = false;

    setLoading(
      UI.searchSubmit,
      false,
      "Search"
    );
  }
}

/* =========================================================
   11. VIDEO THUMBNAIL GENERATION
   ========================================================= */

async function generateMomentThumbnail(
  videoUrl,
  timestamp,
  thumbnailBox
) {
  if (!videoUrl || !thumbnailBox) return;

  const video = document.createElement("video");

  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = "anonymous";
  video.src = videoUrl;

  try {
    await new Promise((resolve, reject) => {
      video.addEventListener(
        "loadedmetadata",
        resolve,
        { once: true }
      );

      video.addEventListener(
        "error",
        reject,
        { once: true }
      );

      video.load();
    });

    const safeTimestamp = Math.max(
      0,
      Math.min(
        timestamp,
        Math.max(0, video.duration - 0.1)
      )
    );

    await new Promise((resolve, reject) => {
      video.addEventListener(
        "seeked",
        resolve,
        { once: true }
      );

      video.addEventListener(
        "error",
        reject,
        { once: true }
      );

      video.currentTime = safeTimestamp;
    });

    const canvas = document.createElement("canvas");

    canvas.width = 240;
    canvas.height = 135;

    const context = canvas.getContext("2d");

    if (!context) return;

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    const image = document.createElement("img");

    image.src = canvas.toDataURL("image/jpeg", 0.8);
    image.alt = "";
    image.loading = "lazy";

    thumbnailBox.replaceChildren(image);

  } catch (error) {
    console.warn(
      "Could not generate video thumbnail:",
      error
    );

  } finally {
    video.removeAttribute("src");
    video.load();
  }
}

/* =========================================================
   12. MATCHING MOMENTS
   ========================================================= */

function renderResults(results) {
  if (!UI.resultsList) return;

  UI.resultsList.replaceChildren();

  if (UI.resultsCount) {
    UI.resultsCount.textContent =
      `${results.length} result${results.length === 1 ? "" : "s"}`;
  }

  if (UI.resultsFooter) {
    UI.resultsFooter.hidden = results.length === 0;
  }

  if (!results.length) {
    clearResults(
      "No matching moments found. Try another query."
    );
    return;
  }

  results.forEach((moment, index) => {
    const start = getMomentStart(moment);
    const end = getMomentEnd(moment);

    const card = createElement(
      "button",
      "result-card"
    );

    card.type = "button";
    card.dataset.resultIndex = String(index);

    card.setAttribute(
      "aria-label",
      `Play matching moment ${index + 1}, ${formatTime(start)} to ${formatTime(end)}`
    );

    const number = createElement(
      "span",
      "result-index",
      String(index + 1).padStart(2, "0")
    );

    const thumbnailBox = createElement(
      "span",
      "result-thumbnail"
    );

    const thumbnail = getValue(
      moment,
      ["thumbnail_url", "thumbnail", "frame_url"],
      null
    );

    if (thumbnail) {
      const image = document.createElement("img");

      image.src = resolveMediaUrl(thumbnail);
      image.alt = "";
      image.loading = "lazy";

      thumbnailBox.appendChild(image);

    } else if (state.currentVideoUrl) {
      generateMomentThumbnail(
        state.currentVideoUrl,
        start,
        thumbnailBox
      );
    }

    const details = createElement(
      "span",
      "result-details"
    );

    details.append(
      createElement(
        "span",
        "result-time",
        `${formatTime(start)} - ${formatTime(end)}`
      ),
      createElement(
        "span",
        "result-description",
        getMomentDescription(moment)
      )
    );

    const play = createElement(
      "span",
      "result-play",
      "\u25B6"
    );

    play.setAttribute("aria-hidden", "true");

    card.append(
      number,
      thumbnailBox,
      details,
      play
    );

    card.addEventListener("click", () => {
      playMoment(index);
    });

    UI.resultsList.appendChild(card);
  });
}

function playMoment(index) {
  const moment = state.results[index];

  if (
    !moment ||
    !state.currentVideoId ||
    !UI.player
  ) {
    return;
  }

  const start = getMomentStart(moment);

  state.activeResult = index;

  UI.resultsList
    ?.querySelectorAll(".result-card")
    .forEach((card, i) => {
      card.classList.toggle("active", i === index);
    });

  const seek = () => {
    UI.player.currentTime = Math.max(0, start);

    UI.player.play().catch((error) => {
      console.warn(
        "Playback could not start:",
        error
      );
    });
  };

  if (
    UI.player.readyState >=
    HTMLMediaElement.HAVE_METADATA
  ) {
    seek();
  } else {
    UI.player.addEventListener(
      "loadedmetadata",
      seek,
      { once: true }
    );
  }
}

/* =========================================================
   13. ANIMATED VIDEO-MOMENT BANNER
   ========================================================= */

function initializeMomentBanner() {
  const timeline = $("bannerTimeline");
  const frames = $("bannerFrames");
  const ruler = $("bannerRuler");
  const playhead = $("bannerPlayhead");
  const timecode = $("bannerTimecode");
  const status = $("bannerStatus");
  const pauseButton = $("bannerPause");

  const timecodePanel =
    document.querySelector(".timecode-panel");

  if (
    !timeline ||
    !frames ||
    !ruler ||
    !playhead ||
    !timecode ||
    !status
  ) {
    return;
  }

  const IMAGE_IDS = [
    10, 11, 14, 15, 16, 17,
    18, 19, 20, 22, 24, 25,
    27, 28, 29, 37, 39, 42
  ];

  const FRAME_WIDTH = 148;
  const FRAME_STEP = 160;
  const CRUISE_SPEED = 560;
  const FRAME_DURATION = 83.7;

  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  let width = 0;
  let playheadX = 0;
  let slots = [];
  let slotCount = 0;

  let position = 500 * FRAME_STEP;
  let velocity = 0;

  let mode = "run";
  let holdTime = 0;
  let targetIndex = null;
  let foundCount = 0;

  let paused = reduceMotion;
  let hovering = false;
  let pointerX = 0;
  let dirty = true;
  let lastTime = 0;
  let currentStatus = "";

  const mod = (a, b) =>
    ((a % b) + b) % b;

  const pad = (value, length = 2) =>
    String(value).padStart(length, "0");

  function buildFrames() {
    width = timeline.clientWidth;

    playheadX = Math.round(width * 0.28);

    playhead.style.transform =
      `translateX(${playheadX}px)`;

    const required =
      Math.ceil(width / FRAME_STEP) + 3;

    if (required === slotCount) {
      dirty = true;
      return;
    }

    frames.replaceChildren();

    slots = [];
    slotCount = required;

    for (let i = 0; i < slotCount; i++) {
      const frame = createElement(
        "div",
        "timeline-frame"
      );

      const inner = createElement(
        "div",
        "timeline-frame-inner"
      );

      const gray = document.createElement("img");

      gray.className = "frame-grayscale";
      gray.alt = "";
      gray.draggable = false;

      const color = document.createElement("img");

      color.className = "frame-color";
      color.alt = "";
      color.draggable = false;

      gray.onerror = () => {
        gray.style.visibility = "hidden";
      };

      color.onerror = () => {
        color.style.visibility = "hidden";
      };

      const number = createElement(
        "div",
        "timeline-frame-number"
      );

      inner.append(gray, color);
      frame.append(inner, number);
      frames.appendChild(frame);

      slots.push({
        element: frame,
        images: [gray, color],
        number,
        index: null
      });
    }

    dirty = true;
  }

  function timestampFor(index) {
    return (
      mod(index, 1000) * FRAME_DURATION +
      mod(index * 13, 24) / 24
    );
  }

  function formatTimecode(time) {
    const seconds = Math.max(
      0,
      Math.floor(time)
    );

    const frame = Math.floor(
      (time - seconds) * 24
    );

    return [
      pad(Math.floor(seconds / 3600) % 24),
      pad(Math.floor(seconds / 60) % 60),
      pad(seconds % 60),
      pad(frame)
    ].join(":");
  }

  function updateStatus(message) {
    if (message === currentStatus) return;

    currentStatus = message;

    status.replaceChildren();

    [...message].forEach((character, index) => {
      const span = document.createElement("span");

      span.textContent = character;

      span.style.animationDelay =
        `${reduceMotion ? 0 : index * 22}ms`;

      status.appendChild(span);
    });
  }

  function render() {
    const base =
      Math.floor(position / FRAME_STEP) - 1;

    for (let i = 0; i < slotCount; i++) {
      const absoluteIndex = base + i;

      const slot = slots[
        mod(absoluteIndex, slotCount)
      ];

      const x =
        absoluteIndex * FRAME_STEP - position;

      if (slot.index !== absoluteIndex) {
        slot.index = absoluteIndex;

        const imageId = IMAGE_IDS[
          mod(absoluteIndex, IMAGE_IDS.length)
        ];

        const source =
          `https://picsum.photos/id/${imageId}/296/192`;

        slot.images.forEach((image) => {
          image.style.visibility = "";
          image.src = source;
        });

        slot.number.textContent =
          pad(mod(absoluteIndex, 1000), 3);
      }

      slot.element.style.transform =
        `translate3d(${x}px, 0, 0)`;

      slot.element.classList.toggle(
        "active",
        mode === "hold" &&
          absoluteIndex === targetIndex
      );
    }

    const fractional =
      (
        position +
        playheadX -
        FRAME_WIDTH / 2
      ) / FRAME_STEP;

    const floor = Math.floor(fractional);

    const value =
      timestampFor(floor) +
      (
        timestampFor(floor + 1) -
        timestampFor(floor)
      ) * (fractional - floor);

    timecode.textContent = formatTimecode(value);

    ruler.style.setProperty(
      "--ox",
      `${-mod(position, FRAME_STEP)}px`
    );
  }

  function highlight(active) {
    slots.forEach((slot) => {
      slot.element.classList.toggle(
        "active",
        active && slot.index === targetIndex
      );
    });

    if (timecodePanel) {
      timecodePanel.classList.toggle(
        "hit",
        active
      );
    }
  }

  function animate(timestamp) {
    const delta = Math.min(
      0.05,
      (timestamp - lastTime) / 1000 || 0
    );

    lastTime = timestamp;

    if (hovering && !paused) {
      const normalized = Math.max(
        -1,
        Math.min(
          1,
          (pointerX - width / 2) / (width / 2)
        )
      );

      const targetVelocity =
        Math.abs(normalized) < 0.08
          ? 0
          : normalized * 900;

      velocity +=
        (targetVelocity - velocity) *
        Math.min(1, delta * 6);

      if (mode === "hold") {
        mode = "run";
        targetIndex = null;
        highlight(false);
      }

      updateStatus(
        Math.abs(velocity) > 8
          ? velocity < 0
            ? "REWINDING"
            : "SCRUBBING"
          : "SCRUB THE TIMELINE"
      );

    } else if (paused) {
      velocity +=
        (0 - velocity) *
        Math.min(1, delta * 8);

      if (Math.abs(velocity) < 0.5) {
        velocity = 0;
      }

      if (mode !== "hold") {
        updateStatus("PAUSED");
      }

    } else {
      if (mode === "run") {
        velocity +=
          (CRUISE_SPEED - velocity) *
          Math.min(1, delta * 2.2);

        updateStatus("SCANNING MOMENTS...");

        if (targetIndex === null) {
          targetIndex =
            Math.floor(
              (position + playheadX) / FRAME_STEP
            ) +
            5 +
            Math.floor(Math.random() * 4);
        }

        const distance =
          targetIndex * FRAME_STEP -
          position +
          FRAME_WIDTH / 2 -
          playheadX;

        if (distance < 0) {
          targetIndex = null;
        } else if (
          distance < velocity * 0.65
        ) {
          mode = "brake";
        }
      }

      if (mode === "brake") {
        const distance =
          targetIndex * FRAME_STEP -
          position +
          FRAME_WIDTH / 2 -
          playheadX;

        velocity = Math.min(
          velocity,
          Math.max(distance * 1.7, 10)
        );

        if (velocity * delta >= distance) {
          position += distance;
          velocity = 0;

          mode = "hold";
          holdTime = 1.15;

          highlight(true);

          updateStatus(
            `MOMENT FOUND #${pad(++foundCount, 3)}`
          );

          dirty = true;
        }
      }

      if (mode === "hold") {
        holdTime -= delta;

        if (holdTime <= 0) {
          mode = "run";
          highlight(false);
          targetIndex = null;
        }
      }
    }

    if (velocity !== 0) {
      position += velocity * delta;
      dirty = true;
    }

    if (dirty) {
      render();
      dirty = false;
    }

    requestAnimationFrame(animate);
  }

  timeline.addEventListener("pointerenter", (event) => {
    if (event.pointerType !== "mouse") return;

    hovering = true;

    pointerX =
      event.clientX -
      timeline.getBoundingClientRect().left;
  });

  timeline.addEventListener("pointermove", (event) => {
    if (event.pointerType !== "mouse") return;

    hovering = true;

    pointerX =
      event.clientX -
      timeline.getBoundingClientRect().left;
  });

  timeline.addEventListener("pointerleave", () => {
    hovering = false;
    targetIndex = null;
    mode = "run";

    highlight(false);
    velocity *= 0.5;
  });

  function updatePauseButton() {
    if (!pauseButton) return;

    pauseButton.innerHTML = paused
      ? 'PLAY <span aria-hidden="true">&#9654;</span>'
      : 'PAUSE <span aria-hidden="true">&#10074;&#10074;</span>';

    pauseButton.setAttribute(
      "aria-pressed",
      String(paused)
    );
  }

  if (pauseButton) {
    pauseButton.addEventListener("click", () => {
      paused = !paused;

      updatePauseButton();

      if (paused && mode !== "hold") {
        mode = "run";
        targetIndex = null;
      }

      if (!paused && mode === "hold") {
        holdTime = 0.3;
      }
    });
  }

  if (typeof ResizeObserver !== "undefined") {
    const resizeObserver =
      new ResizeObserver(buildFrames);

    resizeObserver.observe(timeline);
  } else {
    window.addEventListener(
      "resize",
      buildFrames
    );
  }

  buildFrames();
  updatePauseButton();

  if (reduceMotion) {
    targetIndex = 502;

    position =
      targetIndex * FRAME_STEP +
      FRAME_WIDTH / 2 -
      playheadX;

    mode = "hold";
    holdTime = Number.POSITIVE_INFINITY;

    render();
    highlight(true);
    updateStatus("PAUSED");
  }

  requestAnimationFrame(animate);
}

/* =========================================================
   14. DASHBOARD INITIALIZATION
   ========================================================= */

async function initializeDashboard() {
  clearResults();
  clearAnswer();

  initializePlayerEvents();
  initializeUploadEvents();
  initializeYoutubeEvents();
  initializeExampleQueries();
  initializeSearchEvents();

  initializeMomentBanner();

  try {
    await initializeVisitorSession();
  } catch (error) {
    console.error(
      "Visitor session initialization failed:",
      error
    );

    setMessage(
      UI.sourceStatus,
      error.message,
      true
    );

    return;
  }

  await loadSampleVideos();

  console.info(
    "VideoMoment dashboard initialized."
  );
}

document.addEventListener(
  "DOMContentLoaded",
  initializeDashboard
);





