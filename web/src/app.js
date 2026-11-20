// User interface: queue of movies, options, batch processing with live previews, PNG export.

import { computeColors } from "./barcode.js";
import { METHODS, toByteColor } from "./color.js";
import { drawCircular, drawHorizontal, outputName } from "./layout.js";
import {
  canPickDirectory, download, filesFromDrop, isVideoFile, OUTPUT_FOLDER, pickDirectory, requestWriteAccess,
  saveToDirectory,
} from "./files.js";
import { UnsupportedVideoError } from "./video.js";

const $ = (id) => document.getElementById(id);
const PREVIEW_SIZE = 180;

const state = {
  items: [], // {file?: File, url?: string, path: string}
  dir: null, // FileSystemDirectoryHandle of a picked folder
  controller: null,
  outputs: [], // {name, url}
  results: [], // exposed as window.movieBarcodes for debugging and the smoke test
};
window.movieBarcodes = state.results;

// ---------------------------------------------------------------------------------------------------- inputs

function setItems(items, dir = null) {
  state.items = items;
  state.dir = dir;
  renderQueue();
}

function addItems(items) {
  const known = new Set(state.items.map((i) => i.path));
  state.items.push(...items.filter((i) => !known.has(i.path)));
  state.dir = null; // mixing sources: no folder to write back to
  renderQueue();
}

function renderQueue() {
  const queue = $("queue");
  queue.replaceChildren(...state.items.map((item) => {
    const li = document.createElement("li");
    li.textContent = item.path;
    if (item.file) {
      const size = document.createElement("span");
      size.className = "size";
      size.textContent = formatBytes(item.file.size);
      li.append(" ", size);
    }
    return li;
  }));
  $("clear").hidden = !state.items.length;
  $("save-to-folder-row").hidden = !state.dir;
  $("start").disabled = !state.items.length || !!state.controller;
}

$("file").addEventListener("change", (e) => {
  addItems([...e.target.files].filter(isVideoFile).map((file) => ({ file, path: file.name })));
  e.target.value = "";
});

$("folder").addEventListener("change", (e) => {
  const files = [...e.target.files].filter(isVideoFile).map((file) => ({ file, path: file.webkitRelativePath || file.name }));
  if (!files.length) alert("No video files found in this folder.");
  setItems(files);
  e.target.value = "";
});

if (canPickDirectory()) {
  $("folder-fallback").hidden = true;
  $("pick-folder").addEventListener("click", async () => {
    try {
      const picked = await pickDirectory();
      if (!picked) return;
      if (!picked.files.length) alert("No video files found in this folder.");
      setItems(picked.files, picked.dir);
    } catch (error) {
      alert(`Could not read the folder: ${error.message}`);
    }
  });
} else {
  $("pick-folder").hidden = true;
}

const dropzone = $("dropzone");
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("over"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("over"));
dropzone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropzone.classList.remove("over");
  const files = await filesFromDrop(e.dataTransfer);
  if (!files.length) alert("No video files in what was dropped.");
  addItems(files);
});

$("clear").addEventListener("click", () => setItems([]));

// Same-origin URL for testing, e.g. /web/?src=/tests/sample.mp4 when serving the repository root
const src = new URLSearchParams(location.search).get("src");
if (src) setItems([{ url: src, path: decodeURIComponent(src.split("/").pop()) }]);

// --------------------------------------------------------------------------------------------------- options

function readOptions() {
  const form = new FormData($("options"));
  const int = (name) => {
    const v = parseInt(form.get(name), 10);
    return Number.isFinite(v) && v > 0 ? v : 0;
  };
  return {
    methods: form.getAll("method"),
    types: form.getAll("type"),
    samples: Math.min(int("samples"), 16384),
    height: Math.min(int("height"), 16384),
    diameter: Math.min(int("diameter"), 8192),
    analysisWidth: int("analysisWidth"),
    fps: parseFloat(form.get("fps")) > 0 ? parseFloat(form.get("fps")) : 0,
    saveToFolder: !!state.dir && form.get("saveToFolder") === "on",
  };
}

// ----------------------------------------------------------------------------------------------- processing

function waitUntilVisible() {
  if (!document.hidden) return Promise.resolve();
  $("visibility-warning").hidden = false;
  return new Promise((resolve) => {
    const onChange = () => {
      if (document.hidden) return;
      document.removeEventListener("visibilitychange", onChange);
      $("visibility-warning").hidden = true;
      resolve();
    };
    document.addEventListener("visibilitychange", onChange);
  });
}

$("start").addEventListener("click", start);
$("cancel").addEventListener("click", () => state.controller?.abort(new DOMException("Cancelled", "AbortError")));
$("download-all").addEventListener("click", async () => {
  for (const { url, name } of state.outputs) {
    download(url, name);
    await new Promise((r) => setTimeout(r, 300));
  }
});

async function start() {
  const options = readOptions();
  if (!options.methods.length || !options.types.length) {
    alert("Choose at least one color method and one barcode type.");
    return;
  }
  // Must happen in the click handler (user gesture)
  if (options.saveToFolder && !(await requestWriteAccess(state.dir).catch(() => false))) {
    options.saveToFolder = false;
    alert("No write access to the folder: the barcodes will be downloaded instead.");
  }

  for (const { url } of state.outputs) URL.revokeObjectURL(url);
  state.outputs = [];
  state.results.length = 0;
  const controller = (state.controller = new AbortController());
  $("start").disabled = true;
  $("cancel").hidden = false;
  $("download-all").hidden = true;
  $("progress-panel").hidden = false;
  $("results").replaceChildren();
  const wakeLock = await navigator.wakeLock?.request("screen").catch(() => null);

  const items = [...state.items];
  const cards = items.map((item) => createCard(item));
  const startTime = performance.now();
  let failures = 0;
  const overall = (fraction) => {
    const progress = Math.min(1, fraction / items.length);
    $("overall-progress").value = progress;
    const elapsed = (performance.now() - startTime) / 1000;
    const eta = progress > 0 && progress < 1 ? ` · about ${formatDuration((elapsed * (1 - progress)) / progress)} left` : "";
    const done = Math.min(items.length, Math.floor(fraction));
    $("overall-text").textContent = `${done} / ${items.length} movies · ${formatDuration(elapsed)} elapsed${eta}`;
  };
  overall(0);

  for (let index = 0; index < items.length; index++) {
    const card = cards[index];
    if (controller.signal.aborted) {
      card.setStatus("cancelled");
      continue;
    }
    try {
      await processItem(items[index], card, options, controller.signal, (f) => overall(index + f));
    } catch (error) {
      if (controller.signal.aborted) {
        card.setStatus("cancelled");
      } else {
        failures++;
        card.fail(describeError(error, items[index]));
        console.error(items[index].path, error);
      }
    }
    overall(index + 1);
  }

  const elapsed = formatDuration((performance.now() - startTime) / 1000);
  $("overall-text").textContent = controller.signal.aborted
    ? `Cancelled after ${elapsed}.`
    : `Done in ${elapsed}: ${items.length - failures} / ${items.length} movies` + (failures ? `, ${failures} failed.` : ".");
  wakeLock?.release().catch(() => {});
  state.controller = null;
  $("cancel").hidden = true;
  $("download-all").hidden = !state.outputs.length || options.saveToFolder;
  renderQueue();
}

async function processItem(item, card, options, signal, onFraction) {
  const url = item.url ?? URL.createObjectURL(item.file);
  const startTime = performance.now();
  let previews = null;
  try {
    card.setStatus("opening…");
    const result = await computeColors({
      url,
      host: document.body,
      methods: options.methods,
      samples: options.samples,
      fps: options.fps,
      analysisWidth: options.analysisWidth,
      signal,
      waitUntilVisible,
      onStall: (stalled) => { $("visibility-warning").hidden = !stalled; },
      onInfo: (info) => {
        card.setMeta(info);
        card.setStatus("processing");
        previews = card.createPreviews(options, info.samples);
      },
      onProgress: (done, colors) => {
        const fraction = done / previews.samples;
        const elapsed = (performance.now() - startTime) / 1000;
        card.setProgress(fraction, fraction < 1 ? (elapsed * (1 - fraction)) / fraction : 0);
        previews.update(done, colors);
        onFraction(fraction);
      },
    });
    if (result.colors[options.methods[0]].length === 0) throw new Error("no frame could be read");

    card.setStatus("saving…");
    const saved = [];
    for (const preview of previews.list) {
      const colors = result.colors[preview.method];
      const name = outputName(item.path.split("/").pop(), METHODS[preview.method].fileMethod, preview.type);
      const blob = await renderPng(colors, preview.type, result.info, options);
      const blobUrl = URL.createObjectURL(blob);
      state.outputs.push({ url: blobUrl, name });
      if (options.saveToFolder) {
        await saveToDirectory(state.dir, name, blob);
        saved.push(name);
      }
      preview.finish(blobUrl, name, options.saveToFolder);
    }
    const seconds = (performance.now() - startTime) / 1000;
    state.results.push({ path: item.path, ...result, seconds });
    card.done(seconds, result.missedFrames, saved);
  } finally {
    if (!item.url) URL.revokeObjectURL(url);
  }
}

/** Draws a full size barcode and encodes it as PNG. */
async function renderPng(colors, type, info, options) {
  let canvas;
  if (type === "horizontal") {
    canvas = new OffscreenCanvas(colors.length, options.height || info.height);
    drawHorizontal(canvas.getContext("2d"), colors);
  } else {
    const size = options.diameter || info.width;
    canvas = new OffscreenCanvas(size, size);
    drawCircular(canvas.getContext("2d"), colors);
  }
  return canvas.convertToBlob({ type: "image/png" });
}

function describeError(error, item) {
  const name = item.path.toLowerCase();
  if (error instanceof UnsupportedVideoError) {
    let hint = "";
    const isChromium = !!navigator.userAgentData?.brands?.some((b) => /Chrom/.test(b.brand));
    if (name.endsWith(".mkv") && !isChromium) hint = " MKV files only open in Chrome or Edge.";
    else if (!isChromium) hint = " Try Chrome or Edge, which support more codecs.";
    return `Cannot read this movie: ${error.message}.${hint}`;
  }
  return `Failed: ${error.message}`;
}

// ------------------------------------------------------------------------------------------------- results

function createCard(item) {
  const node = $("result-template").content.firstElementChild.cloneNode(true);
  node.querySelector(".name").textContent = item.path;
  const status = node.querySelector(".status");
  const progress = node.querySelector("progress");
  const eta = node.querySelector(".eta");
  $("results").appendChild(node);
  const setStatus = (text) => { status.textContent = text; node.dataset.status = text; };
  setStatus("queued");
  return {
    setStatus,
    setMeta(info) {
      const fps = `${+info.fps.toFixed(3)} fps${info.fpsSource === "detected" ? "" : ` (${info.fpsSource})`}`;
      node.querySelector(".meta").textContent =
        `${info.width}×${info.height} · ${formatDuration(info.duration)} · ${fps} · ~${info.frameCount} frames · ` +
        `${info.samples} samples`;
    },
    setProgress(fraction, secondsLeft) {
      progress.value = fraction;
      eta.textContent = `${Math.round(fraction * 100)} %` + (secondsLeft ? ` · ${formatDuration(secondsLeft)} left` : "");
    },
    createPreviews(options, samples) {
      const container = node.querySelector(".previews");
      const list = [];
      for (const type of options.types) {
        for (const method of options.methods) list.push(createPreview(container, method, type, samples));
      }
      let lastCircular = 0;
      return {
        samples,
        list,
        update(done, colors) {
          const now = performance.now();
          const redrawCircular = done === samples || now - lastCircular > 250;
          if (redrawCircular) lastCircular = now;
          for (const p of list) {
            if (p.type === "horizontal") p.addColumn(done - 1, colors[p.method][done - 1]);
            else if (redrawCircular) drawCircular(p.ctx, colors[p.method], samples);
          }
        },
      };
    },
    fail(message) {
      setStatus("error");
      const error = node.querySelector(".error");
      error.textContent = message;
      error.hidden = false;
    },
    done(seconds, missedFrames, saved) {
      setStatus("done");
      progress.value = 1;
      eta.textContent = `done in ${formatDuration(seconds)}` +
        (saved.length ? ` · saved in ${OUTPUT_FOLDER}/` : "") +
        (missedFrames ? ` · ${missedFrames} frames may be approximate` : "");
    },
  };
}

function createPreview(container, method, type, samples) {
  const figure = document.createElement("figure");
  figure.className = `preview ${type}`;
  const canvas = document.createElement("canvas");
  if (type === "horizontal") {
    canvas.width = samples;
    canvas.height = 1;
  } else {
    canvas.width = canvas.height = PREVIEW_SIZE * Math.min(2, window.devicePixelRatio || 1);
  }
  const caption = document.createElement("figcaption");
  caption.textContent = `${METHODS[method].label} · ${type}`;
  figure.append(canvas, caption);
  container.appendChild(figure);
  const ctx = canvas.getContext("2d");
  const column = ctx.createImageData(1, 1);
  return {
    method,
    type,
    ctx,
    addColumn(x, color) {
      const [r, g, b] = toByteColor(color);
      column.data.set([r, g, b, 255]);
      ctx.putImageData(column, x, 0);
    },
    finish(url, name, savedToFolder) {
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.textContent = savedToFolder ? `${name} (saved)` : name;
      caption.replaceChildren(`${METHODS[method].label} · ${type} · `, link);
    },
  };
}

// --------------------------------------------------------------------------------------------------- format

function formatDuration(seconds) {
  seconds = Math.round(seconds);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h) return `${h} h ${String(m).padStart(2, "0")} min`;
  if (m) return `${m} min ${String(s).padStart(2, "0")} s`;
  return `${s} s`;
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (bytes >= 1024 && i < units.length - 1) { bytes /= 1024; i++; }
  return `${bytes.toFixed(i ? 1 : 0)} ${units[i]}`;
}
