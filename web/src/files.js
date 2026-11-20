// Getting movies in (file input, drag and drop, folders) and barcodes out (downloads or the picked folder).

const VIDEO_EXTENSIONS = [".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".ogv"];
export const OUTPUT_FOLDER = "barcodes";

export function isVideoFile(file) {
  const name = file.name.toLowerCase();
  return file.type.startsWith("video/") || VIDEO_EXTENSIONS.some((ext) => name.endsWith(ext));
}

export const canPickDirectory = () => typeof window.showDirectoryPicker === "function";

/**
 * Lets the user pick a folder with the File System Access API (Chromium) and lists the videos it contains,
 * subfolders included (except the barcodes output folder).
 *
 * @returns {Promise<{dir: FileSystemDirectoryHandle, files: {file: File, path: string}[]}|null>} null if cancelled
 */
export async function pickDirectory() {
  let dir;
  try {
    dir = await window.showDirectoryPicker({ id: "movie-barcodes", mode: "read" });
  } catch (error) {
    if (error.name === "AbortError") return null;
    throw error;
  }
  const files = [];
  async function walk(handle, prefix) {
    for await (const entry of handle.values()) {
      if (entry.kind === "directory") {
        if (!(prefix === "" && entry.name === OUTPUT_FOLDER) && !entry.name.startsWith(".")) {
          await walk(entry, `${prefix}${entry.name}/`);
        }
      } else if (!entry.name.startsWith(".")) {
        const file = await entry.getFile();
        if (isVideoFile(file)) files.push({ file, path: prefix + file.name });
      }
    }
  }
  await walk(dir, "");
  files.sort((a, b) => a.path.localeCompare(b.path));
  return { dir, files };
}

/** Asks for write access to a picked folder. Must be called from a user gesture (e.g. a click). */
export async function requestWriteAccess(dir) {
  const options = { mode: "readwrite" };
  if ((await dir.queryPermission(options)) === "granted") return true;
  return (await dir.requestPermission(options)) === "granted";
}

/** Writes a PNG into the barcodes/ subfolder of a picked folder (like the CLI's default output folder). */
export async function saveToDirectory(dir, name, blob) {
  const out = await dir.getDirectoryHandle(OUTPUT_FOLDER, { create: true });
  const handle = await out.getFileHandle(name, { create: true });
  const writable = await handle.createWritable();
  await writable.write(blob);
  await writable.close();
}

/** Video files from a drop event, folders included (webkitGetAsEntry). */
export async function filesFromDrop(dataTransfer) {
  const entries = [...dataTransfer.items].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dataTransfer.files].filter(isVideoFile).map((file) => ({ file, path: file.name }));
  const files = [];
  const readAll = (reader) => new Promise((resolve, reject) => {
    const all = [];
    const next = () => reader.readEntries((batch) => (batch.length ? (all.push(...batch), next()) : resolve(all)), reject);
    next();
  });
  async function walk(entry, prefix) {
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      if (isVideoFile(file)) files.push({ file, path: prefix + file.name });
    } else if (entry.isDirectory && entry.name !== OUTPUT_FOLDER) {
      for (const child of await readAll(entry.createReader())) await walk(child, `${prefix}${entry.name}/`);
    }
  }
  for (const entry of entries) await walk(entry, "");
  return files;
}

/** Triggers the download of a blob URL. */
export function download(url, name) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
