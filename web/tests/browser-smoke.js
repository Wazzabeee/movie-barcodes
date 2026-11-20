// End-to-end check in headless Chrome: runs the app on tests/sample.mp4 (all frames, all methods) and compares
// the colors with the ones the Python package computes (fixture.json, "sample_mp4").
//
// Not run in CI (needs Chrome). Usage, from web/:  npm run smoke
// Options (environment): CHROME=<path to the Chrome binary>, ANALYSIS_WIDTH=<256|640|0 (native)>, PORT=<CDP port>,
// SCREENSHOT=<png path> to save a screenshot of the page once done.
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url)); // repository root: serves web/ and tests/sample.mp4
const fixture = JSON.parse(readFileSync(new URL("./fixture.json", import.meta.url), "utf8")).sample_mp4;
const chromePath = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const analysisWidth = process.env.ANALYSIS_WIDTH ?? "256";
const cdpPort = Number(process.env.PORT || 9333);
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".mp4": "video/mp4" };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Static server with Range support (media elements seek with range requests)
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^\/+/, "");
  let file = join(root, path.endsWith("/") || path === "" ? join(path, "index.html") : path);
  if (!file.startsWith(root)) return res.writeHead(403).end();
  let data;
  try { data = readFileSync(file); } catch { return res.writeHead(404).end(); }
  const headers = { "Content-Type": TYPES[extname(file)] || "application/octet-stream", "Accept-Ranges": "bytes" };
  const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range || "");
  if (range) {
    const start = Number(range[1] || 0);
    const end = range[2] ? Number(range[2]) : data.length - 1;
    res.writeHead(206, { ...headers, "Content-Range": `bytes ${start}-${end}/${data.length}` });
    return res.end(data.subarray(start, end + 1));
  }
  res.writeHead(200, headers).end(data);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const appUrl = `http://127.0.0.1:${server.address().port}/web/?src=/tests/sample.mp4`;

const profile = mkdtempSync(join(tmpdir(), "barcode-smoke-"));
const chrome = spawn(chromePath, [
  "--headless=new", `--remote-debugging-port=${cdpPort}`, `--user-data-dir=${profile}`,
  "--autoplay-policy=no-user-gesture-required", "--no-first-run", "about:blank",
], { stdio: "ignore" });

let exitCode = 1;
try {
  let targets;
  for (let i = 0; i < 50 && !targets; i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${cdpPort}/json`)).json(); } catch { await sleep(200); }
  }
  const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r));
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (e) => {
    const m = JSON.parse(e.data);
    pending.get(m.id)?.(m);
    pending.delete(m.id);
  });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const evaluate = async (expression) => {
    const res = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true, timeout: 600000 });
    if (res.result?.exceptionDetails) throw new Error(JSON.stringify(res.result.exceptionDetails));
    return res.result?.result?.value;
  };

  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1100, height: 1300, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: appUrl });
  await sleep(1500);
  const result = await evaluate(`(async () => {
    for (const box of document.querySelectorAll('input[name=method]')) box.checked = true;
    document.querySelector('select[name=analysisWidth]').value = ${JSON.stringify(analysisWidth)};
    document.getElementById('start').click();
    while (!document.querySelector('.result[data-status=done], .result[data-status=error]')) {
      await new Promise((r) => setTimeout(r, 100));
    }
    const card = document.querySelector('.result');
    const r = window.movieBarcodes[0];
    return { status: card.dataset.status, error: card.querySelector('.error').textContent, info: r?.info,
      colors: r?.colors, seconds: r?.seconds, missedFrames: r?.missedFrames,
      outputs: [...card.querySelectorAll('figcaption a')].map((a) => a.download) };
  })()`);
  if (process.env.SCREENSHOT) {
    const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
    writeFileSync(process.env.SCREENSHOT, Buffer.from(shot.result.data, "base64"));
  }
  ws.close();

  if (result.status !== "done") throw new Error(`processing failed: ${result.error}`);
  console.log(`info: ${JSON.stringify(result.info)}`);
  console.log(`time: ${result.seconds.toFixed(2)} s, missed frames: ${result.missedFrames}`);
  console.log(`outputs: ${result.outputs.join(", ")}`);
  if (result.info.frameCount !== fixture.frame_count) throw new Error(`frame count ${result.info.frameCount} != ${fixture.frame_count}`);
  let worst = 0;
  for (const [method, expected] of Object.entries(fixture.colors)) {
    const got = result.colors[method];
    let max = 0, sum = 0, n = 0;
    expected.forEach((c, i) => c.forEach((v, ch) => {
      const d = Math.abs(got[i][ch] - v);
      max = Math.max(max, d); sum += d; n++;
    }));
    worst = Math.max(worst, max);
    console.log(`${method.padEnd(8)} vs Python: mean |diff| ${(sum / n).toFixed(2)}, max |diff| ${max.toFixed(2)} (levels /255)`);
  }
  if (worst > 6) throw new Error(`colors differ from Python by up to ${worst.toFixed(2)} levels`);
  console.log("OK");
  exitCode = 0;
} catch (error) {
  console.error(error);
} finally {
  chrome.kill();
  server.close();
  await sleep(300);
  rmSync(profile, { recursive: true, force: true });
  process.exit(exitCode);
}
