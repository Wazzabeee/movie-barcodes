// Frame access through an HTMLVideoElement: the browser decodes the file (H.264, HEVC, VP9, AV1... whatever it
// supports) straight from disk through an object URL, so multi-GB movies are never loaded in memory.
//
// Each sample seeks to the middle of the wanted frame, waits for "seeked" and then for the new frame to be
// presented (requestVideoFrameCallback), and reads it back through a small canvas.

export class UnsupportedVideoError extends Error {}

const COMMON_FPS = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120];

function once(target, event, signal) {
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    target.addEventListener(event, (e) => { signal?.removeEventListener("abort", onAbort); resolve(e); }, { once: true });
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function timeout(ms, value) {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

const errorMessage = (video) => {
  const code = video.error?.code;
  if (code === 4) return "format or codec not supported by this browser";
  if (code === 3) return "the video could not be decoded";
  if (code === 2) return "the file could not be read";
  return "the video could not be opened";
};

/**
 * Loads a video in a hidden-but-attached <video> element (frames are only presented for elements in the DOM)
 * and primes the decoder.
 *
 * @param {string} url object URL or same-origin URL
 * @param {HTMLElement} host element the <video> is attached to
 * @param {AbortSignal} [signal]
 * @returns {Promise<HTMLVideoElement>}
 * @throws {UnsupportedVideoError} if the browser cannot play the file
 */
export async function openVideo(url, host, signal) {
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "auto";
  video.playsInline = true;
  video.className = "decoder";
  host.appendChild(video);
  try {
    const failed = once(video, "error", signal).then(() => { throw new UnsupportedVideoError(errorMessage(video)); });
    video.src = url;
    const loaded = await Promise.race([once(video, "loadedmetadata", signal), failed, timeout(20000, "timeout")]);
    if (loaded === "timeout") throw new UnsupportedVideoError("timed out while reading the video metadata");
    if (!video.videoWidth || !video.videoHeight) {
      throw new UnsupportedVideoError("no video track this browser can decode (audio only, or unsupported codec)");
    }
    if (!Number.isFinite(video.duration) || video.duration <= 0) {
      throw new UnsupportedVideoError("the video duration is unknown");
    }
    // Prime the decoder: a paused, never played element may stay at HAVE_METADATA and never decode a frame
    await video.play().catch(() => {});
    video.pause();
    if (video.readyState < 2) {
      const ready = await Promise.race([once(video, "loadeddata", signal), failed, timeout(20000, "timeout")]);
      if (ready === "timeout") throw new UnsupportedVideoError("the first frame could not be decoded");
    }
    failed.catch(() => {});
    return video;
  } catch (error) {
    closeVideo(video);
    throw error;
  }
}

/** Releases the decoder and detaches the element. */
export function closeVideo(video) {
  video.pause();
  video.removeAttribute("src");
  video.load();
  video.remove();
}

/**
 * Estimates the frame rate from the media times of the frames presented while playing for a short moment
 * (<video> exposes neither the frame rate nor the frame count). Snapped to a common rate when close.
 *
 * @returns {Promise<number|null>} frames per second, or null if it could not be measured
 */
export async function detectFps(video) {
  if (!("requestVideoFrameCallback" in video)) return null;
  const times = [];
  const start = video.currentTime;
  await new Promise((resolve) => {
    const onFrame = (_now, meta) => {
      times.push(meta.mediaTime);
      if (times.length >= 10) resolve();
      else video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
    video.play().catch(resolve);
    setTimeout(resolve, 1500);
  });
  video.pause();
  video.currentTime = start;
  // Smallest positive gap between consecutive presented frames (dropped frames only make gaps larger)
  let gap = Infinity;
  for (let i = 1; i < times.length; i++) {
    const d = times[i] - times[i - 1];
    if (d > 1e-4 && d < gap) gap = d;
  }
  if (!Number.isFinite(gap)) return null;
  const fps = 1 / gap;
  const snapped = COMMON_FPS.find((f) => Math.abs(f - fps) / f < 0.01);
  return snapped ?? Math.round(fps * 1000) / 1000;
}

/**
 * Seeks to `time` and waits for the corresponding frame to be presented.
 *
 * @returns {Promise<{mediaTime: number|null}>} media time of the presented frame (null if rVFC is unavailable or
 *   did not fire, e.g. in a background tab)
 */
export async function seekTo(video, time, signal) {
  const presented = "requestVideoFrameCallback" in video
    ? new Promise((resolve) => video.requestVideoFrameCallback((_now, meta) => resolve(meta.mediaTime)))
    : null;
  const seeked = once(video, "seeked", signal);
  video.currentTime = time;
  await seeked;
  // The timeout starts once the seek is done, so slow seeks on big files are not mistaken for a missing frame
  const mediaTime = presented ? await Promise.race([presented, timeout(1000, null)]) : null;
  return { mediaTime };
}

/**
 * Reads the currently presented frame, downscaled to the analysis canvas.
 *
 * @param {HTMLVideoElement} video
 * @param {CanvasRenderingContext2D|OffscreenCanvasRenderingContext2D} ctx analysis canvas context
 * @returns {Uint8ClampedArray} RGBA pixels
 */
export function readFrame(video, ctx) {
  const { width, height } = ctx.canvas;
  ctx.drawImage(video, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height).data;
}

/** Analysis size: at most `maxWidth` pixels wide (0 = native size), keeping the video aspect ratio. */
export function analysisSize(videoWidth, videoHeight, maxWidth) {
  if (!maxWidth || maxWidth >= videoWidth) return { width: videoWidth, height: videoHeight };
  return { width: maxWidth, height: Math.max(1, Math.round((videoHeight * maxWidth) / videoWidth)) };
}
