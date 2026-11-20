// One pass over a movie: samples frames evenly and computes the color of each for every requested method.

import { colorsFromStats, frameStats } from "./color.js";
import { defaultSampleCount, frameTime, sampleFrameIndices } from "./layout.js";
import { analysisSize, closeVideo, detectFps, openVideo, readFrame, seekTo } from "./video.js";

const DEFAULT_FPS = 24;

/**
 * @param {object} job
 * @param {string} job.url object URL (or same-origin URL) of the movie
 * @param {HTMLElement} job.host element the decoding <video> is attached to
 * @param {string[]} job.methods keys of METHODS
 * @param {number} [job.samples] number of samples (0 = min(video width, frame count))
 * @param {number} [job.fps] frame rate override (0 = detect)
 * @param {number} [job.analysisWidth] width frames are downscaled to before averaging (0 = native)
 * @param {AbortSignal} [job.signal]
 * @param {() => Promise<void>} [job.waitUntilVisible] resolves when the tab is visible (decoding stalls otherwise)
 * @param {(info: object) => void} [job.onInfo] called once the video is opened
 * @param {(done: number, colors: object) => void} [job.onProgress] called after every sample
 * @param {(stalled: boolean) => void} [job.onStall] called when frames stop (or start again) being presented, e.g.
 *   when the window is minimized or covered: the visibility API does not report every case
 * @returns {Promise<{info: object, colors: Record<string, number[][]>, missedFrames: number}>}
 */
export async function computeColors(job) {
  const { url, host, methods, signal } = job;
  const video = await openVideo(url, host, signal);
  try {
    const detected = job.fps ? null : await detectFps(video);
    const fps = job.fps || detected || DEFAULT_FPS;
    const frameCount = Math.max(1, Math.round(video.duration * fps));
    const samples = job.samples || defaultSampleCount(video.videoWidth, frameCount);
    const size = analysisSize(video.videoWidth, video.videoHeight, job.analysisWidth);
    const info = {
      width: video.videoWidth,
      height: video.videoHeight,
      duration: video.duration,
      fps,
      fpsSource: job.fps ? "set" : detected ? "detected" : "default",
      frameCount,
      samples,
      analysis: size,
    };
    job.onInfo?.(info);

    const canvas = new OffscreenCanvas(size.width, size.height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";

    const colors = Object.fromEntries(methods.map((m) => [m, []]));
    const indices = sampleFrameIndices(frameCount, samples);
    let missedFrames = 0;
    let consecutiveMisses = 0;
    let previous = null;
    for (let i = 0; i < samples; i++) {
      signal?.throwIfAborted();
      // Asking for more samples than frames: reuse the color instead of seeking to the same frame again
      if (!(previous && indices[i] === indices[i - 1])) {
        await job.waitUntilVisible?.();
        const time = frameTime(indices[i], fps, video.duration);
        let { mediaTime } = await seekTo(video, time, signal);
        // Retry once, unless frames are not presented at all right now (no point waiting twice per sample)
        if (mediaTime === null && consecutiveMisses === 0) ({ mediaTime } = await seekTo(video, time, signal));
        if (mediaTime === null) {
          missedFrames++;
          if (++consecutiveMisses === 2) job.onStall?.(true);
        } else {
          if (consecutiveMisses >= 2) job.onStall?.(false);
          consecutiveMisses = 0;
        }
        previous = colorsFromStats(frameStats(readFrame(video, ctx)));
      }
      for (const m of methods) colors[m].push(previous[m]);
      job.onProgress?.(i + 1, colors);
    }
    return { info, colors, missedFrames };
  } finally {
    closeVideo(video);
  }
}
