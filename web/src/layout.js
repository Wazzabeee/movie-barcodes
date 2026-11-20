// Sampling positions and barcode drawing. Drawing works on any CanvasRenderingContext2D-like object
// (HTMLCanvasElement or OffscreenCanvas); the index math has no DOM dependency and is unit tested in Node.

import { toByteColor } from "./color.js";

/** Default number of samples: min(video width, frame count), like get_target_frames in the Python API. */
export function defaultSampleCount(videoWidth, frameCount) {
  return Math.max(1, Math.min(videoWidth, frameCount));
}

/**
 * Indices of the frames to sample, spread evenly over the whole video (the end is sampled too), like
 * iter_sampled_frames in Python: i * frameCount // n.
 */
export function sampleFrameIndices(frameCount, n) {
  const indices = new Array(n);
  for (let i = 0; i < n; i++) indices[i] = Math.floor((i * frameCount) / n);
  return indices;
}

/**
 * Time (seconds) to seek to for a frame index: the middle of the frame, so rounding never lands on the previous
 * frame. Clamped inside the video.
 */
export function frameTime(frameIndex, fps, duration) {
  const t = (frameIndex + 0.5) / fps;
  return Math.max(0, Math.min(t, duration - 0.5 / fps));
}

/** Index of the color drawn in column `column` of a `width` pixels wide barcode made from `colorCount` colors. */
export function columnColorIndex(column, width, colorCount) {
  return Math.floor((column * colorCount) / width);
}

/** Outer radius of ring `idx` (0 = center disc) of a circular barcode of `size` pixels with `count` rings. */
export function ringRadius(idx, count, size) {
  return ((idx + 1) * size) / 2 / count;
}

/**
 * Draws a horizontal barcode: one column per color, stretched to the canvas height.
 *
 * @param {CanvasRenderingContext2D|OffscreenCanvasRenderingContext2D} ctx
 * @param {number[][]} colors RGB colors (may be shorter than the canvas width while the barcode fills in)
 * @param {number} total Final number of colors: columns not computed yet stay transparent
 */
export function drawHorizontal(ctx, colors, total = colors.length) {
  const { width, height } = ctx.canvas;
  if (!colors.length) return ctx.clearRect(0, 0, width, height);
  const row = ctx.createImageData(width, 1);
  for (let x = 0; x < width; x++) {
    const idx = columnColorIndex(x, width, total);
    if (idx >= colors.length) break;
    const [r, g, b] = toByteColor(colors[idx]);
    row.data.set([r, g, b, 255], x * 4);
  }
  // Repeat the row instead of scaling it: exact colors, no interpolation
  for (let y = 0; y < height; y++) ctx.putImageData(row, 0, y);
}

/**
 * Draws a circular barcode: concentric filled discs, first color at the center, last one on the outer ring,
 * transparent background (generate_circular_barcode in Python).
 *
 * @param {CanvasRenderingContext2D|OffscreenCanvasRenderingContext2D} ctx square canvas
 * @param {number[][]} colors RGB colors (may be shorter than `total` while the barcode fills in)
 * @param {number} total Final number of rings, which sets the ring width
 */
export function drawCircular(ctx, colors, total = colors.length) {
  const size = ctx.canvas.width;
  ctx.clearRect(0, 0, size, size);
  // Outside in: each disc covers the inner part of the previous one, so rings never leave gaps
  for (let idx = colors.length - 1; idx >= 0; idx--) {
    const [r, g, b] = toByteColor(colors[idx]);
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, ringRadius(idx, total, size), 0, 2 * Math.PI);
    ctx.fill();
  }
}

/** Output file name, like the CLI: <movie>_<method>_<type>.png. */
export function outputName(fileName, fileMethod, barcodeType) {
  const dot = fileName.lastIndexOf(".");
  const base = dot > 0 ? fileName.slice(0, dot) : fileName;
  return `${base}_${fileMethod}_${barcodeType}.png`;
}
