// Color math shared by the app and the Node tests. No DOM access here.
//
// Mirrors src/movie_barcodes/color_extraction.py:
// - "avg": mean of every pixel (cv2.mean), kept as floats until drawing.
// - "smoothed": area resize of the frame to 1x1 pixel, i.e. the same mean rounded to 8 bits.
// - "linear" (the CLI's --linear, files named "avg_linear"): mean in linear light, re-encoded to sRGB.

/** Exact sRGB decoding of a value in [0, 1] to linear light. */
export function srgbToLinear(v) {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

/** Exact sRGB encoding of a linear light value (clipped to [0, 1]). */
export function linearToSrgb(v) {
  v = Math.min(1, Math.max(0, v));
  return v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
}

/** Linear light value of every 8-bit sRGB code value (SRGB_TO_LINEAR_LUT in Python). */
export const SRGB_TO_LINEAR_LUT = Float64Array.from({ length: 256 }, (_, i) => srgbToLinear(i / 255));

/** Methods offered by the web app, with the method name used in output file names (like the CLI). */
export const METHODS = {
  avg: { label: "Average", fileMethod: "avg" },
  smoothed: { label: "Smoothed", fileMethod: "smoothed" },
  linear: { label: "Linear light average", fileMethod: "avg_linear" },
};

/**
 * Per-channel statistics of an RGBA pixel buffer (e.g. ImageData.data), computed in one pass.
 * Alpha is ignored (video frames are opaque).
 *
 * @param {Uint8ClampedArray|Uint8Array} rgba
 * @returns {{count: number, sum: number[], linearSum: number[]}}
 */
export function frameStats(rgba) {
  const lut = SRGB_TO_LINEAR_LUT;
  let r = 0, g = 0, b = 0, lr = 0, lg = 0, lb = 0;
  for (let p = 0; p < rgba.length; p += 4) {
    const pr = rgba[p], pg = rgba[p + 1], pb = rgba[p + 2];
    r += pr; g += pg; b += pb;
    lr += lut[pr]; lg += lut[pg]; lb += lut[pb];
  }
  return { count: rgba.length / 4, sum: [r, g, b], linearSum: [lr, lg, lb] };
}

/**
 * Color of a frame for each method, from its statistics.
 *
 * @param {{count: number, sum: number[], linearSum: number[]}} stats
 * @returns {{avg: number[], smoothed: number[], linear: number[]}} RGB colors: floats for avg, 8-bit integers otherwise
 */
export function colorsFromStats(stats) {
  const avg = stats.sum.map((s) => s / stats.count);
  return {
    avg,
    smoothed: avg.map((v) => Math.round(v)),
    linear: stats.linearSum.map((s) => Math.round(linearToSrgb(s / stats.count) * 255)),
  };
}

/** Rounds and clips a color to 8-bit integers, as the barcode drawing does. */
export function toByteColor(color) {
  return color.map((v) => Math.min(255, Math.max(0, Math.round(v))));
}
