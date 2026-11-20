"""
Generates web/tests/fixture.json: reference colors computed by the movie_barcodes Python package on small synthetic
frames, compared with the JavaScript color math by color.test.js.

Run from the repository root:  UV_NO_CONFIG=1 uv run python web/tests/generate_fixture.py
"""

import json
from pathlib import Path

import numpy as np

from movie_barcodes.utility import get_dominant_color_function
from movie_barcodes.video_processing import extract_colors, load_video
from movie_barcodes.color_extraction import (
    SRGB_TO_LINEAR_LUT,
    get_dominant_color_mean,
    get_dominant_color_mean_linear,
    get_smoothed_frame,
)


def frames() -> dict:
    rng = np.random.default_rng(42)
    half = np.zeros((6, 8, 3), dtype=np.uint8)
    half[:, 4:] = 255
    gradient = np.repeat(np.linspace(0, 255, 16, dtype=np.uint8)[None, :, None], 9, axis=0).repeat(3, axis=2)
    return {
        "random": rng.integers(0, 256, (9, 16, 3), dtype=np.uint8),
        "random_dark": rng.integers(0, 40, (12, 20, 3), dtype=np.uint8),
        "half_black_white": half,
        "single_color": np.full((5, 7, 3), (12, 200, 99), dtype=np.uint8),
        "gradient": gradient,
    }


def sample_video() -> dict:
    """Colors of every frame of tests/sample.mp4, for the browser smoke test (browser-smoke.js)."""
    path = str(Path(__file__).parents[2] / "tests" / "sample.mp4")
    video, frame_count, _, _ = load_video(path)
    video.release()
    functions = {
        "avg": get_dominant_color_function("avg"),
        "smoothed": get_dominant_color_function("smoothed"),
        "linear": get_dominant_color_function("avg", linear=True),
    }
    colors = extract_colors(path, 0, frame_count - 1, functions, frame_count)

    def rgb(color) -> list:
        color = np.asarray(color, dtype=np.float64)
        if color.ndim > 1:  # smoothed frame column
            color = color[0, 0]
        return [round(float(v), 4) for v in color[::-1]]

    return {"frame_count": frame_count, "colors": {m: [rgb(c) for c in colors[m]] for m in functions}}


def main() -> None:
    cases = []
    for name, rgb in frames().items():
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)], axis=2)
        cases.append(
            {
                "name": name,
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "rgba": rgba.ravel().tolist(),
                # Python works in BGR: reverse to RGB
                "avg": [float(v) for v in get_dominant_color_mean(bgr)[::-1]],
                "smoothed": [int(v) for v in get_smoothed_frame(bgr)[0, 0, ::-1]],
                "linear": [int(v) for v in get_dominant_color_mean_linear(bgr)[::-1]],
            }
        )
    # Frame indices sampled by iter_sampled_frames over a whole video: i * frame_count // n_samples
    sampling = [
        {"frame_count": count, "samples": n, "indices": [i * count // n for i in range(n)]}
        for count, n in [(93, 93), (93, 10), (1000, 7), (25, 40)]
    ]
    fixture = {"lut": SRGB_TO_LINEAR_LUT.tolist(), "cases": cases, "sampling": sampling, "sample_mp4": sample_video()}
    out = Path(__file__).with_name("fixture.json")
    out.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
