"""
End-to-end regression tests: run the real CLI on a synthetic video whose expected colors are known.

The video has letterbox bars, three segments of distinct colors (blue, green, red thirds) and a black fade in
the middle, so the checks cover sampling order, sequential/parallel equivalence, border cropping, the handling
of dark frames, the single-pass --all_methods mode and the circular output.
"""

import logging
import os
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from movie_barcodes import cli

WIDTH, HEIGHT, FRAMES, BAR = 320, 180, 300, 30
# BGR colors of the three thirds of the video: blue, green, red
SEGMENT_COLORS = [(200, 40, 40), (40, 200, 40), (40, 40, 200)]
FADE = range(140, 160)  # Fully black frames (bars included)
VIDEO_NAME = "movie"


def _write_video(path: str) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (WIDTH, HEIGHT))
    assert writer.isOpened(), "mp4v video writer is not available"
    for index in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        if index not in FADE:
            frame[BAR : HEIGHT - BAR] = SEGMENT_COLORS[min(index * 3 // FRAMES, 2)]
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> str:
    """
    Directory holding the synthetic video; the CLI runs from it and saves its barcodes in ./barcodes.
    """
    directory = tmp_path_factory.mktemp("e2e")
    _write_video(str(directory / f"{VIDEO_NAME}.mp4"))
    return str(directory)


def run_cli(workdir: str, *args: str, max_processes: int = 4) -> None:
    """
    Run the CLI from workdir. MAX_PROCESSES is fixed so that the worker counts used here don't depend on the
    number of cores of the machine running the tests.
    """
    argv = ["movie-barcodes", "-i", f"{VIDEO_NAME}.mp4", *args]
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        with patch.object(sys, "argv", argv), patch.object(cli, "MAX_PROCESSES", max_processes):
            cli.main()
    finally:
        os.chdir(cwd)


def load(workdir: str, name: str, mode: str = "RGB") -> np.ndarray:
    with Image.open(os.path.join(workdir, "barcodes", f"{name}.png")) as image:
        return np.asarray(image.convert(mode)).astype(int)


@pytest.fixture(scope="module")
def avg_sequential(workdir: str) -> np.ndarray:
    run_cli(workdir, "-m", "avg", "-w", "1", "-n", "avg_seq")
    return load(workdir, "avg_seq")


@pytest.fixture(scope="module")
def avg_cropped(workdir: str) -> np.ndarray:
    run_cli(workdir, "-m", "avg", "-w", "2", "--crop-borders", "-n", "avg_crop")
    return load(workdir, "avg_crop")


@pytest.fixture(scope="module")
def bgr_cropped(workdir: str) -> np.ndarray:
    run_cli(workdir, "-m", "bgr", "-w", "2", "--crop-borders", "-n", "bgr_crop")
    return load(workdir, "bgr_crop")


def test_default_width_is_min_of_video_width_and_frame_count(avg_sequential: np.ndarray) -> None:
    assert avg_sequential.shape == (HEIGHT, min(WIDTH, FRAMES), 3)


def test_sequential_and_parallel_are_identical(workdir: str, avg_sequential: np.ndarray) -> None:
    run_cli(workdir, "-m", "avg", "-w", "2", "-n", "avg_par")
    np.testing.assert_array_equal(load(workdir, "avg_par"), avg_sequential)


def test_segment_order_is_preserved(avg_sequential: np.ndarray) -> None:
    first, middle, last = avg_sequential[0, 10], avg_sequential[0, 120], avg_sequential[0, -10]
    assert first[2] > first[0] and first[2] > first[1]  # Blue first
    assert middle[1] > middle[0] and middle[1] > middle[2]  # Then green
    assert last[0] > last[1] and last[0] > last[2]  # Red last


def test_crop_borders_restores_content_color(avg_sequential: np.ndarray, avg_cropped: np.ndarray) -> None:
    # Without cropping, the bars darken the average; with it, the average is the content color (RGB)
    assert avg_cropped[0, 10].sum() > avg_sequential[0, 10].sum() + 50
    np.testing.assert_allclose(avg_cropped[0, 10], [40, 40, 200], atol=8)
    np.testing.assert_allclose(avg_cropped[0, -10], [200, 40, 40], atol=8)


def test_hsv_on_black_frames_is_dark(workdir: str) -> None:
    run_cli(workdir, "-m", "hsv", "-w", "2", "-n", "hsv")
    hsv = load(workdir, "hsv")
    column = FADE.start + len(FADE) // 2  # One column per frame: this sample is inside the black fade
    assert hsv[0, column].max() < 40


def test_bgr_with_crop_returns_content_color(bgr_cropped: np.ndarray) -> None:
    np.testing.assert_allclose(bgr_cropped[0, 10], [40, 40, 200], atol=8)


def test_all_methods_match_single_method_runs(workdir: str, avg_cropped: np.ndarray, bgr_cropped: np.ndarray) -> None:
    run_cli(workdir, "-a", "-w", "2", "--crop-borders")
    methods = ["avg", "bgr", "hsv", "kmeans", "smoothed"]
    for method in methods:
        assert os.path.exists(os.path.join(workdir, "barcodes", f"{VIDEO_NAME}_{method}_horizontal_workers_2.png"))
    np.testing.assert_array_equal(load(workdir, f"{VIDEO_NAME}_avg_horizontal_workers_2"), avg_cropped)
    np.testing.assert_array_equal(load(workdir, f"{VIDEO_NAME}_bgr_horizontal_workers_2"), bgr_cropped)


def test_too_many_workers_are_capped_with_warning(workdir: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        run_cli(workdir, "-m", "avg", "-w", "999", "-n", "capped", max_processes=2)
    assert any("999" in record.getMessage() and "using 2" in record.getMessage() for record in caplog.records)
    assert os.path.exists(os.path.join(workdir, "barcodes", "capped.png"))


def test_circular_output(workdir: str) -> None:
    run_cli(workdir, "-m", "avg", "-t", "circular", "-w", "2", "-n", "circular")
    circular = load(workdir, "circular", mode="RGBA")
    assert circular.shape == (WIDTH, WIDTH, 4)
    assert circular[WIDTH // 2, WIDTH // 2, 3] == 255  # Opaque center
    assert circular[0, 0, 3] == 0  # Transparent corner
