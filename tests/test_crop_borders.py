import functools
import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from movie_barcodes import cli as main
from movie_barcodes import video_processing
from movie_barcodes.color_extraction import get_dominant_color_mean
from movie_barcodes.utility import get_dominant_color_function

WIDTH, HEIGHT = 320, 240
CONTENT_VALUE = 200


def make_frame(top: int = 0, bottom: int = 0, left: int = 0, right: int = 0, value: int = CONTENT_VALUE) -> np.ndarray:
    """
    Build a BGR frame filled with `value`, with black bars of the given sizes on each side.
    """
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[top : HEIGHT - bottom, left : WIDTH - right] = value
    return frame


class TestFindContentArea(unittest.TestCase):
    """
    Detection of the content area on synthetic frames.
    """

    def test_letterbox(self) -> None:
        frames = [make_frame(top=40, bottom=40)] * 5
        self.assertEqual(video_processing.find_content_area(frames), (0, 40, WIDTH, HEIGHT - 80))

    def test_pillarbox(self) -> None:
        frames = [make_frame(left=48, right=48)] * 5
        self.assertEqual(video_processing.find_content_area(frames), (48, 0, WIDTH - 96, HEIGHT))

    def test_no_bars_returns_full_frame(self) -> None:
        frames = [make_frame()] * 5
        self.assertEqual(video_processing.find_content_area(frames), (0, 0, WIDTH, HEIGHT))

    def test_all_black_returns_none(self) -> None:
        frames = [make_frame(value=0)] * 5
        self.assertIsNone(video_processing.find_content_area(frames))

    def test_no_frames_returns_none(self) -> None:
        self.assertIsNone(video_processing.find_content_area([]))

    def test_dark_scene_does_not_over_crop(self) -> None:
        """
        Rows that are dark in one frame but bright in another are kept: frames are combined with a max.
        """
        dark_scene = make_frame(top=100, bottom=40)
        bright_scene = make_frame(top=40, bottom=40)
        black_frame = make_frame(value=0)
        frames = [dark_scene, black_frame, bright_scene]
        self.assertEqual(video_processing.find_content_area(frames), (0, 40, WIDTH, HEIGHT - 80))

    def test_compression_noise_in_bars_is_ignored(self) -> None:
        frame = make_frame(top=40, bottom=40)
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 12, size=(40, WIDTH, 3), dtype=np.uint8)
        frame[:40] = noise
        frame[0, 0] = 255  # Isolated artifact
        self.assertEqual(video_processing.find_content_area([frame]), (0, 40, WIDTH, HEIGHT - 80))

    def test_tiny_content_returns_none(self) -> None:
        frames = [make_frame(top=110, bottom=110)]
        self.assertIsNone(video_processing.find_content_area(frames))


class TestDetectContentArea(unittest.TestCase):
    """
    Detection on a small video written with cv2.VideoWriter.
    """

    def _write_video(self, directory: str, frame: np.ndarray, frame_count: int = 40) -> str:
        video_path = os.path.join(directory, "video.avi")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (WIDTH, HEIGHT))
        if not writer.isOpened():
            self.skipTest("MJPG video writer is not available")
        for _ in range(frame_count):
            writer.write(frame)
        writer.release()
        return video_path

    def test_letterboxed_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = self._write_video(directory, make_frame(top=48, bottom=48))
            x, y, w, h = video_processing.detect_content_area(video_path, 40)
        self.assertEqual((x, w), (0, WIDTH))
        self.assertAlmostEqual(y, 48, delta=3)
        self.assertAlmostEqual(h, HEIGHT - 96, delta=6)

    def test_all_black_video_falls_back_to_full_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = self._write_video(directory, make_frame(value=0))
            self.assertEqual(video_processing.detect_content_area(video_path, 40), (0, 0, WIDTH, HEIGHT))


class TestCropAndExtract(unittest.TestCase):
    """
    Colors extracted from cropped frames.
    """

    def test_avg_is_not_darkened_by_bars(self) -> None:
        frame = make_frame(top=40, bottom=40)
        rect = video_processing.find_content_area([frame])

        uncropped = get_dominant_color_mean(frame)
        cropped = video_processing.crop_and_extract(frame, get_dominant_color_mean, rect)

        self.assertLess(uncropped[0], CONTENT_VALUE - 50)
        np.testing.assert_allclose(cropped, [CONTENT_VALUE] * 3)

    def test_all_methods_accept_cropped_frames(self) -> None:
        frame = make_frame(left=48, right=48)  # Side crop: non-contiguous view
        rect = video_processing.find_content_area([frame])
        for method in ["avg", "hsv", "bgr", "kmeans", "smoothed"]:
            with self.subTest(method=method):
                color = video_processing.crop_and_extract(frame, get_dominant_color_function(method), rect)
                self.assertGreater(float(np.mean(color)), CONTENT_VALUE - 10)

    def test_cropping_extractor_is_picklable(self) -> None:
        extractor = main.get_color_function("avg", (0, 40, WIDTH, HEIGHT - 80))
        self.assertIsInstance(extractor, functools.partial)
        restored = pickle.loads(pickle.dumps(extractor))
        np.testing.assert_allclose(restored(make_frame(top=40, bottom=40)), [CONTENT_VALUE] * 3)

    def test_no_crop_returns_plain_extractor(self) -> None:
        self.assertIs(main.get_color_function("avg"), get_dominant_color_mean)


class TestCropBordersIntegration(unittest.TestCase):
    """
    End-to-end run with --crop-borders, sequential and parallel.
    """

    def _run(self, workers: int) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = [
                "movie-barcodes",
                "--input_video_path",
                "tests/sample.mp4",
                "--workers",
                str(workers),
                "--width",
                "20",
                "--crop-borders",
                "--destination_path",
                os.path.join(directory, "barcode.png"),
            ]
            with patch.object(sys, "argv", args):
                main.main()
            self.assertTrue(os.path.exists(os.path.join(directory, "barcode.png")))

    def test_sequential(self) -> None:
        self._run(1)

    def test_parallel(self) -> None:
        self._run(2)


if __name__ == "__main__":
    unittest.main()
