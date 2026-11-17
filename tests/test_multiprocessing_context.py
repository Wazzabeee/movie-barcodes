import os
import subprocess
import sys
import textwrap
import unittest

from movie_barcodes import video_processing

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Reproduces the deadlock seen on Linux CI: the parent runs K-means (OpenMP thread pool) and then runs
# K-means again in parallel workers. With "fork" workers, they inherit the parent's OpenMP pool in an
# inconsistent state and hang forever. The script forces "fork" as the global default start method, so the
# test also catches the bug on macOS, where the default is "spawn".
SCRIPT = textwrap.dedent(
    """
    import multiprocessing
    import sys

    sys.path.insert(0, {src!r})

    if __name__ == "__main__":
        if "fork" in multiprocessing.get_all_start_methods():
            multiprocessing.set_start_method("fork", force=True)

        import numpy as np
        from movie_barcodes.color_extraction import get_dominant_color_kmeans
        from movie_barcodes.video_processing import parallel_extract_colors

        # Initialize the OpenMP thread pool in the parent process
        get_dominant_color_kmeans(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8))

        colors = parallel_extract_colors({video!r}, 93, get_dominant_color_kmeans, 2, 6)
        print("OK", len(colors))
    """
)


class TestMultiprocessingContext(unittest.TestCase):
    def test_workers_use_spawn(self) -> None:
        """
        Worker processes must not be forked from the parent (see MP_CONTEXT in video_processing).
        """
        self.assertEqual(video_processing.MP_CONTEXT.get_start_method(), "spawn")

    def test_parallel_kmeans_after_kmeans_in_parent_does_not_hang(self) -> None:
        """
        K-means in the parent, then K-means in parallel workers, completes (it deadlocked with fork workers).
        """
        script = SCRIPT.format(
            src=os.path.join(REPO_ROOT, "src"),
            video=os.path.join(REPO_ROOT, "tests", "sample.mp4"),
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, timeout=180, cwd=REPO_ROOT
            )
        except subprocess.TimeoutExpired:
            self.fail("parallel K-means extraction hung after running K-means in the parent process")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("OK 6", result.stdout)


if __name__ == "__main__":
    unittest.main()
