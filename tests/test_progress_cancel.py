"""
Progress callback and cancellation of the decoding functions (on_progress / cancel).
"""

import multiprocessing
import os
import threading
from typing import List, Tuple

import numpy as np
import pytest

from movie_barcodes import Cancelled, create_barcodes, extract_movie_colors, iter_barcodes
from movie_barcodes.video_processing import extract_colors, parallel_extract_colors

SAMPLE = os.path.join(os.path.dirname(__file__), "sample.mp4")
WIDTH = 24


def mean_color(frame: np.ndarray) -> np.ndarray:
    return frame.mean(axis=(0, 1))


class Recorder:
    """
    on_progress callback recording its calls, optionally setting a cancel event after a number of calls.
    """

    def __init__(self, cancel_after: int = 0) -> None:
        self.calls: List[Tuple[int, int]] = []
        self.cancel = threading.Event()
        self.cancel_after = cancel_after

    def __call__(self, done: int, total: int) -> None:
        self.calls.append((done, total))
        if self.cancel_after and len(self.calls) >= self.cancel_after:
            self.cancel.set()


def assert_no_worker_left() -> None:
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("workers", [1, 2])
def test_progress_counts_every_sampled_frame(workers: int) -> None:
    recorder = Recorder()
    colors = extract_movie_colors(SAMPLE, ["avg"], width=WIDTH, workers=workers, on_progress=recorder)
    assert len(colors.colors["avg"]) == WIDTH
    done = [call[0] for call in recorder.calls]
    assert {call[1] for call in recorder.calls} == {WIDTH}
    assert done == sorted(done) and done[-1] == WIDTH
    if workers == 1:
        assert done == list(range(1, WIDTH + 1))


def test_progress_of_parallel_extraction_ends_at_total() -> None:
    recorder = Recorder()
    colors = parallel_extract_colors(SAMPLE, 93, mean_color, 2, 10, on_progress=recorder)
    assert len(colors) == 10
    assert recorder.calls[-1] == (10, 10)


def test_cancel_sequential_stops_at_next_frame() -> None:
    recorder = Recorder(cancel_after=3)
    with pytest.raises(Cancelled):
        extract_colors(SAMPLE, 0, 92, mean_color, 20, on_progress=recorder, cancel=recorder.cancel)
    assert recorder.calls == [(1, 20), (2, 20), (3, 20)]


def test_cancel_set_before_does_not_decode() -> None:
    recorder = Recorder()
    recorder.cancel.set()
    for workers in (1, 2):
        with pytest.raises(Cancelled):
            create_barcodes(SAMPLE, ["avg"], width=WIDTH, workers=workers, on_progress=recorder, cancel=recorder.cancel)
    assert recorder.calls == []
    assert_no_worker_left()


def test_cancel_parallel_terminates_workers() -> None:
    recorder = Recorder(cancel_after=1)
    with pytest.raises(Cancelled):
        # Every frame, so that the workers are still busy when the cancel is seen
        parallel_extract_colors(SAMPLE, 93, mean_color, 2, None, on_progress=recorder, cancel=recorder.cancel)
    assert recorder.calls and recorder.calls[-1][0] < 93
    assert_no_worker_left()


def test_cancel_from_another_thread() -> None:
    cancel = threading.Event()
    started = threading.Event()
    outcome = {}

    def wait() -> None:
        started.set()
        cancel.wait(60)

    def job() -> None:
        try:
            # The callback waits for the cancel, so the video can't be finished before it is seen
            create_barcodes(SAMPLE, ["avg"], workers=1, on_progress=lambda done, total: wait(), cancel=cancel)
            outcome["result"] = "finished"
        except Cancelled:
            outcome["result"] = "cancelled"

    thread = threading.Thread(target=job)
    thread.start()
    assert started.wait(60)
    cancel.set()
    thread.join(60)
    assert outcome["result"] == "cancelled"


def test_iter_barcodes_cancel_is_raised_not_reported_as_failed(tmp_path) -> None:
    recorder = Recorder()
    results = iter_barcodes([SAMPLE, SAMPLE], width=WIDTH, workers=1, on_progress=recorder, cancel=recorder.cancel)
    first = next(results)
    assert first.ok
    assert recorder.calls[-1] == (WIDTH, WIDTH)
    recorder.cancel_after = len(recorder.calls) + 2
    with pytest.raises(Cancelled):
        next(results)
    assert len(recorder.calls) == WIDTH + 2


@pytest.mark.parametrize("workers", [1, 2])
def test_hooks_do_not_change_outputs(workers: int) -> None:
    kwargs = {"methods": ["avg", "smoothed"], "width": WIDTH, "workers": workers}
    plain = create_barcodes(SAMPLE, **kwargs)
    hooked = create_barcodes(SAMPLE, **kwargs, on_progress=Recorder(), cancel=threading.Event())
    for method, image in plain.items():
        assert np.array_equal(np.asarray(image), np.asarray(hooked[method]))
