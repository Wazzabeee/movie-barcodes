"""
Several movies in one run: input expansion (directories, list files), per-movie errors and summary,
--skip-existing, several barcode types from one decoding, the colors cache (--save-colors / -i colors.npz) and the
iter_barcodes Python API.

The videos are tiny synthetic clips (cv2.VideoWriter), processed with one worker, so these tests stay fast.
"""

import logging
import os
import shutil
import sys
from typing import List, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from movie_barcodes import api, cli, utility

WIDTH, HEIGHT, FRAMES = 64, 36, 40


def write_video(path: str, color: Tuple[int, int, int], frames: int = FRAMES) -> str:
    """
    Write a small video whose frames go from black to color, with a white stripe (so methods differ).
    """
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (WIDTH, HEIGHT))
    assert writer.isOpened(), "mp4v video writer is not available"
    for index in range(frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        frame[:] = [int(c * index / (frames - 1)) for c in color]
        frame[: HEIGHT // 4, (index % WIDTH) :] = 255
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def library(tmp_path_factory) -> str:
    """
    movies/red.mp4, movies/green.mkv, movies/notes.txt, movies/sub/blue.avi and a broken movies/broken.mp4.
    """
    root = tmp_path_factory.mktemp("library")
    movies = root / "movies"
    (movies / "sub").mkdir(parents=True)
    write_video(str(movies / "red.mp4"), (40, 40, 200))
    write_video(str(movies / "green.mp4"), (40, 200, 40))
    os.replace(movies / "green.mp4", movies / "green.mkv")
    write_video(str(movies / "sub" / "blue.mp4"), (200, 40, 40))
    os.replace(movies / "sub" / "blue.mp4", movies / "sub" / "blue.avi")
    (movies / "notes.txt").write_text("not a video")
    (movies / "broken.mp4").write_bytes(b"not a video either")
    return str(root)


@pytest.fixture()
def workdir(tmp_path, library: str) -> str:
    """
    An empty working directory with a 'movies' symlink-free copy of the library; the CLI runs from it.
    """
    shutil.copytree(os.path.join(library, "movies"), tmp_path / "movies")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield str(tmp_path)
    os.chdir(cwd)


def run_cli(*args: str) -> int:
    """
    Run the CLI with the given arguments (one worker by default) and return its exit code.
    """
    argv = ["movie-barcodes", *args]
    if "-w" not in args:
        argv += ["-w", "1"]
    with patch.object(sys, "argv", argv):
        try:
            cli.main()
        except SystemExit as exit_:
            return int(exit_.code or 0)
    return 0


def load(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def outputs(directory: str = "barcodes") -> List[str]:
    return sorted(os.listdir(directory)) if os.path.isdir(directory) else []


# Input expansion


def test_find_videos_expands_directories(library: str) -> None:
    movies = os.path.join(library, "movies")
    found = utility.find_videos([movies])
    assert [os.path.relpath(path, movies) for path in found] == ["broken.mp4", "green.mkv", "red.mp4"]
    recursive = utility.find_videos([movies], recursive=True)
    assert [os.path.relpath(path, movies) for path in recursive] == [
        "broken.mp4",
        "green.mkv",
        "red.mp4",
        os.path.join("sub", "blue.avi"),
    ]


def test_find_videos_keeps_files_and_removes_duplicates(library: str) -> None:
    red = os.path.join(library, "movies", "red.mp4")
    missing = os.path.join(library, "missing.mp4")
    found = utility.find_videos([red, missing, os.path.join(library, "movies"), red])
    assert found[0] == red and found[1] == missing
    assert found.count(red) == 1 and len(found) == 4


def test_read_path_list(tmp_path) -> None:
    list_file = tmp_path / "list.txt"
    list_file.write_text("# my movies\n\nmovies/a.mp4\n  /abs/b.mkv  \n# movies/c.mp4\n")
    assert utility.read_path_list(str(list_file)) == [str(tmp_path / "movies" / "a.mp4"), "/abs/b.mkv"]


def test_base_names_disambiguate_same_file_names() -> None:
    names = cli.base_names(["s1/E01.mkv", "s2/E01.mkv", "x/movie.mp4", "out/movie_colors.npz", "s1/E01.mp4"])
    assert len(set(names.values())) == 5
    assert names["s2/E01.mkv"] == "s2_E01"
    assert cli.base_names(["a/movie_colors.npz"]) == {"a/movie_colors.npz": "movie"}


def test_get_output_path_type_suffix_and_output_dir() -> None:
    assert utility.get_output_path("m", "avg", "circular", output_name="x", type_suffix=True) == os.path.join(
        "barcodes", "x_circular.png"
    )
    assert (
        utility.get_output_path("m", "avg", "circular", destination_path="o.png", method_suffix=True, type_suffix=True)
        == "o_avg_circular.png"
    )
    assert utility.get_output_path("m", "avg", "horizontal", output_dir="out") == os.path.join(
        "out", "m_avg_horizontal.png"
    )


def test_more_video_extensions_are_accepted(tmp_path) -> None:
    for extension in (".mov", ".avi", ".m4v"):
        video_path = tmp_path / f"video{extension}"
        video_path.write_bytes(b"")
        utility.validate_inputs(str(video_path), 10, 2)


# Command line


def test_directory_and_broken_file_continue_with_summary(workdir: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        code = run_cli("-i", "movies", "--recursive", "--width", "10", "--height", "4")
    assert code == 1  # broken.mp4 failed
    assert outputs() == [
        "blue_avg_horizontal_workers_1.png",
        "green_avg_horizontal_workers_1.png",
        "red_avg_horizontal_workers_1.png",
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any("SUMMARY" in message for message in messages)
    assert any(message.startswith("failed") and "broken.mp4" in message for message in messages)
    assert any("3 ok, 1 failed, 0 skipped" in message for message in messages)


def test_from_list_and_output_dir(workdir: str) -> None:
    with open("list.txt", "w", encoding="utf-8") as file:
        file.write("# two movies\nmovies/red.mp4\n\nmovies/sub/blue.avi\n")
    code = run_cli("--from-list", "list.txt", "--output-dir", "out", "--width", "10", "--height", "4")
    assert code == 0
    assert outputs("out") == ["blue_avg_horizontal_workers_1.png", "red_avg_horizontal_workers_1.png"]
    assert not os.path.exists("barcodes")


def test_repeated_input_option_extends(workdir: str) -> None:
    assert run_cli("-i", "movies/red.mp4", "-i", "movies/green.mkv", "--width", "8", "--height", "4") == 0
    assert outputs() == ["green_avg_horizontal_workers_1.png", "red_avg_horizontal_workers_1.png"]


def test_several_inputs_with_output_name_is_an_error(workdir: str) -> None:
    with pytest.raises(SystemExit) as exit_:
        with patch.object(sys, "argv", ["movie-barcodes", "-i", "movies/red.mp4", "movies/green.mkv", "-n", "x"]):
            cli.main()
    assert exit_.value.code == 2


def test_skip_existing(workdir: str) -> None:
    args = ["-i", "movies/red.mp4", "movies/green.mkv", "--width", "10", "--height", "4", "--save-colors"]
    assert run_cli(*args) == 0
    os.remove(os.path.join("barcodes", "green_colors.npz"))  # green is incomplete: processed again
    with patch.object(api, "extract_colors", wraps=api.extract_colors) as extract_spy:
        assert run_cli(*args, "--skip-existing") == 0
    assert extract_spy.call_count == 1
    assert os.path.exists(os.path.join("barcodes", "green_colors.npz"))


def test_several_types_decode_once_and_match_separate_runs(workdir: str) -> None:
    common = ["-i", "movies/red.mp4", "-a", "--width", "12", "--height", "5"]
    with patch.object(api, "extract_colors", wraps=api.extract_colors) as extract_spy:
        assert run_cli(*common, "-t", "horizontal", "circular", "--output-dir", "both") == 0
    assert extract_spy.call_count == 1
    assert run_cli(*common, "-t", "horizontal", "--output-dir", "horizontal") == 0
    assert run_cli(*common, "-t", "circular", "--output-dir", "circular") == 0

    both = outputs("both")
    assert both == sorted(outputs("horizontal") + outputs("circular"))
    assert len(both) == len(api.METHODS) * 2
    for name in outputs("horizontal"):
        np.testing.assert_array_equal(load(os.path.join("both", name)), load(os.path.join("horizontal", name)))
    for name in outputs("circular"):
        np.testing.assert_array_equal(load(os.path.join("both", name)), load(os.path.join("circular", name)))


def test_several_types_decode_once_with_workers(workdir: str) -> None:
    with patch.object(api, "parallel_extract_colors", wraps=api.parallel_extract_colors) as parallel_spy:
        code = run_cli(
            "-i", "movies/red.mp4", "-w", "2", "-t", "horizontal", "circular", "--width", "8", "--save-colors"
        )
    assert code == 0 and parallel_spy.call_count == 1
    assert outputs() == ["red_avg_circular_workers_2.png", "red_avg_horizontal_workers_2.png", "red_colors.npz"]
    np.testing.assert_array_equal(
        api.load_colors(os.path.join("barcodes", "red_colors.npz")).frame_indices, [0, 5, 10, 15, 20, 25, 30, 35]
    )


def test_several_types_with_output_name(workdir: str) -> None:
    assert run_cli("-i", "movies/red.mp4", "-t", "horizontal", "circular", "-n", "x", "--width", "8") == 0
    assert outputs() == ["x_circular.png", "x_horizontal.png"]


def test_save_colors_and_render_are_pixel_identical(workdir: str) -> None:
    args = ["-i", "movies/red.mp4", "-a", "-t", "horizontal", "circular", "--width", "12", "--height", "5"]
    assert run_cli(*args, "--save-colors", "--crop-borders", "--output-dir", "decoded") == 0
    colors_file = os.path.join("decoded", "red_colors.npz")
    assert os.path.exists(colors_file)

    with (
        patch.object(api, "extract_colors") as extract_spy,
        patch.object(api, "parallel_extract_colors") as parallel_spy,
    ):
        assert run_cli("-i", colors_file, "-t", "horizontal", "circular", "--height", "5", "--output-dir", "r") == 0
    extract_spy.assert_not_called()
    parallel_spy.assert_not_called()

    # Rendered files have no workers suffix (nothing was decoded)
    decoded = outputs("decoded")
    decoded.remove("red_colors.npz")
    assert outputs("r") == sorted(name.replace("_workers_1", "") for name in decoded)
    for name in decoded:
        np.testing.assert_array_equal(
            load(os.path.join("r", name.replace("_workers_1", ""))), load(os.path.join("decoded", name))
        )


def test_render_one_method_from_colors_file(workdir: str) -> None:
    assert run_cli("-i", "movies/red.mp4", "-m", "kmeans", "--width", "8", "--save-colors", "-n", "k") == 0
    assert run_cli("-i", os.path.join("barcodes", "k_colors.npz"), "-t", "circular", "--output-dir", "r") == 0
    assert outputs("r") == ["k_kmeans_circular.png"]
    # A method that was not extracted is an error for that input
    with pytest.raises(ValueError, match="avg"):
        run_cli("-i", os.path.join("barcodes", "k_colors.npz"), "-m", "avg")


# Python API


def test_save_and_load_colors_round_trip(tmp_path) -> None:
    video = write_video(str(tmp_path / "clip.mp4"), (10, 120, 230))
    colors = api.extract_movie_colors(video, ["avg", "smoothed", "hsv"], width=10, workers=1, linear=True)
    colors_file = str(tmp_path / "colors" / "clip_colors.npz")
    api.save_colors(colors, colors_file)
    loaded = api.load_colors(colors_file)

    assert loaded.methods == ["avg", "smoothed", "hsv"]
    for method in loaded.methods:
        assert loaded.colors[method].dtype == colors.colors[method].dtype
        np.testing.assert_array_equal(loaded.colors[method], colors.colors[method])
    assert loaded.colors["smoothed"].shape == (10, HEIGHT, 1, 3)
    np.testing.assert_array_equal(loaded.frame_indices, np.arange(10) * FRAMES // 10)
    assert (loaded.frame_count, loaded.frame_width, loaded.frame_height, loaded.target_frames) == (
        FRAMES,
        WIDTH,
        HEIGHT,
        10,
    )
    assert loaded.linear and loaded.crop_rect is None and loaded.video_name == "clip.mp4"
    assert loaded.fps == pytest.approx(24)
    for barcode_type in api.BARCODE_TYPES:
        np.testing.assert_array_equal(
            np.asarray(api.render_barcode(loaded, "smoothed", barcode_type, height=7)),
            np.asarray(api.render_barcode(colors, "smoothed", barcode_type, height=7)),
        )


def test_frame_indices_with_several_workers() -> None:
    np.testing.assert_array_equal(api._requested_frame_indices(10, 4, 2), [0, 2, 5, 7])
    np.testing.assert_array_equal(api._requested_frame_indices(10, 4, 1), [0, 2, 5, 7])


def test_load_colors_rejects_other_files(tmp_path) -> None:
    other = str(tmp_path / "other.npz")
    np.savez(other, x=np.zeros(3))
    with pytest.raises(ValueError, match="not a movie-barcodes colors file"):
        api.load_colors(other)
    with pytest.raises(FileNotFoundError):
        api.load_colors(str(tmp_path / "missing.npz"))


def test_iter_barcodes_yields_results_and_errors(library: str) -> None:
    movies = os.path.join(library, "movies")
    paths = [os.path.join(movies, "red.mp4"), os.path.join(movies, "broken.mp4"), os.path.join(movies, "green.mkv")]
    results = list(
        api.iter_barcodes(
            paths,
            methods=["avg", "hsv"],
            barcode_types=["horizontal", "circular"],
            width=6,
            height=3,
            workers=1,
            skip=lambda path: path.endswith("green.mkv"),
        )
    )
    assert [result.status for result in results] == ["ok", "failed", "skipped"]
    ok, failed, skipped = results
    assert sorted(ok.barcodes) == [
        ("avg", "circular"),
        ("avg", "horizontal"),
        ("hsv", "circular"),
        ("hsv", "horizontal"),
    ]
    assert ok.barcodes[("avg", "horizontal")].size == (6, 3)
    assert ok.barcodes[("avg", "circular")].size == (WIDTH, WIDTH)
    assert ok.colors is not None and ok.duration > 0
    assert isinstance(failed.error, ValueError) and not failed.ok
    assert skipped.barcodes == {} and skipped.error is None


def test_iter_barcodes_matches_create_barcodes(library: str) -> None:
    red = os.path.join(library, "movies", "red.mp4")
    expected = api.create_barcodes(red, ["avg", "smoothed"], barcode_type="circular", width=6, workers=1)
    (result,) = api.iter_barcodes([red], methods=["avg", "smoothed"], barcode_types=["circular"], width=6, workers=1)
    for method, image in expected.items():
        np.testing.assert_array_equal(np.asarray(result.barcodes[(method, "circular")]), np.asarray(image))


def test_iter_barcodes_rejects_invalid_arguments_upfront() -> None:
    with pytest.raises(ValueError):
        next(api.iter_barcodes(["a.mp4"], methods=["nope"]))
    with pytest.raises(ValueError):
        next(api.iter_barcodes(["a.mp4"], barcode_types=["square"]))
