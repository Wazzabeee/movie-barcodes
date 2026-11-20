"""
Desktop app: the queue, options and job runner run without a display; the page itself is tested with NiceGUI's
simulated user when the "ui" extra is installed.
"""

import asyncio
import importlib.util
import multiprocessing
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image

from movie_barcodes.ui import Job, MovieQueue, Options, fix_std_streams, make_preview, open_command

SAMPLE = os.path.join(os.path.dirname(__file__), "sample.mp4")
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
HAS_NICEGUI = importlib.util.find_spec("nicegui") is not None


@pytest.fixture
def movies(tmp_path):
    """
    A folder with two copies of the sample, a broken video, a subfolder with another copy and a non-video file.
    """
    library = tmp_path / "library"
    (library / "sub").mkdir(parents=True)
    shutil.copy(SAMPLE, library / "a.mp4")
    shutil.copy(SAMPLE, library / "c.mkv")
    (library / "b_broken.mp4").write_bytes(b"not a video")
    shutil.copy(SAMPLE, library / "sub" / "d.mp4")
    (library / "notes.txt").write_text("not a movie")
    return library


def options_for(tmp_path, **kwargs) -> Options:
    values = {"width": 12, "workers": 1, "output_dir": str(tmp_path / "out")}
    values.update(kwargs)
    return Options(**values)


def test_options_errors() -> None:
    assert Options(output_dir="out").errors() == []
    problems = Options(methods=[], barcode_types=["square"], width=0, workers=0, output_dir=" ").errors()
    assert "Choose at least one method." in problems
    assert "Unknown barcode type: square." in problems
    assert "Width must be at least 1." in problems and "Workers must be at least 1." in problems
    assert "Choose an output folder." in problems


def test_options_to_args_matches_the_command_line() -> None:
    args = Options(barcode_types=["circular", "horizontal", "circular"], width=50, output_dir="~/bc").to_args()
    assert args.barcode_types == ["circular", "horizontal"]
    assert args.width == 50 and args.output_dir == os.path.expanduser("~/bc")
    assert args.method is None and not args.all_methods and args.destination_path is None


def test_queue_adds_folders_without_duplicates(movies) -> None:
    queue = MovieQueue()
    assert queue.add([str(movies)]) == 3
    assert [os.path.basename(p) for p in queue.paths] == ["a.mp4", "b_broken.mp4", "c.mkv"]
    assert queue.add([str(movies / "a.mp4"), str(movies)]) == 0
    assert queue.add([str(movies)], recursive=True) == 1
    assert os.path.basename(queue.paths[-1]) == "d.mp4"
    queue.remove(str(movies / "c.mkv"))
    assert len(queue.paths) == 3 and not any(p.endswith("c.mkv") for p in queue.paths)
    queue.clear()
    assert queue.paths == []


def test_make_preview_sizes() -> None:
    horizontal = make_preview(Image.new("RGB", (3000, 1080)), "horizontal")
    assert horizontal.size == (1200, 120)
    assert make_preview(Image.new("RGB", (40, 1080)), "horizontal").size == (40, 120)
    assert make_preview(Image.new("RGBA", (1920, 1920)), "circular").size == (240, 240)


def test_job_saves_each_movie_and_continues_after_a_failure(movies, tmp_path) -> None:
    queue = MovieQueue()
    queue.add([str(movies)])
    options = options_for(tmp_path, methods=["avg", "hsv"], barcode_types=["horizontal", "circular"])
    job = Job(queue.paths, options)
    job.run()

    assert job.state == "finished", job.message
    assert [item.status for item in job.items] == ["ok", "failed", "ok"]
    assert "b_broken.mp4" in job.items[1].path and job.items[1].error
    assert job.overall_progress == 1.0 and job.counts() == {"ok": 2, "failed": 1}
    first = job.items[0]
    # Same names as the command line with --output-dir (no workers suffix in the app)
    expected = {
        str(tmp_path / "out" / f"a_{method}_{kind}.png")
        for method in ("avg", "hsv")
        for kind in ("horizontal", "circular")
    }
    assert set(first.outputs) == expected and all(os.path.exists(p) for p in expected)
    assert [(m, k) for m, k, _, _ in first.previews] == [
        ("avg", "horizontal"),
        ("avg", "circular"),
        ("hsv", "horizontal"),
        ("hsv", "circular"),
    ]
    assert first.previews[0][2].size == (12, 120)
    assert "2 ok, 1 failed" in job.message


def test_job_skip_existing_and_save_colors(tmp_path) -> None:
    first = Job([SAMPLE], options_for(tmp_path, save_colors=True))
    first.run()
    assert first.items[0].status == "ok"
    assert str(tmp_path / "out" / "sample_colors.npz") in first.items[0].outputs
    assert os.path.exists(tmp_path / "out" / "sample_colors.npz")

    again = Job([SAMPLE], options_for(tmp_path, save_colors=True, skip_existing=True))
    again.run()
    assert again.items[0].status == "skipped" and again.state == "finished"


@pytest.mark.parametrize("workers", [1, 2])
def test_job_cancel(tmp_path, workers: int) -> None:
    job = Job([SAMPLE, SAMPLE], options_for(tmp_path, width=None, workers=workers))
    on_progress = job._on_progress

    def cancel_after_a_few_frames(done: int, total: int) -> None:
        on_progress(done, total)
        if done >= 3:
            job.cancel()

    job._on_progress = cancel_after_a_few_frames  # type: ignore[method-assign]
    job.run()
    assert job.state == "cancelled"
    assert [item.status for item in job.items] == ["cancelled", "cancelled"]
    assert not os.path.exists(tmp_path / "out" / "sample_avg_horizontal.png")
    assert multiprocessing.active_children() == []


def test_job_reports_invalid_options_and_output_folder(tmp_path) -> None:
    job = Job([SAMPLE], options_for(tmp_path, methods=[]))
    job.run()
    assert job.state == "error" and "method" in job.message

    blocker = tmp_path / "file"
    blocker.write_text("")
    job = Job([SAMPLE], options_for(tmp_path, output_dir=str(blocker / "out")))
    job.run()
    assert job.state == "error" and "output folder" in job.message


def test_open_command() -> None:
    assert open_command("/m/a.png", platform="darwin") == ["open", "/m/a.png"]
    assert open_command("/m/a.png", reveal=True, platform="darwin") == ["open", "-R", "/m/a.png"]
    assert open_command("C:\\m\\a.png", reveal=True, platform="win32") == ["explorer", "/select,C:\\m\\a.png"]
    assert open_command("/m/a.png", reveal=True, platform="linux") == ["xdg-open", "/m"]


def test_fix_std_streams(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stderr", None)
    fix_std_streams()
    assert sys.stderr is not None
    sys.stderr.write("ignored")


def test_importing_the_package_or_the_app_module_does_not_import_nicegui() -> None:
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); import movie_barcodes, movie_barcodes.ui; "
        "print('nicegui' in sys.modules, 'webview' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script, SRC], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "False"]


@pytest.mark.skipif(not HAS_NICEGUI, reason='needs the "ui" extra')
def test_page_adds_a_movie_and_generates_its_barcode(tmp_path) -> None:
    from nicegui import ui
    from nicegui.testing.user_simulation import user_simulation

    from movie_barcodes.ui import create_root

    async def scenario() -> None:
        async with user_simulation(create_root(native=False)) as user:
            await user.open("/")
            await user.should_see("No movie yet")
            user.find("Output folder").clear().type(str(tmp_path / "out"))
            user.find("Workers").type("1")
            user.find("Width").type("8")
            user.find(kind=ui.input, marker=None, content="Path of a movie or folder").type(SAMPLE)
            user.find("Add").click()
            await user.should_see(kind=ui.item_label, content="sample.mp4")  # In the list, not just typed
            user.find("Generate").click()
            await user.should_see("sample_avg_horizontal.png", retries=300)

    asyncio.run(scenario())
    assert Image.open(tmp_path / "out" / "sample_avg_horizontal.png").size == (8, 1080)


def test_job_counts_as_running_until_it_ends(tmp_path) -> None:
    job = Job([SAMPLE], options_for(tmp_path))
    assert job.running and job.overall_progress == 0.0  # Created, its thread not started: no second job may start
    job.run()
    assert not job.running and job.state == "finished" and job.overall_progress == 1.0
