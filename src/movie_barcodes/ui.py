"""
Desktop app: generate barcodes from a window instead of the command line.

Launch it with ``movie-barcodes-ui`` (a native window) or ``movie-barcodes-ui --browser`` (a tab of the default
browser, served on 127.0.0.1 only). It needs the ``ui`` extra: ``pip install "movie-barcodes[ui]"``.

The app works on file paths: movies are read where they are, never uploaded or copied, and the barcodes are written
to an output folder with the same names as the command line gives them.

This module is split in two: the queue, the options and the job runner below are plain Python (tested without a
display), and :func:`main` builds the NiceGUI page on top of them. NiceGUI is only imported inside :func:`main`:
worker processes are started with "spawn" and re-import the main module, which must stay light and must not start
the app again.
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from .api import BARCODE_TYPES, MAX_PROCESSES, METHODS, BarcodeResult, Cancelled, iter_barcodes
from .cli import COLORS_FILE_SUFFIX, BatchRunner, build_parser
from .utility import VIDEO_EXTENSIONS, find_videos

logger = logging.getLogger(__name__)

#: Height of the previews of horizontal barcodes, and size of the previews of circular barcodes, in pixels.
PREVIEW_HEIGHT = 120
PREVIEW_MAX_WIDTH = 1200
PREVIEW_CIRCLE_SIZE = 240
#: Seconds to wait, when the app stops during a job, for the job to cancel and terminate its worker processes (starting
#: the workers can't be interrupted and may take a few seconds).
SHUTDOWN_TIMEOUT = 20


def default_output_dir() -> str:
    """
    Default output folder: ~/Pictures/movie-barcodes, or ~/movie-barcodes without a Pictures folder.
    """
    pictures = Path.home() / "Pictures"
    return str((pictures if pictures.is_dir() else Path.home()) / "movie-barcodes")


@dataclass
class Options:
    """
    What to generate and where, as chosen in the app. Same meaning as the command-line options.
    """

    methods: List[str] = field(default_factory=lambda: ["avg"])
    barcode_types: List[str] = field(default_factory=lambda: ["horizontal"])
    width: Optional[int] = None
    height: Optional[int] = None
    workers: Optional[int] = None
    crop_borders: bool = False
    linear: bool = False
    skip_existing: bool = False
    save_colors: bool = False
    output_dir: str = field(default_factory=default_output_dir)

    def errors(self) -> List[str]:
        """
        Problems that prevent a run, as messages for the user (empty if the options are valid).
        """
        problems = []
        if not self.methods:
            problems.append("Choose at least one method.")
        problems += [f"Unknown method: {method}." for method in self.methods if method not in METHODS]
        if not self.barcode_types:
            problems.append("Choose at least one barcode type.")
        problems += [f"Unknown barcode type: {kind}." for kind in self.barcode_types if kind not in BARCODE_TYPES]
        for name in ("width", "height", "workers"):
            value = getattr(self, name)
            if value is not None and value < 1:
                problems.append(f"{name.capitalize()} must be at least 1.")
        if not self.output_dir.strip():
            problems.append("Choose an output folder.")
        return problems

    def to_args(self) -> argparse.Namespace:
        """
        The equivalent command-line arguments, so that outputs are named and skipped exactly like the command line.
        """
        args = build_parser().parse_args([])
        args.barcode_types = list(dict.fromkeys(self.barcode_types))
        args.width, args.height, args.workers = self.width, self.height, self.workers
        args.crop_borders, args.linear = self.crop_borders, self.linear
        args.save_colors, args.skip_existing = self.save_colors, self.skip_existing
        args.output_dir = os.path.expanduser(self.output_dir.strip())
        return args


class MovieQueue:
    """
    The movies to process, in order, without duplicates.
    """

    def __init__(self) -> None:
        self.paths: List[str] = []

    @staticmethod
    def _key(file_path: str) -> str:
        return os.path.normcase(os.path.abspath(file_path))

    def add(self, paths: Iterable[str], recursive: bool = False) -> int:
        """
        Add files, and the videos found in folders (by extension, without opening them).

        :param paths: Files and/or folders.
        :param recursive: Also look for videos in the subfolders of folders.
        :return: The number of movies added (already queued ones are ignored).
        """
        known = {self._key(file_path) for file_path in self.paths}
        added = 0
        for file_path in find_videos([os.fspath(p) for p in paths], recursive=recursive):
            key = self._key(file_path)
            if key not in known:
                known.add(key)
                self.paths.append(file_path)
                added += 1
        return added

    def remove(self, file_path: str) -> None:
        """
        Remove a movie from the queue (no-op if it is not queued).
        """
        key = self._key(file_path)
        self.paths = [queued for queued in self.paths if self._key(queued) != key]

    def clear(self) -> None:
        """
        Remove every movie.
        """
        self.paths.clear()


def make_preview(image: Image.Image, barcode_type: str) -> Image.Image:
    """
    A small copy of a barcode for the gallery: horizontal barcodes are squeezed to PREVIEW_HEIGHT pixels (and at most
    PREVIEW_MAX_WIDTH wide), circular ones to PREVIEW_CIRCLE_SIZE pixels.
    """
    if barcode_type == "circular":
        preview = image.copy()
        preview.thumbnail((PREVIEW_CIRCLE_SIZE, PREVIEW_CIRCLE_SIZE))
        return preview
    return image.resize((min(image.width, PREVIEW_MAX_WIDTH), PREVIEW_HEIGHT), Image.Resampling.BILINEAR)


@dataclass
class MovieItem:
    """
    State of one movie of a job, as shown in the app.

    :ivar status: "pending", "running", "ok", "failed", "skipped" or "cancelled".
    :ivar previews: (method, barcode type, small image, saved file) for each barcode.
    """

    path: str
    status: str = "pending"
    error: str = ""
    outputs: List[str] = field(default_factory=list)
    previews: List[Tuple[str, str, Image.Image, str]] = field(default_factory=list)
    duration: float = 0.0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


class _AppRunner(BatchRunner):
    """
    The command line's batch runner, used for its output names, --skip-existing and saving, with the app's choice
    of methods (any subset, where the command line takes one method or all of them).
    """

    def __init__(self, args: argparse.Namespace, inputs: Sequence[str], methods: Sequence[str]) -> None:
        super().__init__(args, inputs, batch=True)
        self.methods = list(methods)
        # File names mention the number of workers only when it is chosen on the command line: not in the app
        self.workers = None

    def methods_for(self, input_path: str, movie_colors=None) -> List[str]:
        return list(self.methods)


class Job:
    """
    Process a list of movies with the given options, saving each movie's barcodes as soon as it is done.

    :meth:`run` blocks: call it from a background thread. The other threads read the public attributes to show the
    progress (plain attribute reads and writes, safe under the GIL) and call :meth:`cancel` to stop it.

    :ivar state: "ready", "running", "finished", "cancelled" or "error" (invalid options or output folder).
    :ivar current: Index of the movie being processed.
    :ivar frames_done: Frames decoded so far in the current movie, out of frames_total (0 until decoding starts).
    :ivar version: Incremented each time a movie's status changes, to know when to redraw the list and the gallery.
    """

    def __init__(self, paths: Sequence[str], options: Options) -> None:
        self.options = options
        self.items = [MovieItem(os.fspath(p)) for p in paths]
        self.state = "ready"
        self.message = ""
        self.current = 0
        self.frames_done = 0
        self.frames_total = 0
        self.version = 0
        self.started = 0.0
        self.finished = 0.0
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        """
        Whether the job is running or about to run (created, its thread not started yet).
        """
        return self.state in ("ready", "running")

    def cancel(self) -> None:
        """
        Stop the job: the movie being decoded is abandoned (its worker processes are terminated), the next ones are
        not processed. Barcodes already saved are kept.
        """
        self._cancel.set()

    @property
    def cancelling(self) -> bool:
        """
        Whether cancel() was called.
        """
        return self._cancel.is_set()

    @property
    def movie_progress(self) -> float:
        """
        Share of the current movie decoded, from 0 to 1.
        """
        return self.frames_done / self.frames_total if self.frames_total else 0.0

    @property
    def overall_progress(self) -> float:
        """
        Share of the job done, from 0 to 1, counting the current movie's progress.
        """
        if not self.items:
            return 1.0
        finished = sum(item.status in ("ok", "failed", "skipped") for item in self.items)
        partial = self.movie_progress if self.running and self.current < len(self.items) else 0.0
        return min(1.0, (finished + partial) / len(self.items))

    def counts(self) -> dict:
        """
        Number of movies per status.
        """
        counts: dict = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def _on_progress(self, done: int, total: int) -> None:
        self.frames_done, self.frames_total = done, total

    def _set_status(self, item: MovieItem, status: str) -> None:
        item.status = status
        self.version += 1

    def _start(self, index: int) -> None:
        self.current, self.frames_done, self.frames_total = index, 0, 0
        if index < len(self.items):
            self._set_status(self.items[index], "running")

    def _finish_item(self, item: MovieItem, result: BarcodeResult, runner: BatchRunner) -> None:
        item.duration = result.duration
        if result.ok:
            try:
                runner.save(result)
                assert result.colors is not None
                plan = runner.plan(result.path, result.colors)
                item.outputs = plan.all_paths(runner.save_colors_for(result.path))
                item.previews = [
                    (method, barcode_type, make_preview(image, barcode_type), plan.images[(method, barcode_type)])
                    for (method, barcode_type), image in result.barcodes.items()
                ]
            except Exception as error:  # e.g. the output folder is not writable: report it on this movie
                result.status, result.error = "failed", error
        if result.status == "failed":
            item.error = f"{type(result.error).__name__}: {result.error}"
            logger.error("Failed to process '%s': %s", result.path, result.error)
        # Free the full-size images and colors before the next movie
        result.barcodes, result.colors = {}, None
        self._set_status(item, result.status)

    def run(self) -> None:
        """
        Process every movie in order (blocking). A movie that fails is reported and the next ones are processed.
        """
        options = self.options
        self.state, self.started = "running", time.time()
        problems = options.errors()
        if problems:
            self.state, self.message = "error", " ".join(problems)
            return
        args = options.to_args()
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except OSError as error:
            self.state, self.message = "error", f"Cannot create the output folder: {error}"
            return

        paths = [item.path for item in self.items]
        runner = _AppRunner(args, paths, options.methods)
        results = iter_barcodes(
            paths,
            methods=options.methods,
            barcode_types=args.barcode_types,
            width=options.width,
            height=options.height,
            workers=options.workers,
            crop_borders=options.crop_borders,
            linear=options.linear,
            skip=runner.outputs_exist if options.skip_existing else None,
            on_progress=self._on_progress,
            cancel=self._cancel,
        )
        self._start(0)
        try:
            for index, result in enumerate(results):
                self._finish_item(self.items[index], result, runner)
                self._start(index + 1)
        except Cancelled:
            for item in self.items[self.current :]:
                self._set_status(item, "cancelled")
            self.state = "cancelled"
        except Exception as error:  # Unexpected: keep the app alive and show it
            logger.exception("Job failed")
            self.state, self.message = "error", f"{type(error).__name__}: {error}"
        else:
            self.state = "finished"
        finally:
            self.finished = time.time()
        counts = self.counts()
        summary = ", ".join(f"{counts[status]} {status}" for status in ("ok", "failed", "skipped") if status in counts)
        if self.state != "error":
            self.message = (
                f"{self.state.capitalize()}: {summary or 'nothing done'} in {self.finished - self.started:.0f} s"
            )


def open_command(file_path: str, reveal: bool = False, platform: str = sys.platform) -> List[str]:
    """
    Command opening a file or folder with the default application, or showing a file in the file manager.

    :param file_path: The file or folder.
    :param reveal: Select the file in its folder (Finder / Explorer) instead of opening it. On Linux, the folder is
        opened.
    :param platform: sys.platform value.
    :return: The command, for subprocess.
    """
    if platform == "darwin":
        return ["open", "-R", file_path] if reveal else ["open", file_path]
    if platform.startswith("win"):
        return ["explorer", f"/select,{file_path}"] if reveal else ["explorer", file_path]
    return ["xdg-open", os.path.dirname(file_path) if reveal else file_path]


def open_path(file_path: str, reveal: bool = False) -> None:
    """
    Open a file or folder (or show a file in its folder) without waiting.
    """
    try:
        subprocess.Popen(open_command(file_path, reveal))  # pylint: disable=consider-using-with
    except OSError as error:
        logger.error("Cannot open '%s': %s", file_path, error)


def fix_std_streams() -> None:
    """
    Give sys.stdout and sys.stderr a sink when they are None (an app started without a console, e.g. a packaged
    windowed app): tqdm progress bars and logging write to them and would fail.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # pylint: disable=consider-using-with


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Arguments of the movie-barcodes-ui command.
    """
    parser = argparse.ArgumentParser(
        prog="movie-barcodes-ui",
        description="Desktop app generating movie barcodes from the movies on this computer.",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open the app in a tab of the default browser instead of a native window (served on 127.0.0.1 only; "
        "movies are then added by typing or pasting their path).",
    )
    parser.add_argument("--port", type=int, default=None, help="Port of the local server (default: a free port).")
    parser.add_argument("--no-show", action="store_true", help="With --browser: don't open the browser tab.")
    return parser


STATUS_COLORS = {
    "pending": "grey",
    "running": "primary",
    "ok": "positive",
    "failed": "negative",
    "skipped": "grey-7",
    "cancelled": "warning",
}


def create_root(native: bool) -> Callable[[], None]:  # pylint: disable=too-many-statements
    """
    Build the app's page function, with its own movie list, options and job.

    :param native: Whether the page is shown in a native window (OS file dialogs) or a browser tab (paths are typed).
    :return: The root function to pass to ``ui.run``.
    """
    # pylint: disable=import-outside-toplevel
    from nicegui import app, run, ui

    queue = MovieQueue()
    options = Options()
    jobs: List[Job] = []  # The current (last) job, if any
    recursive = {"value": False}

    def current_job() -> Optional[Job]:
        return jobs[-1] if jobs else None

    def job_running() -> bool:
        job = current_job()
        return job is not None and job.running

    @app.on_shutdown
    async def stop_job() -> None:
        # Closing the window (or stopping the app) must not leave worker processes behind: NiceGUI hard-exits after
        # the shutdown handlers, so wait for the job to see the cancel and terminate its workers
        job = current_job()
        if job is None or not job.running:
            return
        job.cancel()
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT
        while job.running and time.monotonic() < deadline:
            await asyncio.sleep(0.1)

    def root() -> None:  # pylint: disable=too-many-statements
        ui.page_title("Movie Barcodes")
        shown = {"versions": None}

        # ----- Movies ------------------------------------------------------------------------------------------
        async def add(paths: Iterable[str]) -> None:
            paths = list(paths)
            if not paths:
                return
            # Listing a large (or network) folder can take a while: not on the event loop
            added = await run.io_bound(queue.add, paths, recursive["value"])
            ui.notify(f"{added} movie(s) added" if added else "No new movie found", type="info")
            queue_view.refresh()
            tick()  # Enable Generate now, not at the next timer tick

        async def add_files() -> None:
            import webview  # pylint: disable=import-outside-toplevel

            extensions = ";".join(f"*{extension}" for extension in VIDEO_EXTENSIONS)
            result = await app.native.main_window.create_file_dialog(
                webview.FileDialog.OPEN,
                allow_multiple=True,
                file_types=(f"Movies ({extensions})", "All files (*.*)"),
            )
            await add(result or ())

        async def add_folder() -> None:
            import webview  # pylint: disable=import-outside-toplevel

            result = await app.native.main_window.create_file_dialog(webview.FileDialog.FOLDER)
            await add(result or ())

        async def choose_output_dir() -> None:
            import webview  # pylint: disable=import-outside-toplevel

            result = await app.native.main_window.create_file_dialog(
                webview.FileDialog.FOLDER, directory=os.path.expanduser(options.output_dir)
            )
            if result:
                options.output_dir = result[0]

        def remove(file_path: str) -> None:
            queue.remove(file_path)
            queue_view.refresh()
            tick()

        def clear() -> None:
            queue.clear()
            queue_view.refresh()
            tick()

        with ui.header().classes("items-center justify-between"):
            ui.label("Movie Barcodes").classes("text-xl font-medium")
            ui.label("Movies are read where they are, nothing is uploaded").classes("text-sm opacity-80")

        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center"):
                ui.label("Movies").classes("text-lg font-medium")
                ui.space()
                if native:
                    ui.button("Add files", icon="movie", on_click=add_files).props("unelevated")
                    ui.button("Add folder", icon="folder", on_click=add_folder).props("unelevated")
                else:
                    path_input = (
                        ui.input(placeholder="Path of a movie or folder")
                        .props("dense clearable")
                        .classes("w-96")
                        .tooltip("In a browser, paste the path: files are read from this computer, not uploaded")
                    )

                    async def add_typed() -> None:
                        value = (path_input.value or "").strip().strip('"').strip("'")
                        if value:
                            path_input.value = ""
                            await add([os.path.expanduser(value)])

                    path_input.on("keydown.enter", add_typed)
                    ui.button("Add", icon="add", on_click=add_typed).props("unelevated")
                ui.checkbox("Include subfolders").bind_value(recursive, "value").tooltip(
                    "When adding a folder, also add the movies of its subfolders"
                )
                clear_button = ui.button("Clear", icon="clear_all", on_click=clear).props("flat")

            @ui.refreshable
            def queue_view() -> None:
                job = current_job()
                statuses = {item.path: item for item in job.items} if job is not None else {}
                if not queue.paths:
                    ui.label("No movie yet: add files or a folder.").classes("text-gray-500")
                    return
                with ui.list().props("dense separator").classes("w-full"):
                    for file_path in queue.paths:
                        item = statuses.get(file_path)
                        with ui.item():
                            with ui.item_section():
                                ui.item_label(os.path.basename(file_path))
                                ui.item_label(os.path.dirname(file_path)).props("caption")
                            if item is not None:
                                with ui.item_section().props("side"):
                                    ui.badge(item.status, color=STATUS_COLORS[item.status])
                            with ui.item_section().props("side"):
                                ui.button(icon="close", on_click=lambda p=file_path: remove(p)).props(
                                    "flat round dense"
                                ).tooltip("Remove from the list").set_enabled(not job_running())

            queue_view()

        # ----- Options -----------------------------------------------------------------------------------------
        with ui.card().classes("w-full"):
            ui.label("Options").classes("text-lg font-medium")
            with ui.row().classes("w-full items-end gap-4"):
                ui.select(list(METHODS), multiple=True, label="Methods").props("use-chips").classes(
                    "min-w-48"
                ).bind_value(options, "methods").tooltip(
                    "avg: average color, smoothed: two-step resize, hsv/bgr: histogram peak, kmeans: dominant cluster"
                )
                ui.select(list(BARCODE_TYPES), multiple=True, label="Barcode types").props("use-chips").classes(
                    "min-w-48"
                ).bind_value(options, "barcode_types")

                def optional_int(value: object) -> Optional[int]:
                    return int(value) if value not in (None, "") else None  # type: ignore[call-overload]

                ui.number("Width", min=1, precision=0, placeholder="auto").classes("w-28").bind_value(
                    options, "width", forward=optional_int
                ).tooltip("Number of sampled frames (default: min(video width, frame count))")
                ui.number("Height", min=1, precision=0, placeholder="auto").classes("w-28").bind_value(
                    options, "height", forward=optional_int
                ).tooltip("Height of horizontal barcodes (default: video height)")
                ui.number("Workers", min=1, max=MAX_PROCESSES, precision=0, placeholder=str(MAX_PROCESSES)).classes(
                    "w-28"
                ).bind_value(options, "workers", forward=optional_int).tooltip(
                    f"Parallel decoding processes (default: {MAX_PROCESSES}, the number of CPU cores)"
                )
            with ui.row().classes("w-full items-center gap-4"):
                ui.checkbox("Crop black borders").bind_value(options, "crop_borders")
                ui.checkbox("Linear light").bind_value(options, "linear").tooltip(
                    "Average avg and smoothed colors in linear light (brighter)"
                )
                ui.checkbox("Skip existing").bind_value(options, "skip_existing").tooltip(
                    "Skip the movies whose barcodes are already in the output folder"
                )
                ui.checkbox("Save colors").bind_value(options, "save_colors").tooltip(
                    f"Also save the extracted colors (<movie>{COLORS_FILE_SUFFIX}) to render them again later "
                    "with the command line, without decoding"
                )
            with ui.row().classes("w-full items-center gap-2"):
                ui.input("Output folder").classes("grow").bind_value(options, "output_dir")
                if native:
                    ui.button(icon="folder_open", on_click=choose_output_dir).props("flat round").tooltip("Choose")

                def open_output_dir() -> None:
                    output_dir = os.path.expanduser(options.output_dir.strip())
                    if os.path.isdir(output_dir):
                        open_path(output_dir)
                    else:
                        ui.notify("The output folder does not exist yet", type="warning")

                ui.button("Open", icon="open_in_new", on_click=open_output_dir).props("flat")

        # ----- Run ---------------------------------------------------------------------------------------------
        async def generate() -> None:
            if job_running():
                return
            problems = options.errors()
            if problems:
                ui.notify(" ".join(problems), type="negative")
                return
            if not queue.paths:
                ui.notify("Add movies first", type="warning")
                return
            job = Job(list(queue.paths), Options(**vars(options)))
            jobs[:] = [job]
            # In a worker thread: the event loop keeps serving the page (Job.run catches its own errors)
            await run.io_bound(job.run)
            kind = {"finished": "positive", "cancelled": "warning", "error": "negative"}.get(job.state, "info")
            if job.state == "finished" and job.counts().get("failed"):
                kind = "warning"
            ui.notify(job.message, type=kind)

        def cancel() -> None:
            job = current_job()
            if job is not None and job.running:
                job.cancel()
                status_label.text = "Cancelling..."

        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center"):
                generate_button = ui.button("Generate", icon="play_arrow", on_click=generate).props("unelevated")
                cancel_button = ui.button("Cancel", icon="stop", color="negative", on_click=cancel).props("outline")
                status_label = ui.label().classes("text-sm")
            overall_bar = ui.linear_progress(value=0, show_value=False, size="10px").classes("w-full")
            movie_label = ui.label().classes("text-sm text-gray-600")
            movie_bar = ui.linear_progress(value=0, show_value=False, color="secondary").classes("w-full")

        # ----- Results -----------------------------------------------------------------------------------------
        def render_card(item: MovieItem) -> None:
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center"):
                    ui.badge(item.status, color=STATUS_COLORS[item.status])
                    ui.label(item.name).classes("font-medium")
                    ui.label(f"{item.duration:.1f} s").classes("text-sm text-gray-500")
                    ui.space()
                    if item.outputs:
                        ui.button(
                            "Show in folder", icon="folder", on_click=lambda f=item.outputs[0]: open_path(f, True)
                        ).props("flat dense")
                if item.error:
                    ui.label(item.error).classes("text-negative text-sm")
                if item.status == "skipped":
                    ui.label("Its barcodes are already in the output folder.").classes("text-sm text-gray-500")
                # Horizontal barcodes stacked at full width, circular ones side by side below
                horizontal = [preview for preview in item.previews if preview[1] == "horizontal"]
                circular = [preview for preview in item.previews if preview[1] != "horizontal"]
                for method, _, preview, saved in horizontal:
                    with ui.column().classes("w-full gap-1"):
                        ui.image(preview).props("fit=fill no-spinner").classes("w-full cursor-pointer").style(
                            f"height: {PREVIEW_HEIGHT}px"
                        ).on("click", lambda f=saved: open_path(f)).tooltip("Open")
                        ui.label(f"{method} · {os.path.basename(saved)}").classes("text-xs text-gray-500")
                if circular:
                    with ui.row().classes("gap-4"):
                        for method, _, preview, saved in circular:
                            with ui.column().classes("gap-1 items-center"):
                                ui.image(preview).props("fit=contain no-spinner").classes("cursor-pointer").style(
                                    f"width: {PREVIEW_CIRCLE_SIZE}px; height: {PREVIEW_CIRCLE_SIZE}px"
                                ).on("click", lambda f=saved: open_path(f)).tooltip("Open")
                                ui.label(f"{method} · {os.path.basename(saved)}").classes("text-xs text-gray-500")

        ui.label("Results").classes("text-lg font-medium")
        gallery = ui.column().classes("w-full")
        # Cards already shown for the current job: each movie's card is added once, when the movie is done
        rendered = {"job": None, "items": set()}

        def update_gallery(job: Optional[Job]) -> None:
            if rendered["job"] is not job:
                gallery.clear()
                rendered["job"], rendered["items"] = job, set()
            if job is None:
                return
            for index, item in enumerate(job.items):
                if item.status in ("ok", "failed", "skipped") and index not in rendered["items"]:
                    rendered["items"].add(index)
                    with gallery:
                        render_card(item)

        def tick() -> None:
            job = current_job()
            running = job_running()
            generate_button.set_enabled(not running and bool(queue.paths))
            clear_button.set_enabled(not running)
            cancel_button.set_enabled(running)
            if job is None:
                status_label.text = f"{len(queue.paths)} movie(s) in the list"
                overall_bar.value, movie_bar.value, movie_label.text = 0, 0, ""
            else:
                overall_bar.value = job.overall_progress
                movie_bar.value = job.movie_progress if running else 0
                if running:
                    if not job.cancelling:
                        status_label.text = f"Movie {min(job.current + 1, len(job.items))} of {len(job.items)}"
                    if job.current < len(job.items):
                        item = job.items[job.current]
                        movie_label.text = (
                            f"{item.name}: {job.frames_done} / {job.frames_total} frames"
                            if job.frames_total
                            else f"{item.name}: opening..."
                        )
                else:
                    status_label.text, movie_label.text = job.message, ""
            # Redraw the list and add finished movies to the gallery only when a movie changed status
            versions = (id(job), job.version if job else 0, running)
            if shown["versions"] != versions:
                shown["versions"] = versions
                update_gallery(job)
                queue_view.refresh()

        ui.timer(0.2, tick)

    return root


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Start the desktop app (entry point of movie-barcodes-ui).
    """
    multiprocessing.freeze_support()
    fix_std_streams()
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(message)s")

    try:
        # pylint: disable=import-outside-toplevel
        from nicegui import native as nicegui_native
        from nicegui import ui
    except ImportError:
        sys.exit('The desktop app needs the "ui" extra: pip install "movie-barcodes[ui]"')

    native = not args.browser
    root = create_root(native)
    port = args.port or nicegui_native.find_open_port()
    ui.run(
        root,
        title="Movie Barcodes",
        native=native,
        window_size=(1200, 900) if native else None,
        host="127.0.0.1",
        port=port,
        reload=False,
        show=not native and not args.no_show,
        uvicorn_logging_level="warning",
        show_welcome_message=not native,
    )


if __name__ == "__main__":
    main()
