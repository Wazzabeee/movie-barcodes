import argparse
import logging
import sys
from collections import Counter
from os import path
from typing import Dict, List, Optional, Sequence, Tuple

from .api import (
    BARCODE_TYPES,
    MAX_PROCESSES,
    METHODS,
    BarcodeResult,
    MovieColors,
    is_colors_file,
    iter_barcodes,
    load_colors,
    save_colors,
)
from .utility import (
    LINEAR_METHODS,
    find_videos,
    format_time,
    get_output_path,
    read_path_list,
    resolve_workers,
    save_barcode_image,
    validate_destination_path,
)

#: Default output directory, relative to the current working directory.
DEFAULT_OUTPUT_DIR = "barcodes"
#: Suffix of the colors files written by --save-colors, after the movie name.
COLORS_FILE_SUFFIX = "_colors.npz"


def collect_inputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[str]:
    """
    Expand the -i paths (directories included) and the --from-list file into the list of inputs to process.

    :param args: The parsed command-line arguments.
    :param parser: The parser, to report usage errors.
    :return: Video files and colors files, without duplicates, in order.
    """
    paths = list(args.input_video_path or [])
    if args.from_list:
        try:
            paths += read_path_list(args.from_list)
        except OSError as error:
            parser.error(f"cannot read --from-list file: {error}")
    if not paths:
        parser.error("no input: give videos or directories with -i, or a list file with --from-list")
    inputs = find_videos(paths, recursive=args.recursive)
    if not inputs:
        parser.error("no video found in the given directories")
    return inputs


def is_batch(args: argparse.Namespace, inputs: Sequence[str]) -> bool:
    """
    Whether to run in batch mode: errors are reported per movie, with a summary and an exit code, instead of
    being raised. A single file given with -i keeps the single-movie behavior.
    """
    return len(inputs) > 1 or bool(args.from_list) or any(path.isdir(p) for p in args.input_video_path or [])


def base_names(inputs: Sequence[str]) -> Dict[str, str]:
    """
    Name used for the output files of each input: the movie file name without extension.

    Colors files are named after the movie they were extracted from (<movie>_colors.npz gives <movie>). When two
    inputs share a name (e.g. Season 1/E01.mkv and Season 2/E01.mkv), their parent directory name is prepended,
    then a number if they still clash, so that no output overwrites another.

    :param inputs: The input paths.
    :return: {input path: base name}.
    """

    def stem(input_path: str) -> str:
        name = path.splitext(path.basename(input_path))[0]
        if is_colors_file(input_path) and name.endswith(COLORS_FILE_SUFFIX[: -len(".npz")]):
            name = name[: -len(COLORS_FILE_SUFFIX) + len(".npz")]
        return name

    names = {input_path: stem(input_path) for input_path in inputs}
    counts = Counter(names.values())
    for input_path, name in names.items():
        if counts[name] > 1:
            parent = path.basename(path.dirname(path.abspath(input_path)))
            names[input_path] = f"{parent}_{name}" if parent else name
    seen: Counter = Counter()
    counts = Counter(names.values())
    for input_path, name in names.items():
        if counts[name] > 1:
            seen[name] += 1
            if seen[name] > 1:
                names[input_path] = f"{name}_{seen[name]}"
    return names


class OutputPlan:
    """
    Where the outputs of one input go: one image per (method, barcode type), and the colors file.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        base_name: str,
        methods: Sequence[str],
        linear: bool,
        workers: Optional[int],
        output_dir: str,
    ) -> None:
        self.images: Dict[Tuple[str, str], str] = {}
        for method in methods:
            for barcode_type in args.barcode_types:
                # With --linear, avg and smoothed barcodes get a distinct name so they don't overwrite the gamma
                # averaged ones
                output_method = f"{method}_linear" if linear and method in LINEAR_METHODS else method
                self.images[(method, barcode_type)] = get_output_path(
                    base_name,
                    output_method,
                    barcode_type,
                    workers=workers,
                    destination_path=args.destination_path,
                    output_name=args.output_name,
                    # With several images, -n / -d would give them all the same path: append the method / type
                    method_suffix=len(methods) > 1,
                    type_suffix=len(args.barcode_types) > 1,
                    output_dir=output_dir,
                )
        if args.destination_path:
            colors_path = path.splitext(args.destination_path)[0] + COLORS_FILE_SUFFIX
        else:
            colors_path = path.join(output_dir, (args.output_name or base_name) + COLORS_FILE_SUFFIX)
        self.colors = colors_path

    def all_paths(self, with_colors: bool) -> List[str]:
        """
        Every output file of this input.
        """
        return list(self.images.values()) + ([self.colors] if with_colors else [])


class BatchRunner:
    """
    Run the command line on a list of inputs: decode each movie once, save its barcodes (and colors), and keep
    track of the outcome of each input for the summary.
    """

    def __init__(self, args: argparse.Namespace, inputs: Sequence[str], batch: bool) -> None:
        self.args = args
        self.inputs = list(inputs)
        self.batch = batch
        self.output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        # Workers above the number of CPU cores are capped (with a warning); the file name mentions the number of
        # workers actually used, only when the user chose it
        self.workers = resolve_workers(args.workers, MAX_PROCESSES)
        self.names = base_names(self.inputs)
        self.results: List[BarcodeResult] = []

    def methods_for(self, input_path: str, movie_colors: Optional[MovieColors] = None) -> List[str]:
        """
        Methods rendered for an input: all with --all_methods, else --method, else avg for videos and every stored
        method for colors files.
        """
        if self.args.all_methods:
            return list(METHODS)
        if self.args.method:
            return [self.args.method]
        if is_colors_file(input_path):
            movie_colors = movie_colors or load_colors(input_path)
            return movie_colors.methods
        return ["avg"]

    def plan(self, input_path: str, movie_colors: Optional[MovieColors] = None) -> OutputPlan:
        """
        Output paths of an input. Colors files need their metadata (methods, linear flag): they are read if not
        given.
        """
        colors_file = is_colors_file(input_path)
        if colors_file and movie_colors is None:
            movie_colors = load_colors(input_path)
        linear = movie_colors.linear if colors_file else self.args.linear
        return OutputPlan(
            self.args,
            self.names[input_path],
            self.methods_for(input_path, movie_colors),
            linear,
            # Colors files are not decoded: no workers in their names
            None if colors_file else self.workers,
            self.output_dir,
        )

    def outputs_exist(self, input_path: str) -> bool:
        """
        --skip-existing: whether every output of an input already exists.
        """
        try:
            outputs = self.plan(input_path).all_paths(self.save_colors_for(input_path))
        except (OSError, ValueError):
            return False  # Unreadable colors file: process it, to report the error
        if all(path.exists(output) for output in outputs):
            logging.info("Skipping '%s': its outputs already exist", input_path)
            return True
        return False

    def save_colors_for(self, input_path: str) -> bool:
        """
        Whether the colors of an input are saved: with --save-colors, for videos (colors files already are).
        """
        return self.args.save_colors and not is_colors_file(input_path)

    def save(self, result: BarcodeResult) -> None:
        """
        Save the barcodes (and colors) of a processed input, and log its properties.
        """
        assert result.colors is not None
        outputs = self.plan(result.path, result.colors)
        for key, image in result.barcodes.items():
            save_barcode_image(image, outputs.images[key])
        if self.save_colors_for(result.path):
            save_colors(result.colors, outputs.colors)
            logging.info("Colors saved at '%s'", path.abspath(outputs.colors))

        colors = result.colors
        logging.info("Processed File: %s", self.names[result.path])
        logging.info("Number of Frames: %d", colors.frame_count)
        if not is_colors_file(result.path):
            video_duration = colors.frame_count / colors.fps if colors.fps > 0 else 0
            logging.info("Video Duration: %s", format_time(video_duration))
            logging.info("Video Size: %.2f MB", path.getsize(result.path) / (1024 * 1024))
        logging.info("Processing Time: %s", format_time(result.duration))

    def run(self) -> List[BarcodeResult]:
        """
        Process every input in order. In batch mode, an input that fails is logged and the next ones are processed;
        otherwise its error is raised.

        :return: One result per input.
        """
        args = self.args
        results = iter_barcodes(
            self.inputs,
            methods=list(METHODS) if args.all_methods else ([args.method] if args.method else None),
            barcode_types=args.barcode_types,
            width=args.width,
            height=args.height,
            workers=self.workers,
            crop_borders=args.crop_borders,
            linear=args.linear,
            skip=self.outputs_exist if args.skip_existing else None,
        )
        for index, result in enumerate(results, start=1):
            if self.batch:
                logging.info("[%d/%d] %s: %s", index, len(self.inputs), result.status, result.path)
            if result.ok:
                try:
                    self.save(result)
                except Exception as error:
                    if not self.batch:
                        raise
                    result.status, result.error, result.barcodes = "failed", error, {}
            if result.status == "failed":
                if not self.batch:
                    assert result.error is not None
                    raise result.error
                logging.error("Failed to process '%s': %s", result.path, result.error)
            # Images and colors are saved: free them before decoding the next movie (a long batch would otherwise
            # keep every movie's colors in memory)
            result.barcodes, result.colors = {}, None
            self.results.append(result)
        return self.results

    def log_summary(self) -> None:
        """
        Log one line per input (status, duration, path, error) and the totals.
        """
        logging.info("=" * 40 + " SUMMARY " + "=" * 40)
        for result in self.results:
            line = f"{result.status:<8} {format_time(result.duration):>12}  {result.path}"
            if result.error is not None:
                line += f"  ({type(result.error).__name__}: {result.error})"
            logging.info("%s", line)
        counts = Counter(result.status for result in self.results)
        logging.info(
            "%d ok, %d failed, %d skipped, total time %s",
            counts["ok"],
            counts["failed"],
            counts["skipped"],
            format_time(sum(result.duration for result in self.results)),
        )


def generate_and_save_barcodes(args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None) -> int:
    """
    Generate and save the barcodes of every input.

    Each video is decoded once for all the requested methods (--all_methods) and barcode types (-t horizontal
    circular). Colors files (.npz, written by --save-colors) are rendered without decoding.

    :param args: argparse.Namespace object containing the command-line arguments
    :param parser: The parser, to report usage errors. Default is the command-line parser.
    :return: The exit code: 1 if an input failed in batch mode, else 0. With a single input, errors are raised.
    """
    parser = parser or build_parser()
    if args.barcode_types is None:
        args.barcode_types = ["horizontal"]
    args.barcode_types = list(dict.fromkeys(args.barcode_types))
    inputs = collect_inputs(args, parser)
    if len(inputs) > 1 and (args.output_name or args.destination_path):
        parser.error("-n/--output_name and -d/--destination_path need a single input; use --output-dir instead")

    # Fail before decoding anything if the images cannot be saved
    validate_destination_path(args.destination_path)
    if not args.destination_path:
        validate_destination_path(path.join(args.output_dir or DEFAULT_OUTPUT_DIR, "barcode.png"))

    batch = is_batch(args, inputs)
    if batch:
        logging.info("Processing %d inputs", len(inputs))
    runner = BatchRunner(args, inputs, batch)
    results = runner.run()
    if batch:
        runner.log_summary()
    return 1 if any(result.status == "failed" for result in results) else 0


def build_parser() -> argparse.ArgumentParser:
    """
    Command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate color barcodes from one or several video files.",
        epilog="Several movies: give several files or directories to -i (and/or --from-list). They are processed one "
        "after another, each with all the workers; a movie that fails is reported and the others are still "
        "processed. Images are saved in --output-dir (default ./barcodes), named "
        "<movie>_<method>_<type>[_workers_<n>].png.",
    )
    parser.add_argument(
        "-i",
        "--input_video_path",
        type=str,
        nargs="+",
        action="extend",
        default=None,
        metavar="INPUT",
        help="Video file(s) and/or directories of videos to process. Colors files (.npz) written by --save-colors are "
        "rendered without decoding the video. Required unless --from-list is given.",
    )
    parser.add_argument(
        "--from-list",
        type=str,
        default=None,
        metavar="FILE",
        help="Text file listing inputs, one path per line (relative paths are relative to the file; blank lines and "
        "lines starting with # are ignored). Can be combined with -i.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also look for videos in the subdirectories of the directories given as input.",
    )
    parser.add_argument(
        "-d",
        "--destination_path",
        type=str,
        nargs="?",
        help="Path to save the output image. If not provided, the image will be saved in a default location. With "
        "--all_methods, the method is appended to the file name (e.g. out_avg.png, out_kmeans.png), and with several "
        "barcode types the type. Single input only.",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory where the images (and colors files) are saved with their generated names. Default is "
        "./barcodes.",
    )
    parser.add_argument(
        "-t",
        "--barcode_type",
        dest="barcode_types",
        choices=list(BARCODE_TYPES),
        nargs="+",
        default=None,
        help="Type(s) of barcode to generate: horizontal and/or circular (e.g. -t horizontal circular, from a single "
        "decoding of the video). Default is horizontal.",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=list(METHODS),
        default=None,
        help="Method to extract dominant color: avg (average), kmeans (K-Means clustering), hsv (HSV histogram), "
        "bgr (BGR histogram) or smoothed version (averaging the colors with two-step resize). Default is avg (for a "
        "colors file: every method it contains).",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of workers for parallel processing. Default behavior uses all available CPU cores. Setting this "
        "to 1 will use sequential processing. Values above the number of CPU cores are capped to it (with a warning).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Number of frames sampled: the output image width for horizontal barcodes, the number of rings for "
        "circular barcodes. If not provided, defaults to the video width (capped by the number of frames).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of the output image. If not provided, the height will be the same as the video",
    )
    parser.add_argument(
        "-n",
        "--output_name",
        type=str,
        nargs="?",
        help="Custom name for the output barcode image. If not provided, a name will be automatically generated. "
        "With --all_methods, the method is appended (e.g. NAME_avg.png, NAME_kmeans.png), and with several barcode "
        "types the type. Single input only.",
        default=None,
    )
    parser.add_argument(
        "-a",
        "--all_methods",
        action="store_true",
        help="If provided, all methods to extract dominant color will be used to create barcodes. Overrides --method "
        "argument.",
    )
    parser.add_argument(
        "--crop-borders",
        action="store_true",
        help="If provided, detect black borders (letterbox / pillarbox) once and ignore them when extracting "
        "colors. Default is False.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Average colors in linear light (perceptually correct, brighter on high-contrast scenes). Applies to "
        "the avg and smoothed methods only; their output files get a '_linear' suffix after the method name. "
        "Default is False.",
    )
    parser.add_argument(
        "--save-colors",
        action="store_true",
        help="Also save the extracted colors of each video next to its images (<movie>_colors.npz), to render other "
        "barcodes later without decoding the video again: movie-barcodes -i <movie>_colors.npz -t circular.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip the inputs whose output files (images, and colors file with --save-colors) all exist already.",
    )
    return parser


def main() -> None:
    """
    Main function to generate barcodes from video files.
    """
    # Logging configuration
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    header_msg = "=" * 40 + " NEW RUN " + "=" * 40
    logging.info("%s", header_msg)

    parser = build_parser()
    args = parser.parse_args()

    exit_code = generate_and_save_barcodes(args, parser)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
