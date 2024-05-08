# Movie Barcodes
The Lodger: A Story of the London Fog (1927) - Alfred Hitchcock - [Public Domain](https://archive.org/details/TheLodgerAStoryOfTheLondonFog_579)

Circular Barcode           |  Horizontal Barcode
:-------------------------:|:-------------------------:
![](https://github.com/Wazzabeee/movie_color_barcode/blob/main/examples/thelodgerastoryofthelondonfog_circular.png?raw=true)  |  ![](https://github.com/Wazzabeee/movie_color_barcode/blob/main/examples/thelodgerastoryofthelondonfog_horizontal.png?raw=true)

![PyPI - Version](https://img.shields.io/pypi/v/movie-barcodes) ![PyPI - License](https://img.shields.io/pypi/l/movie-barcodes)
![Python](https://img.shields.io/badge/python-3.11-blue)

# Overview

Compress every frame of a movie in a single color barcode.

This project is a robust and highly configurable utility designed to extract dominant colors from video files and generate color barcodes. Built with Python and OpenCV, the tool offers multiple algorithms for color extraction, including average color, K-means clustering, and HSV/BGR histograms. The output can be generated in various forms, like horizontal and circular barcodes, providing a visually intuitive summary of the color distribution in the video.

Designed with performance in mind, the application supports both sequential and parallel processing. It scales automatically based on the available CPU cores but can be fine-tuned for a specified number of workers. This makes it suitable for analyzing both short clips and full-length movies with high efficiency.

# Features
- Horizontal and Circular Barcodes
- Fast frame skipping for efficiency.
- Multiprocessing support for parallel processing.
- Customizable color extraction function (Average or K-means).
- Progress tracking and estimated time remaining.

# Usage
```bash
# Install the package
$ pip install movie-barcodes

# Generate a movie barcode
$ movie-barcodes --input_video_path "path/to/video"
```

***Mandatory Arguments:***
- `input_video_path`: The path to the input video file. (Required, type: str)

***Optional Arguments:***
- `--destination_path`: The path where the output image will be saved. If not provided, defaults to a pre-defined location. (Optional, type: str)

- `--barcode_type`: The type of barcode to generate. Options are horizontal or circular. Default is horizontal. (Optional, type: str)

- `--method`: The algorithm for extracting the dominant color from frames. Options are avg (average), kmeans (K-Means clustering), hsv (HSV histogram), and bgr (BGR histogram). Default is avg. (Optional, type: str)

- `--workers`: Number of parallel workers for processing. By default, the script will use all available CPU cores. Setting this to 1 will use sequential processing. (Optional, type: int)

- `--width`: The output image's width in pixels. If not specified, the width will be the same as the input video. (Optional, type: int)

- `--output_name`: Custom name for the output barcode image. If not provided, a name will be automatically generated. (Optional, type: str)

- `--all_methods`: If set to True, all methods for color extraction will be employed, overriding the --method argument. Default is False. (Optional, type: bool)

# Examples
## Sequential Processing
```python
python -m src.main --input_video_path "path/to/video" --width 200 --workers 1
```
## Parallel Processing
```python
python -m src.main --input_video_path "path/to/video" --width 200 --workers 8
```

# Development Setup
```bash
# Clone this repository
$ git clone https://github.com/Wazzabeee/movie-barcodes

# Go into the repository
$ cd movie-barcodes

# Install requirements
$ pip install -r requirements.txt
$ pip install -r requirements_lint.txt

# Install precommit
$ pip install pre-commit
$ pre-commit install

# Run tests
$ pip install pytest
$ pytest tests/

# Run package locally
$ python -m src.main --input_video_path "path_to_video.mp4"
```

# Todo

- [ ] Optimize K-means to speed up the process
- [ ] Add a small GUI with all options available
- [ ] Add option to modify the barcode's height (current is frame's height)
- [ ] Ensure the software can handle various video formats beyond MP4
- [ ] Allow the software to process multiple videos at once
- [ ] Add examples to Readme
- [ ] Develop POC on Hugging Face Space

# More Examples
## Circular Barcodes
## Horizontal Barcodes
