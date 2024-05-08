import subprocess
from setuptools import setup, find_packages


def get_version():
    return subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"]).strip().decode("utf-8")[1:]


setup(
    name="movie-barcodes",
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        "colorama==0.4.6",
        "joblib==1.3.2",
        "numpy==1.26.0",
        "opencv-python==4.8.0.76",
        "Pillow==10.3.0",
        "scikit-learn==1.3.1",
        "scipy==1.11.2",
        "threadpoolctl==3.2.0",
        "tqdm==4.66.3",
    ],
    extras_require={
        "lint": [
            "pylint==2.6.0",
            "mypy==1.6.0",
            "flake8==4.0.0",
            "black==24.3.0",
            "types-Pillow",
            "types-pytz",
            "types-tqdm",
            "types-PyYAML",
        ],
        "dev": ["pytest", "pre-commit"],
    },
    author="Clément Delteil",
    author_email="clement45.delteil45@gmail.com",
    description="Compress every frame of a movie in a single color barcode."
    "Transform entire movies into stunning single-barcode visualizations."
    "Capture the essence of cinematic storytelling through dominant color extraction from each frame.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Wazzabeee/movie-barcodes",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "movie-barcodes=src.main:main",
        ],
    },
)
