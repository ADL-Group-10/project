"""Editable install for the project. Run once: `pip install -e .`"""

from setuptools import setup, find_packages

setup(
    name        = "nvd-snow-yolov9",
    version     = "0.1.0",
    description = "Car detection in snow — YOLOv9 + snow-aware augmentation on NVD",
    packages    = find_packages(),
    python_requires = ">=3.10",
    install_requires = [
        "omegaconf>=2.3",
        "torch>=2.0",
        "torchvision>=0.15",
        "albumentations>=1.3",
        "optuna>=3.6",
        "wandb>=0.16",
        "numpy>=1.24",
        "opencv-python>=4.8",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "tqdm>=4.65",
    ],
)
