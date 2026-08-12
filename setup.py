"""Setup script for content moderation system."""

from setuptools import setup, find_packages

setup(
    name="content-moderation-system",
    version="0.1.0",
    description="Multilingual Content Moderation System with Human-in-the-Loop",
    author="Research Project",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        'console_scripts': [
            'train-classifier=phase1_text_baseline.train_classifier:main',
            'calibrate-model=phase1_text_baseline.calibrate_thresholds:main',
        ],
    },
)
