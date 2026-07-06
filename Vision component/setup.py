from setuptools import setup, find_packages

setup(
    name="gender_detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "timm",
        "opencv-python",
        "PyYAML",
        "python-dotenv",
        "numpy",
        "scipy",
        "Pillow",
        "tqdm",
        "requests",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"],
    },
)
