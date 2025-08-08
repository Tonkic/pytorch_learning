from setuptools import setup, find_packages

setup(
    name="pytorch_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "pandas",
        "pillow",
        "tqdm",
    ],
    python_requires=">=3.8",
)
