from setuptools import setup, find_packages

setup(
    name="boolrl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "sympy>=1.12",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "torch_geometric>=2.3.0",
        "tqdm>=4.64.0",
    ],
    python_requires=">=3.8",
)