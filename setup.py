#!/usr/bin/env python3
# Copyright © 2025 Manus AI

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Base requirements
install_requires = [
    "mlx>=0.0.1",
    "mlx-lm>=0.0.1", 
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "pydantic>=2.0.0",
    "numpy>=1.21.0",
    "transformers>=4.30.0",
    "requests>=2.28.0",
]

# Development requirements
dev_requires = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "coverage>=7.0.0",
]

# Performance requirements
perf_requires = [
    "psutil>=5.9.0",
    "memory-profiler>=0.60.0",
]

setup(
    name="mlx-speculative-server",
    version="0.1.0",
    author="Manus AI",
    author_email="contact@manus.im",
    description="High-performance speculative decoding server for Apple Silicon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlx-speculative-server",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/mlx-speculative-server/issues",
        "Documentation": "https://github.com/yourusername/mlx-speculative-server#readme",
        "Source Code": "https://github.com/yourusername/mlx-speculative-server",
    },
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "perf": perf_requires,
        "all": dev_requires + perf_requires,
    },
    entry_points={
        "console_scripts": [
            "mlx-speculative=mlx_speculative.cli:main",
            "mlx-spec-server=mlx_speculative.server_v2:run_enhanced_server",
        ],
    },
    include_package_data=True,
    package_data={
        "mlx_speculative": ["*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords=[
        "mlx",
        "apple-silicon",
        "llm",
        "language-model",
        "speculative-decoding",
        "inference",
        "serving",
        "vllm",
        "performance",
        "concurrent",
        "batching",
    ],
)
