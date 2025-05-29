#!/usr/bin/env python3
"""
BERT英日翻訳モデル - セットアップスクリプト
本番環境対応版
"""

from setuptools import setup, find_packages
import os

# README.mdを読み込み
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "BERT英日翻訳モデル - 本番環境対応版"

# requirements.txtを読み込み
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        return []

setup(
    name="bert-en-ja-translation",
    version="1.0.0",
    description="BERT英日翻訳モデル - TAID蒸留 + EfQAT量子化（本番環境対応版）",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="BERT Translation Team",
    author_email="contact@example.com",
    url="https://github.com/your-org/bert-en-ja-translation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
            "coverage>=7.0.0",
        ],
        "monitoring": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
        "full": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
            "coverage>=7.0.0",
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bert-translation=main:main",
            "bert-train-teacher=scripts.train_teacher:main",
            "bert-distill-student=scripts.distill_student:main",
            "bert-quantize=scripts.quantize_model:main",
            "bert-evaluate=scripts.evaluate_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md", "*.txt"],
    },
    keywords=[
        "bert",
        "translation",
        "english",
        "japanese",
        "distillation",
        "quantization",
        "transformer",
        "nlp",
        "deep learning",
        "production"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/bert-en-ja-translation/issues",
        "Source": "https://github.com/your-org/bert-en-ja-translation",
        "Documentation": "https://github.com/your-org/bert-en-ja-translation/wiki",
    },
    zip_safe=False,
)