"""
BERT英日翻訳モデル - セットアップスクリプト
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bert-en-ja-translation",
    version="1.0.0",
    author="BERT Translation Team",
    author_email="team@example.com",
    description="BERT英日翻訳モデル - TAID蒸留 + EfQAT量子化",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/bert-en-ja-translation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bert-train-teacher=scripts.train_teacher:main",
            "bert-distill-student=scripts.distill_student:main",
            "bert-quantize-model=scripts.quantize_model:main",
            "bert-evaluate-model=scripts.evaluate_model:main",
            "bert-pipeline=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
)