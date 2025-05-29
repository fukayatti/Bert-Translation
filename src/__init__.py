"""
BERT英日翻訳モデル - TAID蒸留 + EfQAT量子化

このパッケージは、BERT2BERTからTinyBERT-4Lへの蒸留とEfQAT量子化を使用した
英日翻訳モデルの実装を提供します。
"""

__version__ = "1.0.0"
__author__ = "BERT Translation Team"
__email__ = "team@example.com"

from .data import dataset_loader
from .models import teacher_model, student_model
from .training import taid_distillation, efqat_quantization
from .evaluation import metrics
from .utils import preprocessing

__all__ = [
    "dataset_loader",
    "teacher_model", 
    "student_model",
    "taid_distillation",
    "efqat_quantization",
    "metrics",
    "preprocessing"
]