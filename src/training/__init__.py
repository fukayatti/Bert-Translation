"""
訓練モジュール

TAID蒸留とEfQAT量子化の実装を提供します。
"""

from .taid_distillation import TAIDDistillation
from .efqat_quantization import EfQATQuantization

__all__ = ["TAIDDistillation", "EfQATQuantization"]