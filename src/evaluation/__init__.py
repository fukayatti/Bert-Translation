"""
評価モジュール

BLEU、BERTScoreなどの評価指標の実装を提供します。
"""

from .metrics import ModelEvaluator

__all__ = ["ModelEvaluator"]