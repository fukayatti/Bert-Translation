#!/usr/bin/env python3
"""
モデル量子化スクリプト

EfQAT量子化を使用してStudentモデルを量子化するスクリプトです。
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset_loader import DatasetLoader
from src.utils.preprocessing import TextPreprocessor
from src.models.student_model import StudentModel
from src.training.efqat_quantization import EfQATQuantization

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='モデル量子化')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--student-model',
        type=str,
        default='models/student',
        help='Studentモデルのパス'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/quantized',
        help='量子化モデル保存ディレクトリ'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='使用デバイス (cuda/cpu/auto)'
    )
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用デバイス: {device}")
    
    # 設定読み込み
    logger.info(f"設定ファイル読み込み: {args.config}")
    config = load_config(args.config)
    
    # データセット読み込み（量子化用）
    logger.info("データセット読み込み開始...")
    dataset_loader = DatasetLoader(config['dataset'])
    dataset = dataset_loader.load_combined_dataset()
    
    # 前処理
    logger.info("データ前処理開始...")
    preprocessor = TextPreprocessor(config['tokenizer'])
    train_dataset = preprocessor.preprocess_dataset(dataset['train'])
    
    # Studentモデル読み込み
    logger.info(f"Studentモデル読み込み: {args.student_model}")
    student_model = StudentModel(config['student'], preprocessor.tokenizer, device)
    student_model.load_model(args.student_model)
    
    # モデル情報表示（量子化前）
    student_info_before = student_model.get_model_info()
    logger.info(f"量子化前Studentモデル情報: {student_info_before}")
    
    # EfQAT量子化設定
    logger.info("EfQAT量子化初期化...")
    efqat_quantization = EfQATQuantization(
        config['quantization'],
        student_model,
        device
    )
    
    # 量子化情報表示
    quantization_info = efqat_quantization.get_quantization_info()
    logger.info(f"EfQAT量子化情報: {quantization_info}")
    
    # 量子化実行
    logger.info("EfQAT量子化開始...")
    history = efqat_quantization.quantize(train_dataset)
    
    # モデル情報表示（量子化後）
    student_info_after = student_model.get_model_info()
    logger.info(f"量子化後Studentモデル情報: {student_info_after}")
    
    # モデル保存
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    student_model.save_model(str(save_path))
    preprocessor.tokenizer.save_pretrained(str(save_path))
    
    # 量子化履歴保存
    import json
    history_path = save_path / 'quantization_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    # 量子化情報保存
    quantization_info_after = efqat_quantization.get_quantization_info()
    info_path = save_path / 'quantization_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(quantization_info_after, f, indent=2, ensure_ascii=False)
    
    logger.info(f"量子化モデル保存完了: {save_path}")
    
    # 結果表示
    logger.info("=" * 50)
    logger.info("モデル量子化完了")
    logger.info("=" * 50)
    logger.info(f"最終損失: {history['losses'][-1]:.4f}")
    logger.info(f"圧縮率: {quantization_info_after['compression_ratio']:.1%}")
    logger.info(f"量子化: {quantization_info_after['quantization']['weights']}重み / {quantization_info_after['quantization']['activations']}活性化")
    logger.info(f"保存先: {save_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()