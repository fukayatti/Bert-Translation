#!/usr/bin/env python3
"""
Teacherモデル学習スクリプト

BERT2BERTベースのTeacherモデルを学習するスクリプトです。
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
from src.models.teacher_model import TeacherModel

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
    parser = argparse.ArgumentParser(description='Teacherモデル学習')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/teacher',
        help='モデル保存ディレクトリ'
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
    
    # データセット読み込み
    logger.info("データセット読み込み開始...")
    dataset_loader = DatasetLoader(config['dataset'])
    dataset = dataset_loader.load_combined_dataset()
    
    # データセット情報表示
    dataset_info = dataset_loader.get_dataset_info(dataset)
    logger.info(f"データセット情報: {dataset_info}")
    
    # 前処理
    logger.info("データ前処理開始...")
    preprocessor = TextPreprocessor(config['tokenizer'])
    
    train_dataset = preprocessor.preprocess_dataset(dataset['train'])
    eval_dataset = preprocessor.preprocess_dataset(dataset['test'])
    
    # トークナイザー情報表示
    tokenizer_info = preprocessor.get_tokenizer_info()
    logger.info(f"トークナイザー情報: {tokenizer_info}")
    
    # Teacherモデル作成
    logger.info("Teacherモデル初期化...")
    teacher_config = config['teacher'].copy()
    teacher_config['output_dir'] = args.output_dir
    
    teacher_model = TeacherModel(teacher_config, preprocessor.tokenizer)
    
    # モデル情報表示
    model_info = teacher_model.get_model_info()
    logger.info(f"Teacherモデル情報: {model_info}")
    
    # 学習実行
    logger.info("Teacherモデル学習開始...")
    train_result = teacher_model.train(train_dataset, eval_dataset)
    
    # モデル保存
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    teacher_model.save_model(str(save_path))
    preprocessor.tokenizer.save_pretrained(str(save_path))
    
    logger.info(f"Teacherモデル保存完了: {save_path}")
    
    # 結果表示
    logger.info("=" * 50)
    logger.info("Teacherモデル学習完了")
    logger.info("=" * 50)
    logger.info(f"最終損失: {train_result.training_loss:.4f}")
    logger.info(f"保存先: {save_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()