#!/usr/bin/env python3
"""
Student蒸留スクリプト

TAID蒸留を使用してStudentモデルを学習するスクリプトです。
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
from src.models.student_model import StudentModel
from src.training.taid_distillation import TAIDDistillation

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
    parser = argparse.ArgumentParser(description='Student蒸留学習')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--teacher-model',
        type=str,
        default='models/teacher',
        help='Teacherモデルのパス'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/student',
        help='Studentモデル保存ディレクトリ'
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
    
    # 前処理
    logger.info("データ前処理開始...")
    preprocessor = TextPreprocessor(config['tokenizer'])
    train_dataset = preprocessor.preprocess_dataset(dataset['train'])
    
    # Teacherモデル読み込み
    logger.info(f"Teacherモデル読み込み: {args.teacher_model}")
    teacher_model = TeacherModel(config['teacher'], preprocessor.tokenizer)
    teacher_model.load_model(args.teacher_model)
    
    # Teacherモデル情報表示
    teacher_info = teacher_model.get_model_info()
    logger.info(f"Teacherモデル情報: {teacher_info}")
    
    # Studentモデル作成
    logger.info("Studentモデル初期化...")
    student_model = StudentModel(config['student'], preprocessor.tokenizer, device)
    
    # Studentモデル情報表示
    student_info = student_model.get_model_info()
    logger.info(f"Studentモデル情報: {student_info}")
    
    # TAID蒸留設定
    logger.info("TAID蒸留初期化...")
    distillation_config = config['distillation'].copy()
    distillation_config['learning_rate'] = config['student']['learning_rate']
    
    taid_distillation = TAIDDistillation(
        distillation_config,
        teacher_model,
        student_model,
        device
    )
    
    # 蒸留情報表示
    distillation_info = taid_distillation.get_distillation_info()
    logger.info(f"TAID蒸留情報: {distillation_info}")
    
    # 蒸留実行
    logger.info("TAID蒸留開始...")
    history = taid_distillation.distill(train_dataset)
    
    # モデル保存
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    student_model.save_model(str(save_path))
    preprocessor.tokenizer.save_pretrained(str(save_path))
    
    # 学習履歴保存
    import json
    history_path = save_path / 'distillation_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Studentモデル保存完了: {save_path}")
    
    # 結果表示
    logger.info("=" * 50)
    logger.info("Student蒸留完了")
    logger.info("=" * 50)
    logger.info(f"最終損失: {history['losses'][-1]:.4f}")
    logger.info(f"最終α: {history['alphas'][-1]:.4f}")
    logger.info(f"保存先: {save_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()