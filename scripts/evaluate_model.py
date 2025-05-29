#!/usr/bin/env python3
"""
モデル評価スクリプト

TeacherモデルとStudentモデルの性能を評価するスクリプトです。
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path
import sys
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset_loader import DatasetLoader
from src.utils.preprocessing import TextPreprocessor
from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.evaluation.metrics import ModelEvaluator

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
    parser = argparse.ArgumentParser(description='モデル評価')
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
        '--student-model',
        type=str,
        default='models/quantized',
        help='Studentモデルのパス'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='評価結果保存ディレクトリ'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='使用デバイス (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='TeacherとStudentを比較評価'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['teacher', 'student', 'both'],
        default='both',
        help='評価するモデルタイプ'
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
    logger.info("テストデータセット読み込み開始...")
    dataset_loader = DatasetLoader(config['dataset'])
    dataset = dataset_loader.load_combined_dataset()
    
    # 前処理
    logger.info("データ前処理開始...")
    preprocessor = TextPreprocessor(config['tokenizer'])
    test_dataset = dataset['test']  # 前処理前のデータセットを使用（評価時に個別に処理）
    
    # 評価器初期化
    logger.info("評価器初期化...")
    evaluator = ModelEvaluator(config['evaluation'], preprocessor.tokenizer, device)
    
    # 結果保存ディレクトリ作成
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    teacher_model = None
    student_model = None
    
    # Teacherモデル読み込み・評価
    if args.model_type in ['teacher', 'both']:
        logger.info(f"Teacherモデル読み込み: {args.teacher_model}")
        try:
            teacher_model = TeacherModel(config['teacher'], preprocessor.tokenizer)
            teacher_model.load_model(args.teacher_model)
            logger.info("Teacherモデル評価開始...")
            teacher_results = evaluator.evaluate_model(teacher_model, test_dataset, "Teacher")
            
            # 結果保存
            teacher_results_path = output_path / 'teacher_evaluation.json'
            with open(teacher_results_path, 'w', encoding='utf-8') as f:
                json.dump(teacher_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Teacher評価結果保存: {teacher_results_path}")
            
        except Exception as e:
            logger.error(f"Teacherモデル評価エラー: {e}")
            teacher_model = None
    
    # Studentモデル読み込み・評価
    if args.model_type in ['student', 'both']:
        logger.info(f"Studentモデル読み込み: {args.student_model}")
        try:
            student_model = StudentModel(config['student'], preprocessor.tokenizer, device)
            student_model.load_model(args.student_model)
            logger.info("Studentモデル評価開始...")
            student_results = evaluator.evaluate_model(student_model, test_dataset, "Student")
            
            # 結果保存
            student_results_path = output_path / 'student_evaluation.json'
            with open(student_results_path, 'w', encoding='utf-8') as f:
                json.dump(student_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Student評価結果保存: {student_results_path}")
            
        except Exception as e:
            logger.error(f"Studentモデル評価エラー: {e}")
            student_model = None
    
    # 比較評価
    if args.compare and teacher_model and student_model:
        logger.info("モデル比較評価開始...")
        comparison_results = evaluator.compare_models(teacher_model, student_model, test_dataset)
        
        # 比較結果保存
        comparison_path = output_path / 'model_comparison.json'
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        logger.info(f"比較評価結果保存: {comparison_path}")
        
        # サマリーレポート作成
        create_summary_report(comparison_results, output_path / 'evaluation_summary.txt')
    
    logger.info("=" * 50)
    logger.info("モデル評価完了")
    logger.info("=" * 50)
    logger.info(f"結果保存先: {output_path}")
    logger.info("=" * 50)


def create_summary_report(comparison_results: dict, output_path: Path):
    """評価サマリーレポートを作成"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("BERT英日翻訳モデル評価レポート\n")
        f.write("=" * 50 + "\n\n")
        
        # Teacher結果
        teacher = comparison_results['teacher']
        f.write("■ Teacherモデル (BERT2BERT)\n")
        f.write(f"  BLEU Score: {teacher['bleu_score']:.2f}\n")
        f.write(f"  BERTScore F1: {teacher['bert_score']['f1']:.4f}\n")
        f.write(f"  サンプル数: {teacher['sample_size']}\n\n")
        
        # Student結果
        student = comparison_results['student']
        f.write("■ Studentモデル (TinyBERT-4L + W4A8量子化)\n")
        f.write(f"  BLEU Score: {student['bleu_score']:.2f}\n")
        f.write(f"  BERTScore F1: {student['bert_score']['f1']:.4f}\n")
        f.write(f"  サンプル数: {student['sample_size']}\n\n")
        
        # 比較結果
        comp = comparison_results['comparison']
        f.write("■ 性能比較\n")
        f.write(f"  BLEU劣化: {comp['bleu_degradation']:.2f}\n")
        f.write(f"  BERTScore F1劣化: {comp['bert_f1_degradation']:.4f}\n")
        f.write(f"  BLEU保持率: {comp['performance_retention']['bleu']:.1%}\n")
        f.write(f"  BERTScore F1保持率: {comp['performance_retention']['bert_f1']:.1%}\n\n")
        
        # サンプル予測
        f.write("■ 予測サンプル\n")
        for i in range(min(3, len(teacher['sample_predictions']))):
            f.write(f"サンプル {i+1}:\n")
            f.write(f"  参照: {teacher['sample_references'][i]}\n")
            f.write(f"  Teacher: {teacher['sample_predictions'][i]}\n")
            f.write(f"  Student: {student['sample_predictions'][i]}\n\n")


if __name__ == "__main__":
    main()