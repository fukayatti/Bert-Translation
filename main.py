#!/usr/bin/env python3
"""
BERT英日翻訳モデル - メインスクリプト

TAID蒸留 + EfQAT量子化の完全なパイプラインを実行するスクリプトです。
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path
import sys
import time
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data.dataset_loader import DatasetLoader
from src.utils.preprocessing import TextPreprocessor
from src.models.teacher_model import TeacherModel
from src.models.student_model import StudentModel
from src.training.taid_distillation import TAIDDistillation
from src.training.efqat_quantization import EfQATQuantization
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


def save_pipeline_info(info: dict, output_path: Path):
    """パイプライン情報を保存"""
    info_path = output_path / 'pipeline_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logger.info(f"パイプライン情報保存: {info_path}")


def main():
    parser = argparse.ArgumentParser(description='BERT英日翻訳モデル - 完全パイプライン')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='使用デバイス (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--skip-teacher',
        action='store_true',
        help='Teacher学習をスキップ'
    )
    parser.add_argument(
        '--skip-distillation',
        action='store_true',
        help='蒸留をスキップ'
    )
    parser.add_argument(
        '--skip-quantization',
        action='store_true',
        help='量子化をスキップ'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='評価をスキップ'
    )
    
    args = parser.parse_args()
    
    # パイプライン開始時刻
    start_time = time.time()
    
    # デバイス設定
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info("=" * 60)
    logger.info("BERT英日翻訳モデル - TAID蒸留 + EfQAT量子化")
    logger.info("=" * 60)
    logger.info(f"使用デバイス: {device}")
    
    # 出力ディレクトリ作成
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # モデル保存パス
    teacher_path = output_path / 'teacher'
    student_path = output_path / 'student'
    quantized_path = output_path / 'quantized'
    results_path = output_path / 'results'
    
    # 設定読み込み
    logger.info(f"設定ファイル読み込み: {args.config}")
    config = load_config(args.config)
    
    # パイプライン情報初期化
    pipeline_info = {
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        'config': config,
        'device': str(device),
        'output_dir': str(output_path),
        'steps_completed': [],
        'execution_times': {}
    }
    
    try:
        # ステップ1: データセット読み込み
        logger.info("\n" + "="*50)
        logger.info("ステップ1: データセット読み込み")
        logger.info("="*50)
        
        step_start = time.time()
        dataset_loader = DatasetLoader(config['dataset'])
        dataset = dataset_loader.load_combined_dataset()
        
        # データセット情報表示
        dataset_info = dataset_loader.get_dataset_info(dataset)
        logger.info(f"データセット情報: {dataset_info}")
        
        # 前処理
        preprocessor = TextPreprocessor(config['tokenizer'])
        train_dataset = preprocessor.preprocess_dataset(dataset['train'])
        test_dataset = dataset['test']  # 評価用は前処理前を使用
        
        pipeline_info['execution_times']['data_loading'] = time.time() - step_start
        pipeline_info['steps_completed'].append('data_loading')
        pipeline_info['dataset_info'] = dataset_info
        
        # ステップ2: Teacher学習
        if not args.skip_teacher:
            logger.info("\n" + "="*50)
            logger.info("ステップ2: Teacherモデル学習")
            logger.info("="*50)
            
            step_start = time.time()
            teacher_config = config['teacher'].copy()
            teacher_config['output_dir'] = str(teacher_path)
            
            teacher_model = TeacherModel(teacher_config, preprocessor.tokenizer)
            teacher_info = teacher_model.get_model_info()
            logger.info(f"Teacherモデル情報: {teacher_info}")
            
            # 学習実行
            eval_dataset = preprocessor.preprocess_dataset(dataset['test'])
            train_result = teacher_model.train(train_dataset, eval_dataset)
            
            # モデル保存
            teacher_path.mkdir(parents=True, exist_ok=True)
            teacher_model.save_model(str(teacher_path))
            preprocessor.tokenizer.save_pretrained(str(teacher_path))
            
            pipeline_info['execution_times']['teacher_training'] = time.time() - step_start
            pipeline_info['steps_completed'].append('teacher_training')
            pipeline_info['teacher_info'] = teacher_info
            
        else:
            logger.info("Teacher学習をスキップしました")
            # 既存のTeacherモデルを読み込み
            teacher_model = TeacherModel(config['teacher'], preprocessor.tokenizer)
            teacher_model.load_model(str(teacher_path))
        
        # ステップ3: Student蒸留
        if not args.skip_distillation:
            logger.info("\n" + "="*50)
            logger.info("ステップ3: TAID蒸留")
            logger.info("="*50)
            
            step_start = time.time()
            student_model = StudentModel(config['student'], preprocessor.tokenizer, device)
            student_info = student_model.get_model_info()
            logger.info(f"Studentモデル情報: {student_info}")
            
            # TAID蒸留実行
            distillation_config = config['distillation'].copy()
            distillation_config['learning_rate'] = config['student']['learning_rate']
            
            taid_distillation = TAIDDistillation(
                distillation_config, teacher_model, student_model, device
            )
            
            distillation_history = taid_distillation.distill(train_dataset)
            
            # モデル保存
            student_path.mkdir(parents=True, exist_ok=True)
            student_model.save_model(str(student_path))
            
            # 履歴保存
            with open(student_path / 'distillation_history.json', 'w') as f:
                json.dump(distillation_history, f, indent=2)
            
            pipeline_info['execution_times']['distillation'] = time.time() - step_start
            pipeline_info['steps_completed'].append('distillation')
            pipeline_info['student_info'] = student_info
            
        else:
            logger.info("蒸留をスキップしました")
            # 既存のStudentモデルを読み込み
            student_model = StudentModel(config['student'], preprocessor.tokenizer, device)
            student_model.load_model(str(student_path))
        
        # ステップ4: EfQAT量子化
        if not args.skip_quantization:
            logger.info("\n" + "="*50)
            logger.info("ステップ4: EfQAT量子化")
            logger.info("="*50)
            
            step_start = time.time()
            efqat_quantization = EfQATQuantization(
                config['quantization'], student_model, device
            )
            
            quantization_history = efqat_quantization.quantize(train_dataset)
            quantization_info = efqat_quantization.get_quantization_info()
            
            # 量子化モデル保存
            quantized_path.mkdir(parents=True, exist_ok=True)
            student_model.save_model(str(quantized_path))
            
            # 情報保存
            with open(quantized_path / 'quantization_history.json', 'w') as f:
                json.dump(quantization_history, f, indent=2)
            with open(quantized_path / 'quantization_info.json', 'w') as f:
                json.dump(quantization_info, f, indent=2)
            
            pipeline_info['execution_times']['quantization'] = time.time() - step_start
            pipeline_info['steps_completed'].append('quantization')
            pipeline_info['quantization_info'] = quantization_info
            
        else:
            logger.info("量子化をスキップしました")
            # 既存の量子化モデルを読み込み
            student_model.load_model(str(quantized_path))
        
        # ステップ5: 評価
        if not args.skip_evaluation:
            logger.info("\n" + "="*50)
            logger.info("ステップ5: モデル評価")
            logger.info("="*50)
            
            step_start = time.time()
            evaluator = ModelEvaluator(config['evaluation'], preprocessor.tokenizer, device)
            
            # 比較評価実行
            comparison_results = evaluator.compare_models(
                teacher_model, student_model, test_dataset
            )
            
            # 結果保存
            results_path.mkdir(parents=True, exist_ok=True)
            with open(results_path / 'evaluation_results.json', 'w') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            
            pipeline_info['execution_times']['evaluation'] = time.time() - step_start
            pipeline_info['steps_completed'].append('evaluation')
            pipeline_info['evaluation_results'] = comparison_results
        
        # パイプライン完了
        total_time = time.time() - start_time
        pipeline_info['total_execution_time'] = total_time
        pipeline_info['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # パイプライン情報保存
        save_pipeline_info(pipeline_info, output_path)
        
        # 結果サマリー表示
        logger.info("\n" + "="*60)
        logger.info("パイプライン完了")
        logger.info("="*60)
        logger.info(f"総実行時間: {total_time/60:.1f}分")
        logger.info(f"完了ステップ: {', '.join(pipeline_info['steps_completed'])}")
        logger.info(f"出力ディレクトリ: {output_path}")
        
        if 'evaluation_results' in pipeline_info:
            comp = pipeline_info['evaluation_results']['comparison']
            logger.info(f"性能保持率 - BLEU: {comp['performance_retention']['bleu']:.1%}, "
                       f"BERTScore: {comp['performance_retention']['bert_f1']:.1%}")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"パイプライン実行エラー: {e}")
        pipeline_info['error'] = str(e)
        pipeline_info['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save_pipeline_info(pipeline_info, output_path)
        raise


if __name__ == "__main__":
    main()