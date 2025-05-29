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
import traceback
import os
import psutil
from typing import Optional, Dict, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from src.data.dataset_loader import DatasetLoader
    from src.utils.preprocessing import TextPreprocessor
    from src.models.teacher_model import TeacherModel
    from src.models.student_model import StudentModel
    from src.training.taid_distillation import TAIDDistillation
    from src.training.efqat_quantization import EfQATQuantization
    from src.evaluation.metrics import ModelEvaluator
except ImportError as e:
    print(f"重要なモジュールのインポートに失敗しました: {e}")
    print("requirements.txtの依存関係がインストールされているか確認してください")
    sys.exit(1)

# ログ設定
def setup_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
    """ロギングを設定"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # ログディレクトリの作成
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ログファイル設定
    log_file = log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # フォーマッター設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ルートロガー設定
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # ファイルハンドラー
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # ハンドラー追加
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"ログ設定完了: {log_file}")
    return logger

logger = logging.getLogger(__name__)


def log_system_resources():
    """システムリソース使用状況をログ出力"""
    try:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # メモリ使用状況
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_used_gb = memory.used / (1024**3)
        memory_percent = memory.percent
        
        # ディスク使用状況
        disk = psutil.disk_usage('/')
        disk_gb = disk.total / (1024**3)
        disk_used_gb = disk.used / (1024**3)
        disk_percent = (disk.used / disk.total) * 100
        
        logger.info("システムリソース使用状況:")
        logger.info(f"  CPU使用率: {cpu_percent:.1f}%")
        logger.info(f"  メモリ使用量: {memory_used_gb:.1f}GB / {memory_gb:.1f}GB ({memory_percent:.1f}%)")
        logger.info(f"  ディスク使用量: {disk_used_gb:.1f}GB / {disk_gb:.1f}GB ({disk_percent:.1f}%)")
        
        # GPUメモリ（利用可能な場合）
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"  GPU{i} メモリ: {memory_allocated:.2f}GB使用 / {memory_reserved:.2f}GB予約")
    
    except Exception as e:
        logger.warning(f"システムリソース監視でエラー: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """設定ファイルの妥当性を検証"""
    required_sections = ['dataset', 'tokenizer', 'teacher', 'student', 'distillation', 'quantization', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"必須設定セクション '{section}' が見つかりません")
            return False
    
    # データセット設定の検証
    dataset_config = config['dataset']
    if not all(key in dataset_config for key in ['jpara_samples', 'jesc_samples']):
        logger.error("データセット設定が不完全です")
        return False
    
    # 学習率などの数値パラメータの検証
    try:
        # Student学習率の検証
        student_lr = config['student'].get('learning_rate', 0)
        if isinstance(student_lr, str):
            student_lr = float(student_lr)
        if student_lr <= 0:
            logger.error("Student学習率は正の値である必要があります")
            return False
            
        # 量子化学習率の検証
        quant_lr = config['quantization'].get('learning_rate', 0)
        if isinstance(quant_lr, str):
            quant_lr = float(quant_lr)
        if quant_lr <= 0:
            logger.error("量子化学習率は正の値である必要があります")
            return False
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"設定パラメータの検証エラー: {e}")
        return False
    
    return True


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """設定ファイルを読み込みと検証"""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"設定ファイルが見つかりません: {config_path}")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            logger.error("設定ファイルが空です")
            return None
        
        if not validate_config(config):
            logger.error("設定ファイルの検証に失敗しました")
            return None
        
        logger.info("設定ファイルの検証完了")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"YAML解析エラー: {e}")
        return None
    except Exception as e:
        logger.error(f"設定ファイル読み込みエラー: {e}")
        return None


def save_pipeline_info(info: dict, output_path: Path):
    """パイプライン情報を保存"""
    info_path = output_path / 'pipeline_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logger.info(f"パイプライン情報保存: {info_path}")

def setup_device(device_arg: str) -> torch.device:
    """デバイスを安全に設定"""
    try:
        if device_arg == 'auto':
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"CUDA利用可能: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("CUDA利用不可、CPUを使用します")
        else:
            device = torch.device(device_arg)
            if device.type == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDAが指定されましたが利用できません。CPUに切り替えます")
                device = torch.device("cpu")
        
        # デバイスのテスト
        test_tensor = torch.randn(1).to(device)
        logger.info(f"デバイステスト成功: {device}")
        return device
        
    except Exception as e:
        logger.error(f"デバイス設定エラー: {e}")
        logger.info("CPUにフォールバックします")
        return torch.device("cpu")


def cleanup_on_error(output_path: Path, pipeline_info: Dict[str, Any], error: Exception):
    """エラー時のクリーンアップ処理"""
    try:
        pipeline_info['error'] = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc()
        }
        pipeline_info['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save_pipeline_info(pipeline_info, output_path)
        
        # GPUメモリのクリアアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU メモリをクリアしました")
            
    except Exception as cleanup_error:
        logger.error(f"クリーンアップ処理エラー: {cleanup_error}")

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
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='ログレベル'
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
    
    # 出力ディレクトリを早期に作成
    try:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"出力ディレクトリの作成に失敗: {e}")
        sys.exit(1)
    
    # ログ設定
    global logger
    logger = setup_logging(output_path, args.log_level)
    
    # パイプライン開始時刻
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("BERT英日翻訳モデル - TAID蒸留 + EfQAT量子化")
    logger.info("=" * 60)
    
    # システム情報のログ出力
    logger.info(f"Python バージョン: {sys.version}")
    logger.info(f"PyTorch バージョン: {torch.__version__}")
    logger.info(f"CUDA 利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA デバイス数: {torch.cuda.device_count()}")
    
    # システムリソース監視
    log_system_resources()
    
    # デバイス設定
    device = setup_device(args.device)
    logger.info(f"使用デバイス: {device}")
    
    # モデル保存パス
    teacher_path = output_path / 'teacher'
    student_path = output_path / 'student'
    quantized_path = output_path / 'quantized'
    results_path = output_path / 'results'
    
    # 設定読み込み
    logger.info(f"設定ファイル読み込み: {args.config}")
    config = load_config(args.config)
    if config is None:
        logger.error("設定ファイルの読み込みに失敗しました")
        sys.exit(1)
    
    # パイプライン情報初期化
    pipeline_info = {
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        'args': vars(args),
        'config': config,
        'device': str(device),
        'output_dir': str(output_path),
        'system_info': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        'steps_completed': [],
        'execution_times': {}
    }
    
    try:
        # ステップ1: データセット読み込み
        logger.info("\n" + "="*50)
        logger.info("ステップ1: データセット読み込み")
        logger.info("="*50)
        
        step_start = time.time()
        try:
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
            
        except Exception as e:
            logger.error(f"データセット読み込みでエラーが発生: {e}")
            cleanup_on_error(output_path, pipeline_info, e)
            raise
        
        # ステップ2: Teacher学習
        if not args.skip_teacher:
            logger.info("\n" + "="*50)
            logger.info("ステップ2: Teacherモデル学習")
            logger.info("="*50)
            
            step_start = time.time()
            try:
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
                
            except Exception as e:
                logger.error(f"Teacher学習でエラーが発生: {e}")
                cleanup_on_error(output_path, pipeline_info, e)
                raise
            
        else:
            logger.info("Teacher学習をスキップしました")
            try:
                # 既存のTeacherモデルを読み込み
                if not teacher_path.exists():
                    logger.error(f"Teacherモデルが見つかりません: {teacher_path}")
                    raise FileNotFoundError(f"Teacherモデルが見つかりません: {teacher_path}")
                
                teacher_model = TeacherModel(config['teacher'], preprocessor.tokenizer)
                teacher_model.load_model(str(teacher_path))
                logger.info("既存のTeacherモデルを読み込みました")
                
            except Exception as e:
                logger.error(f"Teacherモデル読み込みでエラーが発生: {e}")
                cleanup_on_error(output_path, pipeline_info, e)
                raise
        
        # ステップ3: Student蒸留
        if not args.skip_distillation:
            logger.info("\n" + "="*50)
            logger.info("ステップ3: TAID蒸留")
            logger.info("="*50)
            
            step_start = time.time()
            try:
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
                
            except Exception as e:
                logger.error(f"蒸留処理でエラーが発生: {e}")
                cleanup_on_error(output_path, pipeline_info, e)
                raise
            
        else:
            logger.info("蒸留をスキップしました")
            try:
                # 既存のStudentモデルを読み込み
                if not student_path.exists():
                    logger.error(f"Studentモデルが見つかりません: {student_path}")
                    raise FileNotFoundError(f"Studentモデルが見つかりません: {student_path}")
                
                student_model = StudentModel(config['student'], preprocessor.tokenizer, device)
                student_model.load_model(str(student_path))
                logger.info("既存のStudentモデルを読み込みました")
                
            except Exception as e:
                logger.error(f"Studentモデル読み込みでエラーが発生: {e}")
                cleanup_on_error(output_path, pipeline_info, e)
                raise
        
        # ステップ4: EfQAT量子化
        if not args.skip_quantization:
            logger.info("\n" + "="*50)
            logger.info("ステップ4: EfQAT量子化")
            logger.info("="*50)
            
            step_start = time.time()
            try:
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
                
            except Exception as e:
                logger.error(f"量子化処理でエラーが発生: {e}")
                cleanup_on_error(output_path, pipeline_info, e)
                raise
            
        else:
            logger.info("量子化をスキップしました")
            try:
                # 既存の量子化モデルを読み込み
                if not quantized_path.exists():
                    logger.error(f"量子化モデルが見つかりません: {quantized_path}")
                    raise FileNotFoundError(f"量子化モデルが見つかりません: {quantized_path}")
                
                student_model.load_model(str(quantized_path))
                logger.info("既存の量子化モデルを読み込みました")
                
            except Exception as e:
                logger.error(f"量子化モデル読み込みでエラーが発生: {e}")
                cleanup_on_error(output_path, pipeline_info, e)
                raise
        
        # ステップ5: 評価
        if not args.skip_evaluation:
            logger.info("\n" + "="*50)
            logger.info("ステップ5: モデル評価")
            logger.info("="*50)
            
            step_start = time.time()
            try:
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
                
            except Exception as e:
                logger.error(f"評価処理でエラーが発生: {e}")
                cleanup_on_error(output_path, pipeline_info, e)
                raise
        
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
        
    except KeyboardInterrupt:
        logger.warning("ユーザーによってパイプラインが中断されました")
        pipeline_info['status'] = 'interrupted'
        pipeline_info['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        cleanup_on_error(output_path, pipeline_info, KeyboardInterrupt("ユーザー中断"))
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"パイプライン実行エラー: {e}")
        logger.error(f"詳細なトレースバック:\n{traceback.format_exc()}")
        cleanup_on_error(output_path, pipeline_info, e)
        sys.exit(1)


if __name__ == "__main__":
    main()