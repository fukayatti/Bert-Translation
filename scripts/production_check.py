#!/usr/bin/env python3
"""
本番環境用品質チェックスクリプト

このスクリプトは本番環境での実行前に必要なチェックを行います。
"""

import sys
import os
import yaml
import torch
import psutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Pythonバージョンをチェック"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} - Python 3.8以上が必要です"

def check_dependencies() -> Tuple[bool, str]:
    """依存関係をチェック"""
    try:
        import torch
        import transformers
        import datasets
        import sacrebleu
        import bert_score
        return True, "すべての必須依存関係が利用可能 ✓"
    except ImportError as e:
        return False, f"依存関係エラー: {e}"

def check_gpu_availability() -> Tuple[bool, str]:
    """GPU利用可能性をチェック"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        return True, f"GPU利用可能: {gpu_count}台 - {', '.join(gpu_names)} ✓"
    else:
        return False, "GPU利用不可 - CPUで実行されます"

def check_system_resources() -> Tuple[bool, str]:
    """システムリソースをチェック"""
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    issues = []
    if memory_gb < 8:
        issues.append(f"メモリ不足: {memory_gb:.1f}GB (推奨: 8GB以上)")
    if cpu_count < 4:
        issues.append(f"CPU不足: {cpu_count}コア (推奨: 4コア以上)")
    
    if issues:
        return False, "; ".join(issues)
    else:
        return True, f"システムリソース十分: {memory_gb:.1f}GB RAM, {cpu_count}コア ✓"

def check_config_file() -> Tuple[bool, str]:
    """設定ファイルをチェック"""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        return False, "設定ファイルが見つかりません: config/config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 必須セクションのチェック
        required_sections = ['dataset', 'teacher', 'student', 'distillation', 'quantization', 'evaluation']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            return False, f"設定ファイルに必須セクションがありません: {', '.join(missing_sections)}"
        
        # 本番環境用設定のチェック
        dataset_config = config.get('dataset', {})
        jpara_samples = dataset_config.get('jpara_samples', 0)
        jesc_samples = dataset_config.get('jesc_samples', 0)
        
        if jpara_samples < 100000 or jesc_samples < 100000:
            return False, f"データセットサイズが小さすぎます: JParaCrawl={jpara_samples}, JESC={jesc_samples} (推奨: 各100,000以上)"
        
        return True, "設定ファイル検証完了 ✓"
        
    except yaml.YAMLError as e:
        return False, f"設定ファイル解析エラー: {e}"
    except Exception as e:
        return False, f"設定ファイルチェックエラー: {e}"

def check_disk_space() -> Tuple[bool, str]:
    """ディスク容量をチェック"""
    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024**3)
    
    if free_gb < 10:
        return False, f"ディスク容量不足: {free_gb:.1f}GB (推奨: 10GB以上)"
    else:
        return True, f"ディスク容量十分: {free_gb:.1f}GB ✓"

def check_internet_connection() -> Tuple[bool, str]:
    """インターネット接続をチェック"""
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        return True, "インターネット接続正常 ✓"
    except Exception:
        return False, "インターネット接続エラー - Hugging Faceにアクセスできません"

def check_code_quality() -> Tuple[bool, str]:
    """コード品質をチェック（開発環境のみ）"""
    try:
        # Black フォーマットチェック
        result = subprocess.run(['black', '--check', '.'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, "コードフォーマットエラー: 'black .' を実行してください"
        
        # Flake8 品質チェック
        result = subprocess.run(['flake8', '.'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, f"コード品質エラー: {result.stdout}"
        
        return True, "コード品質チェック完了 ✓"
        
    except subprocess.TimeoutExpired:
        return False, "コード品質チェックがタイムアウトしました"
    except FileNotFoundError:
        return True, "開発ツールが見つかりません（本番環境では不要）"
    except Exception as e:
        return False, f"コード品質チェックエラー: {e}"

def main():
    """メイン関数"""
    print("=" * 60)
    print("BERT英日翻訳モデル - 本番環境品質チェック")
    print("=" * 60)
    
    checks = [
        ("Pythonバージョン", check_python_version),
        ("依存関係", check_dependencies),
        ("GPU利用可能性", check_gpu_availability),
        ("システムリソース", check_system_resources),
        ("設定ファイル", check_config_file),
        ("ディスク容量", check_disk_space),
        ("インターネット接続", check_internet_connection),
        ("コード品質", check_code_quality),
    ]
    
    results = []
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}をチェック中...")
        try:
            passed, message = check_func()
            results.append((check_name, passed, message))
            if passed:
                print(f"  {message}")
            else:
                print(f"  ❌ {message}")
                all_passed = False
        except Exception as e:
            message = f"チェック実行エラー: {e}"
            results.append((check_name, False, message))
            print(f"  ❌ {message}")
            all_passed = False
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("チェック結果サマリー")
    print("=" * 60)
    
    for check_name, passed, message in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check_name:20} : {status}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 すべてのチェックに合格しました！本番環境で実行可能です。")
        print("\n実行コマンド:")
        print("python main.py --config config/config.yaml --log-level INFO")
        sys.exit(0)
    else:
        print("❌ 一部のチェックに失敗しました。上記の問題を解決してから再実行してください。")
        sys.exit(1)

if __name__ == "__main__":
    main()