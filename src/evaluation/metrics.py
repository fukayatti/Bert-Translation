"""
評価指標

BLEU、BERTScoreなどの評価指標を計算するクラスです。
"""

import torch
from sacrebleu import corpus_bleu
from bert_score import score as bert_score
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """モデル評価クラス"""
    
    def __init__(self, config: Dict[str, Any], tokenizer, device: torch.device):
        """
        初期化
        
        Args:
            config: 評価設定辞書
            tokenizer: トークナイザー
            device: 使用デバイス
        """
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.sample_size = config.get('sample_size', 100)
        self.max_generation_length = config.get('max_generation_length', 128)
        
        logger.info("モデル評価器初期化完了")
        logger.info(f"評価サンプル数: {self.sample_size}")
    
    def generate_predictions(self, model, test_dataset) -> Tuple[List[str], List[str]]:
        """
        モデルの予測を生成
        
        Args:
            model: 評価するモデル
            test_dataset: テストデータセット
            
        Returns:
            (予測結果のリスト, 参照結果のリスト)
        """
        model.eval_mode() if hasattr(model, 'eval_mode') else model.eval()
        
        # サンプルデータの選択
        samples = test_dataset.select(range(min(self.sample_size, len(test_dataset))))
        
        predictions = []
        references = []
        
        logger.info(f"予測生成中... (サンプル数: {len(samples)})")
        
        for i, example in enumerate(samples):
            # 入力のトークナイズ
            input_encoded = self.tokenizer(
                example["en"], 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )
            input_ids = input_encoded.input_ids.to(self.device)
            
            # 予測生成
            with torch.no_grad():
                if hasattr(model, 'model') and hasattr(model.model, 'generate'):
                    # TeacherModelの場合（model.model.generateを使用）
                    generated_ids = model.generate(
                        input_ids,
                        max_length=self.max_generation_length
                    )
                elif hasattr(model, 'generate'):
                    # StudentModelの場合（直接generateを使用）
                    generated_ids = model.generate(
                        input_ids,
                        max_length=self.max_generation_length
                    )
                else:
                    # フォールバック：モデルのgenerateメソッドを直接呼び出し
                    generated_ids = model.generate(
                        input_ids,
                        max_length=self.max_generation_length
                    )
            
            # デコード
            prediction = self.tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            predictions.append(prediction)
            references.append(example["ja"])
            
            # 進捗表示
            if (i + 1) % 20 == 0:
                logger.info(f"予測生成進捗: {i + 1}/{len(samples)}")
        
        logger.info("予測生成完了")
        return predictions, references
    
    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """
        BLEU スコアを計算
        
        Args:
            predictions: 予測結果のリスト
            references: 参照結果のリスト
            
        Returns:
            BLEU スコア
        """
        logger.info("BLEU スコア計算中...")
        
        # corpus_bleuは参照を二次元リストとして受け取る
        bleu_score = corpus_bleu(predictions, [references]).score
        
        logger.info(f"BLEU スコア: {bleu_score:.2f}")
        return bleu_score
    
    def compute_bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        BERTScore を計算
        
        Args:
            predictions: 予測結果のリスト
            references: 参照結果のリスト
            
        Returns:
            BERTScore の辞書 (P, R, F1)
        """
        logger.info("BERTScore 計算中...")
        
        try:
            P, R, F1 = bert_score(
                predictions, 
                references, 
                lang='ja', 
                model_type='bert-base-multilingual-cased'
            )
            
            bert_scores = {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
            
            logger.info(f"BERTScore F1: {bert_scores['f1']:.4f}")
            return bert_scores
            
        except Exception as e:
            logger.error(f"BERTScore計算エラー: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def evaluate_model(self, model, test_dataset, model_name: str = "Model") -> Dict[str, Any]:
        """
        モデルの総合評価
        
        Args:
            model: 評価するモデル
            test_dataset: テストデータセット
            model_name: モデル名
            
        Returns:
            評価結果の辞書
        """
        logger.info(f"{model_name}の評価を開始...")
        
        # 予測生成
        predictions, references = self.generate_predictions(model, test_dataset)
        
        # BLEU スコア計算
        bleu_score = self.compute_bleu_score(predictions, references)
        
        # BERTScore 計算
        bert_scores = self.compute_bert_score(predictions, references)
        
        # 結果の整理
        evaluation_results = {
            'model_name': model_name,
            'sample_size': len(predictions),
            'bleu_score': bleu_score,
            'bert_score': bert_scores,
            'sample_predictions': predictions[:5],  # 最初の5つのサンプル
            'sample_references': references[:5]
        }
        
        logger.info(f"{model_name}の評価完了")
        self._log_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def compare_models(self, teacher_model, student_model, test_dataset) -> Dict[str, Any]:
        """
        TeacherモデルとStudentモデルを比較評価
        
        Args:
            teacher_model: Teacherモデル
            student_model: Studentモデル
            test_dataset: テストデータセット
            
        Returns:
            比較評価結果の辞書
        """
        logger.info("モデル比較評価を開始...")
        
        # 各モデルの評価
        teacher_results = self.evaluate_model(teacher_model, test_dataset, "Teacher")
        student_results = self.evaluate_model(student_model, test_dataset, "Student")
        
        # 比較結果の作成
        comparison = {
            'teacher': teacher_results,
            'student': student_results,
            'comparison': {
                'bleu_degradation': teacher_results['bleu_score'] - student_results['bleu_score'],
                'bert_f1_degradation': teacher_results['bert_score']['f1'] - student_results['bert_score']['f1'],
                'performance_retention': {
                    'bleu': student_results['bleu_score'] / teacher_results['bleu_score'] if teacher_results['bleu_score'] > 0 else 0,
                    'bert_f1': student_results['bert_score']['f1'] / teacher_results['bert_score']['f1'] if teacher_results['bert_score']['f1'] > 0 else 0
                }
            }
        }
        
        logger.info("モデル比較評価完了")
        self._log_comparison_results(comparison)
        
        return comparison
    
    def _log_evaluation_results(self, results: Dict[str, Any]):
        """評価結果をログ出力"""
        logger.info("=" * 50)
        logger.info(f"モデル評価結果: {results['model_name']}")
        logger.info("=" * 50)
        logger.info(f"サンプル数: {results['sample_size']}")
        logger.info(f"BLEU スコア: {results['bleu_score']:.2f}")
        logger.info(f"BERTScore:")
        logger.info(f"  - Precision: {results['bert_score']['precision']:.4f}")
        logger.info(f"  - Recall: {results['bert_score']['recall']:.4f}")
        logger.info(f"  - F1: {results['bert_score']['f1']:.4f}")
        logger.info("=" * 50)
    
    def _log_comparison_results(self, comparison: Dict[str, Any]):
        """比較結果をログ出力"""
        comp = comparison['comparison']
        logger.info("=" * 60)
        logger.info("モデル比較結果")
        logger.info("=" * 60)
        logger.info(f"BLEU 劣化: {comp['bleu_degradation']:.2f}")
        logger.info(f"BERTScore F1 劣化: {comp['bert_f1_degradation']:.4f}")
        logger.info(f"性能保持率:")
        logger.info(f"  - BLEU: {comp['performance_retention']['bleu']:.1%}")
        logger.info(f"  - BERTScore F1: {comp['performance_retention']['bert_f1']:.1%}")
        logger.info("=" * 60)