"""
TAID蒸留

Temporally Adaptive Interpolated Distillation の実装です。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def filter_none_collate_fn(batch):
    """None値をフィルタリングするカスタムcollate関数"""
    # None値を除去
    filtered_batch = [item for item in batch if item is not None and
                     all(v is not None for v in item.values() if isinstance(item, dict))]
    
    # 空のバッチの場合は小さなダミーバッチを作成（完全なスキップを避ける）
    if not filtered_batch:
        logger.debug("空のバッチが検出されました")
        return None
    
    # 各フィールドでNone値をチェック
    clean_batch = []
    for item in filtered_batch:
        if isinstance(item, dict):
            # 全てのフィールドがNoneでないことを確認
            required_keys = ['input_ids', 'attention_mask', 'labels']
            if all(item.get(key) is not None for key in required_keys):
                # さらに、各フィールドが空でないことを確認
                if all(len(item.get(key, [])) > 0 for key in required_keys):
                    clean_batch.append(item)
    
    if not clean_batch:
        logger.debug("有効なアイテムがありません")
        return None
    
    # 手動でバッチを構成
    try:
        result = {}
        for key in ['input_ids', 'attention_mask', 'labels']:
            # 各キーの値を収集
            values = [item[key] for item in clean_batch if key in item and item[key] is not None]
            if not values:
                continue
            
            # リストからテンソルに変換
            if isinstance(values[0], list):
                # すべての値が同じ長さになるようにパディング
                if values:  # 空でないことを確認
                    max_length = max(len(v) for v in values if v)
                    if max_length > 0:
                        padded_values = []
                        for v in values:
                            if v and len(v) < max_length:
                                # パディング（0で埋める）
                                v = v + [0] * (max_length - len(v))
                            elif not v:
                                # 完全に空の場合はmax_lengthの0で埋める
                                v = [0] * max_length
                            padded_values.append(v)
                        result[key] = torch.tensor(padded_values, dtype=torch.long)
            else:
                # 既にテンソルの場合
                result[key] = torch.stack([torch.tensor(v, dtype=torch.long) for v in values if v is not None])
        
        # 結果が有効かチェック
        if result and all(len(v) > 0 for v in result.values()):
            return result
        else:
            return None
        
    except (TypeError, ValueError, RuntimeError) as e:
        logger.warning(f"Collate関数でエラーが発生、バッチをスキップ: {e}")
        return None


class TAIDDistillation:
    """TAID (Temporally Adaptive Interpolated Distillation) クラス"""
    
    def __init__(self, config: Dict[str, Any], teacher_model, student_model, device: torch.device):
        """
        初期化
        
        Args:
            config: 蒸留設定辞書
            teacher_model: Teacherモデル
            student_model: Studentモデル
            device: 使用デバイス
        """
        self.config = config
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        
        # TAID パラメータ
        self.alpha_start = float(config.get('alpha_start', 0.2))
        self.alpha_end = float(config.get('alpha_end', 1.0))
        self.momentum = float(config.get('momentum', 0.0))
        self.beta = float(config.get('beta', 0.9))
        self.steps = int(config.get('steps', 1000))
        self.batch_size = int(config.get('batch_size', 8))
        
        # 学習率の型変換（文字列の場合もfloatに変換）
        learning_rate = config.get('learning_rate', 5e-5)
        if isinstance(learning_rate, str):
            self.learning_rate = float(learning_rate)
        else:
            self.learning_rate = float(learning_rate)
        
        # 初期化
        self.alpha = self.alpha_start
        self.m = self.momentum
        
        logger.info("TAID蒸留初期化完了")
        logger.info(f"パラメータ: α={self.alpha_start}→{self.alpha_end}, β={self.beta}, ステップ数={self.steps}")
    
    def setup_training(self, train_dataset):
        """学習の準備"""
        # Teacherモデルを評価モードに設定
        self.teacher_model.eval()
        self.teacher_model.to(self.device)
        
        # Studentモデルを学習モードに設定
        self.student_model.train()
        
        # データローダー作成（カスタムcollate関数を使用）
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=filter_none_collate_fn,
            drop_last=True  # 不完全なバッチを除去
        )
        
        # オプティマイザー作成
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(), 
            lr=self.learning_rate
        )
        
        logger.info("TAID蒸留の準備完了")
    
    def compute_taid_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        TAID損失を計算
        
        Args:
            student_logits: Studentモデルのロジット
            teacher_logits: Teacherモデルのロジット
            
        Returns:
            TAID損失
        """
        # ソフトマックス確率の計算
        p_student = F.softmax(student_logits, dim=-1)
        p_teacher = F.softmax(teacher_logits, dim=-1)
        
        # 補間された確率分布
        p_interpolated = self.alpha * p_teacher + (1 - self.alpha) * p_student
        
        # KL発散損失
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            p_interpolated,
            reduction="batchmean"
        )
        
        return loss
    
    def update_alpha(self, step: int):
        """αパラメータを更新（TAIDアルゴリズム）"""
        # 目標値の計算
        target = self.alpha_end * (step + 1) / self.steps
        
        # モメンタムの更新
        self.m = self.beta * self.m + (1 - self.beta) * (target - self.alpha)
        
        # αの更新
        self.alpha += 0.005 * self.m
        
        # αの範囲制限
        self.alpha = max(0.0, min(1.0, self.alpha))
    
    def distill(self, train_dataset) -> Dict[str, Any]:
        """
        蒸留を実行
        
        Args:
            train_dataset: 学習データセット
            
        Returns:
            学習結果の辞書
        """
        logger.info("TAID蒸留を開始...")
        
        # 学習準備
        self.setup_training(train_dataset)
        
        # 学習履歴
        history = {
            'losses': [],
            'alphas': [],
            'steps': []
        }
        
        step = 0
        for batch in self.dataloader:
            if step >= self.steps:
                break
            
            # Noneバッチをスキップ
            if batch is None:
                logger.warning(f"ステップ {step}: Noneバッチを検出、スキップします")
                continue
            
            try:
                # バッチデータの準備（リストからテンソルに変換）
                if isinstance(batch["input_ids"], list):
                    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).to(self.device)
                    attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).to(self.device)
                    labels = torch.tensor(batch["labels"], dtype=torch.long).to(self.device)
                else:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                
                # バッチサイズの検証
                if input_ids.size(0) == 0:
                    logger.warning(f"ステップ {step}: 空のバッチを検出、スキップします")
                    continue
                
            except (KeyError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"ステップ {step}: バッチデータの準備でエラー: {e}")
                continue
            
            try:
                # Student forward
                student_output = self.student_model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                student_logits = student_output.logits
                
                # Teacher forward (勾配計算なし)
                with torch.no_grad():
                    teacher_output = self.teacher_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    teacher_logits = teacher_output.logits
                
                # TAID損失の計算
                loss = self.compute_taid_loss(student_logits, teacher_logits)
                
                # 損失の妥当性チェック
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"ステップ {step}: 異常な損失値を検出: {loss.item()}")
                    continue
                
                # 勾配更新
                self.optimizer.zero_grad()
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # αパラメータの更新
                self.update_alpha(step)
                
                # ログ記録
                history['losses'].append(loss.item())
                history['alphas'].append(self.alpha)
                history['steps'].append(step)
                
                # 進捗表示
                if step % 100 == 0:
                    logger.info(f"ステップ {step}/{self.steps}: 損失={loss.item():.4f}, α={self.alpha:.4f}")
                
                step += 1
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"ステップ {step} で蒸留処理エラー: {e}")
                    # CUDAメモリクリア
                    torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"ステップ {step} で蒸留処理エラー: {e}")
                    continue
            except Exception as e:
                logger.error(f"ステップ {step} で蒸留処理エラー: {e}")
                continue
        
        logger.info("TAID蒸留完了")
        logger.info(f"最終損失: {history['losses'][-1]:.4f}")
        logger.info(f"最終α: {history['alphas'][-1]:.4f}")
        
        return history
    
    def get_distillation_info(self) -> Dict[str, Any]:
        """蒸留情報を取得"""
        return {
            "method": "TAID",
            "alpha_start": self.alpha_start,
            "alpha_end": self.alpha_end,
            "current_alpha": self.alpha,
            "beta": self.beta,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "momentum": self.m
        }