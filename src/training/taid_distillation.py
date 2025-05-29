"""
TAID蒸留

Temporally Adaptive Interpolated Distillation の実装です。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


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
        
        # データローダー作成
        self.dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
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
            
            # バッチデータの準備
            input_ids = torch.tensor(batch["input_ids"]).to(self.device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(self.device)
            labels = torch.tensor(batch["labels"]).to(self.device)
            
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
            
            # 勾配更新
            self.optimizer.zero_grad()
            loss.backward()
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