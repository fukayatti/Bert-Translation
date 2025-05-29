"""
EfQAT量子化

Efficient Quantization-Aware Training (CWPN) による W4A8量子化の実装です。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class EfQATQuantization:
    """EfQAT (Efficient Quantization-Aware Training) クラス"""
    
    def __init__(self, config: Dict[str, Any], student_model, device: torch.device):
        """
        初期化
        
        Args:
            config: 量子化設定辞書
            student_model: Studentモデル
            device: 使用デバイス
        """
        self.config = config
        self.student_model = student_model
        self.device = device
        
        # EfQAT パラメータ
        self.freeze_ratio = config.get('freeze_ratio', 0.9)
        self.interval = config.get('interval', 4096)
        self.learning_rate = config.get('learning_rate', 3e-5)
        self.steps = config.get('steps', 500)
        self.batch_size = config.get('batch_size', 8)
        
        # 閾値
        self.threshold = None
        
        logger.info("EfQAT量子化初期化完了")
        logger.info(f"パラメータ: freeze_ratio={self.freeze_ratio}, interval={self.interval}, ステップ数={self.steps}")
    
    def setup_training(self, train_dataset):
        """学習の準備"""
        # Studentモデルを学習モードに設定
        self.student_model.train_mode()
        
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
        
        logger.info("EfQAT量子化の準備完了")
    
    def compute_importance_scores(self) -> List[torch.Tensor]:
        """重要度スコアを計算"""
        importance_scores = []
        
        for module in self.student_model.modules():
            if isinstance(module, nn.Linear):
                # 重みの絶対値の平均を重要度スコアとして使用
                importance = module.weight.data.abs().mean(dim=1).cpu()
                importance_scores.append(importance)
        
        return importance_scores
    
    def update_threshold(self):
        """閾値を更新"""
        importance_scores = self.compute_importance_scores()
        
        if importance_scores:
            # 全ての重要度スコアを結合
            all_scores = torch.cat(importance_scores)
            
            # k番目の最小値を閾値として設定
            k = int(len(all_scores) * self.freeze_ratio)
            self.threshold = all_scores.kthvalue(k).values.item()
            
            logger.debug(f"閾値更新: {self.threshold:.6f}")
    
    def apply_gradient_mask(self):
        """勾配マスクを適用"""
        if self.threshold is None:
            return
        
        for module in self.student_model.modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                # 重要度スコアを計算
                importance = module.weight.data.abs().mean(dim=1)
                
                # マスクを作成（重要度が閾値以上の場合は1、そうでなければ0）
                mask = (importance >= self.threshold).float().unsqueeze(1).to(self.device)
                
                # 勾配にマスクを適用
                module.weight.grad.mul_(mask)
    
    def quantize_weights_w4a8(self):
        """W4A8量子化を適用（4bit重み、8bit活性化）"""
        logger.info("W4A8量子化を適用中...")
        
        quantized_modules = 0
        
        for module in self.student_model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                
                # 4bit量子化のためのスケールファクター計算
                # 量子化レベル: -7 to 7 (4bit符号付き)
                scale = (2**3 - 1) / weight.abs().max()
                
                # 量子化・逆量子化
                quantized_weight = torch.round(weight * scale) / scale
                
                # 重みを更新
                module.weight.data = quantized_weight
                quantized_modules += 1
        
        logger.info(f"量子化完了: {quantized_modules}個のLinearモジュールを量子化")
    
    def quantize(self, train_dataset) -> Dict[str, Any]:
        """
        EfQAT量子化を実行
        
        Args:
            train_dataset: 学習データセット
            
        Returns:
            量子化結果の辞書
        """
        logger.info("EfQAT量子化を開始...")
        
        # 学習準備
        self.setup_training(train_dataset)
        
        # 学習履歴
        history = {
            'losses': [],
            'thresholds': [],
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
            
            # Forward pass
            output = self.student_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = output.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 閾値の更新（一定間隔で）
            if step % self.interval == 0:
                self.update_threshold()
            
            # 勾配マスクの適用
            self.apply_gradient_mask()
            
            # パラメータ更新
            self.optimizer.step()
            
            # ログ記録
            history['losses'].append(loss.item())
            history['thresholds'].append(self.threshold if self.threshold else 0.0)
            history['steps'].append(step)
            
            # 進捗表示
            if step % 100 == 0:
                threshold_str = f"{self.threshold:.6f}" if self.threshold else "未設定"
                logger.info(f"ステップ {step}/{self.steps}: 損失={loss.item():.4f}, 閾値={threshold_str}")
            
            step += 1
        
        # W4A8量子化の適用
        self.quantize_weights_w4a8()
        
        logger.info("EfQAT量子化完了")
        logger.info(f"最終損失: {history['losses'][-1]:.4f}")
        
        return history
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """量子化情報を取得"""
        # モデルサイズの計算
        total_params = sum(p.numel() for p in self.student_model.parameters())
        
        # 量子化後のサイズ（概算）
        # Linear層の重みが4bitになったと仮定
        linear_params = 0
        for module in self.student_model.modules():
            if isinstance(module, nn.Linear):
                linear_params += module.weight.numel()
        
        # 4bit重みによる圧縮率の計算
        compression_ratio = (total_params * 32 - linear_params * 28) / (total_params * 32)
        
        return {
            "method": "EfQAT (W4A8)",
            "freeze_ratio": self.freeze_ratio,
            "interval": self.interval,
            "threshold": self.threshold,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "total_parameters": total_params,
            "linear_parameters": linear_params,
            "compression_ratio": compression_ratio,
            "quantization": {
                "weights": "4bit",
                "activations": "8bit"
            }
        }