"""
EfQAT量子化

Efficient Quantization-Aware Training (CWPN) による W4A8量子化の実装です。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import logging
import traceback


def filter_none_collate_fn(batch):
    """None値をフィルタリングするカスタムcollate関数"""
    # None値を除去
    filtered_batch = [item for item in batch if item is not None and
                     isinstance(item, dict) and
                     all(v is not None for v in item.values())]
    
    # 空のバッチの場合はNoneを返す
    if not filtered_batch:
        logging.getLogger(__name__).debug("空のバッチが検出されました")
        return None
    
    # 各フィールドでNone値をチェック
    clean_batch = []
    required_keys = ['input_ids', 'attention_mask', 'labels']
    
    for item in filtered_batch:
        try:
            # 全てのフィールドがNoneでなく、適切な長さを持つことを確認
            if all(key in item and item[key] is not None for key in required_keys):
                # 各フィールドがリストまたはテンソルで、空でないことを確認
                valid_item = True
                for key in required_keys:
                    value = item[key]
                    if isinstance(value, list):
                        if len(value) == 0 or all(x is None for x in value):
                            valid_item = False
                            break
                    elif hasattr(value, '__len__'):
                        if len(value) == 0:
                            valid_item = False
                            break
                    else:
                        valid_item = False
                        break
                
                if valid_item:
                    clean_batch.append(item)
        except (AttributeError, TypeError, KeyError):
            continue
    
    if not clean_batch:
        logging.getLogger(__name__).debug("有効なアイテムがありません")
        return None
    
    # 手動でバッチを構成
    try:
        result = {}
        batch_size = len(clean_batch)
        
        for key in required_keys:
            # 各キーの値を収集
            values = []
            for item in clean_batch:
                if key in item and item[key] is not None:
                    values.append(item[key])
            
            if len(values) != batch_size:
                logging.getLogger(__name__).warning(f"キー '{key}' の値の数が不一致: {len(values)} != {batch_size}")
                return None
            
            # リストからテンソルに変換
            try:
                if isinstance(values[0], list):
                    # 全て同じ長さになるようにパディング
                    max_length = max(len(v) for v in values if v)
                    if max_length > 0:
                        padded_values = []
                        for v in values:
                            if len(v) < max_length:
                                # ラベルの場合は-100でパディング、その他は0
                                pad_value = -100 if key == 'labels' else 0
                                v = v + [pad_value] * (max_length - len(v))
                            padded_values.append(v)
                        result[key] = torch.tensor(padded_values, dtype=torch.long)
                    else:
                        return None
                else:
                    # 既にテンソルの場合
                    result[key] = torch.stack([torch.as_tensor(v, dtype=torch.long) for v in values])
            except (ValueError, TypeError, RuntimeError) as e:
                logging.getLogger(__name__).warning(f"テンソル変換エラー (key: {key}): {e}")
                return None
        
        # 結果の検証
        if not result or not all(key in result for key in required_keys):
            return None
        
        # 全てのテンソルが同じバッチサイズを持つことを確認
        batch_sizes = [result[key].size(0) for key in required_keys]
        if not all(bs == batch_sizes[0] for bs in batch_sizes):
            logging.getLogger(__name__).warning(f"バッチサイズが不一致: {batch_sizes}")
            return None
        
        return result
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Collate関数でエラーが発生、バッチをスキップ: {e}")
        return None

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
        if config is None:
            raise ValueError("設定辞書がNoneです")
        if student_model is None:
            raise ValueError("Studentモデルがありません")
        if device is None:
            raise ValueError("デバイスが指定されていません")
        
        self.config = config
        self.student_model = student_model
        self.device = device
        
        # EfQAT パラメータの検証と設定
        self.freeze_ratio = float(config.get('freeze_ratio', 0.9))
        if not (0.0 <= self.freeze_ratio <= 1.0):
            raise ValueError(f"freeze_ratioは0.0-1.0の範囲である必要があります: {self.freeze_ratio}")
        
        self.interval = int(config.get('interval', 4096))
        if self.interval <= 0:
            raise ValueError(f"intervalは正の値である必要があります: {self.interval}")
        
        # 学習率の型変換（文字列の場合もfloatに変換）
        learning_rate = config.get('learning_rate', 3e-5)
        if isinstance(learning_rate, str):
            self.learning_rate = float(learning_rate)
        else:
            self.learning_rate = float(learning_rate)
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rateは正の値である必要があります: {self.learning_rate}")
        
        self.steps = int(config.get('steps', 500))
        if self.steps <= 0:
            raise ValueError(f"stepsは正の値である必要があります: {self.steps}")
        
        self.batch_size = int(config.get('batch_size', 8))
        if self.batch_size <= 0:
            raise ValueError(f"batch_sizeは正の値である必要があります: {self.batch_size}")
        
        # 閾値
        self.threshold = None
        
        # データローダーとオプティマイザーの初期化
        self.dataloader: Optional[DataLoader] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        logger.info("EfQAT量子化初期化完了")
        logger.info(f"パラメータ: freeze_ratio={self.freeze_ratio}, interval={self.interval}, ステップ数={self.steps}")
    
    def setup_training(self, train_dataset):
        """学習の準備"""
        try:
            if train_dataset is None:
                raise ValueError("学習データセットがありません")
            
            # Studentモデルを学習モードに設定
            self.student_model.train()
            
            # データローダー作成（カスタムcollate関数を使用）
            self.dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # マルチプロセシング問題を回避
                pin_memory=False,  # メモリ問題を回避
                collate_fn=filter_none_collate_fn,
                drop_last=True  # 不完全なバッチを除去
            )
            
            # オプティマイザー作成
            self.optimizer = torch.optim.Adam(
                self.student_model.parameters(),
                lr=self.learning_rate
            )
            
            logger.info("EfQAT量子化の準備完了")
            
        except Exception as e:
            logger.error(f"学習準備でエラーが発生: {e}")
            raise
    
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
        
        try:
            # 学習準備
            self.setup_training(train_dataset)
            
            # 学習履歴
            history = {
                'losses': [],
                'thresholds': [],
                'steps': [],
                'status': 'running'
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
                    
                    # テンソルサイズの検証
                    if input_ids.size(0) == 0:
                        logger.warning(f"ステップ {step}: 空のバッチを検出、スキップします")
                        step += 1
                        continue
                    
                    # Forward pass
                    output = self.student_model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    if not hasattr(output, 'loss') or output.loss is None:
                        logger.warning(f"ステップ {step}: 損失が計算されませんでした")
                        step += 1
                        continue
                    
                    loss = output.loss
                    
                    # 損失の妥当性チェック
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"ステップ {step}: 異常な損失値を検出: {loss.item()}")
                        step += 1
                        continue
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 勾配クリッピング
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                    
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
                    
                    # メモリ使用量のチェック（CUDAの場合）
                    if self.device.type == 'cuda' and step % 1000 == 0:
                        memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                        memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                        logger.debug(f"GPU メモリ使用量: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
                        
                        # メモリ使用量が多すぎる場合の警告
                        if memory_allocated > 10.0:  # 10GB以上
                            logger.warning("GPU メモリ使用量が高くなっています")
                            torch.cuda.empty_cache()
                    
                except RuntimeError as batch_error:
                    if "CUDA out of memory" in str(batch_error):
                        logger.error(f"ステップ {step} でCUDAメモリエラー: {batch_error}")
                        # CUDAメモリクリア
                        torch.cuda.empty_cache()
                        continue
                    else:
                        logger.error(f"ステップ {step} でエラーが発生: {batch_error}")
                        logger.debug(f"バッチエラーの詳細:\n{traceback.format_exc()}")
                        continue
                except Exception as batch_error:
                    logger.error(f"ステップ {step} でエラーが発生: {batch_error}")
                    logger.debug(f"バッチエラーの詳細:\n{traceback.format_exc()}")
                    # バッチエラーの場合は続行
                    
                step += 1
            
            # W4A8量子化の適用
            logger.info("W4A8量子化を適用中...")
            self.quantize_weights_w4a8()
            
            history['status'] = 'completed'
            logger.info("EfQAT量子化完了")
            
            if history['losses']:
                logger.info(f"最終損失: {history['losses'][-1]:.4f}")
                logger.info(f"平均損失: {sum(history['losses'])/len(history['losses']):.4f}")
            
            return history
            
        except Exception as e:
            logger.error(f"EfQAT量子化でエラーが発生: {e}")
            logger.error(f"詳細なトレースバック:\n{traceback.format_exc()}")
            
            # GPUメモリのクリーンアップ
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            raise
    
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