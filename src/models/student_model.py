"""
Studentモデル

TinyBERT-4LベースのStudentモデルの定義を行うクラスです。
"""

from transformers import (
    BertConfig, 
    EncoderDecoderConfig, 
    EncoderDecoderModel
)
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class StudentModel:
    """TinyBERT-4LベースのStudentモデルクラス"""
    
    def __init__(self, config: Dict[str, Any], tokenizer, device: torch.device):
        """
        初期化
        
        Args:
            config: Student モデル設定辞書
            tokenizer: トークナイザー
            device: 使用デバイス
        """
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.num_hidden_layers = config.get('num_hidden_layers', 4)
        
        logger.info(f"Studentモデルを初期化中 (レイヤー数: {self.num_hidden_layers})")
        self.model = self._create_model()
        logger.info("Studentモデル初期化完了")
    
    def _create_model(self) -> EncoderDecoderModel:
        """TinyBERT-4Lモデルを作成"""
        # エンコーダー設定（4層）
        tiny_enc_config = BertConfig.from_pretrained(
            "bert-base-multilingual-cased",
            num_hidden_layers=self.num_hidden_layers
        )
        
        # デコーダー設定（4層）
        tiny_dec_config = BertConfig.from_pretrained(
            "bert-base-multilingual-cased",
            num_hidden_layers=self.num_hidden_layers,
            is_decoder=True,
            add_cross_attention=True
        )
        
        # エンコーダー・デコーダーモデル設定
        student_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            tiny_enc_config, tiny_dec_config
        )
        
        # モデル作成
        model = EncoderDecoderModel(config=student_config).to(self.device)
        
        # 特殊トークンの設定
        model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        model.config.eos_token_id = self.tokenizer.sep_token_id
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        return model
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: torch.Tensor = None):
        """順伝播"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """テキスト生成"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=kwargs.get('max_length', 128),
                num_beams=kwargs.get('num_beams', 1),
                early_stopping=kwargs.get('early_stopping', True),
                **kwargs
            )
        return outputs
    
    def save_model(self, save_path: str):
        """モデルを保存"""
        logger.info(f"Studentモデルを保存中: {save_path}")
        self.model.save_pretrained(save_path)
        logger.info("Studentモデル保存完了")
    
    def load_model(self, model_path: str):
        """モデルを読み込み"""
        logger.info(f"Studentモデルを読み込み中: {model_path}")
        self.model = EncoderDecoderModel.from_pretrained(model_path).to(self.device)
        logger.info("Studentモデル読み込み完了")
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        return {
            "model_type": "TinyBERT-4L",
            "encoder_layers": self.model.config.encoder.num_hidden_layers,
            "decoder_layers": self.model.config.decoder.num_hidden_layers,
            "hidden_size": self.model.config.encoder.hidden_size,
            "vocab_size": self.model.config.encoder.vocab_size,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "device": str(self.device)
        }
    
    def set_device(self, device: torch.device):
        """デバイスを設定"""
        self.device = device
        self.model = self.model.to(device)
        logger.info(f"Studentモデルのデバイスを設定: {device}")
    
    def train_mode(self):
        """学習モードに設定"""
        self.model.train()
    
    def eval_mode(self):
        """評価モードに設定"""
        self.model.eval()
    
    def parameters(self):
        """モデルのパラメータを取得"""
        return self.model.parameters()
    
    def modules(self):
        """モデルのモジュールを取得"""
        return self.model.modules()
    
    def state_dict(self):
        """モデルの状態辞書を取得"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """モデルの状態辞書を読み込み"""
        self.model.load_state_dict(state_dict)