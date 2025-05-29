"""
Teacherモデル

BERT2BERTベースのTeacherモデルの定義と学習を行うクラスです。
"""

from transformers import (
    BertConfig, 
    EncoderDecoderConfig, 
    EncoderDecoderModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TeacherModel:
    """BERT2BERTベースのTeacherモデルクラス"""
    
    def __init__(self, config: Dict[str, Any], tokenizer):
        """
        初期化
        
        Args:
            config: Teacher モデル設定辞書
            tokenizer: トークナイザー
        """
        self.config = config
        self.tokenizer = tokenizer
        self.model_name = config.get('model_name', 'bert-base-multilingual-cased')
        self.output_dir = config.get('output_dir', 'teacher_out')
        
        logger.info(f"Teacherモデルを初期化中: {self.model_name}")
        self.model = self._create_model()
        logger.info("Teacherモデル初期化完了")
    
    def _create_model(self) -> EncoderDecoderModel:
        """BERT2BERTモデルを作成"""
        # エンコーダー設定
        enc_config = BertConfig.from_pretrained(self.model_name)
        
        # デコーダー設定
        dec_config = BertConfig.from_pretrained(self.model_name)
        dec_config.is_decoder = True
        dec_config.add_cross_attention = True
        
        # エンコーダー・デコーダーモデル設定
        model_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            enc_config, dec_config
        )
        
        # モデル作成
        model = EncoderDecoderModel(config=model_config)
        
        # 特殊トークンの設定
        model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        model.config.eos_token_id = self.tokenizer.sep_token_id
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        return model
    
    def create_trainer(self, train_dataset, eval_dataset=None):
        """トレーナーを作成"""
        # データコレクター
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model
        )
        
        # 学習引数
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=self.config.get('per_device_eval_batch_size', 8),
            num_train_epochs=self.config.get('num_train_epochs', 1),
            fp16=self.config.get('fp16', True),
            logging_steps=self.config.get('logging_steps', 100),
            save_total_limit=self.config.get('save_total_limit', 1),
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
        )
        
        # トレーナー作成
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        return trainer
    
    def train(self, train_dataset, eval_dataset=None):
        """モデルの学習を実行"""
        logger.info("Teacherモデルの学習を開始...")
        
        trainer = self.create_trainer(train_dataset, eval_dataset)
        
        # 学習実行
        train_result = trainer.train()
        
        logger.info("Teacherモデルの学習完了")
        logger.info(f"最終損失: {train_result.training_loss:.4f}")
        
        return train_result
    
    def save_model(self, save_path: str):
        """モデルを保存"""
        logger.info(f"Teacherモデルを保存中: {save_path}")
        self.model.save_pretrained(save_path)
        logger.info("Teacherモデル保存完了")
    
    def load_model(self, model_path: str):
        """モデルを読み込み"""
        logger.info(f"Teacherモデルを読み込み中: {model_path}")
        self.model = EncoderDecoderModel.from_pretrained(model_path)
        logger.info("Teacherモデル読み込み完了")
    
    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """テキスト生成"""
        self.model.eval()
        with torch.no_grad():
            # kwargsから個別パラメータを取り出してデフォルト値を設定
            generation_kwargs = {
                'input_ids': input_ids,
                'max_length': kwargs.get('max_length', 128),
                'num_beams': kwargs.get('num_beams', 1),
                'early_stopping': kwargs.get('early_stopping', True),
            }
            
            # その他のkwargsを追加（重複を避ける）
            for key, value in kwargs.items():
                if key not in ['max_length', 'num_beams', 'early_stopping']:
                    generation_kwargs[key] = value
            
            outputs = self.model.generate(**generation_kwargs)
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        return {
            "model_name": self.model_name,
            "model_type": "BERT2BERT",
            "encoder_layers": self.model.config.encoder.num_hidden_layers,
            "decoder_layers": self.model.config.decoder.num_hidden_layers,
            "hidden_size": self.model.config.encoder.hidden_size,
            "vocab_size": self.model.config.encoder.vocab_size,
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def eval(self):
        """モデルを評価モードに設定"""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """モデルを学習/評価モードに設定"""
        self.model.train(mode)
        return self
    
    def to(self, device):
        """モデルを指定デバイスに移動"""
        self.model = self.model.to(device)
        return self
    
    def parameters(self):
        """モデルのパラメータを取得"""
        return self.model.parameters()
    
    def __call__(self, *args, **kwargs):
        """モデルを呼び出し可能にする"""
        return self.model(*args, **kwargs)