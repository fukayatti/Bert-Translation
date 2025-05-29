"""
テキスト前処理

トークナイゼーションと前処理を行うクラスです。
"""

from transformers import BertTokenizer
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """テキストの前処理を行うクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: トークナイザー設定辞書
        """
        self.config = config
        self.model_name = config.get('model_name', 'bert-base-multilingual-cased')
        self.max_length = config.get('max_length', 128)
        
        logger.info(f"トークナイザーを初期化中: {self.model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        logger.info("トークナイザー初期化完了")
    
    def preprocess_batch(self, batch: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        バッチデータの前処理
        
        Args:
            batch: 英語("en")と日本語("ja")のテキストリストを含む辞書
            
        Returns:
            前処理済みバッチデータ
        """
        # ソース（英語）のトークナイズ
        src_encoded = self.tokenizer(
            batch["en"], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length
        )
        
        # ターゲット（日本語）のトークナイズ
        tgt_encoded = self.tokenizer(
            batch["ja"], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length
        )
        
        # バッチデータの作成
        batch["input_ids"] = src_encoded.input_ids
        batch["attention_mask"] = src_encoded.attention_mask
        
        # ラベルの作成（パディングトークンを-100に置換）
        labels = tgt_encoded.input_ids
        batch["labels"] = [
            [tok if tok != self.tokenizer.pad_token_id else -100 for tok in seq] 
            for seq in labels
        ]
        
        return batch
    
    def preprocess_dataset(self, dataset, remove_columns: List[str] = None):
        """
        データセット全体の前処理
        
        Args:
            dataset: 前処理するデータセット
            remove_columns: 削除する列名のリスト
            
        Returns:
            前処理済みデータセット
        """
        if remove_columns is None:
            remove_columns = ["en", "ja"]
            
        logger.info("データセットの前処理を開始...")
        
        processed_dataset = dataset.map(
            self.preprocess_batch,
            batched=True,
            remove_columns=remove_columns
        )
        
        logger.info("データセットの前処理完了")
        return processed_dataset
    
    def encode_text(self, text: str, return_tensors: str = "pt"):
        """
        単一テキストのエンコード
        
        Args:
            text: エンコードするテキスト
            return_tensors: 返すテンソルの形式
            
        Returns:
            エンコード済みテキスト
        """
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
    
    def decode_text(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        トークンIDからテキストにデコード
        
        Args:
            token_ids: デコードするトークンIDのリスト
            skip_special_tokens: 特殊トークンをスキップするかどうか
            
        Returns:
            デコード済みテキスト
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """トークナイザーの情報を取得"""
        return {
            "model_name": self.model_name,
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": self.max_length,
            "pad_token_id": self.tokenizer.pad_token_id,
            "cls_token_id": self.tokenizer.cls_token_id,
            "sep_token_id": self.tokenizer.sep_token_id,
            "special_tokens": {
                "pad": self.tokenizer.pad_token,
                "cls": self.tokenizer.cls_token,
                "sep": self.tokenizer.sep_token,
                "unk": self.tokenizer.unk_token
            }
        }