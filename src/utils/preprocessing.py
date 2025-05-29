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
        # 入力データの検証とクリーニング
        en_texts = []
        ja_texts = []
        
        for en, ja in zip(batch["en"], batch["ja"]):
            # None、空文字列、非文字列をフィルタリング
            if (en is not None and ja is not None and
                isinstance(en, str) and isinstance(ja, str) and
                len(en.strip()) > 0 and len(ja.strip()) > 0):
                en_texts.append(en.strip())
                ja_texts.append(ja.strip())
        
        # 有効なデータがない場合は空のバッチを返す
        if not en_texts or not ja_texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}
            
        try:
            # ソース（英語）のトークナイズ
            src_encoded = self.tokenizer(
                en_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )
            
            # ターゲット（日本語）のトークナイズ
            tgt_encoded = self.tokenizer(
                ja_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
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
            
        except Exception as e:
            logger.warning(f"前処理中にエラーが発生: {e}")
            return {"input_ids": [], "attention_mask": [], "labels": []}
    
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
        
        # カスタムフィルタ関数を定義
        def filter_and_preprocess(examples):
            # 有効なペアのみをフィルタリング
            valid_indices = []
            for i, (en, ja) in enumerate(zip(examples["en"], examples["ja"])):
                if (en is not None and ja is not None and
                    isinstance(en, str) and isinstance(ja, str) and
                    len(en.strip()) > 0 and len(ja.strip()) > 0):
                    valid_indices.append(i)
            
            # 有効なデータがない場合はNoneを返す代わりに、最小限のダミーデータを作成
            if not valid_indices:
                # 最小限のダミーデータで警告を出す
                logger.debug("有効なデータペアが見つかりません - ダミーデータを生成")
                dummy_text = "dummy"
                return {
                    "input_ids": [[self.tokenizer.cls_token_id, self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id] * (self.max_length - 2)],
                    "attention_mask": [[1, 1] + [0] * (self.max_length - 2)],
                    "labels": [[-100, -100] + [-100] * (self.max_length - 2)]
                }
            
            filtered_examples = {
                "en": [examples["en"][i] for i in valid_indices],
                "ja": [examples["ja"][i] for i in valid_indices]
            }
            
            # 前処理を実行
            return self.preprocess_batch(filtered_examples)
        
        processed_dataset = dataset.map(
            filter_and_preprocess,
            batched=True,
            remove_columns=remove_columns,
            batch_size=1000  # バッチサイズを指定
        )
        
        # 有効でないデータを除去（より堅牢なフィルタリング）
        def is_valid_sample(example):
            try:
                return (
                    example["input_ids"] is not None and
                    example["attention_mask"] is not None and
                    example["labels"] is not None and
                    len(example["input_ids"]) > 0 and
                    len(example["attention_mask"]) > 0 and
                    len(example["labels"]) > 0 and
                    len(example["input_ids"]) == len(example["attention_mask"]) == len(example["labels"])
                )
            except (KeyError, TypeError, AttributeError):
                return False
        
        processed_dataset = processed_dataset.filter(is_valid_sample)
        
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