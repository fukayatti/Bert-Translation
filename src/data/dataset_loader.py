"""
データセットローダー

JParaCrawlとJESCデータセットの読み込みと前処理を行うクラスです。
"""

from datasets import load_dataset, concatenate_datasets
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """データセットの読み込みと前処理を行うクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: データセット設定辞書
        """
        self.config = config
        self.jpara_samples = config.get('jpara_samples', 50000)
        self.jesc_samples = config.get('jesc_samples', 50000)
        self.test_size = config.get('test_size', 0.1)
        self.seed = config.get('seed', 42)
        
    def load_jpara_dataset(self):
        """JParaCrawlデータセットを読み込む"""
        logger.info(f"JParaCrawlデータセットを読み込み中... (サンプル数: {self.jpara_samples})")
        
        jpara = (
            load_dataset(
                "Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus",
                split="train"
            )
            .shuffle(seed=self.seed)
            .select(range(self.jpara_samples))
            .rename_columns({"english": "en", "japanese": "ja"})
        )
        
        logger.info(f"JParaCrawl読み込み完了: {len(jpara)}件")
        return jpara
    
    def load_jesc_dataset(self):
        """JESCデータセットを読み込む"""
        logger.info(f"JESCデータセットを読み込み中... (サンプル数: {self.jesc_samples})")
        
        jesc_raw = (
            load_dataset("Hoshikuzu/JESC", split="train")
            .shuffle(seed=self.seed)
            .select(range(self.jesc_samples))
        )
        
        def split_translation(example):
            """翻訳データを英語と日本語に分離"""
            return {
                "en": example["translation"]["en"],
                "ja": example["translation"]["ja"]
            }
        
        jesc = jesc_raw.map(split_translation, remove_columns=["translation"])
        
        logger.info(f"JESC読み込み完了: {len(jesc)}件")
        return jesc
    
    def load_combined_dataset(self):
        """JParaCrawlとJESCを結合したデータセットを読み込む"""
        logger.info("データセットを結合中...")
        
        # 各データセットを読み込み
        jpara = self.load_jpara_dataset()
        jesc = self.load_jesc_dataset()
        
        # データセットを結合
        raw_ds = concatenate_datasets([jpara, jesc])
        
        # train/test分割
        dataset = raw_ds.train_test_split(test_size=self.test_size, seed=self.seed)
        
        logger.info(f"データセット結合完了:")
        logger.info(f"  - 訓練データ: {len(dataset['train'])}件")
        logger.info(f"  - テストデータ: {len(dataset['test'])}件")
        
        return dataset
    
    def get_dataset_info(self, dataset):
        """データセットの情報を表示"""
        info = {
            "train_size": len(dataset["train"]),
            "test_size": len(dataset["test"]),
            "total_size": len(dataset["train"]) + len(dataset["test"]),
            "columns": dataset["train"].column_names
        }
        
        # サンプルデータの表示
        sample = dataset["train"][0]
        info["sample"] = sample
        
        return info