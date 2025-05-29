"""
データセットローダー

JParaCrawlとJESCデータセットの読み込みと前処理を行うクラスです。
"""

from datasets import load_dataset, concatenate_datasets
from typing import Dict, Any, Optional
import logging
import traceback

logger = logging.getLogger(__name__)


class DatasetLoader:
    """データセットの読み込みと前処理を行うクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: データセット設定辞書
        """
        if config is None:
            raise ValueError("設定辞書がNoneです")
        
        self.config = config
        
        # パラメータの検証
        self.jpara_samples = config.get('jpara_samples', 50000)
        if self.jpara_samples <= 0:
            raise ValueError(f"jpara_samplesは正の値である必要があります: {self.jpara_samples}")
        
        self.jesc_samples = config.get('jesc_samples', 50000)
        if self.jesc_samples <= 0:
            raise ValueError(f"jesc_samplesは正の値である必要があります: {self.jesc_samples}")
        
        self.test_size = config.get('test_size', 0.1)
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(f"test_sizeは0.0-1.0の範囲である必要があります: {self.test_size}")
        
        self.seed = config.get('seed', 42)
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seedは非負の整数である必要があります: {self.seed}")
        
        logger.info(f"データセットローダー初期化完了: JParaCrawl={self.jpara_samples}, JESC={self.jesc_samples}")
        
    def load_jpara_dataset(self):
        """JParaCrawlデータセットを読み込む"""
        logger.info(f"JParaCrawlデータセットを読み込み中... (サンプル数: {self.jpara_samples})")
        
        try:
            jpara = load_dataset(
                "Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus",
                split="train",
                trust_remote_code=False  # セキュリティのため
            )
            
            if jpara is None or len(jpara) == 0:
                raise ValueError("JParaCrawlデータセットが空です")
            
            # サンプル数の調整
            actual_samples = min(self.jpara_samples, len(jpara))
            if actual_samples < self.jpara_samples:
                logger.warning(f"要求されたサンプル数({self.jpara_samples})より実際のデータ数({len(jpara)})が少ないです")
            
            jpara = (
                jpara
                .shuffle(seed=self.seed)
                .select(range(actual_samples))
                .rename_columns({"english": "en", "japanese": "ja"})
            )
            
            # データの妥当性チェック
            sample = jpara[0]
            if 'en' not in sample or 'ja' not in sample:
                raise ValueError("データセットに必要なカラム（en, ja）がありません")
            
            logger.info(f"JParaCrawl読み込み完了: {len(jpara)}件")
            return jpara
            
        except Exception as e:
            logger.error(f"JParaCrawlデータセットの読み込みでエラーが発生: {e}")
            logger.error(f"詳細:\n{traceback.format_exc()}")
            raise
    
    def load_jesc_dataset(self):
        """JESCデータセットを読み込む"""
        logger.info(f"JESCデータセットを読み込み中... (サンプル数: {self.jesc_samples})")
        
        try:
            jesc_raw = load_dataset(
                "Hoshikuzu/JESC",
                split="train",
                trust_remote_code=False  # セキュリティのため
            )
            
            if jesc_raw is None or len(jesc_raw) == 0:
                raise ValueError("JESCデータセットが空です")
            
            # サンプル数の調整
            actual_samples = min(self.jesc_samples, len(jesc_raw))
            if actual_samples < self.jesc_samples:
                logger.warning(f"要求されたサンプル数({self.jesc_samples})より実際のデータ数({len(jesc_raw)})が少ないです")
            
            jesc_raw = (
                jesc_raw
                .shuffle(seed=self.seed)
                .select(range(actual_samples))
            )
            
            def split_translation(example):
                """翻訳データを英語と日本語に分離"""
                try:
                    if "translation" not in example:
                        raise KeyError("translationキーがありません")
                    
                    translation = example["translation"]
                    if not isinstance(translation, dict):
                        raise ValueError("translationが辞書型ではありません")
                    
                    if "en" not in translation or "ja" not in translation:
                        raise KeyError("翻訳データにenまたはjaキーがありません")
                    
                    return {
                        "en": str(translation["en"]).strip(),
                        "ja": str(translation["ja"]).strip()
                    }
                except Exception as e:
                    logger.warning(f"翻訳データの処理でエラー: {e}")
                    return {"en": "", "ja": ""}  # 空文字で代替
            
            jesc = jesc_raw.map(
                split_translation,
                remove_columns=["translation"],
                desc="翻訳データの分離"
            )
            
            # 空のデータを除去
            jesc = jesc.filter(lambda x: len(x["en"]) > 0 and len(x["ja"]) > 0)
            
            logger.info(f"JESC読み込み完了: {len(jesc)}件")
            return jesc
            
        except Exception as e:
            logger.error(f"JESCデータセットの読み込みでエラーが発生: {e}")
            logger.error(f"詳細:\n{traceback.format_exc()}")
            raise
    
    def load_combined_dataset(self):
        """JParaCrawlとJESCを結合したデータセットを読み込む"""
        logger.info("データセットを結合中...")
        
        try:
            # 各データセットを読み込み
            jpara = self.load_jpara_dataset()
            jesc = self.load_jesc_dataset()
            
            if jpara is None or jesc is None:
                raise ValueError("データセットの読み込みに失敗しました")
            
            # データセットサイズの確認
            jpara_len = len(jpara)
            jesc_len = len(jesc)
            total_len = jpara_len + jesc_len
            
            logger.info(f"データセットサイズ: JParaCrawl={jpara_len}, JESC={jesc_len}, 合計={total_len}")
            
            if total_len == 0:
                raise ValueError("結合するデータセットが空です")
            
            # データセットを結合
            raw_ds = concatenate_datasets([jpara, jesc])
            
            # train/test分割
            if total_len < 10:  # 最小限のデータ数チェック
                logger.warning("データセットが非常に小さいです。テスト分割を調整します")
                test_size = max(1, int(total_len * 0.2))  # 最低1件はテストデータ
                dataset = raw_ds.train_test_split(test_size=test_size, seed=self.seed)
            else:
                dataset = raw_ds.train_test_split(test_size=self.test_size, seed=self.seed)
            
            # 分割結果の検証
            train_len = len(dataset['train'])
            test_len = len(dataset['test'])
            
            if train_len == 0:
                raise ValueError("訓練データが空です")
            if test_len == 0:
                raise ValueError("テストデータが空です")
            
            logger.info(f"データセット結合完了:")
            logger.info(f"  - 訓練データ: {train_len}件")
            logger.info(f"  - テストデータ: {test_len}件")
            logger.info(f"  - 分割比率: {test_len/total_len:.1%}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"データセット結合でエラーが発生: {e}")
            logger.error(f"詳細:\n{traceback.format_exc()}")
            raise
    
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