# BERT 英日翻訳モデル - TAID 蒸留 + EfQAT 量子化

このプロジェクトは、BERT2BERT から TinyBERT-4L への蒸留と EfQAT 量子化を使用した英日翻訳モデルの実装です。

## 特徴

- **Teacher Model**: BERT2BERT（12 層エンコーダ・デコーダ）
- **Student Model**: TinyBERT-4L（4 層エンコーダ・デコーダ）
- **蒸留手法**: TAID (Temporally Adaptive Interpolated Distillation)
- **量子化**: EfQAT (CWPN) による W4A8 量子化
- **データセット**: JParaCrawl + JESC
- **評価指標**: BLEU、BERTScore

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 完全なパイプライン実行

```bash
python main.py
```

### 個別実行

```bash
# 1. Teacher モデル学習
python scripts/train_teacher.py

# 2. Student モデル蒸留
python scripts/distill_student.py

# 3. モデル量子化
python scripts/quantize_model.py

# 4. モデル評価
python scripts/evaluate_model.py
```

## 設定

`config/config.yaml`でハイパーパラメータや設定を変更できます。

## ライセンス

MIT License

## 参考文献

- TAID: Temporally Adaptive Interpolated Distillation
- EfQAT: Efficient Quantization-Aware Training
