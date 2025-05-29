# BERT 英日翻訳モデル - 本番環境対応版

このプロジェクトは、BERT2BERT から TinyBERT-6L への蒸留と EfQAT 量子化を使用した本番品質の英日翻訳モデルの実装です。

## 特徴

- **Teacher Model**: BERT2BERT（12 層エンコーダ・デコーダ）
- **Student Model**: TinyBERT-6L（6 層エンコーダ・デコーダ）- 本番品質向上
- **蒸留手法**: TAID (Temporally Adaptive Interpolated Distillation)
- **量子化**: EfQAT (CWPN) による W4A8 量子化
- **データセット**: JParaCrawl + JESC（各 50 万サンプル）
- **評価指標**: BLEU、BERTScore、ROUGE
- **本番環境対応**: 包括的な監視、品質管理、セキュリティ機能

## 安定性の向上

このプロジェクトには以下の安定性機能が含まれています：

### エラーハンドリング

- 各ステップでの個別エラーキャッチ
- 詳細なエラーログとトレースバック
- 適切なフォールバック処理
- GPU メモリ不足時の自動クリーンアップ

### 入力検証

- 設定ファイルの妥当性チェック
- データセットサイズの検証
- パラメータ範囲の確認
- ファイル存在チェック

### ログ機能

- 構造化されたログ出力
- ファイルとコンソールへの同時出力
- 実行時間の詳細記録
- 進捗状況の可視化

### リソース管理

- GPU メモリ使用量の監視
- 自動メモリクリーンアップ
- デバイス設定の自動調整
- 中断時の適切なクリーンアップ

## 本番環境の新機能

### リソース監視

- リアルタイム CPU・メモリ・GPU 使用率監視
- 自動リソース最適化とアラート
- システム負荷に応じた動的バッチサイズ調整

### 品質管理

- 開発ツール統合（pytest、black、mypy）
- コードカバレッジレポート
- 自動品質チェック

### セキュリティ

- 暗号化通信対応
- セキュアな依存関係管理
- 設定ファイル検証強化

### 運用支援

- TensorBoard 統合
- Weights & Biases 実験管理
- 詳細な実行履歴とメトリクス

## インストール

```bash
# 基本依存関係のインストール
pip install -r requirements.txt

# 開発環境の場合、追加で以下を実行
pip install -e .
```

## 使用方法

### 本番環境での完全なパイプライン実行

```bash
# 本番設定での実行（推奨）
python main.py --config config/config.yaml --log-level INFO

# 高性能サーバーでの実行
python main.py --device cuda --log-level INFO --output-dir ./production_output
```

### 開発・テスト環境での実行

```bash
# デバッグモードでの実行
python main.py --log-level DEBUG --config config/debug_config.yaml

# 特定のステップをスキップして実行（開発時のみ）
python main.py --skip-teacher --skip-evaluation

# カスタム設定ファイルを使用
python main.py --config custom_config.yaml

# CPU環境での実行
python main.py --device cpu --output-dir ./cpu_output
```

### 本番環境での段階的実行

```bash
# 1. システムリソースチェック
python -c "import psutil; print(f'CPU: {psutil.cpu_count()}cores, RAM: {psutil.virtual_memory().total//1024**3}GB')"

# 2. 設定ファイル検証
python -c "import yaml; print('設定OK' if yaml.safe_load(open('config/config.yaml')) else '設定エラー')"

# 3. 本番パイプライン実行
python main.py --config config/config.yaml --log-level INFO --output-dir ./production_models
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

### 主要設定項目

- `dataset`: データセットサイズとテスト分割比
- `teacher`: Teacher モデルの学習パラメータ
- `student`: Student モデルの構造設定
- `distillation`: TAID 蒸留のパラメータ
- `quantization`: EfQAT 量子化の設定
- `evaluation`: 評価メトリックの設定

## トラブルシューティング

### 一般的な問題と解決策

1. **GPU メモリ不足**

   - バッチサイズを削減: `config.yaml`の`batch_size`を減らす
   - 自動クリーンアップが実行されます

2. **データセット読み込みエラー**

   - インターネット接続を確認
   - Hugging Face データセットアクセスを確認
   - ログで詳細なエラー内容を確認

3. **設定ファイルエラー**
   - YAML 構文を確認
   - 必須フィールドの存在を確認
   - 数値パラメータの範囲を確認

### ログファイル

実行ログは `output/logs/` ディレクトリに保存されます：

- パイプライン情報: `pipeline_info.json`
- 詳細ログ: `pipeline_YYYYMMDD_HHMMSS.log`

## 出力ファイル

実行後、以下のファイルが生成されます：

```
output/
├── logs/                    # ログファイル
├── teacher/                 # Teacherモデル
├── student/                 # 蒸留後Studentモデル
├── quantized/              # 量子化モデル
├── results/                # 評価結果
└── pipeline_info.json     # パイプライン実行情報
```

## ライセンス

MIT License

## 参考文献

- TAID: Temporally Adaptive Interpolated Distillation
- EfQAT: Efficient Quantization-Aware Training
