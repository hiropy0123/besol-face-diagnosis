# 顔タイプ診断 (Face Type Diagnosis)

MediaPipe Face Mesh で顔ランドマークを抽出し、特徴量から8タイプの顔タイプに分類する Streamlit アプリです。

## セットアップ

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 構成

```
.
├── app.py                    # Streamlit エントリポイント
├── requirements.txt
├── src/
│   ├── constants.py          # ランドマークインデックス・FaceType Enum
│   ├── landmarks.py          # MediaPipe によるランドマーク検出
│   ├── metrics.py            # 特徴量算出 (FaceMetrics)
│   ├── classifier.py         # 2軸スコア & 8タイプ分類
│   └── visualizer.py         # 描画・Plotlyチャート
├── assets/
│   └── type_descriptions/descriptions.json
└── tests/
    ├── test_metrics.py
    └── test_classifier.py
```

## 診断軸

- **maturity_score**: -1 (子供顔) 〜 +1 (大人顔)
- **linearity_score**: -1 (曲線的) 〜 +1 (直線的)

## テスト

```bash
pytest tests/
```
