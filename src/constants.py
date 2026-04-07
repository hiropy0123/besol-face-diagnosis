"""ランドマークインデックス定義・閾値・顔タイプEnum"""
from enum import Enum


class FaceType(Enum):
    CUTE = "キュート"
    ACTIVE_CUTE = "アクティブキュート"
    FRESH = "フレッシュ"
    COOL_CASUAL = "クールカジュアル"
    FEMININE = "フェミニン"
    SOFT_ELEGANT = "ソフトエレガント"
    ELEGANT = "エレガント"
    COOL = "クール"


# MediaPipe Face Mesh 468点の主要インデックス
LANDMARKS = {
    # 輪郭
    "jaw_left": 234,
    "jaw_right": 454,
    "chin": 152,
    "forehead_top": 10,
    # 左目（画像上の右目）
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    # 右目（画像上の左目）
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    # 眉
    "left_eyebrow_inner": 107,
    "left_eyebrow_peak": 105,
    "left_eyebrow_outer": 46,
    "right_eyebrow_inner": 336,
    "right_eyebrow_peak": 334,
    "right_eyebrow_outer": 276,
    # 鼻
    "nose_top": 6,
    "nose_tip": 1,
    "nose_left": 129,
    "nose_right": 358,
    # 口
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
}

# 顎ライン（輪郭の丸み計算用）
JAW_LINE_INDICES = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 152,
    400, 378, 379, 365, 397, 288, 361, 323, 454,
]

# 分類閾値
MATURITY_THRESHOLD = 0.2
LINEARITY_THRESHOLD = 0.2
