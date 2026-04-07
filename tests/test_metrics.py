"""extract_metrics の簡易テスト（ダミーランドマークで動作確認）"""
from dataclasses import dataclass

from src.metrics import extract_metrics


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0


def _make_landmarks(n: int = 478) -> list:
    # 顔中心 (0.5, 0.5) 付近のダミー座標で埋める
    return [_LM(0.5, 0.5) for _ in range(n)]


def _place(lms, idx, x, y):
    lms[idx] = _LM(x, y)


def test_extract_metrics_runs_with_dummy():
    lms = _make_landmarks()
    # 最低限のポイントを配置
    _place(lms, 234, 0.3, 0.5)   # jaw_left
    _place(lms, 454, 0.7, 0.5)   # jaw_right
    _place(lms, 152, 0.5, 0.85)  # chin
    _place(lms, 10, 0.5, 0.15)   # forehead_top

    # 目
    _place(lms, 33, 0.35, 0.45)   # left_eye_outer
    _place(lms, 133, 0.45, 0.45)  # left_eye_inner
    _place(lms, 159, 0.4, 0.43)   # left_eye_top
    _place(lms, 145, 0.4, 0.47)   # left_eye_bottom
    _place(lms, 263, 0.65, 0.45)  # right_eye_outer
    _place(lms, 362, 0.55, 0.45)  # right_eye_inner
    _place(lms, 386, 0.6, 0.43)   # right_eye_top
    _place(lms, 374, 0.6, 0.47)   # right_eye_bottom

    m = extract_metrics(lms, img_w=1000, img_h=1000)

    assert m.face_width > 0
    assert m.face_height > 0
    assert 0.5 < m.aspect_ratio < 3.0
    assert 0.0 <= m.eye_width_ratio <= 1.0
    assert 0.0 <= m.face_roundness <= 1.0
