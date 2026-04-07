"""classify の分類ロジックテスト"""
from src.classifier import classify
from src.constants import FaceType
from src.metrics import FaceMetrics


def _base_metrics(**overrides) -> FaceMetrics:
    defaults = dict(
        face_width=100.0,
        face_height=135.0,
        aspect_ratio=1.35,
        eye_position_ratio=0.45,
        forehead_ratio=0.45,
        lower_face_ratio=0.50,
        eye_width_ratio=0.24,
        eye_height_ratio=0.30,
        nose_length_ratio=0.27,
        mouth_width_ratio=0.40,
        eyebrow_eye_distance=1.2,
        jaw_angle=120.0,
        face_roundness=0.80,
        eye_roundness=0.30,
        eyebrow_curvature=0.010,
        nose_width_ratio=0.25,
    )
    defaults.update(overrides)
    return FaceMetrics(**defaults)


def test_cool_classification():
    """大人顔×直線 → クール"""
    m = _base_metrics(
        aspect_ratio=1.60,
        eye_position_ratio=0.50,
        eye_width_ratio=0.20,
        nose_length_ratio=0.33,
        lower_face_ratio=0.60,
        jaw_angle=100.0,
        face_roundness=0.60,
        eye_roundness=0.22,
        eyebrow_curvature=0.005,
    )
    face_type, m = classify(m)
    assert m.maturity_score > 0
    assert m.linearity_score > 0
    assert face_type == FaceType.COOL


def test_cute_classification():
    """子供顔×曲線 → キュート"""
    m = _base_metrics(
        aspect_ratio=1.20,
        eye_position_ratio=0.40,
        eye_width_ratio=0.28,
        nose_length_ratio=0.22,
        lower_face_ratio=0.45,
        jaw_angle=140.0,
        face_roundness=0.90,
        eye_roundness=0.40,
        eyebrow_curvature=0.020,
    )
    face_type, m = classify(m)
    assert m.maturity_score < 0
    assert m.linearity_score < 0
    assert face_type == FaceType.CUTE


def test_elegant_neutral():
    """中間的メトリクス → エレガント付近"""
    m = _base_metrics()
    face_type, _ = classify(m)
    assert face_type in {FaceType.ELEGANT, FaceType.SOFT_ELEGANT, FaceType.COOL_CASUAL}
