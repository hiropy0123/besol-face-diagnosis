"""結果の可視化: ランドマーク描画・レーダーチャート・ポジションチャート"""
import cv2
import numpy as np
import plotly.graph_objects as go
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from .constants import FaceType, LANDMARKS
from .metrics import FaceMetrics


def _connection_pairs(connection_set) -> list[tuple[int, int]]:
    """FaceLandmarksConnections の Connection リストから (start, end) タプルを抽出"""
    pairs = []
    for c in connection_set:
        start = getattr(c, "start", None)
        end = getattr(c, "end", None)
        if start is None:
            # タプル/リストの場合
            start, end = c[0], c[1]
        pairs.append((start, end))
    return pairs


_TESSELATION = _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)
_CONTOURS = (
    _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_NOSE)
)


def draw_landmarks_on_image(
    image: np.ndarray, landmarks, mode: str = "mesh"
) -> np.ndarray:
    """
    mode: "mesh" | "points" | "none"
    landmarks: NormalizedLandmark のリスト ([.x, .y] は 0〜1 正規化)
    """
    out = image.copy()
    if mode == "none" or landmarks is None:
        return out

    h, w = out.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    n = len(pts)

    if mode == "mesh":
        # テッセレーション線を薄く
        for s, e in _TESSELATION:
            if s < n and e < n:
                cv2.line(out, pts[s], pts[e], (180, 180, 180), 1, cv2.LINE_AA)
        # 輪郭・目・眉・口・鼻を強調
        for s, e in _CONTOURS:
            if s < n and e < n:
                cv2.line(out, pts[s], pts[e], (0, 255, 120), 1, cv2.LINE_AA)
    elif mode == "points":
        for name, idx in LANDMARKS.items():
            if idx < n:
                cv2.circle(out, pts[idx], 3, (0, 255, 0), -1)
    return out


def create_radar_chart(metrics: FaceMetrics) -> go.Figure:
    categories = [
        "縦横比", "目の大きさ", "目の丸み", "顎の鋭さ", "輪郭の丸み", "眉の曲率",
    ]
    values = [
        min(max((metrics.aspect_ratio - 1.1) / 0.5, 0), 1),
        min(max((metrics.eye_width_ratio - 0.18) / 0.12, 0), 1),
        min(max((metrics.eye_roundness - 0.15) / 0.3, 0), 1),
        min(max((140 - metrics.jaw_angle) / 40, 0), 1),
        min(max(metrics.face_roundness, 0), 1),
        min(max(metrics.eyebrow_curvature / 0.02, 0), 1),
    ]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values, theta=categories, fill="toself", name="特徴量",
            line=dict(color="#ff6b9d"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="特徴量レーダーチャート",
        height=400,
    )
    return fig


_TYPE_POSITIONS = {
    FaceType.CUTE: (-0.6, -0.6),
    FaceType.FRESH: (0.0, -0.6),
    FaceType.ACTIVE_CUTE: (0.6, -0.6),
    FaceType.FEMININE: (-0.6, 0.6),
    FaceType.SOFT_ELEGANT: (-0.3, 0.3),
    FaceType.ELEGANT: (0.0, 0.0),
    FaceType.COOL_CASUAL: (0.3, 0.0),
    FaceType.COOL: (0.6, 0.6),
}


def create_position_chart(maturity: float, linearity: float, face_type: FaceType) -> go.Figure:
    fig = go.Figure()
    xs, ys, texts = [], [], []
    for t, (x, y) in _TYPE_POSITIONS.items():
        xs.append(x)
        ys.append(y)
        texts.append(t.value)
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text", text=texts, textposition="top center",
            marker=dict(size=12, color="lightgray"), name="タイプ",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[linearity], y=[maturity], mode="markers+text",
            text=[f"★ {face_type.value}"], textposition="bottom center",
            marker=dict(size=22, color="#ff1744", symbol="star"),
            name="あなた",
        )
    )
    fig.update_layout(
        title="8タイプ マトリクス",
        xaxis=dict(title="曲線 ← → 直線", range=[-1, 1], zeroline=True),
        yaxis=dict(title="子供 ← → 大人", range=[-1, 1], zeroline=True),
        height=400,
        showlegend=False,
    )
    return fig
