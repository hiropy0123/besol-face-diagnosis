"""MediaPipe Tasks API (FaceLandmarker) によるランドマーク検出"""
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# モデルは 478 landmarks (refine_landmarks 相当)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_DIR = Path(__file__).parent.parent / "assets" / "models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def get_face_mesh() -> mp_vision.FaceLandmarker:
    """FaceLandmarker インスタンス（画像モード）を生成"""
    model_path = _ensure_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(image_bgr: np.ndarray, landmarker=None) -> Optional[list]:
    """
    BGR画像から478点ランドマークを検出。
    戻り値は NormalizedLandmark のリスト (.x, .y, .z) または None。
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    owns = landmarker is None
    if owns:
        landmarker = get_face_mesh()
    try:
        result = landmarker.detect(mp_image)
    finally:
        if owns:
            landmarker.close()

    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]
