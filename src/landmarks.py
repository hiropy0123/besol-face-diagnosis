"""MediaPipe solutions.face_mesh によるランドマーク検出 (478点, refine)"""
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

_mp_face_mesh = mp.solutions.face_mesh


class _FaceMeshWrapper:
    """Tasks API 互換のインターフェースを提供するラッパー"""

    def __init__(self, face_mesh):
        self._fm = face_mesh

    def detect(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)
        lm_list = []
        if res.multi_face_landmarks:
            lm_list = [res.multi_face_landmarks[0].landmark]
        return type("Result", (), {"face_landmarks": lm_list})()

    def close(self):
        self._fm.close()


def get_face_mesh() -> _FaceMeshWrapper:
    fm = _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return _FaceMeshWrapper(fm)


def detect_landmarks(image_bgr: np.ndarray, landmarker=None) -> Optional[list]:
    """BGR画像から478点ランドマークを検出。戻り値は NormalizedLandmark のリスト。"""
    owns = landmarker is None
    if owns:
        landmarker = get_face_mesh()
    try:
        result = landmarker.detect(image_bgr)
    finally:
        if owns:
            landmarker.close()
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]
