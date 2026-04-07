"""
顔タイプ診断 Streamlit アプリ
MediaPipe Face Mesh で顔ランドマークを検出し、8タイプに分類する。
"""
import io
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src.classifier import classify
from src.landmarks import get_face_mesh
from src.metrics import extract_metrics
from src.visualizer import (
    create_position_chart,
    create_radar_chart,
    draw_landmarks_on_image,
)

st.set_page_config(page_title="顔タイプ診断", page_icon="👤", layout="wide")

DESCRIPTIONS_PATH = Path(__file__).parent / "assets" / "type_descriptions" / "descriptions.json"


@st.cache_resource
def load_face_mesh():
    return get_face_mesh()


@st.cache_data
def load_descriptions() -> dict:
    if DESCRIPTIONS_PATH.exists():
        with open(DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """EXIF補正 + BGR変換"""
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ============================================================
# サイドバー
# ============================================================
with st.sidebar:
    st.header("⚙️ 設定")

    landmark_mode_label = st.radio(
        "ランドマーク表示",
        ["メッシュ全体", "主要ポイントのみ", "非表示"],
        index=0,
    )
    landmark_mode = {
        "メッシュ全体": "mesh",
        "主要ポイントのみ": "points",
        "非表示": "none",
    }[landmark_mode_label]

    sensitivity = st.slider("診断感度", 0.5, 1.5, 1.0, 0.1)

    st.markdown("---")
    with st.expander("ℹ️ 顔タイプとは？"):
        st.markdown(
            """
            「子供顔⇔大人顔」と「曲線⇔直線」の2軸で顔を分類し、
            似合うファッション・メイクの方向性を導くフレームワークです。

            本アプリは MediaPipe で抽出した顔ランドマークから
            特徴量を数値化し、8タイプに分類します。
            """
        )
    st.caption("🔒 アップロード画像はセッション内のみで処理され、サーバーには保存されません。")

# ============================================================
# メイン
# ============================================================
st.title("👤 顔タイプ診断 - AI Face Type Analysis")
st.caption("MediaPipe Face Mesh による顔ランドマーク解析で、8タイプの顔タイプに分類します。")

tab1, tab2 = st.tabs(["📁 アップロード", "📷 カメラ撮影"])
with tab1:
    uploaded = st.file_uploader(
        "写真をアップロード (jpg/png/webp, 最大10MB)",
        type=["jpg", "jpeg", "png", "webp"],
    )
with tab2:
    camera = st.camera_input("カメラで撮影")

image_source = uploaded or camera

run = st.button("診断する", type="primary", disabled=image_source is None)

if run and image_source is not None:
    with st.spinner("顔を解析中..."):
        pil = Image.open(image_source)
        bgr = pil_to_bgr(pil)
        h, w = bgr.shape[:2]

        landmarker = load_face_mesh()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            st.error("😢 顔が検出できませんでした。正面を向いた、明るくはっきり写った写真を使ってください。")
            st.stop()

        landmarks = result.face_landmarks[0]

        metrics = extract_metrics(landmarks, w, h)
        face_type, metrics = classify(metrics, sensitivity=sensitivity)

        # ランドマーク描画
        overlay_bgr = draw_landmarks_on_image(bgr, landmarks, mode=landmark_mode)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # 結果表示
    st.markdown("---")
    st.subheader("🔍 診断結果")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(overlay_rgb, caption="ランドマーク検出結果", use_container_width=True)

    with col2:
        st.markdown(f"### 顔タイプ：**{face_type.value}**")
        maturity = metrics.maturity_score
        linearity = metrics.linearity_score
        st.metric("子供 ↔ 大人", f"{maturity:+.2f}",
                  "大人寄り" if maturity > 0 else "子供寄り")
        st.metric("曲線 ↔ 直線", f"{linearity:+.2f}",
                  "直線寄り" if linearity > 0 else "曲線寄り")

        descriptions = load_descriptions()
        desc = descriptions.get(face_type.value, {})
        if desc.get("summary"):
            st.info(desc["summary"])

    # チャート
    col3, col4 = st.columns([1, 1])
    with col3:
        st.plotly_chart(create_radar_chart(metrics), use_container_width=True)
    with col4:
        st.plotly_chart(
            create_position_chart(maturity, linearity, face_type),
            use_container_width=True,
        )

    # 詳細データ
    with st.expander("📊 詳細データ"):
        data = metrics.to_dict()
        rows = [
            ("縦横比 (height/width)", f"{data['aspect_ratio']:.3f}"),
            ("目の位置比率", f"{data['eye_position_ratio']:.3f}"),
            ("下顔面比率", f"{data['lower_face_ratio']:.3f}"),
            ("目の幅 / 顔幅", f"{data['eye_width_ratio']:.3f}"),
            ("目の縦横比 (丸み)", f"{data['eye_height_ratio']:.3f}"),
            ("鼻の長さ比", f"{data['nose_length_ratio']:.3f}"),
            ("鼻幅比", f"{data['nose_width_ratio']:.3f}"),
            ("口幅比", f"{data['mouth_width_ratio']:.3f}"),
            ("眉目間距離", f"{data['eyebrow_eye_distance']:.3f}"),
            ("顎の角度 (°)", f"{data['jaw_angle']:.1f}"),
            ("輪郭の丸み", f"{data['face_roundness']:.3f}"),
            ("眉の曲率", f"{data['eyebrow_curvature']:.4f}"),
        ]
        st.dataframe(
            {"特徴量": [r[0] for r in rows], "値": [r[1] for r in rows]},
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "maturity_score は縦横比・目の位置/サイズ・鼻長・下顔面比から、"
            "linearity_score は顎角度・輪郭/目の丸み・眉曲率から算出しています。"
        )

    # タイプ説明
    with st.expander(f"💡 「{face_type.value}」タイプの特徴"):
        if desc:
            if desc.get("impression"):
                st.markdown("**印象キーワード**: " + " / ".join(desc["impression"]))
            if desc.get("fashion_keywords"):
                st.markdown("**似合うファッション**: " + " / ".join(desc["fashion_keywords"]))
            if desc.get("makeup_tips"):
                st.markdown(f"**メイクのポイント**: {desc['makeup_tips']}")
            if desc.get("celebrities"):
                st.markdown("**同タイプの著名人例**: " + " / ".join(desc["celebrities"]))
        else:
            st.write("説明データが見つかりませんでした。")

    # ダウンロード
    result_pil = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button(
        "📥 結果画像をダウンロード",
        data=buf.getvalue(),
        file_name=f"face_type_{face_type.name.lower()}.png",
        mime="image/png",
    )
