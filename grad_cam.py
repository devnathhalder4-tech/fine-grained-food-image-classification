from __future__ import annotations

import streamlit as st
from PIL import Image

from predict import load_checkpoint, predict_image
from src.utils import device_from_arg


st.set_page_config(page_title="Food-101 1B Classifier", layout="centered")
st.title("Food-101 1B 亚洲主食分类")

checkpoint = st.sidebar.text_input("模型权重路径", "outputs/resnet18/best.pt")
device_name = st.sidebar.selectbox("设备", ["auto", "cpu", "cuda"], index=0)
topk = st.sidebar.slider("Top-K", 1, 5, 3)
uploaded = st.file_uploader("上传食物图片", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="输入图片", use_container_width=True)
    try:
        device = device_from_arg(device_name)
        model, ckpt = load_checkpoint(checkpoint, device)
        results = predict_image(model, ckpt["classes"], image, ckpt.get("image_size", 224), device, topk)
        st.subheader("预测结果")
        for label, score in results:
            st.progress(score, text=f"{label}: {score:.2%}")
    except Exception as exc:
        st.error(f"预测失败：{exc}")
