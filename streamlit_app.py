import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import json

# MUST be the first Streamlit command
st.set_page_config(page_title="🧠 BrainTumorNet", layout="centered")

@st.cache_resource
def load_trained_model():
    return load_model("braintumor.h5")

# Load model and labels once
model = load_trained_model()
with open("class_names.json", "r") as f:
    labels_dict = json.load(f)
labels = [labels_dict[str(i)] for i in range(len(labels_dict))]

# --- Streamlit UI ---

st.title("🧠 BrainTumorNet")
st.markdown("""
Welcome! Upload an MRI scan image, and the model will predict the type of brain tumor (if any) with confidence scores.
""")

uploaded_file = st.file_uploader("Upload MRI image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image in sidebar
    st.sidebar.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)
    st.sidebar.markdown("---")

    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((150, 150))
    img_array = np.array(img_resized)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('🔍 Running prediction...'):
        predictions = model.predict(img_array)[0]

    if not np.isfinite(predictions).all():
        st.error("⚠️ Model returned invalid prediction values. Please retrain or verify the model.")
    else:
        pred_idx = int(np.argmax(predictions))
        pred_label = labels[pred_idx]
        confidence = predictions[pred_idx] * 100

        st.success(f"🎯 **Prediction:** {pred_label.upper()}")
        st.markdown(f"### ✅ Confidence: **{confidence:.2f}%**")

        st.divider()
        st.subheader("📊 Confidence Scores by Class:")

        cols = st.columns(2)
        for i, label in enumerate(labels):
            bar_label = label.replace("_", " ").title()
            progress = float(predictions[i])  # Ensure float64
            col = cols[i % 2]
            col.markdown(f"**{bar_label}**")
            col.progress(progress)

        st.divider()
        st.caption("🧠 Model developed by Shrayash. Best results with clear MRI images.")
else:
    st.info("📤 Please upload an MRI image to start prediction.")
