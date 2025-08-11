import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

st.set_page_config(page_title="AI Image Classifier", page_icon="📷", layout="centered")

st.title("📷 AI Image Classifier")
st.write("तुमचा फोटो upload करा आणि AI सांगेल त्यात काय आहे.")

@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

model, feature_extractor = load_model()

uploaded_file = st.file_uploader("फोटो Upload करा", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict करा"):
        with st.spinner("AI फोटो analyse करत आहे..."):
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]
        st.success(f"📌 AI Prediction: **{predicted_label}**")
