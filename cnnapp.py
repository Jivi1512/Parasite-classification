import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.preprocessing.image import img_to_array
import os
import glob

MODEL_PATH="alexnet_model.keras"

CLASS_NAMES=[
    "Babesia_1173", "Leishmania_2701", "Leukocyte_1000X_461",
    "Leukocyte_400X_915", "Plasmodium_843", "RBCs_8995",
    "Toxoplasma_1000X_2933", "Toxoplasma_400X_3758",
    "Trichomonad_10134", "Trypanosome_2385"
]

IMG_SIZE=(224, 224)
DATASET_PATH="Parasite Dataset"

class CompatibleInputLayer(InputLayer):
    def __init__(self, **kwargs):
        if "batch_shape" in kwargs:
            kwargs["batch_input_shape"]=kwargs.pop("batch_shape")
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        if "batch_shape" in config:
            config["batch_input_shape"]=config.pop("batch_shape")
        return cls(**config)

@st.cache_resource
def load_alexnet():
    custom_objects={"InputLayer": CompatibleInputLayer}
    model=load_model(MODEL_PATH, custom_objects=custom_objects)
    return model

@st.cache_data
def load_reference_images():
    reference={}
    for cls in CLASS_NAMES:
        folder=os.path.join(DATASET_PATH, cls)
        if os.path.exists(folder):
            imgs=glob.glob(os.path.join(folder, "*.jpg"))+glob.glob(os.path.join(folder, "*.png"))+glob.glob(os.path.join(folder, "*.jpeg"))
            if imgs:
                reference[cls]=imgs[0]
    return reference

def preprocess(image):
    image=image.convert("RGB")
    image=image.resize(IMG_SIZE)
    arr=img_to_array(image)/255.0
    arr=np.expand_dims(arr, axis=0)
    return arr

st.set_page_config(page_title="Parasite Classifier", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
    .main-title { background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 700; text-align: center; margin-bottom: 0.2rem; }
    .sub-title { color: #94a3b8; text-align: center; font-size: 1rem; font-weight: 300; margin-bottom: 2rem; }
    .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 1.5rem; backdrop-filter: blur(10px); margin-bottom: 1.2rem; }
    .card-title { color: #a78bfa; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.6rem; }
    .predicted-label { background: linear-gradient(90deg, #7c3aed, #2563eb); border-radius: 12px; padding: 1rem 1.5rem; color: white; font-size: 1.4rem; font-weight: 700; text-align: center; margin-bottom: 1rem; }
    .confidence-bar-container { background: rgba(255,255,255,0.08); border-radius: 999px; height: 14px; width: 100%; overflow: hidden; margin-top: 0.4rem; }
    .confidence-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #7c3aed, #60a5fa, #34d399); }
    .confidence-value { color: #34d399; font-size: 2rem; font-weight: 700; text-align: center; }
    .top-k-row { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); color: #cbd5e1; font-size: 0.9rem; }
    .top-k-label { color: #e2e8f0; }
    .top-k-pct { color: #60a5fa; font-weight: 600; }
    .similar-caption { color: #94a3b8; font-size: 0.8rem; text-align: center; margin-top: 0.4rem; }
    .divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0; }
    section[data-testid="stFileUploadDropzone"] { background: rgba(167,139,250,0.07) !important; border: 2px dashed rgba(167,139,250,0.4) !important; border-radius: 12px !important; }
    .stButton > button { background: linear-gradient(90deg, #7c3aed, #2563eb); color: white; border: none; border-radius: 10px; padding: 0.6rem 2rem; font-weight: 600; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Parasite Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AlexNet-powered microscopy image classification across 10 parasite classes</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

model=load_alexnet()
reference_images=load_reference_images()

col_upload, col_results=st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown('<div class="card-title">Upload Microscopy Image</div>', unsafe_allow_html=True)
    uploaded_file=st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image=Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        run_btn=st.button("Classify Image")
    else:
        st.markdown('<div class="card" style="text-align:center; color:#64748b; padding:3rem 1rem;">Upload an image to begin classification</div>', unsafe_allow_html=True)
        run_btn=False

with col_results:
    if uploaded_file and run_btn:
        processed=preprocess(image)
        preds=model.predict(processed)[0]
        top_idx=int(np.argmax(preds))
        top_label=CLASS_NAMES[top_idx]
        confidence=float(preds[top_idx])*100

        st.markdown(f'<div class="predicted-label">{top_label.replace("_", " ")}</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Confidence Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-value">{confidence:.2f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-bar-container"><div class="confidence-bar-fill" style="width:{confidence}%"></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        top5_idx=np.argsort(preds)[::-1][:5]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top 5 Predictions</div>', unsafe_allow_html=True)
        for i in top5_idx:
            pct=float(preds[i])*100
            st.markdown(f'<div class="top-k-row"><span class="top-k-label">{CLASS_NAMES[i].replace("_", " ")}</span><span class="top-k-pct">{pct:.2f}%</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if top_label in reference_images:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Nearest Reference Image</div>', unsafe_allow_html=True)
            ref_img=Image.open(reference_images[top_label])
            st.image(ref_img, use_column_width=True)
            st.markdown(f'<div class="similar-caption">Reference: {top_label.replace("_", " ")}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown('<div class="card" style="text-align:center; color:#475569; padding:4rem 1rem; margin-top:1rem;">Results will appear here after classification</div>', unsafe_allow_html=True)
